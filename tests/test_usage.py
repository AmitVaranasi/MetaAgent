"""Cost/usage capture, and treating an errored run as a failure.

The SDK reports total_cost_usd, usage, num_turns and stop_reason on every run.
All four were discarded, so there was no way to answer what a workflow cost —
the question the Opus-plans/Sonnet-executes design exists to answer. is_error
was ignored too, so a run that exhausted max_turns was stored as a success.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from meta_agent.agent_manager import AgentManager
from meta_agent.agent_runner import AgentRunError, AgentRunner
from meta_agent.db import Database
from meta_agent.models import AgentConfig, Task, Workflow, WorkflowStatus


def _result(**kw) -> ResultMessage:
    base = dict(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=3,
        session_id="sess-1",
        result="done",
        total_cost_usd=0.0125,
        usage={"input_tokens": 1200, "output_tokens": 340},
    )
    base.update(kw)
    return ResultMessage(**base)


def _fake_query(messages):
    async def _iter(**kwargs):
        for m in messages:
            yield m

    return _iter


@pytest.fixture()
def manager(db: Database, config) -> AgentManager:
    mgr = AgentManager(db, config.log_dir)
    mgr.start()
    yield mgr
    mgr.shutdown()


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --- the runner reads the ResultMessage ---


@pytest.mark.asyncio
async def test_runner_captures_cost_and_usage():
    runner = AgentRunner(AgentConfig(id="r", name="R", system_prompt="x", allowed_tools=[]))
    task = Task(agent_id="r", prompt="p")
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query([_result()])):
        await runner.run_task(task)

    assert runner.stats.cost_usd == 0.0125
    assert runner.stats.num_turns == 3
    assert runner.stats.usage == {"input_tokens": 1200, "output_tokens": 340}


@pytest.mark.asyncio
async def test_error_context_reports_turns_and_cost():
    runner = AgentRunner(AgentConfig(id="r", name="R", system_prompt="x", allowed_tools=[]))
    task = Task(agent_id="r", prompt="p")
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query([_result()])):
        await runner.run_task(task)

    ctx = runner.get_error_context()
    assert "turns=3" in ctx
    assert "cost_usd=0.0125" in ctx


@pytest.mark.asyncio
async def test_an_errored_result_raises_instead_of_returning_a_partial_answer():
    """max_turns exhaustion ended the loop normally, so the task was stored
    `completed` with truncated work and the Brain assembled on it."""
    runner = AgentRunner(AgentConfig(id="r", name="R", system_prompt="x", allowed_tools=[]))
    task = Task(agent_id="r", prompt="p")
    messages = [
        AssistantMessage(content=[TextBlock(text="half an answer")], model="claude-sonnet-5"),
        _result(subtype="error_max_turns", is_error=True, result=None, stop_reason="max_turns"),
    ]
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query(messages)):
        with pytest.raises(AgentRunError) as excinfo:
            await runner.run_task(task)

    assert excinfo.value.subtype == "error_max_turns"
    assert excinfo.value.stop_reason == "max_turns"
    # the partial work survives the failure
    assert excinfo.value.partial_result == "half an answer"
    # and the stats are still captured
    assert runner.stats.cost_usd == 0.0125


# --- the manager stores it ---


def test_completed_task_records_model_cost_and_usage(manager: AgentManager):
    manager.register_agent(
        AgentConfig(id="u1", name="U", system_prompt="x", allowed_tools=[], model="claude-sonnet-5")
    )
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query([_result()])):
        task = manager.submit_task("u1", "do something")
        assert _wait_for(lambda: manager.get_task(task.id).status == "completed")

    stored = manager.get_task(task.id)
    assert stored.model == "claude-sonnet-5"
    assert stored.cost_usd == 0.0125
    assert stored.num_turns == 3
    assert stored.usage["input_tokens"] == 1200


def test_max_turns_is_recorded_as_a_failure_keeping_the_partial_result(manager: AgentManager):
    manager.register_agent(AgentConfig(id="u2", name="U", system_prompt="x", allowed_tools=[]))
    messages = [
        AssistantMessage(content=[TextBlock(text="half an answer")], model="claude-sonnet-5"),
        _result(subtype="error_max_turns", is_error=True, result=None, stop_reason="max_turns"),
    ]
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query(messages)):
        task = manager.submit_task("u2", "do something")
        assert _wait_for(lambda: manager.get_task(task.id).status == "failed")

    stored = manager.get_task(task.id)
    assert "error_max_turns" in stored.error
    assert stored.result == "half an answer"
    # spend is recorded even though the run failed
    assert stored.cost_usd == 0.0125
    assert stored.stop_reason == "max_turns"


# --- per-workflow aggregation ---


def test_workflow_usage_totals_and_splits_by_model(manager: AgentManager):
    wf = Workflow(prompt="p", brain_agent_id="brain", status=WorkflowStatus.EXECUTING)
    manager.db.save_workflow(wf)
    for tid, model, cost, tokens in [
        ("t1", "claude-opus-5", 0.40, 5000),
        ("t2", "claude-sonnet-5", 0.05, 2000),
        ("t3", "claude-sonnet-5", 0.03, 1000),
    ]:
        manager.db.save_task(
            Task(
                id=tid,
                agent_id="a",
                prompt="p",
                status="completed",
                workflow_id=wf.id,
                model=model,
                cost_usd=cost,
                num_turns=2,
                usage={"input_tokens": tokens, "output_tokens": 100},
            )
        )

    usage = manager.workflow_usage(wf.id)

    assert usage["task_count"] == 3
    assert usage["totals"]["cost_usd"] == pytest.approx(0.48)
    assert usage["totals"]["input_tokens"] == 8000
    assert usage["totals"]["num_turns"] == 6
    assert usage["by_model"]["claude-sonnet-5"]["tasks"] == 2
    assert usage["by_model"]["claude-sonnet-5"]["cost_usd"] == pytest.approx(0.08)
    assert usage["by_model"]["claude-opus-5"]["cost_usd"] == pytest.approx(0.40)


def test_workflow_usage_counts_tasks_the_brain_forgot_to_register(manager: AgentManager):
    """Workflow.subtask_ids only holds what the Brain remembered to add, so the
    tally reads the tasks table instead."""
    wf = Workflow(prompt="p", brain_agent_id="brain")
    manager.db.save_workflow(wf)
    manager.db.save_task(
        Task(id="unregistered", agent_id="a", prompt="p", workflow_id=wf.id, cost_usd=0.10)
    )

    assert wf.subtask_ids == []
    assert manager.workflow_usage(wf.id)["totals"]["cost_usd"] == pytest.approx(0.10)


def test_workflow_usage_of_an_empty_workflow_is_zero(manager: AgentManager):
    wf = Workflow(prompt="p", brain_agent_id="brain")
    manager.db.save_workflow(wf)
    usage = manager.workflow_usage(wf.id)
    assert usage["task_count"] == 0
    assert usage["totals"]["cost_usd"] == 0.0
    assert usage["by_model"] == {}
