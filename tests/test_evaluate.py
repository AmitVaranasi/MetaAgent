"""Tests for the evaluation harness — no real API calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from claude_agent_sdk import ResultMessage
from click.testing import CliRunner

from meta_agent.agent_manager import AgentManager
from meta_agent.cli import main
from meta_agent.config import Config
from meta_agent.db import Database
from meta_agent.evaluate import (
    EvalTask,
    RunResult,
    format_report,
    load_tasks,
    run_suite,
    summarise,
)


@pytest.fixture()
def manager(db: Database, config) -> AgentManager:
    mgr = AgentManager(db, config.log_dir)
    mgr.retry_base_delay_s = 0.0
    mgr.start()
    yield mgr
    mgr.shutdown()


def _answering(text: str, cost: float = 0.01):
    async def _query(**kwargs):
        yield ResultMessage(
            subtype="success", duration_ms=5, duration_api_ms=4, is_error=False,
            num_turns=1, session_id="s", result=text, total_cost_usd=cost,
            usage={"input_tokens": 100, "output_tokens": 20},
        )

    return _query


TASKS = [
    EvalTask(id="t1", prompt="say the magic word", expect=["banana"]),
    EvalTask(id="t2", prompt="anything", expect=[]),
]


# --- task loading ---


@pytest.mark.parametrize("name", ["tasks.json", "tasks-readonly.json"])
def test_the_shipped_task_sets_load(name: str):
    tasks = load_tasks(Path("evals") / name)
    assert len(tasks) >= 4
    assert all(t.id and t.prompt for t in tasks)
    assert all(t.expect for t in tasks)


def test_load_tasks_from_a_file(tmp_path: Path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps([{"id": "a", "prompt": "p", "expect": ["x"]}]))
    tasks = load_tasks(path)
    assert tasks == [EvalTask(id="a", prompt="p", expect=["x"])]


def test_expect_defaults_to_empty(tmp_path: Path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps([{"id": "a", "prompt": "p"}]))
    assert load_tasks(path)[0].expect == []


# --- scoring ---


def test_a_run_passes_only_when_every_expectation_appears(manager: AgentManager):
    with patch("meta_agent.agent_runner.query", side_effect=_answering("the word is BANANA")):
        report = run_suite(manager, TASKS, arms=("solo",))

    by_task = {r["task_id"]: r for r in report["runs"]}
    assert by_task["t1"]["passed"] is True  # matched case-insensitively
    assert by_task["t2"]["passed"] is True  # no expectations, status alone


def test_a_run_fails_when_an_expectation_is_missing(manager: AgentManager):
    with patch("meta_agent.agent_runner.query", side_effect=_answering("no fruit here")):
        report = run_suite(manager, [TASKS[0]], arms=("solo",))

    assert report["runs"][0]["passed"] is False
    assert report["runs"][0]["status"] == "completed"


def test_a_failed_run_never_passes(manager: AgentManager):
    async def boom(**kwargs):
        raise RuntimeError("sdk died")
        yield

    with patch("meta_agent.agent_runner.query", side_effect=boom):
        report = run_suite(manager, [TASKS[1]], arms=("solo",))

    run = report["runs"][0]
    assert run["status"] == "failed" and run["passed"] is False
    assert "sdk died" in run["error"]


# --- arms ---


def test_every_task_runs_once_per_arm(manager: AgentManager):
    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana")):
        report = run_suite(manager, TASKS, arms=("brain", "solo"))

    assert len(report["runs"]) == 4
    assert sorted(r["arm"] for r in report["runs"]) == ["brain", "brain", "solo", "solo"]
    assert set(report["summary"]) == {"brain", "solo"}


def test_each_arm_records_its_own_cost(manager: AgentManager):
    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana", cost=0.02)):
        report = run_suite(manager, [TASKS[0]], arms=("brain", "solo"))

    for run in report["runs"]:
        assert run["cost_usd"] == pytest.approx(0.02)
        assert run["input_tokens"] == 100
        assert run["seconds"] >= 0


def test_the_brain_arm_reports_the_whole_workflow_cost(manager: AgentManager):
    """The brain arm must count its sub-agents' spend, not just its own turn."""
    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana", cost=0.05)):
        report = run_suite(manager, [TASKS[1]], arms=("brain",))

    # one brain task in the workflow here, but the figure comes from
    # workflow_usage, which sums every task carrying the workflow id
    assert report["runs"][0]["cost_usd"] == pytest.approx(0.05)


# --- summary ---


def test_summarise_computes_rates_and_means():
    runs = [
        RunResult("t1", "brain", "completed", True, 10.0, 0.40, 500, 50),
        RunResult("t2", "brain", "completed", False, 20.0, 0.60, 500, 50),
        RunResult("t1", "solo", "completed", True, 5.0, 0.10, 200, 20),
        RunResult("t2", "solo", "failed", False, 5.0, 0.10, 200, 20),
    ]
    summary = summarise(runs)

    assert summary["brain"]["pass_rate"] == 0.5
    assert summary["brain"]["total_cost_usd"] == pytest.approx(1.0)
    assert summary["brain"]["mean_seconds"] == 15.0
    assert summary["solo"]["total_cost_usd"] == pytest.approx(0.20)
    assert summary["solo"]["input_tokens"] == 400


def test_summarise_skips_arms_that_did_not_run():
    runs = [RunResult("t1", "solo", "completed", True, 1.0, 0.1, 10, 1)]
    assert set(summarise(runs)) == {"solo"}


def test_format_report_shows_both_arms_and_every_run():
    runs = [
        RunResult("t1", "brain", "completed", True, 10.0, 0.40, 500, 50),
        RunResult("t1", "solo", "completed", False, 5.0, 0.10, 200, 20),
    ]
    text = format_report({"runs": [r.__dict__ for r in runs], "summary": summarise(runs)})
    assert "brain" in text and "solo" in text
    assert "t1" in text
    assert "0.4000" in text and "0.1000" in text


# --- the CLI wrapper ---


def test_eval_command_refuses_an_unknown_arm(tmp_path: Path):
    Config.reset()
    result = CliRunner().invoke(
        main, ["--data-dir", str(tmp_path), "eval", "--arms", "wizard", "--yes"]
    )
    assert result.exit_code == 1
    assert "Unknown arm" in result.output


def test_eval_command_asks_before_spending(tmp_path: Path):
    Config.reset()
    result = CliRunner().invoke(main, ["--data-dir", str(tmp_path), "eval"], input="n\n")
    assert result.exit_code == 0
    # rich wraps the warning, so match a fragment that cannot break across lines
    assert "real agent session(s)" in result.output
    assert "Aborted" in result.output


def test_eval_command_writes_a_report(tmp_path: Path):
    Config.reset()
    tasks = tmp_path / "t.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p", "expect": ["banana"]}]))
    out = tmp_path / "report.json"

    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana")):
        result = CliRunner().invoke(
            main,
            ["--data-dir", str(tmp_path), "eval", "--tasks", str(tasks),
             "--arms", "solo", "--out", str(out), "--yes"],
        )

    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["summary"]["solo"]["passed"] == 1


# --- delegation tracking ---


def test_the_solo_arm_never_reports_delegation(manager: AgentManager):
    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana")):
        report = run_suite(manager, [TASKS[0]], arms=("solo",))
    assert report["runs"][0]["subtasks"] == 0
    assert report["runs"][0]["agents_used"] == 1


def test_the_brain_arm_counts_the_subtasks_it_created(manager: AgentManager):
    """Without this the brain arm can answer everything itself — it holds
    Read/Glob/Grep — and the comparison silently measures one agent against
    one agent."""
    from meta_agent.brain import BRAIN_AGENT_ID
    from meta_agent.models import AgentConfig

    manager.register_agent(
        AgentConfig(id="worker", name="W", system_prompt="x", allowed_tools=[])
    )
    real_submit = manager.submit_task
    spawned: list[str] = []

    def submit_and_delegate(agent_id, prompt, **kwargs):
        task = real_submit(agent_id, prompt, **kwargs)
        # emulate the Brain delegating once, into the same workflow
        if agent_id == BRAIN_AGENT_ID and not spawned:
            spawned.append("x")
            real_submit("worker", "a subtask", workflow_id=kwargs.get("workflow_id"))
        return task

    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana")):
        with patch.object(manager, "submit_task", side_effect=submit_and_delegate):
            report = run_suite(manager, [TASKS[0]], arms=("brain",))

    run = report["runs"][0]
    assert run["subtasks"] == 1
    assert run["agents_used"] == 2
    assert report["summary"]["brain"]["runs_that_delegated"] == 1


def test_the_report_warns_when_the_brain_never_delegated(manager: AgentManager):
    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana")):
        report = run_suite(manager, [TASKS[0]], arms=("brain",))

    assert report["summary"]["brain"]["runs_that_delegated"] == 0
    text = format_report(report)
    assert "delegated on zero tasks" in text
    assert "not" in text and "orchestration" in text


def test_the_report_has_no_warning_when_only_solo_ran(manager: AgentManager):
    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana")):
        report = run_suite(manager, [TASKS[0]], arms=("solo",))
    assert "delegated on zero tasks" not in format_report(report)


def test_the_shipped_default_task_set_requires_writing_files():
    """A read-only task set cannot test the premise: the Brain answers it alone."""
    tasks = load_tasks(Path("evals/tasks.json"))
    assert len(tasks) >= 4
    for task in tasks:
        assert "creat" in task.prompt.lower(), f"{task.id} does not require producing files"


def test_the_brain_arm_counts_work_filed_under_a_workflow_it_invented(manager: AgentManager):
    """Observed live: given a workflow ID in its prompt, the Brain called
    create_workflow anyway and filed all four sub-agents under its own record.
    Reading workflow_usage(our id) then reported the Brain's turn alone and
    missed every sub-agent's spend."""
    from meta_agent.brain import BRAIN_AGENT_ID
    from meta_agent.models import AgentConfig, Workflow

    manager.register_agent(AgentConfig(id="worker", name="W", system_prompt="x", allowed_tools=[]))
    real_submit = manager.submit_task
    spawned: list[str] = []

    def submit_and_delegate(agent_id, prompt, **kwargs):
        task = real_submit(agent_id, prompt, **kwargs)
        if agent_id == BRAIN_AGENT_ID and not spawned:
            spawned.append("x")
            rogue = Workflow(prompt="the Brain's own", brain_agent_id=BRAIN_AGENT_ID)
            manager.db.save_workflow(rogue)
            real_submit("worker", "a subtask", workflow_id=rogue.id)
        return task

    with patch("meta_agent.agent_runner.query", side_effect=_answering("banana", cost=0.03)):
        with patch.object(manager, "submit_task", side_effect=submit_and_delegate):
            report = run_suite(manager, [TASKS[0]], arms=("brain",))

    run = report["runs"][0]
    assert run["subtasks"] == 1, "the sub-agent's task was not counted"
    assert run["cost_usd"] == pytest.approx(0.06), "sub-agent spend was dropped"
