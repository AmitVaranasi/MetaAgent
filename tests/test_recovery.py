"""Startup reaping, workflow failure, and draining on shutdown.

Nothing in the shipped system could mark a workflow failed or reconcile tasks
left behind by a dead process: 21 of 33 workflows and 13 tasks in the shipped
database are stranded for exactly those two reasons.
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from meta_agent.agent_manager import AgentManager
from meta_agent.db import Database
from meta_agent.models import AgentConfig, AgentStatus, Task, Workflow, WorkflowStatus

DEAD_PID = 2_000_000  # above every platform's pid_max


@pytest.fixture()
def manager(db: Database, config) -> AgentManager:
    mgr = AgentManager(db, config.log_dir)
    mgr.start()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def agent_config() -> AgentConfig:
    return AgentConfig(id="rec_test", name="Recovery Test", system_prompt="x", allowed_tools=[])


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --- reaping orphans ---


@pytest.mark.parametrize("status", ["pending", "running"])
def test_reaper_fails_tasks_whose_owner_process_is_gone(manager: AgentManager, status: str):
    manager.db.save_task(Task(id="orphan", agent_id="a", prompt="p", status=status, owner_pid=DEAD_PID))

    assert manager.reap_orphaned_tasks() == ["orphan"]

    task = manager.get_task("orphan")
    assert task.status == "failed"
    assert "Orphaned" in task.error
    assert str(DEAD_PID) in task.error
    assert task.completed_at is not None


def test_reaper_leaves_tasks_owned_by_a_live_process_alone(manager: AgentManager):
    """Every CLI command builds a manager, so reaping must not disturb tasks
    another running process still owns."""
    manager.db.save_task(
        Task(id="live", agent_id="a", prompt="p", status="running", owner_pid=os.getpid())
    )

    assert manager.reap_orphaned_tasks() == []
    assert manager.get_task("live").status == "running"


def test_reaper_leaves_waiting_for_input_alone(manager: AgentManager):
    """waiting_for_input survives a restart on purpose — session_id lets a later
    process resume the conversation."""
    manager.db.save_task(
        Task(
            id="parked",
            agent_id="a",
            prompt="p",
            status="waiting_for_input",
            session_id="sess-1",
            owner_pid=DEAD_PID,
        )
    )

    assert manager.reap_orphaned_tasks() == []
    assert manager.get_task("parked").status == "waiting_for_input"


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_reaper_leaves_terminal_tasks_alone(manager: AgentManager, status: str):
    manager.db.save_task(Task(id="done", agent_id="a", prompt="p", status=status, owner_pid=DEAD_PID))
    assert manager.reap_orphaned_tasks() == []
    assert manager.get_task("done").status == status


def test_start_reaps(db: Database, config):
    db.save_task(Task(id="orphan", agent_id="a", prompt="p", status="running", owner_pid=DEAD_PID))

    mgr = AgentManager(db, config.log_dir)
    mgr.start()
    try:
        assert mgr.get_task("orphan").status == "failed"
    finally:
        mgr.shutdown()


def test_submitted_tasks_record_this_process_as_owner(
    manager: AgentManager, agent_config: AgentConfig
):
    manager.register_agent(agent_config)

    async def fake_query(**kwargs):
        return
        yield

    with patch("meta_agent.agent_runner.query", side_effect=fake_query):
        task = manager.submit_task("rec_test", "do something")
        assert _wait_for(lambda: manager.get_task(task.id).status == "completed")

    assert manager.get_task(task.id).owner_pid == os.getpid()


# --- workflow failure ---


def _workflow(manager: AgentManager, **kw) -> Workflow:
    wf = Workflow(prompt="p", brain_agent_id="rec_test", **kw)
    manager.db.save_workflow(wf)
    return wf


def test_a_failing_brain_task_fails_its_workflow(
    manager: AgentManager, agent_config: AgentConfig
):
    """WorkflowStatus.FAILED was only ever reachable through the Brain's own
    update_workflow tool, so a dead Brain stranded its workflow forever."""
    manager.register_agent(agent_config)
    wf = _workflow(manager, status=WorkflowStatus.EXECUTING)

    async def failing_query(**kwargs):
        raise RuntimeError("brain died")
        yield

    with patch("meta_agent.agent_runner.query", side_effect=failing_query):
        task = manager.submit_task("rec_test", "orchestrate", workflow_id=wf.id)
        assert _wait_for(lambda: manager.get_task(task.id).status == "failed")

    stored = manager.db.get_workflow(wf.id)
    assert stored.status == WorkflowStatus.FAILED
    assert "brain died" in stored.error
    assert stored.completed_at is not None


def test_a_failing_subtask_does_not_fail_the_workflow(
    manager: AgentManager, agent_config: AgentConfig
):
    """Phase 4 tells the Brain to retry a failed subtask with an adjusted
    prompt, so one subtask dying must not tear the workflow down."""
    manager.register_agent(agent_config)
    worker = AgentConfig(id="worker", name="Worker", system_prompt="x", allowed_tools=[])
    manager.register_agent(worker)
    wf = _workflow(manager, status=WorkflowStatus.EXECUTING)

    async def failing_query(**kwargs):
        raise RuntimeError("subtask died")
        yield

    with patch("meta_agent.agent_runner.query", side_effect=failing_query):
        task = manager.submit_task("worker", "a subtask", workflow_id=wf.id)
        assert _wait_for(lambda: manager.get_task(task.id).status == "failed")

    assert manager.db.get_workflow(wf.id).status == WorkflowStatus.EXECUTING


def test_a_completed_workflow_is_not_reopened_by_a_late_failure(
    manager: AgentManager, agent_config: AgentConfig
):
    manager.register_agent(agent_config)
    wf = _workflow(manager, status=WorkflowStatus.COMPLETED, result="ok")

    async def failing_query(**kwargs):
        raise RuntimeError("late failure")
        yield

    with patch("meta_agent.agent_runner.query", side_effect=failing_query):
        task = manager.submit_task("rec_test", "orchestrate", workflow_id=wf.id)
        assert _wait_for(lambda: manager.get_task(task.id).status == "failed")

    stored = manager.db.get_workflow(wf.id)
    assert stored.status == WorkflowStatus.COMPLETED
    assert stored.result == "ok"


def test_reaping_a_brain_task_also_fails_its_workflow(manager: AgentManager):
    wf = _workflow(manager, status=WorkflowStatus.PLANNING)
    manager.db.save_task(
        Task(
            id="orphan",
            agent_id="rec_test",
            prompt="p",
            status="running",
            workflow_id=wf.id,
            owner_pid=DEAD_PID,
        )
    )

    manager.reap_orphaned_tasks()

    assert manager.db.get_workflow(wf.id).status == WorkflowStatus.FAILED


# --- draining on shutdown ---


def test_shutdown_cancels_in_flight_tasks_and_records_them(
    db: Database, config, agent_config: AgentConfig
):
    """shutdown() used to call loop.stop() and return, so tasks died mid-write."""
    import asyncio

    mgr = AgentManager(db, config.log_dir)
    mgr.start()
    mgr.register_agent(agent_config)
    started = threading.Semaphore(0)

    async def never_finishes(**kwargs):
        started.release()
        await asyncio.sleep(30)
        return
        yield

    with patch("meta_agent.agent_runner.query", side_effect=never_finishes):
        task = mgr.submit_task("rec_test", "do something")
        assert started.acquire(timeout=5)

        mgr.shutdown()

    assert mgr.get_task(task.id).status == "cancelled"
    assert mgr._runs == {}
    assert not mgr._loop_thread.is_alive()


def test_shutdown_is_idempotent(manager: AgentManager):
    manager.shutdown()
    manager.shutdown()  # must not raise
