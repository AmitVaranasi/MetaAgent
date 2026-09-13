"""Tests for the agent manager."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from meta_agent.agent_manager import AgentManager
from meta_agent.db import Database
from meta_agent.models import AgentConfig, AgentStatus


@pytest.fixture()
def manager(db: Database, config) -> AgentManager:
    mgr = AgentManager(db, config.log_dir)
    mgr.start()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def agent_config() -> AgentConfig:
    return AgentConfig(
        id="mgr_test",
        name="Manager Test",
        system_prompt="You are a test.",
        allowed_tools=[],
    )


def test_register_agent(manager: AgentManager, agent_config: AgentConfig):
    state = manager.register_agent(agent_config)
    assert state.config.id == "mgr_test"
    assert state.status == AgentStatus.STOPPED


def test_list_agents(manager: AgentManager, agent_config: AgentConfig):
    manager.register_agent(agent_config)
    agents = manager.list_agents()
    assert len(agents) == 1


def test_get_agent(manager: AgentManager, agent_config: AgentConfig):
    manager.register_agent(agent_config)
    state = manager.get_agent("mgr_test")
    assert state is not None
    assert state.config.name == "Manager Test"


def test_get_nonexistent_agent(manager: AgentManager):
    assert manager.get_agent("nope") is None


def test_unregister_agent(manager: AgentManager, agent_config: AgentConfig):
    manager.register_agent(agent_config)
    assert manager.unregister_agent("mgr_test") is True
    assert manager.get_agent("mgr_test") is None


def test_unregister_nonexistent(manager: AgentManager):
    assert manager.unregister_agent("nope") is False


def test_submit_task_unknown_agent(manager: AgentManager):
    with pytest.raises(ValueError, match="not registered"):
        manager.submit_task("nope", "hello")


def test_submit_task(manager: AgentManager, agent_config: AgentConfig):
    manager.register_agent(agent_config)

    async def fake_query(**kwargs):
        return
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=fake_query):
        task = manager.submit_task("mgr_test", "do something")
    assert task.status == "pending"
    assert task.agent_id == "mgr_test"
    # Give the background loop a moment to process
    time.sleep(0.5)


def test_get_logs_empty(manager: AgentManager):
    assert manager.get_logs("noagent") == ""


def test_list_tasks_empty(manager: AgentManager):
    assert manager.list_tasks() == []


# --- lifecycle regressions ---


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll a predicate on the background loop's work. Returns whether it held."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_task_persists_running_status(manager: AgentManager, agent_config: AgentConfig):
    """In-flight tasks used to read as `pending` everywhere outside this process."""
    manager.register_agent(agent_config)
    seen: list[str | None] = []
    release = threading.Event()

    async def slow_query(**kwargs):
        release.wait(5.0)
        return
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=slow_query):
        task = manager.submit_task("mgr_test", "do something")
        assert _wait_for(lambda: manager.get_task(task.id).status == "running")
        seen.append(manager.get_task(task.id).status)
        release.set()
        assert _wait_for(lambda: manager.get_task(task.id).status == "completed")

    assert seen == ["running"]


def test_task_failed_event_fires_even_when_the_agent_was_deleted_mid_run(
    manager: AgentManager, agent_config: AgentConfig
):
    """The Brain deletes agents in Phase 6, sometimes while a task still runs.

    The failure handler re-read self._agents[agent_id] and raised a KeyError
    *inside* the except block, so the task_failed event never reached the CLI
    and the auto_restart branch never ran — the task looked orphaned.
    """
    agent_config.auto_restart = True
    manager.register_agent(agent_config)
    events: list[dict] = []
    manager.add_progress_listener(events.append)
    entered = threading.Event()
    release = threading.Event()

    async def failing_query(**kwargs):
        entered.set()
        release.wait(5.0)
        raise RuntimeError("sdk blew up")
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=failing_query):
        task = manager.submit_task("mgr_test", "do something", on_progress=events.append)
        assert entered.wait(5.0)
        manager.unregister_agent("mgr_test")  # Brain's Phase 6, mid-run
        release.set()
        assert _wait_for(lambda: manager.get_task(task.id).status == "failed")
        assert _wait_for(lambda: any(e.get("kind") == "task_failed" for e in events))

    stored = manager.get_task(task.id)
    assert "sdk blew up" in stored.error
    assert stored.completed_at is not None


@pytest.mark.asyncio
async def test_task_for_an_already_deleted_agent_fails_legibly(manager: AgentManager):
    """Two stored failures read only `'00cd38ae'` — a bare KeyError on an agent
    id, because the agent was gone before its task reached the event loop.

    Driven directly rather than through submit_task: whether the deletion wins
    that race is timing, and the guard is what is under test.
    """
    from meta_agent.agent_runner import AgentRunner
    from meta_agent.models import Task

    config = AgentConfig(id="gone01", name="Gone", system_prompt="x")
    task = Task(agent_id="gone01", prompt="do something")
    manager.db.save_task(task)

    await manager._execute_task("gone01", AgentRunner(config), task)

    stored = manager.get_task(task.id)
    assert stored.status == "failed"
    assert "was deleted before its task could run" in stored.error
    assert not stored.error.startswith("'gone01'")


def test_failed_task_still_auto_restarts_a_surviving_agent(
    manager: AgentManager, agent_config: AgentConfig
):
    agent_config.auto_restart = True
    manager.register_agent(agent_config)

    async def failing_query(**kwargs):
        raise RuntimeError("sdk blew up")
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=failing_query):
        task = manager.submit_task("mgr_test", "do something")
        assert _wait_for(lambda: manager.get_task(task.id).status == "failed")
        assert _wait_for(lambda: manager.get_agent("mgr_test").restart_count == 1)

    state = manager.get_agent("mgr_test")
    assert state.status == AgentStatus.IDLE
    assert state.error is None


def test_error_context_reaches_the_stored_error(
    manager: AgentManager, agent_config: AgentConfig
):
    """A failure record must name the model, so an exit-1 with empty stderr is
    still diagnosable."""
    manager.register_agent(agent_config)

    async def failing_query(**kwargs):
        raise RuntimeError("Command failed with exit code 1")
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=failing_query):
        task = manager.submit_task("mgr_test", "do something")
        assert _wait_for(lambda: manager.get_task(task.id).status == "failed")

    assert f"model={agent_config.model}" in manager.get_task(task.id).error


# --- more than one task in flight per agent ---


def test_two_tasks_on_one_agent_are_both_tracked(
    manager: AgentManager, agent_config: AgentConfig
):
    """self._runners[agent_id] = runner overwrote the first task's runner, so
    the agent looked like it had one task when it had two."""
    manager.register_agent(agent_config)
    release = threading.Event()

    async def slow_query(**kwargs):
        release.wait(5.0)
        return
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=slow_query):
        first = manager.submit_task("mgr_test", "one")
        second = manager.submit_task("mgr_test", "two")
        assert _wait_for(lambda: len(manager.running_task_ids("mgr_test")) == 2)

        state = manager.get_agent("mgr_test")
        assert sorted(state.running_task_ids) == sorted([first.id, second.id])
        assert state.status == AgentStatus.RUNNING

        release.set()
        assert _wait_for(lambda: manager.running_task_ids("mgr_test") == [])

    assert manager.get_task(first.id).status == "completed"
    assert manager.get_task(second.id).status == "completed"
    assert manager.get_agent("mgr_test").status == AgentStatus.IDLE
    assert manager.get_agent("mgr_test").current_task_id is None


def test_agent_stays_running_until_its_last_task_finishes(
    manager: AgentManager, agent_config: AgentConfig
):
    """The first task to finish used to clear current_task_id and flip the
    agent to IDLE while a second task was still going."""
    manager.register_agent(agent_config)
    release_first = threading.Event()
    release_second = threading.Event()
    gates = {}

    async def gated_query(**kwargs):
        prompt = kwargs.get("prompt", "")
        gates.setdefault(prompt, True)
        (release_first if prompt == "one" else release_second).wait(5.0)
        return
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=gated_query):
        first = manager.submit_task("mgr_test", "one")
        second = manager.submit_task("mgr_test", "two")
        assert _wait_for(lambda: len(manager.running_task_ids("mgr_test")) == 2)

        release_first.set()
        assert _wait_for(lambda: manager.get_task(first.id).status == "completed")

        state = manager.get_agent("mgr_test")
        assert state.status == AgentStatus.RUNNING, "went idle with a task still running"
        assert state.current_task_id == second.id

        release_second.set()
        assert _wait_for(lambda: manager.get_agent("mgr_test").status == AgentStatus.IDLE)


def test_cancel_agent_tasks_cancels_every_in_flight_task(
    manager: AgentManager, agent_config: AgentConfig
):
    """AgentRunner._current_task was never assigned, so cancel() was a no-op and
    only the newest runner was reachable anyway."""
    manager.register_agent(agent_config)
    started = threading.Semaphore(0)

    async def never_finishes(**kwargs):
        started.release()
        await asyncio.sleep(30)
        return
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=never_finishes):
        first = manager.submit_task("mgr_test", "one")
        second = manager.submit_task("mgr_test", "two")
        assert started.acquire(timeout=5) and started.acquire(timeout=5)

        cancelled = manager.cancel_agent_tasks("mgr_test")
        assert sorted(cancelled) == sorted([first.id, second.id])

        assert _wait_for(lambda: manager.get_task(first.id).status == "cancelled")
        assert _wait_for(lambda: manager.get_task(second.id).status == "cancelled")

    assert manager.running_task_ids("mgr_test") == []
    assert manager.get_agent("mgr_test").current_task_id is None


def test_unregister_cancels_all_of_the_agents_tasks(
    manager: AgentManager, agent_config: AgentConfig
):
    manager.register_agent(agent_config)
    started = threading.Semaphore(0)

    async def never_finishes(**kwargs):
        started.release()
        await asyncio.sleep(30)
        return
        yield  # make it an async generator

    with patch("meta_agent.agent_runner.query", side_effect=never_finishes):
        first = manager.submit_task("mgr_test", "one")
        second = manager.submit_task("mgr_test", "two")
        assert started.acquire(timeout=5) and started.acquire(timeout=5)

        assert manager.unregister_agent("mgr_test") is True
        assert _wait_for(lambda: manager.get_task(first.id).status == "cancelled")
        assert _wait_for(lambda: manager.get_task(second.id).status == "cancelled")
