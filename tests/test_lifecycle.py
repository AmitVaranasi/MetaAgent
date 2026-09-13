"""Agent lifecycle and log rotation.

AgentStatus was decorative: register_agent produced a STOPPED agent, start_agent
set IDLE, and submit_task ran regardless of either — so stop_agent changed a
label and the agent kept accepting work.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from meta_agent.agent_manager import LOG_BACKUP_COUNT, AgentManager
from meta_agent.db import Database
from meta_agent.models import AgentConfig, AgentStatus


@pytest.fixture()
def manager(db: Database, config) -> AgentManager:
    mgr = AgentManager(db, config.log_dir)
    mgr.retry_base_delay_s = 0.0
    mgr.start()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def agent_config() -> AgentConfig:
    return AgentConfig(id="life", name="Life", system_prompt="x", allowed_tools=[])


async def _fake_query(**kwargs):
    return
    yield


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_registered_agent_is_ready_for_work(manager: AgentManager, agent_config: AgentConfig):
    assert manager.register_agent(agent_config).status == AgentStatus.IDLE


def test_a_stopped_agent_refuses_new_work(manager: AgentManager, agent_config: AgentConfig):
    manager.register_agent(agent_config)
    manager.stop_agent("life")

    assert manager.get_agent("life").status == AgentStatus.STOPPED
    with pytest.raises(ValueError, match="stopped"):
        manager.submit_task("life", "do something")


def test_starting_a_stopped_agent_lets_it_work_again(
    manager: AgentManager, agent_config: AgentConfig
):
    manager.register_agent(agent_config)
    manager.stop_agent("life")
    assert manager.start_agent("life").status == AgentStatus.IDLE

    with patch("meta_agent.agent_runner.query", side_effect=_fake_query):
        task = manager.submit_task("life", "do something")
        assert _wait_for(lambda: manager.get_task(task.id).status == "completed")


def test_start_agent_does_not_disturb_a_busy_agent(
    manager: AgentManager, agent_config: AgentConfig
):
    import threading

    manager.register_agent(agent_config)
    release = threading.Event()

    async def slow_query(**kwargs):
        release.wait(5.0)
        return
        yield

    with patch("meta_agent.agent_runner.query", side_effect=slow_query):
        manager.submit_task("life", "do something")
        assert _wait_for(lambda: manager.get_agent("life").status == AgentStatus.RUNNING)

        assert manager.start_agent("life").status == AgentStatus.RUNNING
        release.set()


def test_start_and_stop_of_an_unknown_agent_return_none(manager: AgentManager):
    assert manager.start_agent("nope") is None
    assert manager.stop_agent("nope") is None


def test_stop_agent_clears_the_error_only_on_restart(
    manager: AgentManager, agent_config: AgentConfig
):
    manager.register_agent(agent_config)

    async def failing(**kwargs):
        raise RuntimeError("boom")
        yield

    with patch("meta_agent.agent_runner.query", side_effect=failing):
        task = manager.submit_task("life", "do something")
        assert _wait_for(lambda: manager.get_task(task.id).status == "failed")

    assert manager.get_agent("life").status == AgentStatus.ERROR
    assert manager.get_agent("life").error is not None
    # an errored agent still accepts work; only STOPPED refuses
    manager.stop_agent("life")
    assert manager.start_agent("life").error is None


# --- log rotation ---


def test_agent_log_rotates_instead_of_growing_without_bound(
    manager: AgentManager, agent_config: AgentConfig, monkeypatch
):
    """brain.log reached 13 MB, and the file was reopened for every message."""
    monkeypatch.setattr("meta_agent.agent_manager.LOG_MAX_BYTES", 2048)
    manager.register_agent(agent_config)

    sink = manager._open_log("life")
    for i in range(400):
        sink("x" * 100 + f" {i}")

    log_dir = manager.log_dir
    assert (log_dir / "life.log").stat().st_size <= 2048 + 200
    rolls = sorted(p.name for p in log_dir.glob("life.log.*"))
    assert rolls, "nothing rotated"
    assert len(rolls) <= LOG_BACKUP_COUNT


def test_the_log_sink_is_opened_once_per_agent(manager: AgentManager, agent_config: AgentConfig):
    manager.register_agent(agent_config)
    manager._open_log("life")
    manager._open_log("life")
    assert len(manager._agent_loggers["life"].handlers) == 1


def test_get_logs_still_reads_what_the_sink_wrote(
    manager: AgentManager, agent_config: AgentConfig
):
    manager.register_agent(agent_config)
    sink = manager._open_log("life")
    sink("first line")
    sink("second line")
    text = manager.get_logs("life")
    assert "first line" in text and "second line" in text


# --- working directory ---


def test_an_agent_without_a_cwd_inherits_the_managers_default(db: Database, config, tmp_path):
    """Observed live: a sub-agent created with cwd=None ran in the launching
    process's directory and overwrote that repo's README.md."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mgr = AgentManager(db, config.log_dir, default_cwd=str(workspace))
    mgr.start()
    try:
        state = mgr.register_agent(
            AgentConfig(id="sub", name="Sub", system_prompt="x", allowed_tools=[])
        )
        assert state.config.cwd == str(workspace)
        assert mgr.db.get_agent("sub").cwd == str(workspace)
    finally:
        mgr.shutdown()


def test_an_explicit_cwd_is_never_overridden(db: Database, config, tmp_path):
    mgr = AgentManager(db, config.log_dir, default_cwd=str(tmp_path / "default"))
    mgr.start()
    try:
        state = mgr.register_agent(
            AgentConfig(id="sub", name="Sub", system_prompt="x", allowed_tools=[], cwd="/explicit")
        )
        assert state.config.cwd == "/explicit"
    finally:
        mgr.shutdown()


def test_without_a_default_the_cwd_stays_none(manager: AgentManager):
    state = manager.register_agent(
        AgentConfig(id="sub", name="Sub", system_prompt="x", allowed_tools=[])
    )
    assert state.config.cwd is None
