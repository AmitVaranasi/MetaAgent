"""Tests for the agent manager."""

from __future__ import annotations

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
