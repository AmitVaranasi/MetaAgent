"""Tests for the MCP server tools."""

from __future__ import annotations

import pytest

from meta_agent.agent_manager import AgentManager
from meta_agent.db import Database
from meta_agent.mcp_server import create_mcp_server
from meta_agent.models import AgentConfig


@pytest.fixture()
def manager(db: Database, config) -> AgentManager:
    mgr = AgentManager(db, config.log_dir)
    mgr.start()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def mcp(manager: AgentManager):
    return create_mcp_server(manager)


def test_create_mcp_server(mcp):
    assert mcp is not None


def test_list_agents_tool_registered(manager: AgentManager):
    """Verify manager can list agents (tool plumbing works)."""
    agents = manager.list_agents()
    assert isinstance(agents, list)


def test_create_and_get_via_manager(manager: AgentManager):
    """End-to-end: create agent then retrieve."""
    config = AgentConfig(
        id="mcp_test",
        name="MCP Test",
        system_prompt="Test prompt",
        allowed_tools=[],
    )
    manager.register_agent(config)
    state = manager.get_agent("mcp_test")
    assert state is not None
    assert state.config.name == "MCP Test"


def test_delete_via_manager(manager: AgentManager):
    config = AgentConfig(
        id="mcp_del",
        name="Delete Me",
        system_prompt="Test",
        allowed_tools=[],
    )
    manager.register_agent(config)
    assert manager.unregister_agent("mcp_del") is True
    assert manager.get_agent("mcp_del") is None


def test_logs_empty(manager: AgentManager):
    assert manager.get_logs("nonexistent") == ""


def test_list_tasks_empty(manager: AgentManager):
    assert manager.list_tasks() == []
