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


# --- the in-process transport ---


def test_both_transports_expose_the_same_tool_surface(manager: AgentManager):
    from meta_agent.mcp_server import _tool_functions

    names = {fn.__name__ for fn in _tool_functions(manager)}
    assert {"create_agent", "submit_task", "task_status", "report_progress"} <= names
    # both builders read the same list, so a tool cannot exist on only one
    assert len(_tool_functions(manager)) == len(names)


def test_inprocess_server_is_an_sdk_server(manager: AgentManager):
    from meta_agent.mcp_server import MCP_SERVER_NAME, create_inprocess_mcp_server

    cfg = create_inprocess_mcp_server(manager)
    assert cfg["type"] == "sdk"
    assert cfg["name"] == MCP_SERVER_NAME
    assert cfg["instance"] is not None


def test_optional_tool_parameters_are_not_marked_required(manager: AgentManager):
    """The SDK's {name: type} schema shorthand marks everything required."""
    from meta_agent.mcp_server import _input_schema, _tool_functions

    tools = {fn.__name__: fn for fn in _tool_functions(manager)}
    schema = _input_schema(tools["create_agent"])
    assert schema["required"] == ["name", "system_prompt"]
    assert schema["properties"]["allowed_tools"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["max_turns"] == {"type": "integer"}

    assert "required" not in _input_schema(tools["list_agents"])
    assert _input_schema(tools["workflow_status"])["properties"]["lightweight"] == {
        "type": "boolean"
    }


@pytest.mark.asyncio
async def test_inprocess_tool_call_mutates_the_callers_manager(manager: AgentManager):
    """The point of the change: a tool call reaches THIS manager, not a copy."""
    from meta_agent.mcp_server import _as_sdk_tool, _tool_functions

    tools = {fn.__name__: _as_sdk_tool(fn) for fn in _tool_functions(manager)}
    result = await tools["create_agent"].handler(
        {"name": "Spawned", "system_prompt": "test", "agent_id": "spawned01"}
    )
    assert "spawned01" in result["content"][0]["text"]
    assert manager.get_agent("spawned01") is not None


@pytest.mark.asyncio
async def test_report_progress_reaches_a_listener_in_this_process(manager: AgentManager):
    """report_progress broadcast into the subprocess's empty listener list."""
    from meta_agent.mcp_server import _as_sdk_tool, _tool_functions

    events: list[dict] = []
    manager.add_progress_listener(events.append)
    tools = {fn.__name__: _as_sdk_tool(fn) for fn in _tool_functions(manager)}
    await tools["report_progress"].handler(
        {"agent_id": "a1", "task_id": "t1", "message": "reading files", "phase": "reading"}
    )
    assert events == [
        {
            "kind": "agent_progress",
            "agent_id": "a1",
            "task_id": "t1",
            "message": "reading files",
            "phase": "reading",
        }
    ]


def test_manager_hands_the_brain_an_inprocess_server(manager: AgentManager):
    from meta_agent.brain import get_brain_config
    from meta_agent.mcp_server import MCP_SERVER_NAME

    plain = AgentConfig(id="plain", name="Plain", system_prompt="x")
    assert manager.mcp_servers_for(plain) is None

    servers = manager.mcp_servers_for(get_brain_config())
    assert servers is not None
    assert servers[MCP_SERVER_NAME]["type"] == "sdk"
    # built once and reused
    assert manager.mcp_servers_for(get_brain_config()) is servers
