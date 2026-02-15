"""Tests for the Brain agent configuration and workflow submission."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from meta_agent.brain import BRAIN_AGENT_ID, BRAIN_SYSTEM_PROMPT, get_brain_config
from meta_agent.models import AgentConfig


def test_brain_agent_id():
    assert BRAIN_AGENT_ID == "brain"


def test_brain_config_defaults():
    config = get_brain_config()
    assert config.id == BRAIN_AGENT_ID
    assert config.name == "Brain Agent"
    assert config.model == "claude-opus-4-6"
    assert config.max_turns == 200
    assert config.mcp_servers == {}
    assert "Read" in config.allowed_tools


def test_brain_config_with_mcp_server():
    config = get_brain_config(["meta-agent", "mcp-server"])
    assert "meta-agent" in config.mcp_servers
    assert config.mcp_servers["meta-agent"]["command"] == "meta-agent"
    assert config.mcp_servers["meta-agent"]["args"] == ["mcp-server"]


def test_brain_system_prompt_content():
    assert "Brain Agent" in BRAIN_SYSTEM_PROMPT
    assert "create_workflow" in BRAIN_SYSTEM_PROMPT
    assert "submit_task" in BRAIN_SYSTEM_PROMPT
    assert "task_status" in BRAIN_SYSTEM_PROMPT


def test_brain_config_is_valid_agent_config():
    config = get_brain_config(["meta-agent", "mcp-server"])
    assert isinstance(config, AgentConfig)
    # Should serialize/deserialize cleanly
    data = config.model_dump()
    restored = AgentConfig.model_validate(data)
    assert restored.id == BRAIN_AGENT_ID
    assert restored.model == "claude-opus-4-6"


def test_brain_workflow_submission(db, sample_config):
    """Test that brain can be registered and a workflow submitted."""
    from meta_agent.agent_manager import AgentManager
    from meta_agent.models import Workflow

    mgr = AgentManager(db, db.db_path.parent / "logs")
    mgr.start()

    brain_config = get_brain_config()
    mgr.register_agent(brain_config)

    assert mgr.get_agent(BRAIN_AGENT_ID) is not None
    assert mgr.get_agent(BRAIN_AGENT_ID).config.model == "claude-opus-4-6"

    # Create and save a workflow
    workflow = Workflow(prompt="Test task", brain_agent_id=BRAIN_AGENT_ID)
    db.save_workflow(workflow)

    retrieved = db.get_workflow(workflow.id)
    assert retrieved is not None
    assert retrieved.prompt == "Test task"
    assert retrieved.status.value == "planning"

    mgr.shutdown()
