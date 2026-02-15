"""Tests for data models."""

from meta_agent.models import AgentConfig, AgentState, AgentStatus, Task, Workflow, WorkflowStatus


def test_agent_config_defaults():
    cfg = AgentConfig(name="Test", system_prompt="Hello")
    assert cfg.name == "Test"
    assert len(cfg.id) == 8
    assert cfg.model == "claude-sonnet-4-5-20250929"
    assert "Read" in cfg.allowed_tools
    assert cfg.max_turns == 50
    assert cfg.auto_restart is False


def test_agent_config_custom_id():
    cfg = AgentConfig(id="myid", name="Test", system_prompt="Hello")
    assert cfg.id == "myid"


def test_agent_state_defaults():
    cfg = AgentConfig(name="Test", system_prompt="Hello")
    state = AgentState(config=cfg)
    assert state.status == AgentStatus.STOPPED
    assert state.session_id is None
    assert state.restart_count == 0


def test_task_defaults():
    t = Task(agent_id="a1", prompt="Do something")
    assert t.status == "pending"
    assert len(t.id) == 12
    assert t.result is None
    assert t.messages == []


def test_agent_config_serialization():
    cfg = AgentConfig(name="Test", system_prompt="Hello", allowed_tools=["Bash"])
    data = cfg.model_dump()
    restored = AgentConfig.model_validate(data)
    assert restored.name == cfg.name
    assert restored.allowed_tools == ["Bash"]


def test_agent_config_json_round_trip():
    cfg = AgentConfig(name="Test", system_prompt="Hello")
    json_str = cfg.model_dump_json()
    restored = AgentConfig.model_validate_json(json_str)
    assert restored.id == cfg.id


def test_task_workflow_fields():
    t = Task(agent_id="a1", prompt="Do something", workflow_id="wf1", parent_task_id="t0")
    assert t.workflow_id == "wf1"
    assert t.parent_task_id == "t0"


def test_task_workflow_fields_default_none():
    t = Task(agent_id="a1", prompt="Do something")
    assert t.workflow_id is None
    assert t.parent_task_id is None


def test_workflow_defaults():
    wf = Workflow(prompt="Build something", brain_agent_id="brain")
    assert len(wf.id) == 12
    assert wf.status == WorkflowStatus.PLANNING
    assert wf.plan is None
    assert wf.subtask_ids == []
    assert wf.result is None
    assert wf.completed_at is None


def test_workflow_status_values():
    assert WorkflowStatus.PLANNING.value == "planning"
    assert WorkflowStatus.EXECUTING.value == "executing"
    assert WorkflowStatus.ASSEMBLING.value == "assembling"
    assert WorkflowStatus.COMPLETED.value == "completed"
    assert WorkflowStatus.FAILED.value == "failed"


def test_workflow_serialization():
    wf = Workflow(prompt="Test", brain_agent_id="brain", subtask_ids=["t1", "t2"])
    data = wf.model_dump()
    restored = Workflow.model_validate(data)
    assert restored.prompt == "Test"
    assert restored.subtask_ids == ["t1", "t2"]
    assert restored.brain_agent_id == "brain"
