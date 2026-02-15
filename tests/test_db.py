"""Tests for the database layer."""

from datetime import datetime, timezone

from meta_agent.db import Database
from meta_agent.models import AgentConfig, Task, Workflow, WorkflowStatus


def test_save_and_get_agent(db: Database, sample_config: AgentConfig):
    db.save_agent(sample_config)
    result = db.get_agent(sample_config.id)
    assert result is not None
    assert result.name == sample_config.name
    assert result.system_prompt == sample_config.system_prompt


def test_list_agents(db: Database, sample_config: AgentConfig):
    db.save_agent(sample_config)
    agents = db.list_agents()
    assert len(agents) == 1
    assert agents[0].id == sample_config.id


def test_delete_agent(db: Database, sample_config: AgentConfig):
    db.save_agent(sample_config)
    assert db.delete_agent(sample_config.id) is True
    assert db.get_agent(sample_config.id) is None


def test_delete_nonexistent_agent(db: Database):
    assert db.delete_agent("nope") is False


def test_save_and_get_task(db: Database, sample_config: AgentConfig):
    db.save_agent(sample_config)
    task = Task(
        agent_id=sample_config.id,
        prompt="Hello",
        created_at=datetime.now(timezone.utc),
    )
    db.save_task(task)
    result = db.get_task(task.id)
    assert result is not None
    assert result.prompt == "Hello"
    assert result.status == "pending"


def test_list_tasks_by_agent(db: Database, sample_config: AgentConfig):
    db.save_agent(sample_config)
    for i in range(3):
        db.save_task(Task(
            agent_id=sample_config.id,
            prompt=f"Task {i}",
            created_at=datetime.now(timezone.utc),
        ))
    tasks = db.list_tasks(agent_id=sample_config.id)
    assert len(tasks) == 3


def test_list_all_tasks(db: Database, sample_config: AgentConfig):
    db.save_agent(sample_config)
    db.save_task(Task(
        agent_id=sample_config.id,
        prompt="A task",
        created_at=datetime.now(timezone.utc),
    ))
    all_tasks = db.list_tasks()
    assert len(all_tasks) == 1


def test_update_task(db: Database, sample_config: AgentConfig):
    db.save_agent(sample_config)
    task = Task(
        agent_id=sample_config.id,
        prompt="Hello",
        created_at=datetime.now(timezone.utc),
    )
    db.save_task(task)
    task.status = "completed"
    task.result = "Done"
    task.completed_at = datetime.now(timezone.utc)
    db.save_task(task)
    result = db.get_task(task.id)
    assert result.status == "completed"
    assert result.result == "Done"


def test_task_workflow_columns(db: Database, sample_config: AgentConfig):
    db.save_agent(sample_config)
    task = Task(
        agent_id=sample_config.id,
        prompt="Hello",
        workflow_id="wf123",
        parent_task_id="t000",
        created_at=datetime.now(timezone.utc),
    )
    db.save_task(task)
    result = db.get_task(task.id)
    assert result.workflow_id == "wf123"
    assert result.parent_task_id == "t000"


def test_save_and_get_workflow(db: Database):
    wf = Workflow(prompt="Build it", brain_agent_id="brain")
    db.save_workflow(wf)
    result = db.get_workflow(wf.id)
    assert result is not None
    assert result.prompt == "Build it"
    assert result.status == WorkflowStatus.PLANNING
    assert result.brain_agent_id == "brain"


def test_workflow_with_subtasks(db: Database):
    wf = Workflow(
        prompt="Complex task",
        brain_agent_id="brain",
        subtask_ids=["t1", "t2", "t3"],
    )
    db.save_workflow(wf)
    result = db.get_workflow(wf.id)
    assert result.subtask_ids == ["t1", "t2", "t3"]


def test_list_workflows(db: Database):
    for i in range(3):
        db.save_workflow(Workflow(prompt=f"Task {i}", brain_agent_id="brain"))
    workflows = db.list_workflows()
    assert len(workflows) == 3


def test_get_nonexistent_workflow(db: Database):
    assert db.get_workflow("nope") is None


def test_update_workflow(db: Database):
    wf = Workflow(prompt="Test", brain_agent_id="brain")
    db.save_workflow(wf)
    wf.status = WorkflowStatus.EXECUTING
    wf.plan = "Step 1: do thing"
    wf.subtask_ids.append("t1")
    db.save_workflow(wf)
    result = db.get_workflow(wf.id)
    assert result.status == WorkflowStatus.EXECUTING
    assert result.plan == "Step 1: do thing"
    assert result.subtask_ids == ["t1"]
