"""Tests for the database layer."""

import threading
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


# --- threading ---


def test_each_thread_gets_its_own_connection(db: Database):
    seen: dict[int, int] = {}
    barrier = threading.Barrier(4)

    def grab() -> None:
        barrier.wait(5)
        seen[threading.get_ident()] = id(db._conn)

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert len(seen) == 4
    assert len(set(seen.values())) == 4, "connections were shared across threads"


def test_the_same_thread_reuses_one_connection(db: Database):
    assert db._conn is db._conn


def test_concurrent_writers_and_readers_do_not_corrupt_each_other(db: Database):
    """A shared connection means a shared transaction: one thread's commit can
    land another thread's half-written statement, and rows go missing."""
    workers, per_worker = 8, 25
    errors: list[Exception] = []
    barrier = threading.Barrier(workers)

    def churn(worker: int) -> None:
        barrier.wait(10)
        try:
            for i in range(per_worker):
                task = Task(id=f"w{worker}t{i}", agent_id=f"a{worker}", prompt="p")
                db.save_task(task)
                assert db.get_task(task.id) is not None
                db.list_tasks()
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=churn, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert errors == []
    assert len(db.list_tasks()) == workers * per_worker


def test_close_releases_connections_opened_on_other_threads(db: Database):
    def touch() -> None:
        db.save_task(Task(id="other-thread", agent_id="a", prompt="p"))

    t = threading.Thread(target=touch)
    t.start()
    t.join(5)

    assert len(db._open) == 2  # this thread's and the worker's
    db.close()
    assert db._open == []
    # and the Database is still usable afterwards
    assert db.get_task("other-thread") is not None
