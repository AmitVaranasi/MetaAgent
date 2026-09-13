"""SQLite storage for agents, tasks and workflows. One connection per thread."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import AgentConfig, Task, Workflow, WorkflowStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    prompt TEXT NOT NULL,
    messages_json TEXT NOT NULL DEFAULT '[]',
    result TEXT,
    error TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    plan TEXT,
    status TEXT NOT NULL DEFAULT 'planning',
    brain_agent_id TEXT NOT NULL,
    brain_task_id TEXT,
    subtask_ids_json TEXT NOT NULL DEFAULT '[]',
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
"""

_MIGRATIONS = [
    "ALTER TABLE tasks ADD COLUMN workflow_id TEXT",
    "ALTER TABLE tasks ADD COLUMN parent_task_id TEXT",
    "ALTER TABLE tasks ADD COLUMN owner_pid INTEGER",
]


class Database:
    """SQLite storage with one connection PER THREAD.

    A single connection opened with ``check_same_thread=False`` and no lock is
    not safe here: AgentManager runs agents on a background event-loop thread
    while the CLI, the MCP tools and the dashboard read from others. Sharing one
    connection means sharing one transaction — a ``commit()`` on one thread can
    commit another thread's half-written statement, and cursor state interleaves.

    WAL is what makes a connection per thread cheap: one writer and any number
    of concurrent readers, which is the concurrency WAL was turned on for in the
    first place. ``busy_timeout`` covers the writer-vs-writer case.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._open_lock = threading.Lock()
        self._open: list[sqlite3.Connection] = []
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._run_migrations()

    def _new_connection(self) -> sqlite3.Connection:
        # check_same_thread=False so close() can reach connections it did not open.
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        with self._open_lock:
            self._open.append(conn)
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    def _run_migrations(self) -> None:
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

    def close(self) -> None:
        """Close every thread's connection."""
        with self._open_lock:
            connections, self._open = self._open, []
        for conn in connections:
            conn.close()
        # Drop the per-thread handles too, so a later call reopens cleanly.
        self._local = threading.local()

    # --- Agent CRUD ---

    def save_agent(self, config: AgentConfig) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO agents (id, config_json) VALUES (?, ?)",
            (config.id, config.model_dump_json()),
        )
        self._conn.commit()

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        row = self._conn.execute(
            "SELECT config_json FROM agents WHERE id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            return None
        return AgentConfig.model_validate_json(row["config_json"])

    def list_agents(self) -> list[AgentConfig]:
        rows = self._conn.execute("SELECT config_json FROM agents").fetchall()
        return [AgentConfig.model_validate_json(r["config_json"]) for r in rows]

    def delete_agent(self, agent_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- Task CRUD ---

    def save_task(self, task: Task) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO tasks
               (id, agent_id, status, prompt, messages_json, result, error,
                session_id, created_at, completed_at, workflow_id, parent_task_id,
                owner_pid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.id,
                task.agent_id,
                task.status,
                task.prompt,
                json.dumps(task.messages),
                task.result,
                task.error,
                task.session_id,
                task.created_at.isoformat(),
                task.completed_at.isoformat() if task.completed_at else None,
                task.workflow_id,
                task.parent_task_id,
                task.owner_pid,
            ),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> Task | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(self, agent_id: str | None = None) -> list[Task]:
        if agent_id:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            agent_id=row["agent_id"],
            status=row["status"],
            prompt=row["prompt"],
            messages=json.loads(row["messages_json"]),
            result=row["result"],
            error=row["error"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            workflow_id=row["workflow_id"],
            parent_task_id=row["parent_task_id"],
            owner_pid=row["owner_pid"],
        )

    # --- Workflow CRUD ---

    def save_workflow(self, workflow: Workflow) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO workflows
               (id, prompt, plan, status, brain_agent_id, brain_task_id,
                subtask_ids_json, result, error, created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                workflow.id,
                workflow.prompt,
                workflow.plan,
                workflow.status.value,
                workflow.brain_agent_id,
                workflow.brain_task_id,
                json.dumps(workflow.subtask_ids),
                workflow.result,
                workflow.error,
                workflow.created_at.isoformat(),
                workflow.completed_at.isoformat() if workflow.completed_at else None,
            ),
        )
        self._conn.commit()

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        row = self._conn.execute(
            "SELECT * FROM workflows WHERE id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_workflow(row)

    def list_workflows(self) -> list[Workflow]:
        rows = self._conn.execute(
            "SELECT * FROM workflows ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_workflow(r) for r in rows]

    def _row_to_workflow(self, row: sqlite3.Row) -> Workflow:
        return Workflow(
            id=row["id"],
            prompt=row["prompt"],
            plan=row["plan"],
            status=WorkflowStatus(row["status"]),
            brain_agent_id=row["brain_agent_id"],
            brain_task_id=row["brain_task_id"],
            subtask_ids=json.loads(row["subtask_ids_json"]),
            result=row["result"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
        )
