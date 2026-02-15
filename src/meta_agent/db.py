"""SQLite database with WAL mode for agents and tasks."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import AgentConfig, Task

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
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

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
                session_id, created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        )
