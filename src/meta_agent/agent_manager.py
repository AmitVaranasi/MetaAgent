"""Agent lifecycle manager. Runs SDK agents as asyncio tasks."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from .agent_runner import AgentRunner
from .db import Database
from .models import AgentConfig, AgentState, AgentStatus, Task

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self, db: Database, log_dir: Path):
        self.db = db
        self.log_dir = log_dir
        self._lock = threading.Lock()
        self._agents: dict[str, AgentState] = {}
        self._runners: dict[str, AgentRunner] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    def start(self) -> None:
        """Load agents from DB, start event loop in background thread."""
        for config in self.db.list_agents():
            with self._lock:
                self._agents[config.id] = AgentState(config=config)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._loop_thread.start()

    def shutdown(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # --- Agent CRUD ---

    def register_agent(self, config: AgentConfig) -> AgentState:
        self.db.save_agent(config)
        state = AgentState(config=config)
        with self._lock:
            self._agents[config.id] = state
        return state

    def unregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            state = self._agents.pop(agent_id, None)
        if state is None:
            return False
        # Cancel any running task
        runner = self._runners.pop(agent_id, None)
        if runner and self._loop:
            asyncio.run_coroutine_threadsafe(runner.cancel(), self._loop)
        self.db.delete_agent(agent_id)
        return True

    def list_agents(self) -> list[AgentState]:
        with self._lock:
            return list(self._agents.values())

    def get_agent(self, agent_id: str) -> AgentState | None:
        with self._lock:
            return self._agents.get(agent_id)

    # --- Task submission ---

    def submit_task(self, agent_id: str, prompt: str) -> Task:
        """Submit a task to an agent. Runs via SDK in the background event loop."""
        with self._lock:
            state = self._agents.get(agent_id)
        if state is None:
            raise ValueError(f"Agent {agent_id} not registered")

        task = Task(
            agent_id=agent_id,
            prompt=prompt,
            created_at=datetime.now(timezone.utc),
        )
        self.db.save_task(task)

        runner = AgentRunner(state.config)
        with self._lock:
            self._runners[agent_id] = runner
            state.status = AgentStatus.RUNNING
            state.current_task_id = task.id
            state.started_at = datetime.now(timezone.utc)

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._execute_task(agent_id, runner, task),
                self._loop,
            )
        return task

    async def _execute_task(
        self, agent_id: str, runner: AgentRunner, task: Task
    ) -> None:
        """Execute a task and update state on completion."""
        log_path = self.log_dir / f"{agent_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def on_message(msg: object) -> None:
            with open(log_path, "a") as f:
                f.write(f"{msg}\n")

        try:
            result = await runner.run_task(task, on_message=on_message)
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now(timezone.utc)
            with self._lock:
                state = self._agents[agent_id]
                state.status = AgentStatus.IDLE
                state.current_task_id = None
        except Exception as e:
            logger.exception("Task %s failed for agent %s", task.id, agent_id)
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc)
            with self._lock:
                state = self._agents[agent_id]
                state.status = AgentStatus.ERROR
                state.error = str(e)
            if (
                state.config.auto_restart
                and state.restart_count < state.config.max_restarts
            ):
                with self._lock:
                    state.restart_count += 1
                    state.status = AgentStatus.IDLE
                    state.error = None
        finally:
            self.db.save_task(task)

    # --- Logs ---

    def get_logs(self, agent_id: str, lines: int = 100) -> str:
        log_path = self.log_dir / f"{agent_id}.log"
        if not log_path.exists():
            return ""
        all_lines = log_path.read_text().splitlines()
        return "\n".join(all_lines[-lines:])

    # --- Task queries ---

    def get_task(self, task_id: str) -> Task | None:
        return self.db.get_task(task_id)

    def list_tasks(self, agent_id: str | None = None) -> list[Task]:
        return self.db.list_tasks(agent_id)
