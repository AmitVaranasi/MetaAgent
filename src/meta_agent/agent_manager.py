"""Agent lifecycle manager. Runs SDK agents as asyncio tasks."""

from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_runner import AgentRunner
from .db import Database
from .external_runner import ExternalModelRunner
from .models import AgentConfig, AgentState, AgentStatus, Task, WorkflowStatus

ProgressCallback = Callable[[dict[str, Any]], None] | None

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
        # External listeners for live progress events (e.g. from report_progress MCP tool)
        self._progress_listeners: list[Callable[[dict[str, Any]], None]] = []

    def add_progress_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback that receives all progress events (tool calls,
        sub-agent status, etc.)."""
        self._progress_listeners.append(callback)

    def remove_progress_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Unregister a previously added progress listener."""
        try:
            self._progress_listeners.remove(callback)
        except ValueError:
            pass

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

    def submit_task(
        self,
        agent_id: str,
        prompt: str,
        workflow_id: str | None = None,
        parent_task_id: str | None = None,
        on_progress: ProgressCallback = None,
    ) -> Task:
        """Submit a task to an agent. Runs via SDK in the background event loop."""
        with self._lock:
            state = self._agents.get(agent_id)
        if state is None:
            raise ValueError(f"Agent {agent_id} not registered")

        task = Task(
            agent_id=agent_id,
            prompt=prompt,
            created_at=datetime.now(timezone.utc),
            workflow_id=workflow_id,
            parent_task_id=parent_task_id,
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
                self._execute_task(agent_id, runner, task, on_progress=on_progress),
                self._loop,
            )
        return task

    def _fire_progress(self, callback: ProgressCallback, event: dict[str, Any]) -> None:
        """Safely invoke the progress callback."""
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            logger.debug("Progress callback error", exc_info=True)

    def resume_task(
        self,
        task_id: str,
        user_response: str,
        on_progress: ProgressCallback = None,
    ) -> Task:
        """Resume a waiting-for-input task with the user's response."""
        task = self.db.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        if task.status != "waiting_for_input":
            raise ValueError(f"Task {task_id} is not waiting for input (status={task.status})")
        if not task.session_id:
            raise ValueError(f"Task {task_id} has no session_id for resume")

        agent_id = task.agent_id
        with self._lock:
            state = self._agents.get(agent_id)
        if state is None:
            raise ValueError(f"Agent {agent_id} not registered")

        runner = AgentRunner(state.config)
        with self._lock:
            self._runners[agent_id] = runner
            state.status = AgentStatus.RUNNING
            state.current_task_id = task.id

        task.status = "running"
        self.db.save_task(task)

        # Reset workflow status so stale waiting_for_input doesn't re-trigger
        if task.workflow_id:
            wf = self.db.get_workflow(task.workflow_id)
            if wf and wf.status == WorkflowStatus.WAITING_FOR_INPUT:
                wf.status = WorkflowStatus.PLANNING
                self.db.save_workflow(wf)

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._execute_resume(agent_id, runner, task, user_response, on_progress),
                self._loop,
            )
        return task

    async def _execute_resume(
        self,
        agent_id: str,
        runner: AgentRunner,
        task: Task,
        user_response: str,
        on_progress: ProgressCallback = None,
    ) -> None:
        """Execute a resumed task and update state on completion."""
        log_path = self.log_dir / f"{agent_id}.log"

        def on_message(msg: object) -> None:
            with open(log_path, "a") as f:
                f.write(f"{msg}\n")

        def _combined_progress(event: dict[str, Any]) -> None:
            self._fire_progress(on_progress, event)
            for cb in self._progress_listeners:
                try:
                    cb(event)
                except Exception:
                    pass

        self._fire_progress(on_progress, {"kind": "status_change", "status": "running", "task_id": task.id})

        try:
            result = await runner.resume_task(
                task, user_response,
                on_message=on_message,
                on_progress=_combined_progress,
            )

            # Check if Brain is asking more questions
            if task.workflow_id:
                wf = self.db.get_workflow(task.workflow_id)
                if wf and wf.status == WorkflowStatus.WAITING_FOR_INPUT:
                    task.status = "waiting_for_input"
                    task.result = result
                    with self._lock:
                        state = self._agents[agent_id]
                        state.status = AgentStatus.IDLE
                        state.current_task_id = None
                    self._fire_progress(on_progress, {"kind": "waiting_for_input", "task_id": task.id})
                    self.db.save_task(task)
                    return

            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now(timezone.utc)
            self.db.save_task(task)
            with self._lock:
                state = self._agents[agent_id]
                state.status = AgentStatus.IDLE
                state.current_task_id = None
            self._fire_progress(on_progress, {"kind": "task_completed", "task_id": task.id})
        except Exception as e:
            logger.exception("Resume task %s failed for agent %s", task.id, agent_id)
            tb = traceback.format_exc()
            error_ctx = runner.get_error_context()
            rich_error = f"{e}\n--- context: {error_ctx}\n--- traceback (last 5 frames):\n"
            tb_lines = tb.strip().splitlines()
            rich_error += "\n".join(tb_lines[-10:])
            task.status = "failed"
            task.error = rich_error
            task.completed_at = datetime.now(timezone.utc)
            self.db.save_task(task)
            with self._lock:
                state = self._agents[agent_id]
                state.status = AgentStatus.ERROR
                state.error = rich_error
            self._fire_progress(on_progress, {"kind": "task_failed", "task_id": task.id, "error": rich_error})
        finally:
            self.db.save_task(task)

    async def _execute_task(
        self,
        agent_id: str,
        runner: AgentRunner,
        task: Task,
        on_progress: ProgressCallback = None,
    ) -> None:
        """Execute a task and update state on completion."""
        log_path = self.log_dir / f"{agent_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def on_message(msg: object) -> None:
            with open(log_path, "a") as f:
                f.write(f"{msg}\n")

        # Merge per-task callback with global listeners so events reach both
        def _combined_progress(event: dict[str, Any]) -> None:
            self._fire_progress(on_progress, event)
            for cb in self._progress_listeners:
                try:
                    cb(event)
                except Exception:
                    pass

        self._fire_progress(on_progress, {"kind": "status_change", "status": "running", "task_id": task.id})

        try:
            state = self._agents[agent_id]
            if state.config.model.startswith("external:"):
                ext_runner = ExternalModelRunner(state.config.model)
                result = await ext_runner.run(task.prompt, state.config.system_prompt)
            else:
                result = await runner.run_task(
                    task,
                    on_message=on_message,
                    on_progress=_combined_progress,
                )

            # Persist session_id immediately so resume works even if we crash later
            self.db.save_task(task)

            # Check if the Brain set workflow to waiting_for_input
            if task.workflow_id:
                wf = self.db.get_workflow(task.workflow_id)
                if wf and wf.status == WorkflowStatus.WAITING_FOR_INPUT:
                    task.status = "waiting_for_input"
                    task.result = result
                    with self._lock:
                        state = self._agents[agent_id]
                        state.status = AgentStatus.IDLE
                        state.current_task_id = None
                    self._fire_progress(on_progress, {"kind": "waiting_for_input", "task_id": task.id})
                    self.db.save_task(task)
                    return

            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now(timezone.utc)
            # Persist BEFORE firing callback so readers see consistent state
            self.db.save_task(task)
            with self._lock:
                state = self._agents[agent_id]
                state.status = AgentStatus.IDLE
                state.current_task_id = None
            self._fire_progress(on_progress, {"kind": "task_completed", "task_id": task.id})
        except Exception as e:
            logger.exception("Task %s failed for agent %s", task.id, agent_id)
            # Build rich error with last tool call context and traceback
            tb = traceback.format_exc()
            error_ctx = runner.get_error_context()
            rich_error = f"{e}\n--- context: {error_ctx}\n--- traceback (last 5 frames):\n"
            tb_lines = tb.strip().splitlines()
            rich_error += "\n".join(tb_lines[-10:])
            task.status = "failed"
            task.error = rich_error
            task.completed_at = datetime.now(timezone.utc)
            self.db.save_task(task)
            with self._lock:
                state = self._agents[agent_id]
                state.status = AgentStatus.ERROR
                state.error = rich_error
            self._fire_progress(on_progress, {"kind": "task_failed", "task_id": task.id, "error": rich_error})
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
