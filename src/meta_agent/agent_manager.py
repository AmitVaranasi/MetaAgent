"""Agent lifecycle manager. Runs SDK agents as asyncio tasks."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_runner import AgentRunError, AgentRunner
from .db import Database
from .external_runner import ExternalModelRunner
from .models import AgentConfig, AgentState, AgentStatus, Task, WorkflowStatus

ProgressCallback = Callable[[dict[str, Any]], None] | None

# A task in one of these states is being worked on by SOME process.
IN_FLIGHT_STATUSES = ("pending", "running")
# waiting_for_input is deliberately excluded: it survives a restart on purpose,
# because task.session_id lets a later process resume the conversation.

ORPHANED_ERROR = (
    "Orphaned: the process that owned this task exited before it finished "
    "(pid {pid} is gone). Marked failed at startup."
)


def _pid_alive(pid: int | None) -> bool:
    """Whether a process id is still running.

    Signal 0 performs the permission and existence checks without delivering a
    signal. Pid reuse could in principle make a dead owner look alive; for a
    local tool that is an acceptable trade against never reaping at all.
    """
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True

logger = logging.getLogger(__name__)


@dataclass
class _Run:
    """One in-flight task and the runner driving it."""

    agent_id: str
    runner: AgentRunner


class AgentManager:
    def __init__(self, db: Database, log_dir: Path):
        self.db = db
        self.log_dir = log_dir
        self._lock = threading.Lock()
        self._agents: dict[str, AgentState] = {}
        # Keyed by TASK id, not agent id: an agent can legitimately have several
        # tasks in flight, and keying by agent silently dropped all but the last.
        self._runs: dict[str, _Run] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        # External listeners for live progress events (e.g. from report_progress MCP tool)
        self._progress_listeners: list[Callable[[dict[str, Any]], None]] = []
        # Built on first use; holds a live server object, so it is never persisted.
        self._mcp_servers: dict[str, Any] | None = None

    def mcp_servers_for(self, config: AgentConfig) -> dict[str, Any] | None:
        """Return the MCP servers an agent should run with.

        Agents that orchestrate (the Brain) get an in-process server bound to
        THIS manager, so their tool calls reach this object graph rather than a
        subprocess with its own Database and event loop.
        """
        if not config.use_meta_agent_mcp:
            return None
        if self._mcp_servers is None:
            # Imported here: mcp_server imports AgentManager for typing.
            from .mcp_server import MCP_SERVER_NAME, create_inprocess_mcp_server

            self._mcp_servers = {MCP_SERVER_NAME: create_inprocess_mcp_server(self)}
        return self._mcp_servers

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
        """Load agents from DB, reap orphans, start the background event loop."""
        for config in self.db.list_agents():
            with self._lock:
                self._agents[config.id] = AgentState(config=config)
        self.reap_orphaned_tasks()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._loop_thread.start()

    def reap_orphaned_tasks(self) -> list[str]:
        """Fail tasks whose owning process is gone. Returns the task ids.

        Nothing used to reconcile the store on startup, so a task left `pending`
        or `running` by a crashed or killed process stayed that way forever —
        13 such rows accumulated in the shipped database.

        Only tasks whose `owner_pid` is dead are touched, so running this in one
        process does not disturb tasks another live process is still working on
        (every CLI command builds a manager, so that case is routine).
        """
        reaped: list[str] = []
        for task in self.db.list_tasks():
            if task.status not in IN_FLIGHT_STATUSES or _pid_alive(task.owner_pid):
                continue
            task.status = "failed"
            task.error = ORPHANED_ERROR.format(pid=task.owner_pid)
            task.completed_at = datetime.now(timezone.utc)
            self.db.save_task(task)
            self._fail_workflow_for(task)
            reaped.append(task.id)
        if reaped:
            logger.info("Reaped %d orphaned task(s): %s", len(reaped), ", ".join(reaped))
        return reaped

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel in-flight runs, let them record, then stop the loop.

        This used to call loop.stop() and return, so tasks died mid-write and
        their rows were left in whatever state they happened to be in.
        """
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            runners = [run.runner for run in self._runs.values()]
        for runner in runners:
            asyncio.run_coroutine_threadsafe(runner.cancel(), loop)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._runs:
                    break
            time.sleep(0.02)

        loop.call_soon_threadsafe(loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        self._loop = None

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
        self.cancel_agent_tasks(agent_id)
        self.db.delete_agent(agent_id)
        return True

    def cancel_agent_tasks(self, agent_id: str) -> list[str]:
        """Cancel every task in flight for an agent. Returns the task ids.

        Previously only the most recently submitted task could be reached, so a
        stop or delete left the agent's earlier tasks running.
        """
        with self._lock:
            runs = [(tid, run.runner) for tid, run in self._runs.items() if run.agent_id == agent_id]
        if self._loop:
            for _, runner in runs:
                asyncio.run_coroutine_threadsafe(runner.cancel(), self._loop)
        return [tid for tid, _ in runs]

    def running_task_ids(self, agent_id: str) -> list[str]:
        """Task ids currently in flight for an agent."""
        with self._lock:
            return [tid for tid, run in self._runs.items() if run.agent_id == agent_id]

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
            owner_pid=os.getpid(),
            model=state.config.model,
        )
        self.db.save_task(task)

        runner = AgentRunner(state.config, mcp_servers=self.mcp_servers_for(state.config))
        self._start_run(agent_id, task.id, runner)

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

        runner = AgentRunner(state.config, mcp_servers=self.mcp_servers_for(state.config))
        self._start_run(agent_id, task.id, runner)

        task.status = "running"
        task.owner_pid = os.getpid()
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

    def _start_run(self, agent_id: str, task_id: str, runner: AgentRunner) -> None:
        """Register an in-flight task and mark its agent running."""
        with self._lock:
            self._runs[task_id] = _Run(agent_id, runner)
            state = self._agents.get(agent_id)
            if state is None:
                return
            if task_id not in state.running_task_ids:
                state.running_task_ids.append(task_id)
            state.status = AgentStatus.RUNNING
            state.current_task_id = task_id
            state.started_at = datetime.now(timezone.utc)

    def _end_run(
        self,
        agent_id: str,
        task_id: str,
        terminal_status: AgentStatus,
        error: str | None = None,
    ) -> AgentState | None:
        """Retire one run and settle the agent around whatever is still running.

        Tolerates a deleted agent: the Brain is instructed to delete every agent
        it created (Phase 6) and can do that while a task is still going, so a
        lookup after the task starts may legitimately miss. Returns the state,
        or None if the agent is gone.
        """
        with self._lock:
            self._runs.pop(task_id, None)
            state = self._agents.get(agent_id)
            if state is None:
                return None
            if task_id in state.running_task_ids:
                state.running_task_ids.remove(task_id)
            if error is not None:
                state.error = error
            if state.running_task_ids:
                # Other tasks are still in flight — the agent is not idle yet.
                state.status = AgentStatus.RUNNING
                if state.current_task_id == task_id:
                    state.current_task_id = state.running_task_ids[-1]
            else:
                state.status = terminal_status
                state.current_task_id = None
                if error is None:
                    state.error = None
            return state

    def _fail_workflow_for(self, task: Task, error: str | None = None) -> None:
        """Fail the workflow this task was orchestrating, if it was.

        WorkflowStatus.FAILED existed but only the Brain could ever set it, via
        the update_workflow tool — so a workflow could only be failed by the
        very agent whose death is the usual reason it needs failing. 21 of 33
        workflows in the shipped database are stranded in planning / executing /
        assembling for exactly that reason.

        Only the workflow's OWN brain task fails it. A subtask failing is
        recoverable — Phase 4 tells the Brain to retry with an adjusted prompt —
        so it must not tear the workflow down.
        """
        if not task.workflow_id:
            return
        wf = self.db.get_workflow(task.workflow_id)
        if wf is None:
            return
        if wf.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED):
            return
        is_brain_task = task.id == wf.brain_task_id or task.agent_id == wf.brain_agent_id
        if not is_brain_task:
            return
        wf.status = WorkflowStatus.FAILED
        wf.error = error or task.error or "The orchestrating task ended without completing."
        wf.completed_at = datetime.now(timezone.utc)
        self.db.save_workflow(wf)

    @staticmethod
    def _apply_stats(task: Task, runner: AgentRunner) -> None:
        """Copy what the run cost onto the task, whether it succeeded or not."""
        stats = runner.stats
        task.cost_usd = stats.cost_usd
        task.num_turns = stats.num_turns
        task.stop_reason = stats.stop_reason
        task.usage = stats.usage

    def workflow_usage(self, workflow_id: str) -> dict[str, Any]:
        """What a workflow has cost so far, in total and per model.

        The per-model split is the number that answers the question the whole
        design rests on: whether delegating to cheaper sub-agents beats doing
        the work in one expensive session.
        """
        tasks = self.db.list_workflow_tasks(workflow_id)
        by_model: dict[str, dict[str, Any]] = {}
        totals = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "num_turns": 0}
        for task in tasks:
            model = task.model or "unknown"
            bucket = by_model.setdefault(
                model,
                {"tasks": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0},
            )
            bucket["tasks"] += 1
            bucket["cost_usd"] += task.cost_usd or 0.0
            totals["cost_usd"] += task.cost_usd or 0.0
            totals["num_turns"] += task.num_turns or 0
            for field in ("input_tokens", "output_tokens"):
                value = task.usage.get(field, 0) or 0
                bucket[field] += value
                totals[field] += value
        return {"task_count": len(tasks), "totals": totals, "by_model": by_model}

    def _mark_running(self, task: Task, on_progress: ProgressCallback) -> None:
        """Persist the running status so readers outside this process see it."""
        task.status = "running"
        self.db.save_task(task)
        self._fire_progress(
            on_progress, {"kind": "status_change", "status": "running", "task_id": task.id}
        )

    def _record_failure(
        self,
        agent_id: str,
        runner: AgentRunner,
        task: Task,
        exc: Exception,
        on_progress: ProgressCallback,
    ) -> AgentState | None:
        """Persist a task failure with rich context. Returns the agent state, if any."""
        self._apply_stats(task, runner)
        if isinstance(exc, AgentRunError) and exc.partial_result:
            # Keep the partial work — a max_turns stop still produced something.
            task.result = exc.partial_result
        tb = traceback.format_exc()
        error_ctx = runner.get_error_context()
        rich_error = f"{exc}\n--- context: {error_ctx}\n--- traceback (last 10 frames):\n"
        rich_error += "\n".join(tb.strip().splitlines()[-10:])
        task.status = "failed"
        task.error = rich_error
        task.completed_at = datetime.now(timezone.utc)
        self.db.save_task(task)
        self._fail_workflow_for(task)
        state = self._end_run(agent_id, task.id, AgentStatus.ERROR, error=rich_error)
        self._fire_progress(
            on_progress, {"kind": "task_failed", "task_id": task.id, "error": rich_error}
        )
        return state

    def _open_log(self, agent_id: str) -> Callable[[object], None]:
        """Return an on_message sink that appends to the agent's log file."""
        log_path = self.log_dir / f"{agent_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def on_message(msg: object) -> None:
            with open(log_path, "a") as f:
                f.write(f"{msg}\n")

        return on_message

    def _combined_progress(self, on_progress: ProgressCallback) -> Callable[[dict[str, Any]], None]:
        """Merge the per-task callback with the global listeners."""

        def fire(event: dict[str, Any]) -> None:
            self._fire_progress(on_progress, event)
            for cb in list(self._progress_listeners):
                try:
                    cb(event)
                except Exception:
                    logger.debug("Progress listener error", exc_info=True)

        return fire

    def _finish_or_pause(
        self,
        agent_id: str,
        task: Task,
        result: str,
        on_progress: ProgressCallback,
        runner: AgentRunner | None = None,
    ) -> None:
        """Complete a task, or park it if the Brain asked the user a question."""
        if runner is not None:
            self._apply_stats(task, runner)
        if task.workflow_id:
            wf = self.db.get_workflow(task.workflow_id)
            if wf and wf.status == WorkflowStatus.WAITING_FOR_INPUT:
                task.status = "waiting_for_input"
                task.result = result
                self.db.save_task(task)
                self._end_run(agent_id, task.id, AgentStatus.IDLE)
                self._fire_progress(on_progress, {"kind": "waiting_for_input", "task_id": task.id})
                return

        task.status = "completed"
        task.result = result
        task.completed_at = datetime.now(timezone.utc)
        # Persist BEFORE firing callback so readers see consistent state
        self.db.save_task(task)
        self._end_run(agent_id, task.id, AgentStatus.IDLE)
        self._fire_progress(on_progress, {"kind": "task_completed", "task_id": task.id})

    async def _execute_resume(
        self,
        agent_id: str,
        runner: AgentRunner,
        task: Task,
        user_response: str,
        on_progress: ProgressCallback = None,
    ) -> None:
        """Execute a resumed task and update state on completion."""
        self._mark_running(task, on_progress)
        try:
            result = await runner.resume_task(
                task, user_response,
                on_message=self._open_log(agent_id),
                on_progress=self._combined_progress(on_progress),
            )
            self._finish_or_pause(agent_id, task, result, on_progress, runner)
        except asyncio.CancelledError:
            # stop_agent / unregister_agent reached this task. Record it as
            # cancelled rather than leaving it mid-flight, then let the
            # cancellation continue to propagate.
            task.status = "cancelled"
            task.completed_at = datetime.now(timezone.utc)
            self.db.save_task(task)
            self._fail_workflow_for(task, error="The orchestrating task was cancelled.")
            self._end_run(agent_id, task.id, AgentStatus.STOPPED)
            self._fire_progress(on_progress, {"kind": "task_cancelled", "task_id": task.id})
            raise
        except Exception as e:
            logger.exception("Resume task %s failed for agent %s", task.id, agent_id)
            self._record_failure(agent_id, runner, task, e, on_progress)
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
        self._mark_running(task, on_progress)
        try:
            state = self.get_agent(agent_id)
            if state is None:
                raise RuntimeError(f"Agent {agent_id} was deleted before its task could run")

            if state.config.model.startswith("external:"):
                ext_runner = ExternalModelRunner(state.config.model)
                result = await ext_runner.run(task.prompt, state.config.system_prompt)
            else:
                result = await runner.run_task(
                    task,
                    on_message=self._open_log(agent_id),
                    on_progress=self._combined_progress(on_progress),
                )

            # Persist session_id immediately so resume works even if we crash later
            self.db.save_task(task)
            self._finish_or_pause(agent_id, task, result, on_progress, runner)
        except asyncio.CancelledError:
            # stop_agent / unregister_agent reached this task. Record it as
            # cancelled rather than leaving it mid-flight, then let the
            # cancellation continue to propagate.
            task.status = "cancelled"
            task.completed_at = datetime.now(timezone.utc)
            self.db.save_task(task)
            self._fail_workflow_for(task, error="The orchestrating task was cancelled.")
            self._end_run(agent_id, task.id, AgentStatus.STOPPED)
            self._fire_progress(on_progress, {"kind": "task_cancelled", "task_id": task.id})
            raise
        except Exception as e:
            logger.exception("Task %s failed for agent %s", task.id, agent_id)
            state = self._record_failure(agent_id, runner, task, e, on_progress)
            # state is None when the agent was deleted mid-run; there is then
            # nothing to restart. Reading self._agents[agent_id] here used to
            # raise a KeyError *inside* the handler and orphan the task.
            if (
                state is not None
                and state.config.auto_restart
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
