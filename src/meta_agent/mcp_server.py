"""FastMCP server exposing 10 tools for agent management."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .agent_manager import AgentManager
from .models import AgentConfig, Workflow


def create_mcp_server(manager: AgentManager) -> FastMCP:
    mcp = FastMCP("meta-agent")

    @mcp.tool()
    def list_agents() -> list[dict]:
        """List all registered agents with their status."""
        return [
            {
                "id": s.config.id,
                "name": s.config.name,
                "status": s.status.value,
                "description": s.config.description,
                "model": s.config.model,
                "current_task_id": s.current_task_id,
            }
            for s in manager.list_agents()
        ]

    @mcp.tool()
    def get_agent(agent_id: str) -> dict:
        """Get detailed information about an agent."""
        state = manager.get_agent(agent_id)
        if state is None:
            return {"error": f"Agent {agent_id} not found"}
        return {
            "id": state.config.id,
            "name": state.config.name,
            "status": state.status.value,
            "description": state.config.description,
            "model": state.config.model,
            "system_prompt": state.config.system_prompt,
            "allowed_tools": state.config.allowed_tools,
            "current_task_id": state.current_task_id,
            "error": state.error,
            "session_id": state.session_id,
            "started_at": str(state.started_at) if state.started_at else None,
        }

    @mcp.tool()
    def create_agent(
        name: str,
        system_prompt: str,
        description: str = "",
        allowed_tools: list[str] | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        agent_id: str | None = None,
        cwd: str | None = None,
        permission_mode: str = "bypassPermissions",
        max_turns: int = 50,
    ) -> dict:
        """Create and register a new agent.

        Args:
            name: Display name for the agent.
            system_prompt: System prompt that defines the agent's behavior.
            description: Short description of the agent's purpose.
            allowed_tools: List of tool names the agent can use.
            model: Model identifier (e.g. claude-sonnet-4-5-20250929).
            agent_id: Optional custom ID. Auto-generated if omitted.
            cwd: Working directory for the agent.
            permission_mode: Permission mode. Use "bypassPermissions" for automated agents.
            max_turns: Maximum conversation turns before the agent stops.
        """
        kwargs: dict = dict(
            name=name,
            system_prompt=system_prompt,
            description=description,
            model=model,
            cwd=cwd,
            permission_mode=permission_mode,
            max_turns=max_turns,
        )
        if allowed_tools is not None:
            kwargs["allowed_tools"] = allowed_tools
        if agent_id:
            kwargs["id"] = agent_id
        config = AgentConfig(**kwargs)
        state = manager.register_agent(config)
        return {"id": state.config.id, "name": state.config.name, "status": state.status.value}

    @mcp.tool()
    def delete_agent(agent_id: str) -> dict:
        """Delete an agent by ID."""
        if manager.unregister_agent(agent_id):
            return {"deleted": True, "agent_id": agent_id}
        return {"error": f"Agent {agent_id} not found"}

    @mcp.tool()
    def start_agent(agent_id: str) -> dict:
        """Start an agent (set to idle, ready for tasks)."""
        from .models import AgentStatus

        state = manager.get_agent(agent_id)
        if state is None:
            return {"error": f"Agent {agent_id} not found"}
        state.status = AgentStatus.IDLE
        return {"id": agent_id, "status": state.status.value}

    @mcp.tool()
    def stop_agent(agent_id: str) -> dict:
        """Stop an agent."""
        from .models import AgentStatus

        state = manager.get_agent(agent_id)
        if state is None:
            return {"error": f"Agent {agent_id} not found"}
        runner = manager._runners.get(agent_id)
        if runner and manager._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(runner.cancel(), manager._loop)
        state.status = AgentStatus.STOPPED
        state.current_task_id = None
        return {"id": agent_id, "status": state.status.value}

    @mcp.tool()
    def agent_logs(agent_id: str, lines: int = 100) -> str:
        """Get recent logs for an agent."""
        return manager.get_logs(agent_id, lines=lines)

    @mcp.tool()
    def submit_task(
        agent_id: str,
        prompt: str,
        workflow_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> dict:
        """Submit a task prompt to an agent for execution. Optionally link to a workflow."""
        try:
            task = manager.submit_task(
                agent_id, prompt,
                workflow_id=workflow_id,
                parent_task_id=parent_task_id,
            )
            return {"task_id": task.id, "agent_id": agent_id, "status": task.status}
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def task_status(task_id: str) -> dict:
        """Get the status and result of a task.

        Returns a lightweight payload while the task is running (status + error
        only) to conserve tokens.  The full prompt and result are included only
        once the task reaches a terminal state (completed / failed).
        """
        task = manager.get_task(task_id)
        if task is None:
            return {"error": f"Task {task_id} not found"}

        # Always-present lightweight fields
        response: dict = {
            "id": task.id,
            "agent_id": task.agent_id,
            "status": task.status,
        }

        # Include full data only for terminal states to save context tokens
        if task.status in ("completed", "failed"):
            response["result"] = task.result
            response["error"] = task.error
            response["completed_at"] = str(task.completed_at) if task.completed_at else None
        elif task.error:
            # Surface errors even while running (e.g. retries)
            response["error"] = task.error

        return response

    @mcp.tool()
    def list_tasks(agent_id: str | None = None) -> list[dict]:
        """List tasks, optionally filtered by agent ID."""
        tasks = manager.list_tasks(agent_id)
        return [
            {
                "id": t.id,
                "agent_id": t.agent_id,
                "status": t.status,
                "prompt": t.prompt[:100],
                "created_at": str(t.created_at),
            }
            for t in tasks
        ]

    # --- Sub-agent progress reporting ---

    @mcp.tool()
    def report_progress(
        agent_id: str,
        task_id: str,
        message: str,
        phase: str = "working",
    ) -> dict:
        """Report progress from a sub-agent.  Sub-agents should call this
        periodically so the Brain and CLI can show live status updates.

        Args:
            agent_id: The reporting agent's ID.
            task_id: The task the agent is working on.
            message: Short human-readable status line (< 120 chars).
            phase: Current phase, e.g. "reading", "writing", "testing", "done".
        """
        event = {
            "kind": "agent_progress",
            "agent_id": agent_id,
            "task_id": task_id,
            "message": message[:120],
            "phase": phase,
        }
        # Broadcast to any registered progress listeners
        for cb in manager._progress_listeners:
            try:
                cb(event)
            except Exception:
                pass
        return {"ok": True}

    # --- Workflow tools ---

    @mcp.tool()
    def create_workflow(prompt: str) -> dict:
        """Create a new workflow record for brain orchestration."""
        from .brain import BRAIN_AGENT_ID

        workflow = Workflow(prompt=prompt, brain_agent_id=BRAIN_AGENT_ID)
        manager.db.save_workflow(workflow)
        return {
            "workflow_id": workflow.id,
            "status": workflow.status.value,
            "prompt": workflow.prompt,
        }

    @mcp.tool()
    def workflow_status(workflow_id: str, lightweight: bool = True) -> dict:
        """Get workflow status and its subtask statuses.

        Args:
            workflow_id: The workflow to inspect.
            lightweight: When True (default) returns only status counters and
                IDs — dramatically reducing token usage during polling.  Set to
                False to retrieve the full plan, prompt, result and per-subtask
                detail (useful once the workflow is complete).
        """
        workflow = manager.db.get_workflow(workflow_id)
        if workflow is None:
            return {"error": f"Workflow {workflow_id} not found"}

        if lightweight:
            # ---------- compact response ----------
            counts: dict[str, int] = {}
            failed_ids: list[str] = []
            for tid in workflow.subtask_ids:
                task = manager.get_task(tid)
                if task:
                    counts[task.status] = counts.get(task.status, 0) + 1
                    if task.status == "failed":
                        failed_ids.append(tid)
            response: dict = {
                "id": workflow.id,
                "status": workflow.status.value,
                "subtask_count": len(workflow.subtask_ids),
                "subtask_status_counts": counts,
            }
            if failed_ids:
                response["failed_task_ids"] = failed_ids
            if workflow.error:
                response["error"] = workflow.error
            # Include result only when workflow is done
            if workflow.status.value in ("completed", "failed") and workflow.result:
                response["result"] = workflow.result
            return response

        # ---------- full response ----------
        subtasks = []
        for tid in workflow.subtask_ids:
            task = manager.get_task(tid)
            if task:
                subtasks.append({
                    "id": task.id,
                    "agent_id": task.agent_id,
                    "status": task.status,
                    "prompt": task.prompt[:200],
                    "result": task.result if task.result else None,
                    "error": task.error,
                })
        return {
            "id": workflow.id,
            "prompt": workflow.prompt,
            "plan": workflow.plan,
            "status": workflow.status.value,
            "brain_agent_id": workflow.brain_agent_id,
            "brain_task_id": workflow.brain_task_id,
            "subtasks": subtasks,
            "result": workflow.result,
            "error": workflow.error,
            "created_at": str(workflow.created_at),
            "completed_at": str(workflow.completed_at) if workflow.completed_at else None,
        }

    @mcp.tool()
    def update_workflow(
        workflow_id: str,
        status: str | None = None,
        plan: str | None = None,
        result: str | None = None,
        error: str | None = None,
        add_subtask_id: str | None = None,
        brain_task_id: str | None = None,
    ) -> dict:
        """Update a workflow's state. Use add_subtask_id to append a subtask."""
        from .models import WorkflowStatus

        workflow = manager.db.get_workflow(workflow_id)
        if workflow is None:
            return {"error": f"Workflow {workflow_id} not found"}
        if status:
            workflow.status = WorkflowStatus(status)
        if plan is not None:
            workflow.plan = plan
        if result is not None:
            workflow.result = result
        if error is not None:
            workflow.error = error
        if brain_task_id is not None:
            workflow.brain_task_id = brain_task_id
        if add_subtask_id:
            workflow.subtask_ids.append(add_subtask_id)
        if status in ("completed", "failed"):
            from datetime import datetime, timezone
            workflow.completed_at = datetime.now(timezone.utc)
        manager.db.save_workflow(workflow)
        return {"id": workflow.id, "status": workflow.status.value}

    @mcp.tool()
    def list_workflows() -> list[dict]:
        """List all workflows."""
        workflows = manager.db.list_workflows()
        return [
            {
                "id": w.id,
                "prompt": w.prompt[:100],
                "status": w.status.value,
                "subtask_count": len(w.subtask_ids),
                "created_at": str(w.created_at),
            }
            for w in workflows
        ]

    return mcp
