"""The meta-agent MCP tool surface.

The tools are declared once, closed over one AgentManager, and exposed two
ways:

* :func:`create_mcp_server` — a FastMCP stdio server, for attaching meta-agent
  to an *external* Claude Code session.
* :func:`create_inprocess_mcp_server` — an SDK in-process server. This is what
  the Brain uses. Spawning the stdio server instead gave the Brain a second OS
  process with its own Database and AgentManager, so sub-agents ran somewhere
  the chat could not see and died when the Brain's session ended (ADR note in
  docs/architecture-review.md).
"""

from __future__ import annotations

import inspect
import json
import types as pytypes
import typing
from typing import Any, Callable

from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool, create_sdk_mcp_server
from mcp.server.fastmcp import FastMCP

from .agent_manager import AgentManager
from .models import AgentConfig, Workflow

MCP_SERVER_NAME = "meta-agent"


def _tool_functions(manager: AgentManager) -> list[Callable[..., Any]]:
    """Build the tool functions, closed over `manager`."""

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
            "running_task_ids": state.running_task_ids,
            "error": state.error,
            "session_id": state.session_id,
            "started_at": str(state.started_at) if state.started_at else None,
        }

    def create_agent(
        name: str,
        system_prompt: str,
        description: str = "",
        allowed_tools: list[str] | None = None,
        model: str = "claude-sonnet-5",
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
            model: Model identifier (e.g. claude-sonnet-5).
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

    def delete_agent(agent_id: str) -> dict:
        """Delete an agent by ID."""
        if manager.unregister_agent(agent_id):
            return {"deleted": True, "agent_id": agent_id}
        return {"error": f"Agent {agent_id} not found"}

    def start_agent(agent_id: str) -> dict:
        """Bring a stopped agent back into service so it accepts tasks again."""
        state = manager.start_agent(agent_id)
        if state is None:
            return {"error": f"Agent {agent_id} not found"}
        return {"id": agent_id, "status": state.status.value}

    def stop_agent(agent_id: str) -> dict:
        """Stop an agent: cancel its in-flight tasks and refuse new ones."""
        cancelled = manager.running_task_ids(agent_id)
        state = manager.stop_agent(agent_id)
        if state is None:
            return {"error": f"Agent {agent_id} not found"}
        return {"id": agent_id, "status": state.status.value, "cancelled_task_ids": cancelled}

    def agent_logs(agent_id: str, lines: int = 100) -> str:
        """Get recent logs for an agent."""
        return manager.get_logs(agent_id, lines=lines)

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
            manager.broadcast_progress({
                "kind": "subtask_submitted",
                "task_id": task.id,
                "agent_id": agent_id,
                "workflow_id": workflow_id,
                "prompt": prompt,
            })
            return {"task_id": task.id, "agent_id": agent_id, "status": task.status}
        except ValueError as e:
            return {"error": str(e)}

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
        if task.status in ("completed", "failed", "cancelled"):
            response["result"] = task.result
            response["error"] = task.error
            response["completed_at"] = str(task.completed_at) if task.completed_at else None
            response["cost_usd"] = task.cost_usd
            response["num_turns"] = task.num_turns
            response["stop_reason"] = task.stop_reason
        elif task.error:
            # Surface errors even while running (e.g. retries)
            response["error"] = task.error

        return response

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
            response["cost_usd"] = round(
                manager.workflow_usage(workflow_id)["totals"]["cost_usd"], 4
            )
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
            "usage": manager.workflow_usage(workflow_id),
            "status": workflow.status.value,
            "brain_agent_id": workflow.brain_agent_id,
            "brain_task_id": workflow.brain_task_id,
            "subtasks": subtasks,
            "result": workflow.result,
            "error": workflow.error,
            "created_at": str(workflow.created_at),
            "completed_at": str(workflow.completed_at) if workflow.completed_at else None,
        }

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
        manager.broadcast_progress({
            "kind": "workflow_update",
            "workflow_id": workflow.id,
            "status": workflow.status.value,
            "plan": plan,
            "result": result,
            "error": error,
        })
        return {"id": workflow.id, "status": workflow.status.value}

    def workflow_usage(workflow_id: str) -> dict:
        """What a workflow has cost so far: totals and a per-model breakdown.

        Use it to decide whether to keep delegating or stop early, and to report
        the cost of the work in your final summary.
        """
        return manager.workflow_usage(workflow_id)

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

    return [
        list_agents,
        get_agent,
        create_agent,
        delete_agent,
        start_agent,
        stop_agent,
        agent_logs,
        submit_task,
        task_status,
        list_tasks,
        report_progress,
        create_workflow,
        workflow_status,
        update_workflow,
        workflow_usage,
        list_workflows,
    ]


# --- schema derivation -------------------------------------------------------

_JSON_TYPES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(annotation: Any) -> dict[str, Any]:
    """Map a parameter annotation to a JSON Schema fragment."""
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}

    origin = typing.get_origin(annotation)
    if origin in (typing.Union, pytypes.UnionType):
        variants = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(variants) == 1:
            return _json_type(variants[0])
        return {"anyOf": [_json_type(v) for v in variants]}
    if origin is list:
        args = typing.get_args(annotation)
        return {"type": "array", "items": _json_type(args[0])} if args else {"type": "array"}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if annotation is list:
        return {"type": "array"}
    return {"type": "string"}


def _input_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema from a tool function's signature.

    Written out rather than handed to the SDK as a {name: type} dict, because
    that shorthand marks every parameter required — most of these tools have
    optional ones.
    """
    hints = typing.get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        properties[name] = _json_type(hints.get(name, str))
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _as_sdk_tool(fn: Callable[..., Any]) -> SdkMcpTool[Any]:
    """Adapt a tool function to the SDK's in-process tool protocol."""

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        result = fn(**args)
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        return {"content": [{"type": "text", "text": text}]}

    return SdkMcpTool(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        input_schema=_input_schema(fn),
        handler=handler,
    )


# --- the two transports ------------------------------------------------------


def create_mcp_server(manager: AgentManager) -> FastMCP:
    """A stdio MCP server, for attaching meta-agent to an external session."""
    mcp = FastMCP(MCP_SERVER_NAME)
    for fn in _tool_functions(manager):
        mcp.tool()(fn)
    return mcp


def create_inprocess_mcp_server(manager: AgentManager) -> McpSdkServerConfig:
    """An in-process MCP server sharing this process's AgentManager.

    Tool calls reach the same object graph as the caller, so sub-agents run on
    the caller's event loop, report_progress reaches listeners that actually
    exist, and there is no second Database to diverge from.
    """
    return create_sdk_mcp_server(
        name=MCP_SERVER_NAME,
        tools=[_as_sdk_tool(fn) for fn in _tool_functions(manager)],
    )
