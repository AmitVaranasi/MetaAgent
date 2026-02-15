"""FastMCP server exposing 10 tools for agent management."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .agent_manager import AgentManager
from .models import AgentConfig


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
    ) -> dict:
        """Create and register a new agent."""
        kwargs: dict = dict(
            name=name,
            system_prompt=system_prompt,
            description=description,
            model=model,
            cwd=cwd,
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
    def submit_task(agent_id: str, prompt: str) -> dict:
        """Submit a task prompt to an agent for execution."""
        try:
            task = manager.submit_task(agent_id, prompt)
            return {"task_id": task.id, "agent_id": agent_id, "status": task.status}
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def task_status(task_id: str) -> dict:
        """Get the status and result of a task."""
        task = manager.get_task(task_id)
        if task is None:
            return {"error": f"Task {task_id} not found"}
        return {
            "id": task.id,
            "agent_id": task.agent_id,
            "status": task.status,
            "prompt": task.prompt,
            "result": task.result,
            "error": task.error,
            "created_at": str(task.created_at),
            "completed_at": str(task.completed_at) if task.completed_at else None,
        }

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

    return mcp
