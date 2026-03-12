"""Run agents using the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Callable

from claude_agent_sdk import ClaudeAgentOptions, query

from .models import AgentConfig, Task

logger = logging.getLogger(__name__)

# Type alias for the structured progress callback
ProgressCallback = Callable[[dict[str, Any]], None] | None


def _parse_sdk_message(message: Any, agent_id: str) -> dict[str, Any] | None:
    """Extract a structured progress event from an SDK message.

    Returns a dict suitable for on_progress callbacks, or None if the message
    is not interesting for observability.
    """
    msg_type = getattr(message, "type", None) or type(message).__name__

    # Tool-use request (agent decided to call a tool)
    if msg_type in ("tool_use", "ToolUseMessage") or hasattr(message, "tool_name"):
        tool_name = getattr(message, "tool_name", None) or getattr(message, "name", None)
        tool_input = getattr(message, "input", None) or getattr(message, "tool_input", None)
        if tool_name:
            return {
                "kind": "tool_call",
                "agent_id": agent_id,
                "tool": tool_name,
                "input_preview": str(tool_input)[:200] if tool_input else None,
            }

    # Tool result (tool finished executing)
    if msg_type in ("tool_result", "ToolResultMessage") or hasattr(message, "tool_result"):
        tool_name = getattr(message, "tool_name", None) or getattr(message, "name", None)
        output = getattr(message, "tool_result", None) or getattr(message, "output", None)
        is_error = getattr(message, "is_error", False)
        return {
            "kind": "tool_result",
            "agent_id": agent_id,
            "tool": tool_name,
            "is_error": is_error,
            "output_preview": str(output)[:200] if output else None,
        }

    return None


class AgentRunner:
    """Executes a task using the Claude Agent SDK."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._current_task: asyncio.Task[Any] | None = None
        # Track the last tool call for richer error context
        self.last_tool_call: str | None = None

    def _build_options(self, resume_session_id: str | None = None) -> ClaudeAgentOptions:
        """Build SDK options, optionally resuming a previous session."""
        options = ClaudeAgentOptions(
            system_prompt=self.config.system_prompt,
            allowed_tools=self.config.allowed_tools,
            disallowed_tools=self.config.disallowed_tools,
            model=self.config.model,
            max_turns=self.config.max_turns,
            permission_mode=self.config.permission_mode,
            cwd=self.config.cwd,
        )

        if resume_session_id:
            options.resume = resume_session_id

        if self.config.max_budget_usd:
            options.max_budget_usd = self.config.max_budget_usd

        if self.config.mcp_servers:
            options.mcp_servers = self.config.mcp_servers

        return options

    async def _run(
        self,
        task: Task,
        prompt: str,
        on_message: Callable[[Any], None] | None = None,
        on_progress: ProgressCallback = None,
        resume_session_id: str | None = None,
    ) -> str:
        """Internal: execute or resume a task. Returns the final result text."""
        options = self._build_options(resume_session_id=resume_session_id)

        result_text = ""
        async for message in query(prompt=prompt, options=options):
            if on_message:
                on_message(message)

            # --- Emit structured progress events ---
            if on_progress:
                event = _parse_sdk_message(message, task.agent_id)
                if event:
                    event["task_id"] = task.id
                    try:
                        on_progress(event)
                    except Exception:
                        pass

            # Track last tool call for error context
            tool_name = getattr(message, "tool_name", None) or getattr(message, "name", None)
            if tool_name:
                self.last_tool_call = tool_name

            # Capture from ResultMessage.result (SDK final message)
            if hasattr(message, "result") and getattr(message, "result", None):
                result_text = message.result
            # Fallback: capture from AssistantMessage.content
            elif hasattr(message, "content") and getattr(message, "type", None) == "assistant":
                result_text = message.content
            if hasattr(message, "session_id"):
                task.session_id = message.session_id

        return result_text

    async def run_task(
        self,
        task: Task,
        on_message: Callable[[Any], None] | None = None,
        on_progress: ProgressCallback = None,
    ) -> str:
        """Execute a task prompt via the SDK. Returns the final result text."""
        return await self._run(task, task.prompt, on_message=on_message, on_progress=on_progress)

    async def resume_task(
        self,
        task: Task,
        user_response: str,
        on_message: Callable[[Any], None] | None = None,
        on_progress: ProgressCallback = None,
    ) -> str:
        """Resume a paused task with the user's response. Returns the final result text."""
        if not task.session_id:
            raise ValueError(f"Task {task.id} has no session_id to resume")
        return await self._run(
            task, user_response,
            on_message=on_message,
            on_progress=on_progress,
            resume_session_id=task.session_id,
        )

    def get_error_context(self) -> str:
        """Return context about the last operation for richer error messages."""
        parts = []
        if self.last_tool_call:
            parts.append(f"last_tool_call={self.last_tool_call}")
        return "; ".join(parts) if parts else "no context"

    async def cancel(self) -> None:
        """Cancel the currently running task."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
