"""Run agents using the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from claude_agent_sdk import ClaudeAgentOptions, query

from .models import AgentConfig, Task

logger = logging.getLogger(__name__)


class AgentRunner:
    """Executes a task using the Claude Agent SDK."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._current_task: asyncio.Task[Any] | None = None

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
        resume_session_id: str | None = None,
    ) -> str:
        """Internal: execute or resume a task. Returns the final result text."""
        options = self._build_options(resume_session_id=resume_session_id)

        result_text = ""
        async for message in query(prompt=prompt, options=options):
            if on_message:
                on_message(message)
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
    ) -> str:
        """Execute a task prompt via the SDK. Returns the final result text."""
        return await self._run(task, task.prompt, on_message=on_message)

    async def resume_task(
        self,
        task: Task,
        user_response: str,
        on_message: Callable[[Any], None] | None = None,
    ) -> str:
        """Resume a paused task with the user's response. Returns the final result text."""
        if not task.session_id:
            raise ValueError(f"Task {task.id} has no session_id to resume")
        return await self._run(
            task, user_response, on_message=on_message, resume_session_id=task.session_id
        )

    async def cancel(self) -> None:
        """Cancel the currently running task."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
