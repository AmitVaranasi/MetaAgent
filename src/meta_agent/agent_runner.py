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

    async def run_task(
        self,
        task: Task,
        on_message: Callable[[Any], None] | None = None,
    ) -> str:
        """Execute a task prompt via the SDK. Returns the final result text."""
        options = ClaudeAgentOptions(
            system_prompt=self.config.system_prompt,
            allowed_tools=self.config.allowed_tools,
            model=self.config.model,
            max_turns=self.config.max_turns,
            permission_mode=self.config.permission_mode,
            cwd=self.config.cwd,
        )

        if self.config.max_budget_usd:
            options.max_budget_usd = self.config.max_budget_usd

        if self.config.mcp_servers:
            options.mcp_servers = self.config.mcp_servers

        result_text = ""
        async for message in query(prompt=task.prompt, options=options):
            if on_message:
                on_message(message)
            if hasattr(message, "content") and getattr(message, "type", None) == "assistant":
                result_text = message.content
            if hasattr(message, "session_id"):
                task.session_id = message.session_id

        return result_text

    async def cancel(self) -> None:
        """Cancel the currently running task."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
