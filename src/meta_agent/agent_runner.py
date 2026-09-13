"""Run agents using the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from .models import AgentConfig, Task

logger = logging.getLogger(__name__)

# Type alias for the structured progress callback
ProgressCallback = Callable[[dict[str, Any]], None] | None

# How many stderr lines from the Claude CLI to keep for error reporting.
STDERR_BUFFER_LINES = 50

_PREVIEW_CHARS = 200


@dataclass
class RunStats:
    """What a run cost, read off the SDK's final ResultMessage.

    The SDK reports all of this on every run and the system used to discard it,
    so there was no way to answer what a workflow cost or whether delegating to
    cheaper sub-agents actually saved anything.
    """

    cost_usd: float | None = None
    num_turns: int | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class AgentRunError(RuntimeError):
    """The SDK ended a run with is_error set.

    Most commonly `error_max_turns`: the run hit max_turns and stopped. That
    used to end the async-for normally, so the task was stored `completed` with
    whatever partial text it had and the Brain assembled on truncated work.
    """

    def __init__(self, subtype: str, stop_reason: str | None = None, partial_result: str = ""):
        self.subtype = subtype
        self.stop_reason = stop_reason
        self.partial_result = partial_result
        detail = f" (stop_reason={stop_reason})" if stop_reason else ""
        super().__init__(f"The run ended with error subtype {subtype!r}{detail}")


def _preview(value: Any) -> str | None:
    """Render a tool input/output as a short single-line preview."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    if not text:
        return None
    return text[:_PREVIEW_CHARS]


def parse_sdk_message(
    message: Any,
    agent_id: str,
    tool_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Extract progress events from one SDK message.

    The SDK yields dataclasses — ``AssistantMessage``, ``UserMessage``,
    ``ResultMessage``. Tool use and tool results are content *blocks* inside
    those messages, never messages of their own, so a single message can carry
    several events (or none).

    ``tool_names`` maps ``tool_use_id`` -> tool name. It is populated from
    ``ToolUseBlock``s and read back when the matching ``ToolResultBlock``
    arrives, since a result block carries only the id.
    """
    events: list[dict[str, Any]] = []

    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                if tool_names is not None:
                    tool_names[block.id] = block.name
                events.append(
                    {
                        "kind": "tool_call",
                        "agent_id": agent_id,
                        "tool": block.name,
                        "tool_use_id": block.id,
                        "input_preview": _preview(block.input),
                    }
                )

    elif isinstance(message, UserMessage):
        content = message.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, ToolResultBlock):
                    events.append(
                        {
                            "kind": "tool_result",
                            "agent_id": agent_id,
                            "tool": (tool_names or {}).get(block.tool_use_id),
                            "tool_use_id": block.tool_use_id,
                            "is_error": bool(block.is_error),
                            "output_preview": _preview(block.content),
                        }
                    )

    return events


def _assistant_text(message: AssistantMessage) -> str:
    """Join the text blocks of an assistant message."""
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))


class AgentRunner:
    """Executes a task using the Claude Agent SDK."""

    def __init__(self, config: AgentConfig, mcp_servers: dict[str, Any] | None = None):
        self.config = config
        # Live MCP server configs, supplied per run. In-process SDK servers hold
        # an object instance and so cannot travel in the persisted AgentConfig.
        self.mcp_servers = mcp_servers
        self._current_task: asyncio.Task[Any] | None = None
        # Track the last tool call for richer error context
        self.last_tool_call: str | None = None
        # tool_use_id -> tool name, so a ToolResultBlock can be named
        self._tool_names: dict[str, str] = {}
        # The Claude CLI reports most startup failures ONLY on stderr — the SDK
        # exception for them is the contentless "Command failed with exit code 1".
        self._stderr: deque[str] = deque(maxlen=STDERR_BUFFER_LINES)
        # Populated from the final ResultMessage, on success and on failure.
        self.stats = RunStats()

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
            stderr=self._stderr.append,
        )

        if resume_session_id:
            options.resume = resume_session_id

        if self.config.max_budget_usd:
            options.max_budget_usd = self.config.max_budget_usd

        mcp_servers = self.mcp_servers or self.config.mcp_servers
        if mcp_servers:
            options.mcp_servers = mcp_servers

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
        # Recorded so cancel() has something to cancel — it was never set, which
        # made stop_agent and unregister_agent silent no-ops.
        self._current_task = asyncio.current_task()
        options = self._build_options(resume_session_id=resume_session_id)

        result_text = ""
        failed_result: ResultMessage | None = None
        async for message in query(prompt=prompt, options=options):
            if on_message:
                on_message(message)

            # --- Emit structured progress events ---
            events = parse_sdk_message(message, task.agent_id, self._tool_names)
            for event in events:
                if event["kind"] == "tool_call":
                    self.last_tool_call = event["tool"]
                if on_progress:
                    event["task_id"] = task.id
                    try:
                        on_progress(event)
                    except Exception:
                        logger.debug("Progress callback error", exc_info=True)

            # ResultMessage carries the final answer; fall back to the last
            # assistant turn's text if the run ends without one.
            if isinstance(message, ResultMessage):
                self.stats = RunStats(
                    cost_usd=message.total_cost_usd,
                    num_turns=message.num_turns,
                    stop_reason=message.stop_reason,
                    usage=message.usage or {},
                )
                if message.result:
                    result_text = message.result
                if message.is_error:
                    failed_result = message
            elif isinstance(message, AssistantMessage):
                text = _assistant_text(message)
                if text:
                    result_text = text

            session_id = getattr(message, "session_id", None)
            if session_id:
                task.session_id = session_id

        if failed_result is not None:
            raise AgentRunError(
                failed_result.subtype,
                stop_reason=failed_result.stop_reason,
                partial_result=result_text,
            )

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
        """Return context about the last operation for richer error messages.

        Always names the model, because a CLI that rejects its own arguments
        exits 1 with an empty stderr and an exception that says nothing.
        """
        parts = [f"model={self.config.model}"]
        if self.stats.num_turns is not None:
            parts.append(f"turns={self.stats.num_turns}")
        if self.stats.cost_usd is not None:
            parts.append(f"cost_usd={self.stats.cost_usd:.4f}")
        if self.last_tool_call:
            parts.append(f"last_tool_call={self.last_tool_call}")
        stderr_tail = [line for line in self._stderr if line.strip()]
        if stderr_tail:
            parts.append("cli_stderr=" + " | ".join(stderr_tail[-10:]))
        else:
            parts.append("cli_stderr=<empty>")
        return "; ".join(parts)

    async def cancel(self) -> None:
        """Cancel the run this runner is driving, if it is still going."""
        task = self._current_task
        if task is not None and not task.done():
            task.cancel()
