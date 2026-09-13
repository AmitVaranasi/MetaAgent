"""Tests for the agent runner.

These build messages out of the REAL SDK dataclasses. The previous version of
this file used a hand-rolled ``FakeMessage`` whose shape matched the parser's
assumptions rather than the SDK's, which is why a progress parser that never
fired in production passed its tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from meta_agent.agent_runner import AgentRunner, parse_sdk_message
from meta_agent.models import AgentConfig, Task


@pytest.fixture()
def runner_config() -> AgentConfig:
    return AgentConfig(
        id="runner_test",
        name="Runner Test",
        system_prompt="You are a test.",
        allowed_tools=[],
    )


@pytest.fixture()
def runner(runner_config: AgentConfig) -> AgentRunner:
    return AgentRunner(runner_config)


def _result_message(result: str) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="sess-123",
        result=result,
    )


def _sdk_conversation() -> list[object]:
    """A realistic SDK message sequence: think -> tool call -> result -> answer."""
    return [
        AssistantMessage(
            content=[ThinkingBlock(thinking="Let me look.", signature="sig")],
            model="claude-sonnet-5",
        ),
        AssistantMessage(
            content=[ToolUseBlock(id="tu-1", name="Read", input={"file_path": "/tmp/a.txt"})],
            model="claude-sonnet-5",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="tu-1", content="file body")]),
        AssistantMessage(content=[TextBlock(text="Hello, world!")], model="claude-sonnet-5"),
        _result_message("Hello, world!"),
    ]


def _fake_query(messages: list[object]):
    async def _iter(**kwargs):
        for m in messages:
            yield m

    return _iter


# --- parse_sdk_message against real SDK types ---


def test_tool_use_block_inside_assistant_message_yields_event():
    msg = AssistantMessage(
        content=[ToolUseBlock(id="tu-1", name="Bash", input={"command": "ls -la"})],
        model="claude-sonnet-5",
    )
    events = parse_sdk_message(msg, "agent-1")
    assert len(events) == 1
    assert events[0]["kind"] == "tool_call"
    assert events[0]["tool"] == "Bash"
    assert "ls -la" in events[0]["input_preview"]


def test_two_tool_uses_in_one_message_yield_two_events():
    msg = AssistantMessage(
        content=[
            ToolUseBlock(id="tu-1", name="Read", input={}),
            ToolUseBlock(id="tu-2", name="Grep", input={}),
        ],
        model="claude-sonnet-5",
    )
    assert [e["tool"] for e in parse_sdk_message(msg, "agent-1")] == ["Read", "Grep"]


def test_tool_result_is_named_by_correlating_the_tool_use_id():
    names: dict[str, str] = {}
    parse_sdk_message(
        AssistantMessage(
            content=[ToolUseBlock(id="tu-1", name="Read", input={})],
            model="claude-sonnet-5",
        ),
        "agent-1",
        names,
    )
    events = parse_sdk_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="tu-1", content="body")]),
        "agent-1",
        names,
    )
    assert len(events) == 1
    assert events[0]["kind"] == "tool_result"
    assert events[0]["tool"] == "Read"
    assert events[0]["is_error"] is False


def test_tool_result_error_flag_is_carried():
    events = parse_sdk_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="tu-9", content="boom", is_error=True)]),
        "agent-1",
    )
    assert events[0]["is_error"] is True


def test_uninteresting_messages_yield_no_events():
    text_only = AssistantMessage(content=[TextBlock(text="hi")], model="claude-sonnet-5")
    assert parse_sdk_message(text_only, "agent-1") == []
    assert parse_sdk_message(_result_message("done"), "agent-1") == []
    assert parse_sdk_message(UserMessage(content="plain string"), "agent-1") == []


# --- the runner driven by a real SDK message sequence ---


@pytest.mark.asyncio
async def test_run_task_captures_result_from_result_message(runner: AgentRunner):
    task = Task(agent_id="runner_test", prompt="Say hello")
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query(_sdk_conversation())):
        result = await runner.run_task(task)
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_run_task_falls_back_to_assistant_text_without_a_result_message(runner: AgentRunner):
    messages = [AssistantMessage(content=[TextBlock(text="partial answer")], model="claude-sonnet-5")]
    task = Task(agent_id="runner_test", prompt="Say hello")
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query(messages)):
        result = await runner.run_task(task)
    assert result == "partial answer"


@pytest.mark.asyncio
async def test_run_task_emits_progress_events(runner: AgentRunner):
    """The regression this file exists for: real SDK messages must fire events."""
    task = Task(agent_id="runner_test", prompt="Say hello")
    events: list[dict] = []
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query(_sdk_conversation())):
        await runner.run_task(task, on_progress=events.append)

    kinds = [e["kind"] for e in events]
    assert kinds == ["tool_call", "tool_result"]
    assert all(e["task_id"] == task.id for e in events)
    assert events[0]["tool"] == "Read"
    assert events[1]["tool"] == "Read"


@pytest.mark.asyncio
async def test_run_task_records_last_tool_call_for_error_context(runner: AgentRunner):
    task = Task(agent_id="runner_test", prompt="Say hello")
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query(_sdk_conversation())):
        await runner.run_task(task)
    assert runner.last_tool_call == "Read"
    assert "last_tool_call=Read" in runner.get_error_context()


@pytest.mark.asyncio
async def test_run_task_calls_on_message_for_every_message(runner: AgentRunner):
    task = Task(agent_id="runner_test", prompt="Say hello")
    messages: list[object] = []
    convo = _sdk_conversation()
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query(convo)):
        await runner.run_task(task, on_message=messages.append)
    assert len(messages) == len(convo)


@pytest.mark.asyncio
async def test_run_task_captures_session_id(runner: AgentRunner):
    task = Task(agent_id="runner_test", prompt="Say hello")
    with patch("meta_agent.agent_runner.query", side_effect=_fake_query(_sdk_conversation())):
        await runner.run_task(task)
    assert task.session_id == "sess-123"


# --- error context ---


def test_error_context_always_names_the_model(runner: AgentRunner):
    """An invalid CLI argument exits 1 with an empty stderr, so the model is
    the only clue left."""
    ctx = runner.get_error_context()
    assert f"model={runner.config.model}" in ctx
    assert "cli_stderr=<empty>" in ctx


def test_error_context_includes_the_captured_cli_stderr(runner: AgentRunner):
    runner._build_options()  # wires options.stderr to the runner's buffer
    runner._stderr.append("error: unknown option --nope")
    assert "error: unknown option --nope" in runner.get_error_context()


def test_stderr_callback_is_wired_into_sdk_options(runner: AgentRunner):
    options = runner._build_options()
    options.stderr("a stderr line")
    assert "a stderr line" in runner.get_error_context()


def test_stderr_buffer_is_bounded(runner: AgentRunner):
    from meta_agent.agent_runner import STDERR_BUFFER_LINES

    for i in range(STDERR_BUFFER_LINES * 2):
        runner._stderr.append(f"line {i}")
    assert len(runner._stderr) == STDERR_BUFFER_LINES


@pytest.mark.asyncio
async def test_cancel(runner: AgentRunner):
    await runner.cancel()
