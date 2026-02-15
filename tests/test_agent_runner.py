"""Tests for the agent runner (mocks the SDK)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from meta_agent.agent_runner import AgentRunner
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


class FakeMessage:
    def __init__(self, content: str, msg_type: str = "assistant", session_id: str | None = None):
        self.content = content
        self.type = msg_type
        if session_id:
            self.session_id = session_id


async def fake_query_iter(**kwargs):
    yield FakeMessage("Thinking...", msg_type="thinking")
    yield FakeMessage("Hello, world!", msg_type="assistant", session_id="sess-123")


@pytest.mark.asyncio
async def test_run_task_captures_result(runner: AgentRunner):
    task = Task(agent_id="runner_test", prompt="Say hello")
    with patch("meta_agent.agent_runner.query", side_effect=fake_query_iter):
        result = await runner.run_task(task)
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_run_task_calls_on_message(runner: AgentRunner):
    task = Task(agent_id="runner_test", prompt="Say hello")
    messages = []
    with patch("meta_agent.agent_runner.query", side_effect=fake_query_iter):
        await runner.run_task(task, on_message=messages.append)
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_run_task_captures_session_id(runner: AgentRunner):
    task = Task(agent_id="runner_test", prompt="Say hello")
    with patch("meta_agent.agent_runner.query", side_effect=fake_query_iter):
        await runner.run_task(task)
    assert task.session_id == "sess-123"


@pytest.mark.asyncio
async def test_cancel(runner: AgentRunner):
    await runner.cancel()
