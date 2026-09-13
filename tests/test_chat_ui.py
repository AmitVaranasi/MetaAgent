"""Tests for the chat UI — 213 lines that had none, and it is the product surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from meta_agent import chat_ui
from meta_agent.chat_ui import (
    ChatProgress,
    _format_duration,
    print_help,
    print_plan_mode_toggle,
    print_progress,
    print_summary,
    print_welcome,
)
from meta_agent.models import Task, Workflow, WorkflowStatus


@pytest.fixture()
def rendered(monkeypatch) -> list[str]:
    """Capture what the UI prints, without a terminal."""
    from rich.panel import Panel

    lines: list[str] = []

    def capture(*args, **kwargs) -> None:
        for arg in args:
            # print_summary renders a Panel; keep its body, not its repr.
            lines.append(str(arg.renderable) if isinstance(arg, Panel) else str(arg))

    monkeypatch.setattr(chat_ui.console, "print", capture)
    return lines


# --- ChatProgress ---


def _progress() -> ChatProgress:
    return ChatProgress(brain_task_id="brain-1", workflow_id="wf-1")


def test_brain_completion_finishes_the_run(rendered):
    p = _progress()
    p({"kind": "task_completed", "task_id": "brain-1"})
    assert p.finished.is_set()
    assert p.outcome == "task_completed"


@pytest.mark.parametrize(
    "kind", ["task_completed", "task_failed", "task_cancelled", "waiting_for_input"]
)
def test_every_terminal_kind_finishes_the_run(rendered, kind: str):
    p = _progress()
    p({"kind": kind, "task_id": "brain-1"})
    assert p.outcome == kind


def test_a_subtask_finishing_does_not_finish_the_run(rendered):
    p = _progress()
    p({"kind": "task_completed", "task_id": "sub-1"})
    assert not p.finished.is_set()
    assert p.outcome is None


def test_subtasks_are_numbered_in_submission_order(rendered):
    p = _progress()
    p({"kind": "subtask_submitted", "task_id": "a", "agent_id": "w1", "prompt": "first"})
    p({"kind": "subtask_submitted", "task_id": "b", "agent_id": "w2", "prompt": "second"})
    p({"kind": "task_completed", "task_id": "b"})

    text = "\n".join(rendered)
    # No denominator while the Brain is still submitting — it is not knowable.
    assert "Subtask 1: first" in text
    assert "Subtask 2: second" in text
    assert "Subtask 2 completed" in text


def test_a_subtask_end_is_reported_once(rendered):
    p = _progress()
    p({"kind": "subtask_submitted", "task_id": "a", "agent_id": "w", "prompt": "x"})
    p({"kind": "task_completed", "task_id": "a"})
    p({"kind": "task_completed", "task_id": "a"})
    assert "\n".join(rendered).count("completed") == 1


def test_a_failing_subtask_renders_its_error(rendered):
    p = _progress()
    p({"kind": "subtask_submitted", "task_id": "a", "agent_id": "w", "prompt": "x"})
    p({"kind": "task_failed", "task_id": "a", "error": "boom\nsecond line"})
    text = "\n".join(rendered)
    assert "Subtask 1 failed" in text and "boom" in text


def test_workflow_update_renders_the_plan(rendered):
    p = _progress()
    p({"kind": "workflow_update", "workflow_id": "wf-1", "status": "executing", "plan": "1. do it"})
    text = "\n".join(rendered)
    assert "Plan:" in text and "1. do it" in text


def test_workflow_update_for_another_workflow_is_ignored(rendered):
    p = _progress()
    p({"kind": "workflow_update", "workflow_id": "other", "status": "assembling", "plan": None})
    assert rendered == []


def test_assembling_is_announced(rendered):
    p = _progress()
    p({"kind": "workflow_update", "workflow_id": "wf-1", "status": "assembling", "plan": None})
    assert "Assembling" in "\n".join(rendered)


def test_tool_calls_from_sub_agents_are_shown_live(rendered):
    p = _progress()
    p({"kind": "tool_call", "task_id": "sub-1", "agent_id": "w1", "tool": "Read",
       "input_preview": "src/auth.py"})
    text = "\n".join(rendered)
    assert "Read" in text and "src/auth.py" in text


def test_a_successful_tool_result_is_not_printed(rendered):
    p = _progress()
    p({"kind": "tool_result", "task_id": "s", "agent_id": "w", "tool": "Read", "is_error": False})
    assert rendered == []


def test_an_errored_tool_result_is_printed(rendered):
    p = _progress()
    p({"kind": "tool_result", "task_id": "s", "agent_id": "w", "tool": "Bash",
       "is_error": True, "output_preview": "permission denied"})
    assert "permission denied" in "\n".join(rendered)


def test_report_progress_from_a_sub_agent_is_shown(rendered):
    p = _progress()
    p({"kind": "agent_progress", "agent_id": "w1", "phase": "testing", "message": "running pytest"})
    assert "running pytest" in "\n".join(rendered)


def test_finished_can_be_reset_for_a_resume_cycle(rendered):
    p = _progress()
    p({"kind": "waiting_for_input", "task_id": "brain-1"})
    assert p.finished.is_set()
    p.finished.clear()
    p.outcome = None
    p({"kind": "task_completed", "task_id": "brain-1"})
    assert p.outcome == "task_completed"


# --- the rest of the UI ---


def test_print_welcome_mentions_plan_mode_when_on(rendered):
    print_welcome(plan_mode=True)
    assert "Plan Mode ON" in "\n".join(rendered)


def test_print_help_reflects_plan_mode(rendered):
    print_help(plan_mode=True)
    text = "\n".join(rendered)
    assert "/plan" in text and "/help" in text


def test_plan_mode_toggle_states(rendered):
    print_plan_mode_toggle(True)
    print_plan_mode_toggle(False)
    text = "\n".join(rendered)
    assert "Plan mode ON" in text and "Plan mode OFF" in text


def test_unknown_event_kind_does_not_raise(rendered):
    print_progress({"kind": "something_new", "message": "hello"})
    assert "hello" in "\n".join(rendered)


def test_summary_includes_cost_when_usage_is_given(rendered):
    wf = Workflow(
        id="wf-1", prompt="p", brain_agent_id="brain", status=WorkflowStatus.COMPLETED,
        result="all done", subtask_ids=["t1"],
        completed_at=datetime.now(timezone.utc),
    )
    tasks = [Task(id="t1", agent_id="w", prompt="do a thing", status="completed")]
    usage = {
        "task_count": 2,
        "totals": {"cost_usd": 0.4812, "input_tokens": 8000, "output_tokens": 900, "num_turns": 6},
        "by_model": {
            "claude-opus-5": {"tasks": 1, "cost_usd": 0.40, "input_tokens": 5000, "output_tokens": 500},
            "claude-sonnet-5": {"tasks": 1, "cost_usd": 0.0812, "input_tokens": 3000, "output_tokens": 400},
        },
    }
    print_summary(wf, tasks, usage)

    text = "\n".join(rendered)
    assert "$0.4812 total" in text
    assert "claude-opus-5: $0.4000" in text
    assert "8,000 in / 900 out tokens" in text


def test_summary_omits_cost_when_nothing_was_spent(rendered):
    wf = Workflow(id="wf-1", prompt="p", brain_agent_id="brain", status=WorkflowStatus.COMPLETED)
    print_summary(wf, [], {"task_count": 0, "totals": {"cost_usd": 0.0}, "by_model": {}})
    assert "Cost:" not in "\n".join(rendered)


@pytest.mark.parametrize(
    "delta,expected",
    [(timedelta(seconds=5), "5s"), (timedelta(seconds=75), "1m 15s"), (timedelta(minutes=10), "10m 00s")],
)
def test_format_duration(delta, expected: str):
    start = datetime.now(timezone.utc)
    assert _format_duration(start, start + delta) == expected


def test_format_duration_without_a_start_is_a_dash():
    assert _format_duration(None, None) == "—"
