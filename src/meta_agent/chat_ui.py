"""Rich-powered chat UI helpers for the meta-agent CLI."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_welcome(plan_mode: bool = False) -> None:
    """Print the welcome banner."""
    console.print()
    console.print("  [bold cyan]Meta-Agent Brain[/bold cyan]")
    if plan_mode:
        console.print("  [yellow][Plan Mode ON][/yellow] — Brain will present plans for approval before executing.")
    console.print("  Type your task and press Enter. Type 'exit' to quit.")
    console.print("  Type '/help' for available commands.")
    console.print()


def get_user_input(plan_mode: bool = False) -> str | None:
    """Prompt for user input. Returns None on EOF/KeyboardInterrupt."""
    try:
        if plan_mode:
            return console.input("[bold green]You [yellow][plan][/yellow] > [/bold green]")
        return console.input("[bold green]You > [/bold green]")
    except (EOFError, KeyboardInterrupt):
        return None


def _subtask_label(event: dict) -> str:
    """"3/7" when the total is known, plain "3" while the Brain is still
    submitting — a streaming UI cannot know the denominator up front."""
    index = event.get("index", "?")
    total = event.get("total")
    return f"{index}/{total}" if total else f"{index}"


def print_progress(event: dict) -> None:
    """Print a formatted progress line.

    Expected event keys:
        kind: 'workflow_created' | 'planning' | 'plan_ready' |
              'subtask_running' | 'subtask_done' | 'subtask_failed' |
              'assembling' | 'completed' | 'failed' | 'status_change' |
              'tool_call' | 'tool_result' | 'agent_progress'
        Plus context-specific keys (workflow_id, index, total, description, etc.)
    """
    kind = event.get("kind", "")

    if kind == "workflow_created":
        console.print(f"  [green]✓[/green] Workflow created (id: {event.get('workflow_id', '?')})")
    elif kind == "planning":
        console.print("  [yellow]◐[/yellow] Planning task decomposition...")
    elif kind == "plan_ready":
        plan = event.get("plan", "")
        total = event.get("total", 0)
        console.print(f"  [green]✓[/green] Plan: {total} subtasks")
        if plan:
            for line in plan.strip().splitlines():
                console.print(f"    {line}")
    elif kind == "subtask_running":
        desc = event.get("description", "")
        agent = event.get("agent_id", "")
        label = f"  [yellow]◐[/yellow] Subtask {_subtask_label(event)}"
        if desc:
            label += f": {desc}"
        if agent:
            label += f" [dim](agent: {agent})[/dim]"
        console.print(label)

    # --- Live tool call feed (new) ---
    elif kind == "tool_call":
        agent = event.get("agent_id", "?")
        tool = event.get("tool") or "?"
        preview = event.get("input_preview", "")
        # Compact one-liner: "  ↳ agent:abc123 → Read(src/auth.py...)"
        line = f"    [dim]↳ {agent} → {tool}[/dim]"
        if preview:
            line += f"[dim]({preview[:80]})[/dim]"
        console.print(line)
    elif kind == "tool_result":
        agent = event.get("agent_id", "?")
        tool = event.get("tool") or "?"
        is_error = event.get("is_error", False)
        if is_error:
            preview = event.get("output_preview", "")
            console.print(f"    [red]↳ {agent} ← {tool} ERROR: {preview[:100]}[/red]")
        # Don't print successful tool results to avoid noise — only errors
    elif kind == "agent_progress":
        agent = event.get("agent_id", "?")
        phase = event.get("phase", "")
        msg = event.get("message", "")
        phase_icon = {
            "reading": "📖", "writing": "✏️", "testing": "🧪",
            "done": "✅", "working": "⚙️",
        }.get(phase, "⚙️")
        console.print(f"    {phase_icon} [dim]{agent}:[/dim] {msg}")

    elif kind == "subtask_done":
        console.print(f"  [green]✓[/green] Subtask {_subtask_label(event)} completed")
    elif kind == "subtask_failed":
        error = event.get("error", "unknown error")
        # Show first 3 lines of error (which now includes context + traceback)
        error_lines = error.strip().splitlines()
        console.print(f"  [red]✗[/red] Subtask {_subtask_label(event)} failed: {error_lines[0]}")
        for line in error_lines[1:4]:
            console.print(f"    [dim red]{line}[/dim red]")
        if len(error_lines) > 4:
            console.print(f"    [dim]... ({len(error_lines) - 4} more lines)[/dim]")
    elif kind == "waiting_for_input":
        console.print("  [cyan]?[/cyan] Brain needs clarification")
    elif kind == "assembling":
        console.print("  [yellow]◐[/yellow] Assembling final result...")
    elif kind == "completed":
        console.print("  [green]✓[/green] Workflow completed")
    elif kind == "failed":
        error = event.get("error", "unknown error")
        console.print(f"  [red]✗[/red] Workflow failed: {error}")
    elif kind == "status_change":
        status = event.get("status", "")
        console.print(f"  [dim]  Status: {status}[/dim]")
    else:
        msg = event.get("message", str(event))
        console.print(f"  [dim]{msg}[/dim]")


class ChatProgress:
    """Renders live progress for one Brain run, and says when it is over.

    Replaces the old poll-and-diff loop in `chat`, which slept 2s, re-read the
    workflow, and reconstructed milestones by comparing snapshots. Everything
    below now arrives as an event: the manager emits task lifecycle and tool
    events, and the in-process MCP tools emit workflow updates and subtask
    submissions as the Brain makes them.

    Call it with each event; wait on `finished` for the run to end.
    """

    TERMINAL = {"task_completed", "task_failed", "task_cancelled", "waiting_for_input"}

    def __init__(self, brain_task_id: str, workflow_id: str):
        self.brain_task_id = brain_task_id
        self.workflow_id = workflow_id
        self.finished = threading.Event()
        self.outcome: str | None = None
        self._subtasks: list[str] = []
        self._settled: set[str] = set()

    def _index(self, task_id: str) -> tuple[int, int]:
        if task_id not in self._subtasks:
            self._subtasks.append(task_id)
        return self._subtasks.index(task_id) + 1, len(self._subtasks)

    def __call__(self, event: dict) -> None:
        kind = event.get("kind", "")
        task_id = event.get("task_id")
        is_brain = task_id == self.brain_task_id

        if kind == "workflow_update":
            if event.get("workflow_id") == self.workflow_id:
                self._render_workflow_update(event)
        elif kind == "subtask_submitted":
            index, _ = self._index(event.get("task_id", ""))
            print_progress({
                "kind": "subtask_running",
                "index": index,
                "description": (event.get("prompt") or "")[:120],
                "agent_id": event.get("agent_id", ""),
            })
        elif kind in ("task_completed", "task_failed", "task_cancelled") and not is_brain:
            self._render_subtask_end(kind, event)
        elif not is_brain or kind not in self.TERMINAL:
            # tool_call, tool_result, agent_progress, status_change, ...
            print_progress(event)

        if is_brain and kind in self.TERMINAL:
            self.outcome = kind
            self.finished.set()

    def _render_workflow_update(self, event: dict) -> None:
        status = event.get("status")
        if event.get("plan"):
            print_progress({
                "kind": "plan_ready",
                "plan": event["plan"],
                "total": len(self._subtasks),
            })
        elif status == "planning":
            print_progress({"kind": "planning"})
        elif status == "assembling":
            print_progress({"kind": "assembling"})
        elif status == "failed" and event.get("error"):
            print_progress({"kind": "failed", "error": event["error"]})

    def _render_subtask_end(self, kind: str, event: dict) -> None:
        task_id = event.get("task_id", "")
        if task_id in self._settled:
            return
        self._settled.add(task_id)
        index, _ = self._index(task_id)
        if kind == "task_completed":
            print_progress({"kind": "subtask_done", "index": index})
        else:
            print_progress({
                "kind": "subtask_failed",
                "index": index,
                "error": event.get("error") or kind.replace("task_", ""),
            })


def print_summary(workflow, tasks: list | None = None, usage: dict | None = None) -> None:
    """Print a rich summary panel for a completed workflow.

    Args:
        workflow: A Workflow model instance.
        tasks: Optional list of Task objects for subtask details.
        usage: Optional AgentManager.workflow_usage() result — cost totals and
            the per-model breakdown.
    """
    lines: list[str] = []

    if workflow.plan:
        lines.append("[bold]Plan:[/bold]")
        lines.append(f"  {workflow.plan}")
        lines.append("")

    if tasks:
        lines.append("[bold]What happened:[/bold]")
        for i, t in enumerate(tasks, 1):
            if t.status == "completed":
                icon = "[green]✓[/green]"
            elif t.status == "failed":
                icon = "[red]✗[/red]"
            else:
                icon = "[yellow]○[/yellow]"
            desc = t.prompt[:120] if t.prompt else "—"
            lines.append(f"  {i}. {icon} {desc}")
        lines.append("")

    if workflow.result:
        lines.append("[bold]Result:[/bold]")
        result_text = workflow.result
        if len(result_text) > 500:
            result_text = result_text[:500] + "..."
        lines.append(f"  {result_text}")
        lines.append("")

    if usage and usage["totals"]["cost_usd"]:
        lines.append("[bold]Cost:[/bold]")
        totals = usage["totals"]
        lines.append(
            f"  ${totals['cost_usd']:.4f} total"
            f" · {totals['input_tokens']:,} in / {totals['output_tokens']:,} out tokens"
            f" · {totals['num_turns']} turns"
        )
        for model, bucket in sorted(
            usage["by_model"].items(), key=lambda kv: -kv[1]["cost_usd"]
        ):
            lines.append(
                f"    {model}: ${bucket['cost_usd']:.4f}"
                f" over {bucket['tasks']} task(s)"
            )
        lines.append("")

    # Timing and stats
    duration_str = _format_duration(workflow.created_at, workflow.completed_at)
    subtask_count = len(workflow.subtask_ids) if workflow.subtask_ids else 0
    agent_count = len({t.agent_id for t in tasks}) if tasks else 0
    stats = f"Duration: {duration_str} | Agents used: {agent_count} | Subtasks: {subtask_count}"
    lines.append(f"[dim]{stats}[/dim]")

    body = "\n".join(lines)

    title = "Task Complete" if workflow.status.value == "completed" else f"Task {workflow.status.value.title()}"
    border_style = "green" if workflow.status.value == "completed" else "red"

    console.print()
    console.print(Panel(body, title=title, border_style=border_style, padding=(1, 2)))
    console.print()


def print_plan_mode_toggle(enabled: bool) -> None:
    """Print plan mode toggle confirmation."""
    if enabled:
        console.print("  [yellow]Plan mode ON[/yellow] — Brain will present plans for approval before executing.")
    else:
        console.print("  [dim]Plan mode OFF[/dim] — Brain will auto-execute (default behavior).")


def print_help(plan_mode: bool = False) -> None:
    """Print available slash commands."""
    console.print()
    console.print("  [bold]Available commands:[/bold]")
    console.print()
    mode_status = "[yellow]ON[/yellow]" if plan_mode else "[dim]OFF[/dim]"
    console.print(f"  /plan      Toggle plan mode (currently {mode_status})")
    console.print("             When on, Brain presents plans for approval before executing.")
    console.print("  /help      Show this help message")
    console.print("  exit       Exit the chat")
    console.print()


def _format_duration(start: datetime | None, end: datetime | None) -> str:
    """Format duration between two datetimes."""
    if not start:
        return "—"
    end = end or datetime.now(timezone.utc)
    delta = end - start
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m {seconds:02d}s"
