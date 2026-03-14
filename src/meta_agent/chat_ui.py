"""Rich-powered chat UI helpers for the meta-agent CLI."""

from __future__ import annotations

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
        idx = event.get("index", "?")
        total = event.get("total", "?")
        desc = event.get("description", "")
        agent = event.get("agent_id", "")
        label = f"  [yellow]◐[/yellow] Subtask {idx}/{total}"
        if desc:
            label += f": {desc}"
        if agent:
            label += f" [dim](agent: {agent})[/dim]"
        console.print(label)

    # --- Live tool call feed (new) ---
    elif kind == "tool_call":
        agent = event.get("agent_id", "?")
        tool = event.get("tool", "?")
        preview = event.get("input_preview", "")
        # Compact one-liner: "  ↳ agent:abc123 → Read(src/auth.py...)"
        line = f"    [dim]↳ {agent} → {tool}[/dim]"
        if preview:
            line += f"[dim]({preview[:80]})[/dim]"
        console.print(line)
    elif kind == "tool_result":
        agent = event.get("agent_id", "?")
        tool = event.get("tool", "?")
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
        idx = event.get("index", "?")
        total = event.get("total", "?")
        console.print(f"  [green]✓[/green] Subtask {idx}/{total} completed")
    elif kind == "subtask_failed":
        idx = event.get("index", "?")
        total = event.get("total", "?")
        error = event.get("error", "unknown error")
        # Show first 3 lines of error (which now includes context + traceback)
        error_lines = error.strip().splitlines()
        console.print(f"  [red]✗[/red] Subtask {idx}/{total} failed: {error_lines[0]}")
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


def print_summary(workflow, tasks: list | None = None) -> None:
    """Print a rich summary panel for a completed workflow.

    Args:
        workflow: A Workflow model instance.
        tasks: Optional list of Task objects for subtask details.
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
