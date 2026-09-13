"""Compare orchestration against a single agent on the same tasks.

The project's premise is that an Opus Brain decomposing work across cheaper
sub-agents beats doing it in one session. Nothing could test that: there was no
way to see what a run cost, and no fixed task set to run twice.

Each task is run once per arm. An arm is a way of doing the work:

    brain  the Brain orchestrator, with the in-process MCP tool surface
    solo   one Sonnet agent with the same tools and no delegation

Every run records success, wall time, cost and tokens, so the arms are
comparable on the three things that matter.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .agent_manager import AgentManager
from .brain import BRAIN_AGENT_ID, get_brain_config
from .models import AgentConfig, Workflow

SOLO_AGENT_ID = "eval-solo"
SOLO_SYSTEM_PROMPT = (
    "You are an expert engineer working alone. Complete the task fully, then "
    "reply with a short summary of what you did."
)
DEFAULT_TOOLS = ["Read", "Glob", "Grep", "Bash", "Edit", "Write"]
ARMS = ("brain", "solo")


@dataclass
class EvalTask:
    """One unit of work, run identically by every arm."""

    id: str
    prompt: str
    # Substrings that must all appear in the final answer for the run to pass.
    expect: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalTask:
        return cls(id=data["id"], prompt=data["prompt"], expect=list(data.get("expect", [])))


@dataclass
class RunResult:
    task_id: str
    arm: str
    status: str
    passed: bool
    seconds: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    result: str | None = None
    error: str | None = None


def load_tasks(path: Path) -> list[EvalTask]:
    data = json.loads(Path(path).read_text())
    return [EvalTask.from_dict(item) for item in data]


def _scores(result_text: str | None, task: EvalTask) -> bool:
    """Whether a run satisfied the task. No expectations means status alone."""
    if not task.expect:
        return True
    text = (result_text or "").lower()
    return all(needle.lower() in text for needle in task.expect)


def _await_task(manager: AgentManager, task_id: str, timeout: float) -> str:
    """Block until a task reaches a terminal state. Returns the final status."""
    done = threading.Event()
    terminal = {"task_completed", "task_failed", "task_cancelled", "waiting_for_input"}

    def listener(event: dict) -> None:
        if event.get("task_id") == task_id and event.get("kind") in terminal:
            done.set()

    manager.add_progress_listener(listener)
    try:
        # A task that finished between submit and subscribe would never fire.
        stored = manager.get_task(task_id)
        if stored is None or stored.status in ("completed", "failed", "cancelled"):
            return stored.status if stored else "missing"
        done.wait(timeout)
    finally:
        manager.remove_progress_listener(listener)

    stored = manager.get_task(task_id)
    return stored.status if stored else "missing"


def _run_solo(manager: AgentManager, task: EvalTask, cwd: str | None, timeout: float) -> RunResult:
    manager.register_agent(
        AgentConfig(
            id=SOLO_AGENT_ID,
            name="Eval Solo",
            system_prompt=SOLO_SYSTEM_PROMPT,
            allowed_tools=DEFAULT_TOOLS,
            permission_mode="bypassPermissions",
            cwd=cwd,
        )
    )
    started = time.monotonic()
    submitted = manager.submit_task(SOLO_AGENT_ID, task.prompt)
    status = _await_task(manager, submitted.id, timeout)
    stored = manager.get_task(submitted.id)
    return RunResult(
        task_id=task.id,
        arm="solo",
        status=status,
        passed=status == "completed" and _scores(stored.result if stored else None, task),
        seconds=round(time.monotonic() - started, 2),
        cost_usd=round(stored.cost_usd or 0.0, 6) if stored else 0.0,
        input_tokens=(stored.usage.get("input_tokens", 0) if stored else 0) or 0,
        output_tokens=(stored.usage.get("output_tokens", 0) if stored else 0) or 0,
        result=(stored.result or "")[:2000] if stored else None,
        error=stored.error if stored else None,
    )


def _run_brain(manager: AgentManager, task: EvalTask, cwd: str | None, timeout: float) -> RunResult:
    config = get_brain_config()
    config.cwd = cwd
    manager.register_agent(config)

    workflow = Workflow(prompt=task.prompt, brain_agent_id=BRAIN_AGENT_ID)
    manager.db.save_workflow(workflow)

    started = time.monotonic()
    submitted = manager.submit_task(
        BRAIN_AGENT_ID,
        f"Workflow ID: {workflow.id}\n\nUser Request: {task.prompt}",
        workflow_id=workflow.id,
    )
    workflow.brain_task_id = submitted.id
    manager.db.save_workflow(workflow)

    status = _await_task(manager, submitted.id, timeout)
    stored = manager.get_task(submitted.id)
    # The Brain's own answer, or the workflow result it assembled.
    final = manager.db.get_workflow(workflow.id)
    answer = (final.result if final and final.result else None) or (stored.result if stored else None)

    usage = manager.workflow_usage(workflow.id)
    return RunResult(
        task_id=task.id,
        arm="brain",
        status=status,
        passed=status == "completed" and _scores(answer, task),
        seconds=round(time.monotonic() - started, 2),
        cost_usd=round(usage["totals"]["cost_usd"], 6),
        input_tokens=usage["totals"]["input_tokens"],
        output_tokens=usage["totals"]["output_tokens"],
        result=(answer or "")[:2000],
        error=stored.error if stored else None,
    )


RUNNERS = {"brain": _run_brain, "solo": _run_solo}


def run_suite(
    manager: AgentManager,
    tasks: list[EvalTask],
    arms: tuple[str, ...] = ARMS,
    cwd: str | None = None,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Run every task through every arm and summarise."""
    runs: list[RunResult] = []
    for task in tasks:
        for arm in arms:
            runs.append(RUNNERS[arm](manager, task, cwd, timeout))
    return {"runs": [asdict(r) for r in runs], "summary": summarise(runs, arms)}


def summarise(runs: list[RunResult], arms: tuple[str, ...] = ARMS) -> dict[str, Any]:
    """Per-arm totals: how often it worked, what it cost, how long it took."""
    summary: dict[str, Any] = {}
    for arm in arms:
        arm_runs = [r for r in runs if r.arm == arm]
        if not arm_runs:
            continue
        passed = sum(1 for r in arm_runs if r.passed)
        summary[arm] = {
            "tasks": len(arm_runs),
            "passed": passed,
            "pass_rate": round(passed / len(arm_runs), 3),
            "total_cost_usd": round(sum(r.cost_usd for r in arm_runs), 6),
            "mean_cost_usd": round(sum(r.cost_usd for r in arm_runs) / len(arm_runs), 6),
            "total_seconds": round(sum(r.seconds for r in arm_runs), 2),
            "mean_seconds": round(sum(r.seconds for r in arm_runs) / len(arm_runs), 2),
            "input_tokens": sum(r.input_tokens for r in arm_runs),
            "output_tokens": sum(r.output_tokens for r in arm_runs),
        }
    return summary


def format_report(report: dict[str, Any]) -> str:
    """A plain-text comparison table."""
    lines = [
        f"{'arm':<8}{'pass':>8}{'rate':>8}{'cost $':>12}{'mean $':>10}{'secs':>9}{'mean s':>9}",
        "-" * 64,
    ]
    for arm, row in report["summary"].items():
        lines.append(
            f"{arm:<8}{row['passed']}/{row['tasks']:<6}{row['pass_rate']:>8.0%}"
            f"{row['total_cost_usd']:>12.4f}{row['mean_cost_usd']:>10.4f}"
            f"{row['total_seconds']:>9.1f}{row['mean_seconds']:>9.1f}"
        )
    lines.append("")
    lines.append(f"{'task':<24}{'arm':<8}{'pass':<6}{'cost $':>10}{'secs':>8}")
    lines.append("-" * 64)
    for run in report["runs"]:
        mark = "yes" if run["passed"] else "no"
        lines.append(
            f"{run['task_id'][:23]:<24}{run['arm']:<8}{mark:<6}"
            f"{run['cost_usd']:>10.4f}{run['seconds']:>8.1f}"
        )
    return "\n".join(lines)
