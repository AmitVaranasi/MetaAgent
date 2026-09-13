"""Data models for the meta-agent system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """An agent's lifecycle state.

    STOPPED means *deliberately* stopped and refusing work — it is not the
    state a freshly registered agent is in. It used to be the default, which
    made the whole enum decorative: submit_task ran regardless of it, so
    start_agent and stop_agent changed a label and nothing else.
    """

    STOPPED = "stopped"
    RUNNING = "running"
    IDLE = "idle"
    ERROR = "error"


class AgentConfig(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    name: str
    description: str = ""
    system_prompt: str
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Read", "Glob", "Grep", "Bash", "Edit", "Write"]
    )
    disallowed_tools: list[str] = Field(default_factory=list)
    model: str = "claude-sonnet-5"
    max_turns: int = 50
    max_budget_usd: float | None = None
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Attach the in-process meta-agent MCP server at run time. A flag, not
    # a server instance, because AgentConfig is persisted as JSON.
    use_meta_agent_mcp: bool = False
    permission_mode: str = "acceptEdits"
    cwd: str | None = None
    auto_restart: bool = False
    max_restarts: int = 3


class AgentState(BaseModel):
    config: AgentConfig
    status: AgentStatus = AgentStatus.IDLE
    session_id: str | None = None
    current_task_id: str | None = None
    # Every task currently in flight for this agent. current_task_id is the most
    # recently started of these, kept for callers that only show one.
    running_task_ids: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    error: str | None = None
    restart_count: int = 0


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    agent_id: str
    status: str = "pending"
    prompt: str
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    parent_task_id: str | None = None
    # PID of the process that started this task, so a later process can tell an
    # orphan (owner gone) from a task another live process is still running.
    owner_pid: int | None = None

    # --- what the run actually cost, from the SDK's ResultMessage ---
    # Recorded per task and summed per workflow. The model is stored here rather
    # than looked up from the agent, because the Brain deletes its agents when
    # it is done (Phase 6) and the spend has to outlive them.
    model: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class WorkflowStatus(str, Enum):
    PLANNING = "planning"
    WAITING_FOR_INPUT = "waiting_for_input"
    EXECUTING = "executing"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"


class Workflow(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    prompt: str
    plan: str | None = None
    status: WorkflowStatus = WorkflowStatus.PLANNING
    brain_agent_id: str
    brain_task_id: str | None = None
    subtask_ids: list[str] = Field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
