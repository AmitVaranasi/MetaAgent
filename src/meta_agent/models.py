"""Data models for the meta-agent system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
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
    model: str = "claude-sonnet-4-5-20250929"
    max_turns: int = 50
    max_budget_usd: float | None = None
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    permission_mode: str = "acceptEdits"
    cwd: str | None = None
    auto_restart: bool = False
    max_restarts: int = 3


class AgentState(BaseModel):
    config: AgentConfig
    status: AgentStatus = AgentStatus.STOPPED
    session_id: str | None = None
    current_task_id: str | None = None
    started_at: datetime | None = None
    error: str | None = None
    restart_count: int = 0


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    agent_id: str
    status: str = "pending"
    prompt: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    parent_task_id: str | None = None


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
