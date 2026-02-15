"""Predefined agent configurations."""

from ..brain import get_brain_config
from ..models import AgentConfig

ECHO_AGENT = AgentConfig(
    id="echo",
    name="Echo Agent",
    description="Simple agent that responds to prompts",
    system_prompt="You are a helpful assistant. Respond concisely.",
    allowed_tools=[],
    model="claude-sonnet-4-5-20250929",
)

CODER_AGENT = AgentConfig(
    id="coder",
    name="Coding Agent",
    description="Agent that can read, write, and edit code",
    system_prompt="You are an expert programmer. Read files, write code, run tests.",
    allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    model="claude-sonnet-4-5-20250929",
)

REVIEWER_AGENT = AgentConfig(
    id="reviewer",
    name="Code Reviewer",
    description="Reviews code for bugs, style, and security issues",
    system_prompt="You are a senior code reviewer. Read code and provide detailed reviews.",
    allowed_tools=["Read", "Glob", "Grep"],
    model="claude-sonnet-4-5-20250929",
)

BRAIN_AGENT = get_brain_config(["meta-agent", "mcp-server"])

ALL_EXAMPLES = [ECHO_AGENT, CODER_AGENT, REVIEWER_AGENT, BRAIN_AGENT]
