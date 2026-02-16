# Meta-Agent

A Python-based meta-agent system that orchestrates multiple Claude-powered agents locally using the [Claude Agent SDK](https://docs.anthropic.com/en/docs/claude-agent-sdk). It features an interactive chat CLI powered by an Opus-class Brain agent, an MCP server for Claude Code integration, and a full management CLI.

```
 You ──(chat)──> meta-agent chat
                   ├── Brain Agent (Opus) — orchestrates everything
                   ├── AgentManager (lifecycle, task scheduling)
                   ├── SQLite DB (WAL mode)
                   └── CLI (Click + Rich)
                         │
             ┌───────────┼───────────┐
             v           v           v
         Agent 1     Agent 2     Agent N
        (SDK query) (SDK query) (SDK query)
        own session own session own session
```

## Features

- **Interactive Chat** — Conversational interface with the Brain agent; type a task, watch live progress, get a rich summary
- **Brain Orchestration** — Opus-powered Brain agent that decomposes complex tasks, spawns specialized agents, and assembles results
- **Agent Management** — Create, start, stop, and delete agents with custom system prompts, tool permissions, and model selection
- **Live Progress** — Real-time status updates as the Brain plans, delegates, and completes subtasks
- **Task Execution** — Submit prompts to agents; tasks run asynchronously via the Claude Agent SDK
- **MCP Server** — 10 tools exposed over stdio for integration with Claude Code
- **CLI** — Full command-line interface for all operations
- **Session Tracking** — SDK sessions with resume support
- **Auto-Restart** — Configurable automatic recovery from agent errors
- **SQLite Storage** — WAL-mode database for concurrent access

## Requirements

- Python 3.11+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated (`claude` must be available on PATH)

## Installation

```bash
# Clone the repository
git clone https://github.com/AmitVaranasi/MetaAgent.git
cd MetaAgent/meta-agents

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize

```bash
meta-agent init
```

This creates the `data/` directory with the SQLite database and log folder.

### 2. Start a Chat Session

This is the primary way to use Meta-Agent. The Brain agent handles everything — planning, agent creation, task delegation, and assembly.

```bash
meta-agent chat
```

Example session:

```
  Meta-Agent Brain
  Type your task and press Enter. Type 'exit' to quit.

You > Build a Python CLI that converts CSV to JSON

  Brain is thinking...
  ✓ Workflow created (id: a1b2c3)
  ◐ Planning task decomposition...
  ✓ Plan: 3 subtasks
    1. Create CSV parser module
    2. Create JSON converter
    3. Add CLI entry point
  ◐ Running subtask 1/3: Create CSV parser module (agent: coder-abc)
  ✓ Subtask 1/3 completed
  ◐ Running subtask 2/3: Create JSON converter (agent: coder-def)
  ✓ Subtask 2/3 completed
  ◐ Running subtask 3/3: Add CLI entry point (agent: coder-ghi)
  ✓ Subtask 3/3 completed
  ◐ Assembling final result...

  ╭─ Task Complete ──────────────────────────────────╮
  │                                                   │
  │  Plan:                                            │
  │  Split into 3 subtasks: parser, converter, CLI    │
  │                                                   │
  │  What happened:                                   │
  │  1. ✓ Created csv_parser.py with streaming reader │
  │  2. ✓ Created json_converter.py with formatting   │
  │  3. ✓ Created main.py with click CLI              │
  │                                                   │
  │  Result:                                          │
  │  Built a CLI tool at ./csv2json that converts...  │
  │                                                   │
  │  Duration: 2m 34s | Agents used: 3 | Subtasks: 3 │
  ╰──────────────────────────────────────────────────╯

You > exit
Goodbye!
```

To exit the chat: type `exit`, `quit`, or press `Ctrl+C`.

### 3. Direct Brain Command (Non-Interactive)

For one-off tasks without entering the chat REPL:

```bash
# Fire and forget
meta-agent brain "Write a Python script that fetches weather data"

# Wait for completion
meta-agent brain --wait "Write a Python script that fetches weather data"
```

### 4. Manual Agent Management

You can also create and manage agents directly:

```bash
# Create agents
meta-agent create --name "Coder" \
  --system-prompt "You are an expert programmer." \
  --tools "Read,Write,Edit,Bash,Glob,Grep"

meta-agent create --name "Reviewer" \
  --system-prompt "You are a senior code reviewer." \
  --tools "Read,Glob,Grep"

# List agents
meta-agent list

# Submit a task
meta-agent submit coder "Write a prime number checker"

# Check status
meta-agent status
meta-agent status coder

# View logs
meta-agent logs coder
meta-agent logs coder -n 200

# Check workflows
meta-agent workflow
meta-agent workflow <workflow-id>

# Delete an agent
meta-agent delete coder
```

#### Agent Create Options

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Agent display name | (required) |
| `--system-prompt` | Instructions for the agent | (required) |
| `--tools` | Comma-separated tool list | `Read,Write,Edit,Bash,Glob,Grep` |
| `--model` | Claude model ID | `claude-sonnet-4-5-20250929` |
| `--id` | Custom agent ID | auto-generated |
| `--description` | Agent description | empty |
| `--cwd` | Working directory for the agent | current directory |

## MCP Server (Claude Code Integration)

Start the MCP server for use with Claude Code:

```bash
meta-agent mcp-server
```

This runs a [FastMCP](https://github.com/modelcontextprotocol/python-sdk) server over stdio. To connect it from Claude Code, add to your MCP config:

```json
{
  "mcpServers": {
    "meta-agent": {
      "command": "meta-agent",
      "args": ["mcp-server"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `list_agents` | List all registered agents with status |
| `get_agent` | Get detailed info about an agent |
| `create_agent` | Create and register a new agent |
| `delete_agent` | Delete an agent by ID |
| `start_agent` | Set an agent to idle (ready for tasks) |
| `stop_agent` | Stop an agent and cancel running tasks |
| `agent_logs` | Get recent log lines for an agent |
| `submit_task` | Submit a prompt to an agent |
| `task_status` | Get status and result of a task |
| `list_tasks` | List tasks, optionally filtered by agent |

## CLI Reference

```
Usage: meta-agent [OPTIONS] COMMAND [ARGS]...

Options:
  --data-dir TEXT  Data directory (env: META_AGENT_DATA)
  --help           Show this message and exit.

Commands:
  brain       Submit a task to the Brain agent for automatic orchestration
  chat        Interactive chat with the Brain agent
  create      Create and register a new agent
  delete      Delete an agent by ID
  init        Initialize the data directory and database
  list        List all registered agents
  logs        View agent logs
  mcp-server  Start the MCP server (stdio transport)
  status      Show agent or task status
  submit      Submit a task to an agent
  workflow    List workflows or show workflow detail with subtask tree
```

## Agent Configuration

Agents are defined by an `AgentConfig` with these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | str | auto-generated | Unique identifier |
| `name` | str | (required) | Display name |
| `description` | str | `""` | Human-readable description |
| `system_prompt` | str | (required) | Instructions sent to the LLM |
| `allowed_tools` | list[str] | `["Read","Glob","Grep","Bash","Edit","Write"]` | SDK tools the agent can use |
| `model` | str | `claude-sonnet-4-5-20250929` | Claude model ID |
| `max_turns` | int | `50` | Max conversation turns per task |
| `max_budget_usd` | float \| None | `None` | Optional spending cap |
| `mcp_servers` | dict | `{}` | External MCP servers the agent can access |
| `permission_mode` | str | `"acceptEdits"` | SDK permission mode |
| `cwd` | str \| None | `None` | Working directory |
| `auto_restart` | bool | `False` | Auto-recover from errors |
| `max_restarts` | int | `3` | Max restart attempts |

### Available Tools

These are the built-in Claude Agent SDK tools you can grant to agents:

| Tool | Description |
|------|-------------|
| `Read` | Read file contents |
| `Write` | Create or overwrite files |
| `Edit` | Make targeted edits to files |
| `Bash` | Execute shell commands |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents with regex |

Use an empty list (`--tools ""`) for chat-only agents with no tool access.

## Agent Lifecycle

```
STOPPED ──(start)──> IDLE ──(task submitted)──> RUNNING
   ^                  ^                            │
   │                  │                            │
   │                  └──(task completes)───────────┘
   │                                               │
   └──(stop)──── ERROR <──(task fails)─────────────┘
                   │
                   └──(auto_restart=true)──> IDLE
```

| Status | Meaning |
|--------|---------|
| `stopped` | Agent is registered but not active |
| `idle` | Agent is started and ready to receive tasks |
| `running` | Agent is actively processing a task |
| `error` | Last task failed (error message stored on agent state) |

## Project Structure

```
meta-agents/
├── pyproject.toml                       # Dependencies and project config
├── src/meta_agent/
│   ├── __init__.py
│   ├── __main__.py                      # python -m meta_agent
│   ├── models.py                        # Pydantic models: AgentConfig, AgentState, Task, Workflow
│   ├── config.py                        # Config singleton (db_path, log_dir)
│   ├── db.py                            # SQLite WAL database, CRUD operations
│   ├── agent_runner.py                  # Wraps claude_agent_sdk.query()
│   ├── agent_manager.py                 # Agent lifecycle, task scheduling, progress callbacks
│   ├── mcp_server.py                    # FastMCP server with 10 tools
│   ├── cli.py                           # Click CLI commands (chat, brain, create, etc.)
│   ├── chat_ui.py                       # Rich chat UI helpers (progress, summary panels)
│   ├── brain.py                         # Brain agent config and system prompt
│   ├── external_runner.py               # External model runner (Gemini, etc.)
│   └── agent_configs/
│       └── examples.py                  # Predefined agent configs
├── data/                                # Runtime data (gitignored)
│   ├── meta_agent.db                    # SQLite database
│   └── logs/                            # Agent log files
└── tests/
    ├── conftest.py                      # Shared fixtures
    ├── test_models.py                   # Model creation and serialization
    ├── test_db.py                       # Database CRUD operations
    ├── test_agent_runner.py             # SDK query mocking
    ├── test_agent_manager.py            # Lifecycle and task scheduling
    ├── test_brain.py                    # Brain config tests
    ├── test_external_runner.py          # External model runner tests
    └── test_mcp_server.py              # MCP tool integration
```

## Concurrency Model

- `AgentManager` runs an asyncio event loop in a background daemon thread
- SDK `query()` calls are scheduled via `asyncio.run_coroutine_threadsafe()`
- `threading.Lock` protects in-memory agent state
- SQLite WAL mode with `busy_timeout=5000` for database concurrency
- CLI is synchronous — it interacts with the manager through thread-safe methods

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Custom Data Directory

```bash
# Via flag
meta-agent --data-dir /path/to/data init

# Via environment variable
export META_AGENT_DATA=/path/to/data
meta-agent init
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `claude-agent-sdk` | Claude Agent SDK for running agents |
| `click` | CLI framework |
| `pydantic` | Data models and validation |
| `mcp` | MCP server (FastMCP) |
| `rich` | Terminal formatting, tables, and chat UI |

## License

MIT
