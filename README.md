# Meta-Agent

A Python-based meta-agent system that orchestrates multiple Claude-powered agents locally using the [Claude Agent SDK](https://docs.anthropic.com/en/docs/claude-agent-sdk). It exposes an MCP server for Claude Code integration, a CLI, and a web dashboard.

```
 Claude Code ──MCP(stdio)──> Meta-Agent Process
                              ├── AgentManager (orchestrates SDK agents)
                              ├── SQLite DB (WAL mode)
                              ├── CLI (Click + Rich)
                              └── Dashboard (Flask + htmx, polling)
                                    │
                        ┌───────────┼───────────┐
                        v           v           v
                    Agent 1     Agent 2     Agent N
                   (SDK query) (SDK query) (SDK query)
                   own session own session own session
```

## Features

- **Agent Management** — Create, start, stop, and delete agents with custom system prompts, tool permissions, and model selection
- **Task Execution** — Submit prompts to agents; tasks run asynchronously via the Claude Agent SDK
- **MCP Server** — 10 tools exposed over stdio for integration with Claude Code
- **Web Dashboard** — Real-time agent monitoring with status badges, task queue, and log viewer (auto-refreshes every 2s)
- **CLI** — Full command-line interface for all operations
- **Session Tracking** — SDK sessions with resume support
- **Auto-Restart** — Configurable automatic recovery from agent errors
- **SQLite Storage** — WAL-mode database for concurrent access

## Requirements

- Python 3.11+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated (`claude` must be available on PATH)

## Installation

```bash
# Clone and install
cd meta-agents
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize

```bash
meta-agent init
```

This creates the `data/` directory with the SQLite database and log folder.

### 2. Create Agents

```bash
# Simple chat agent (no tools)
meta-agent create --name "Echo" --system-prompt "You are helpful. Respond concisely." --tools ""

# Coding agent with file access
meta-agent create --name "Coder" \
  --system-prompt "You are an expert programmer. Read files, write code, run tests." \
  --tools "Read,Write,Edit,Bash,Glob,Grep"

# Read-only code reviewer
meta-agent create --name "Reviewer" \
  --system-prompt "You are a senior code reviewer. Analyze code for bugs and security issues." \
  --tools "Read,Glob,Grep"
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Agent display name | (required) |
| `--system-prompt` | Instructions for the agent | (required) |
| `--tools` | Comma-separated tool list | `Read,Write,Edit,Bash,Glob,Grep` |
| `--model` | Claude model ID | `claude-sonnet-4-5-20250929` |
| `--id` | Custom agent ID | auto-generated |
| `--description` | Agent description | empty |
| `--cwd` | Working directory for the agent | current directory |

### 3. List Agents

```bash
meta-agent list
```

```
                               Agents
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ ID       ┃ Name     ┃ Status  ┃ Model                      ┃ Description ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ echo     │ Echo     │ stopped │ claude-sonnet-4-5-20250929 │             │
│ coder    │ Coder    │ idle    │ claude-sonnet-4-5-20250929 │             │
│ reviewer │ Reviewer │ stopped │ claude-sonnet-4-5-20250929 │             │
└──────────┴──────────┴─────────┴────────────────────────────┴─────────────┘
```

### 4. Submit Tasks

```bash
meta-agent submit echo "What is 2+2?"
meta-agent submit coder "Write a Python function that checks if a number is prime"
```

### 5. Check Status

```bash
# All tasks
meta-agent status

# Specific agent
meta-agent status echo
```

### 6. View Logs

```bash
# Last 50 lines
meta-agent logs echo

# Last 200 lines
meta-agent logs echo -n 200
```

### 7. Delete an Agent

```bash
meta-agent delete echo
```

## Web Dashboard

Start the dashboard:

```bash
meta-agent dashboard --port 5555
```

Open http://localhost:5555 in your browser.

The dashboard provides:
- **Create Agent form** — name, model selector, tools, and system prompt
- **Agent cards** — live status badges (stopped/idle/running/error), Start/Stop/Delete/Logs buttons
- **Submit Task form** — pick an agent and enter a prompt
- **Tasks table** — task ID, agent, status badge, prompt, and timestamp
- **Auto-refresh** — polls every 2 seconds for live updates

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
  init        Initialize the data directory and database
  list        List all registered agents
  create      Create and register a new agent
  delete      Delete an agent by ID
  submit      Submit a task to an agent
  status      Show agent or task status
  logs        View agent logs
  mcp-server  Start the MCP server (stdio transport)
  dashboard   Start the web dashboard
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
│   ├── models.py                        # Pydantic models: AgentConfig, AgentState, Task
│   ├── config.py                        # Config singleton (db_path, log_dir)
│   ├── db.py                            # SQLite WAL database, CRUD operations
│   ├── agent_runner.py                  # Wraps claude_agent_sdk.query()
│   ├── agent_manager.py                 # Agent lifecycle, task scheduling, logs
│   ├── mcp_server.py                    # FastMCP server with 10 tools
│   ├── cli.py                           # Click CLI commands
│   ├── dashboard/
│   │   ├── app.py                       # Flask app factory
│   │   ├── routes.py                    # JSON API + page routes
│   │   └── templates/index.html         # htmx dashboard
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
    └── test_mcp_server.py              # MCP tool integration
```

## Concurrency Model

- `AgentManager` runs an asyncio event loop in a background daemon thread
- SDK `query()` calls are scheduled via `asyncio.run_coroutine_threadsafe()`
- `threading.Lock` protects in-memory agent state
- SQLite WAL mode with `busy_timeout=5000` for database concurrency
- Flask and CLI are synchronous — they interact with the manager through thread-safe methods

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
| `flask` | Web dashboard |
| `pydantic` | Data models and validation |
| `mcp` | MCP server (FastMCP) |
| `rich` | Terminal formatting and tables |

## License

MIT
