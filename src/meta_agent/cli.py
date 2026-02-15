"""CLI for the meta-agent system."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .agent_manager import AgentManager
from .config import Config
from .db import Database
from .models import AgentConfig

console = Console()


def _make_manager(base_dir: str | None = None) -> AgentManager:
    cfg = Config.get(Path(base_dir) if base_dir else None)
    db = Database(cfg.db_path)
    mgr = AgentManager(db, cfg.log_dir)
    mgr.start()
    return mgr


@click.group()
@click.option("--data-dir", envvar="META_AGENT_DATA", default=None, help="Data directory")
@click.pass_context
def main(ctx: click.Context, data_dir: str | None) -> None:
    """Meta-Agent: orchestrate multiple Claude-powered agents."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


@main.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize the meta-agent data directory and database."""
    cfg = Config.get(Path(ctx.obj["data_dir"]) if ctx.obj["data_dir"] else None)
    Database(cfg.db_path)
    console.print(f"[green]Initialized at {cfg.base_dir}[/green]")


@main.command("list")
@click.pass_context
def list_agents(ctx: click.Context) -> None:
    """List all registered agents."""
    mgr = _make_manager(ctx.obj["data_dir"])
    agents = mgr.list_agents()
    if not agents:
        console.print("[dim]No agents registered.[/dim]")
        return
    table = Table(title="Agents")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Description")
    for a in agents:
        table.add_row(a.config.id, a.config.name, a.status.value, a.config.model, a.config.description)
    console.print(table)


@main.command()
@click.option("--name", required=True, help="Agent name")
@click.option("--system-prompt", required=True, help="System prompt")
@click.option("--tools", default="Read,Write,Edit,Bash,Glob,Grep", help="Comma-separated tool list (empty for none)")
@click.option("--model", default="claude-sonnet-4-5-20250929", help="Model ID")
@click.option("--description", default="", help="Agent description")
@click.option("--id", "agent_id", default=None, help="Custom agent ID")
@click.option("--cwd", default=None, help="Working directory")
@click.pass_context
def create(
    ctx: click.Context,
    name: str,
    system_prompt: str,
    tools: str,
    model: str,
    description: str,
    agent_id: str | None,
    cwd: str | None,
) -> None:
    """Create and register a new agent."""
    tool_list = [t.strip() for t in tools.split(",") if t.strip()] if tools else []
    kwargs: dict = dict(
        name=name,
        system_prompt=system_prompt,
        allowed_tools=tool_list,
        model=model,
        description=description,
        cwd=cwd,
    )
    if agent_id:
        kwargs["id"] = agent_id
    config = AgentConfig(**kwargs)
    mgr = _make_manager(ctx.obj["data_dir"])
    state = mgr.register_agent(config)
    console.print(f"[green]Created agent '{state.config.name}' (id={state.config.id})[/green]")


@main.command()
@click.argument("agent_id")
@click.pass_context
def delete(ctx: click.Context, agent_id: str) -> None:
    """Delete an agent by ID."""
    mgr = _make_manager(ctx.obj["data_dir"])
    if mgr.unregister_agent(agent_id):
        console.print(f"[green]Deleted agent {agent_id}[/green]")
    else:
        console.print(f"[red]Agent {agent_id} not found[/red]")
        sys.exit(1)


@main.command()
@click.argument("agent_id")
@click.argument("prompt")
@click.pass_context
def submit(ctx: click.Context, agent_id: str, prompt: str) -> None:
    """Submit a task to an agent."""
    mgr = _make_manager(ctx.obj["data_dir"])
    try:
        task = mgr.submit_task(agent_id, prompt)
        console.print(f"[green]Submitted task {task.id} to agent {agent_id}[/green]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("agent_id", required=False)
@click.pass_context
def status(ctx: click.Context, agent_id: str | None) -> None:
    """Show agent or task status."""
    mgr = _make_manager(ctx.obj["data_dir"])
    if agent_id:
        state = mgr.get_agent(agent_id)
        if state is None:
            console.print(f"[red]Agent {agent_id} not found[/red]")
            sys.exit(1)
        console.print(f"Agent: {state.config.name} ({state.config.id})")
        console.print(f"Status: {state.status.value}")
        if state.current_task_id:
            console.print(f"Current task: {state.current_task_id}")
        if state.error:
            console.print(f"Error: {state.error}")
    else:
        tasks = mgr.list_tasks()
        if not tasks:
            console.print("[dim]No tasks.[/dim]")
            return
        table = Table(title="Tasks")
        table.add_column("ID")
        table.add_column("Agent")
        table.add_column("Status")
        table.add_column("Prompt", max_width=40)
        table.add_column("Created")
        for t in tasks[:20]:
            table.add_row(t.id, t.agent_id, t.status, t.prompt[:40], str(t.created_at)[:19])
        console.print(table)


@main.command()
@click.argument("agent_id")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
@click.option("-n", "--lines", default=50, help="Number of lines")
@click.pass_context
def logs(ctx: click.Context, agent_id: str, follow: bool, lines: int) -> None:
    """View agent logs."""
    mgr = _make_manager(ctx.obj["data_dir"])
    log_text = mgr.get_logs(agent_id, lines=lines)
    if log_text:
        console.print(log_text)
    else:
        console.print(f"[dim]No logs for agent {agent_id}[/dim]")


@main.command("mcp-server")
@click.pass_context
def mcp_server(ctx: click.Context) -> None:
    """Start the MCP server (stdio transport)."""
    from .mcp_server import create_mcp_server

    cfg = Config.get(Path(ctx.obj["data_dir"]) if ctx.obj["data_dir"] else None)
    db = Database(cfg.db_path)
    mgr = AgentManager(db, cfg.log_dir)
    mgr.start()
    server = create_mcp_server(mgr)
    server.run(transport="stdio")


@main.command()
@click.option("--port", default=5555, help="Dashboard port")
@click.option("--host", default="127.0.0.1", help="Dashboard host")
@click.pass_context
def dashboard(ctx: click.Context, port: int, host: str) -> None:
    """Start the web dashboard."""
    from .dashboard.app import create_app

    cfg = Config.get(Path(ctx.obj["data_dir"]) if ctx.obj["data_dir"] else None)
    db = Database(cfg.db_path)
    mgr = AgentManager(db, cfg.log_dir)
    mgr.start()
    app = create_app(mgr)
    console.print(f"[green]Dashboard running at http://{host}:{port}[/green]")
    app.run(host=host, port=port, debug=False)
