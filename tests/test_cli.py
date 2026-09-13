"""Tests for the CLI — 475 lines that had none, and it is how the tool is used."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from meta_agent.cli import main
from meta_agent.config import Config


@pytest.fixture()
def data_dir(tmp_path: Path) -> str:
    Config.reset()
    return str(tmp_path)


@pytest.fixture()
def run(data_dir: str):
    def _run(*args: str):
        return CliRunner().invoke(main, ["--data-dir", data_dir, *args])

    return _run


def test_help_lists_every_command(run):
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ["brain", "chat", "create", "dashboard", "delete", "init",
                    "list", "logs", "mcp-server", "status", "submit", "workflow"]:
        assert command in result.output


def test_init_creates_the_database(run, data_dir: str):
    result = run("init")
    assert result.exit_code == 0
    assert (Path(data_dir) / "meta_agent.db").exists()


def test_list_is_empty_before_anything_is_created(run):
    result = run("list")
    assert result.exit_code == 0
    assert "No agents registered" in result.output


def test_create_then_list_then_delete(run):
    created = run("create", "--name", "Writer", "--system-prompt", "You write.", "--id", "w1")
    assert created.exit_code == 0
    assert "w1" in created.output

    listed = run("list")
    assert "Writer" in listed.output

    deleted = run("delete", "w1")
    assert deleted.exit_code == 0
    assert "No agents registered" in run("list").output


def test_delete_unknown_agent_exits_nonzero(run):
    result = run("delete", "nope")
    assert result.exit_code == 1
    assert "not found" in result.output


def test_submit_to_unknown_agent_exits_nonzero(run):
    result = run("submit", "nope", "do a thing")
    assert result.exit_code == 1
    assert "not registered" in result.output


def test_status_with_no_tasks(run):
    assert "No tasks" in run("status").output


def test_status_of_unknown_agent_exits_nonzero(run):
    result = run("status", "nope")
    assert result.exit_code == 1


def test_status_of_a_known_agent(run):
    run("create", "--name", "Writer", "--system-prompt", "You write.", "--id", "w1")
    result = run("status", "w1")
    assert result.exit_code == 0
    assert "Writer" in result.output


def test_logs_for_an_agent_with_none(run):
    assert "No logs" in run("logs", "w1").output


def test_workflow_list_is_empty(run):
    assert "No workflows" in run("workflow").output


def test_workflow_show_unknown_exits_nonzero(run):
    result = run("workflow", "nope")
    assert result.exit_code == 1
    assert "not found" in result.output


def test_create_accepts_an_empty_tool_list(run):
    result = run("create", "--name", "Thinker", "--system-prompt", "think",
                 "--id", "t1", "--tools", "")
    assert result.exit_code == 0
    from meta_agent.db import Database

    cfg = Config.get()
    assert Database(cfg.db_path).get_agent("t1").allowed_tools == []


def test_data_dir_is_honoured(tmp_path: Path):
    """--data-dir was never propagated into the spawned MCP subprocess, so the
    Brain silently orchestrated against a different database."""
    Config.reset()
    target = tmp_path / "elsewhere"
    result = CliRunner().invoke(main, ["--data-dir", str(target), "init"])
    assert result.exit_code == 0
    assert (target / "meta_agent.db").exists()


@pytest.mark.parametrize("command", ["dashboard", "brain", "chat", "mcp-server"])
def test_command_help_runs(command: str):
    result = CliRunner().invoke(main, [command, "--help"])
    assert result.exit_code == 0
