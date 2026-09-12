"""Smoke tests for the dashboard.

The dashboard shipped as 490 lines with zero references from the CLI and zero
tests, so nothing ever proved it could even be constructed.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from meta_agent.agent_manager import AgentManager
from meta_agent.dashboard.app import create_app
from meta_agent.db import Database
from meta_agent.models import AgentConfig


@pytest.fixture()
def manager(db: Database, config) -> AgentManager:
    mgr = AgentManager(db, config.log_dir)
    mgr.start()
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def client(manager: AgentManager):
    app = create_app(manager)
    app.config.update(TESTING=True)
    return app.test_client()


def test_dashboard_command_is_registered():
    from meta_agent.cli import main

    assert "dashboard" in main.commands


def test_dashboard_command_help_runs():
    from meta_agent.cli import main

    result = CliRunner().invoke(main, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output


@pytest.mark.parametrize("path", ["/", "/kanban", "/kanban/enhanced"])
def test_pages_render(client, path: str):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize(
    "path", ["/api/agents", "/api/tasks", "/api/workflows", "/api/kanban", "/api/kanban/enhanced"]
)
def test_api_endpoints_return_json(client, path: str):
    response = client.get(path)
    assert response.status_code == 200
    assert response.get_json() is not None


def test_api_agents_reflects_the_managers_agents(client, manager: AgentManager):
    manager.register_agent(AgentConfig(id="dash01", name="Dash", system_prompt="x"))
    ids = [a["id"] for a in client.get("/api/agents").get_json()]
    assert "dash01" in ids


def test_api_get_missing_agent_is_404(client):
    assert client.get("/api/agents/nope").status_code == 404
