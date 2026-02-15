"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from meta_agent.config import Config
from meta_agent.db import Database
from meta_agent.models import AgentConfig


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def config(tmp_dir: Path) -> Config:
    Config.reset()
    return Config.get(tmp_dir)


@pytest.fixture()
def db(config: Config) -> Database:
    d = Database(config.db_path)
    yield d
    d.close()


@pytest.fixture()
def sample_config() -> AgentConfig:
    return AgentConfig(
        id="test01",
        name="Test Agent",
        description="A test agent",
        system_prompt="You are a test agent.",
        allowed_tools=["Read", "Grep"],
        model="claude-sonnet-4-5-20250929",
    )
