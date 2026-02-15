"""Configuration singleton for the meta-agent system."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_BASE = Path(__file__).resolve().parent.parent.parent / "data"


class Config:
    _instance: Config | None = None

    def __init__(
        self,
        base_dir: Path | None = None,
    ):
        self.base_dir = base_dir or _DEFAULT_BASE
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.base_dir / "meta_agent.db"
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get(cls, base_dir: Path | None = None) -> Config:
        if cls._instance is None:
            cls._instance = cls(base_dir)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
