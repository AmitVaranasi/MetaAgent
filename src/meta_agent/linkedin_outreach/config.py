"""Configuration for LinkedIn outreach automation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class OutreachConfig:
    """Configuration for LinkedIn outreach automation.

    Attributes:
        num_connections: Number of connection requests to send per session
        outreach_message: Template for connection request message
        log_file: Path to log file for outreach activities
        cooldown_hours: Minimum hours between outreach sessions to avoid spam
        targets_file: Path to JSON file containing target profiles
        last_run_file: Path to file tracking last run timestamp
    """

    num_connections: int = 10
    outreach_message: str = (
        "Hi {name}, I came across your profile and was impressed by your work "
        "at {company}. I'd love to connect and learn more about your experience "
        "in {title}. Looking forward to connecting!"
    )
    log_file: Path = field(default_factory=lambda: Path.home() / ".meta-agent" / "linkedin_outreach.log")
    cooldown_hours: int = 12
    targets_file: Path = field(default_factory=lambda: Path.home() / ".meta-agent" / "linkedin_targets.json")
    last_run_file: Path = field(default_factory=lambda: Path.home() / ".meta-agent" / "linkedin_last_run.json")

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the path to the configuration file."""
        return Path.home() / ".meta-agent" / "linkedin_outreach_config.json"

    @classmethod
    def load(cls) -> OutreachConfig:
        """Load configuration from file.

        Returns:
            OutreachConfig instance with loaded settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is malformed
        """
        config_path = cls.get_config_path()

        if not config_path.exists():
            # Return default config if file doesn't exist
            return cls()

        with open(config_path, "r") as f:
            data = json.load(f)

        # Convert path strings to Path objects
        if "log_file" in data:
            data["log_file"] = Path(data["log_file"])
        if "targets_file" in data:
            data["targets_file"] = Path(data["targets_file"])
        if "last_run_file" in data:
            data["last_run_file"] = Path(data["last_run_file"])

        return cls(**data)

    def save(self) -> None:
        """Save configuration to file.

        Creates the config directory if it doesn't exist.
        """
        config_path = self.get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path objects
        data: dict[str, Any] = {
            "num_connections": self.num_connections,
            "outreach_message": self.outreach_message,
            "log_file": str(self.log_file),
            "cooldown_hours": self.cooldown_hours,
            "targets_file": str(self.targets_file),
            "last_run_file": str(self.last_run_file),
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.targets_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_run_file.parent.mkdir(parents=True, exist_ok=True)

    def get_last_run_time(self) -> datetime | None:
        """Get the timestamp of the last outreach run.

        Returns:
            datetime of last run, or None if never run
        """
        if not self.last_run_file.exists():
            return None

        try:
            with open(self.last_run_file, "r") as f:
                data = json.load(f)
                timestamp = data.get("last_run")
                if timestamp:
                    return datetime.fromisoformat(timestamp)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

        return None

    def update_last_run_time(self, timestamp: datetime | None = None) -> None:
        """Update the last run timestamp.

        Args:
            timestamp: Timestamp to record, defaults to current time
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.last_run_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.last_run_file, "w") as f:
            json.dump({"last_run": timestamp.isoformat()}, f, indent=2)

    def is_cooldown_active(self) -> bool:
        """Check if the cooldown period is still active.

        Returns:
            True if cooldown is active (too soon to run again), False otherwise
        """
        last_run = self.get_last_run_time()
        if last_run is None:
            return False

        now = datetime.now(timezone.utc)
        hours_since_last_run = (now - last_run).total_seconds() / 3600

        return hours_since_last_run < self.cooldown_hours

    def hours_until_next_run(self) -> float:
        """Calculate hours remaining until next allowed run.

        Returns:
            Hours remaining, or 0.0 if ready to run
        """
        last_run = self.get_last_run_time()
        if last_run is None:
            return 0.0

        now = datetime.now(timezone.utc)
        hours_since_last_run = (now - last_run).total_seconds() / 3600
        remaining = self.cooldown_hours - hours_since_last_run

        return max(0.0, remaining)
