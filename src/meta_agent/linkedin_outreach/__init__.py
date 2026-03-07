"""LinkedIn outreach automation module.

This module provides automated LinkedIn outreach functionality that triggers
when a macOS laptop wakes from sleep, with user permission.
"""

from __future__ import annotations

from meta_agent.linkedin_outreach.config import OutreachConfig
from meta_agent.linkedin_outreach.outreach_engine import OutreachEngine
from meta_agent.linkedin_outreach.wake_detector import WakeDetector

__all__ = [
    "OutreachConfig",
    "OutreachEngine",
    "WakeDetector",
]
