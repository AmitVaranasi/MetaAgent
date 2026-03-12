"""macOS wake-from-sleep detection."""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class WakeDetector:
    """Detects when a macOS laptop wakes from sleep.

    Uses the macOS `log stream` command to monitor power management events
    and trigger callbacks when the system wakes from sleep.
    """

    def __init__(self) -> None:
        """Initialize the wake detector."""
        self._callbacks: list[Callable[[], None]] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def on_wake(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when the system wakes from sleep.

        Args:
            callback: Function to call on wake event (takes no arguments)
        """
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start monitoring for wake events.

        This method blocks and continuously monitors system logs for wake events.
        It should typically be run in a separate thread or as the main event loop.

        Raises:
            RuntimeError: If already running
            OSError: If unable to start log stream process
        """
        if self._running:
            raise RuntimeError("WakeDetector is already running")

        self._running = True
        logger.info("Starting wake detection...")

        try:
            # Start the log stream process to monitor wake events
            # The predicate filters for wake-related messages in the power management subsystem
            self._process = subprocess.Popen(
                [
                    "log",
                    "stream",
                    "--predicate",
                    'eventMessage contains "Wake reason" or '
                    'eventMessage contains "DarkWake" or '
                    'subsystem == "com.apple.iokit.power"',
                    "--style",
                    "syslog",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
            )

            logger.info("Wake detector started successfully")

            # Monitor the output stream
            if self._process.stdout is not None:
                for line in iter(self._process.stdout.readline, b""):
                    if not self._running:
                        break

                    line_str = line.decode("utf-8", errors="ignore").strip()

                    # Check if this is a wake event
                    if self._is_wake_event(line_str):
                        logger.info(f"Wake event detected: {line_str}")
                        self._trigger_callbacks()

        except Exception as e:
            logger.error(f"Error in wake detector: {e}")
            raise
        finally:
            self._cleanup()

    def start_async(self) -> None:
        """Start monitoring in a background thread.

        Returns immediately while monitoring continues in the background.
        Use stop() to terminate monitoring.

        Raises:
            RuntimeError: If already running
        """
        if self._running:
            raise RuntimeError("WakeDetector is already running")

        self._thread = threading.Thread(target=self.start, daemon=True)
        self._thread.start()
        logger.info("Wake detector started in background thread")

    def stop(self) -> None:
        """Stop monitoring for wake events.

        Terminates the log stream process and cleans up resources.
        Safe to call multiple times.
        """
        if not self._running:
            return

        logger.info("Stopping wake detector...")
        self._running = False

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate gracefully, killing it")
                self._process.kill()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self._cleanup()
        logger.info("Wake detector stopped")

    def _is_wake_event(self, log_line: str) -> bool:
        """Determine if a log line represents a wake event.

        Args:
            log_line: A line from the system log

        Returns:
            True if this is a wake event, False otherwise
        """
        # Look for common wake indicators
        wake_indicators = [
            "Wake reason",
            "DarkWake",
            "wake",  # Generic wake message
        ]

        line_lower = log_line.lower()

        # Check for wake indicators
        for indicator in wake_indicators:
            if indicator.lower() in line_lower:
                # Additional filtering to avoid false positives
                # Exclude maintenance wakes or network wakes that don't involve user interaction
                if "maintenance" in line_lower:
                    continue
                return True

        return False

    def _trigger_callbacks(self) -> None:
        """Execute all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in wake callback: {e}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._process:
            try:
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        self._process = None
        self._running = False


class SimpleWakeDetector:
    """A simpler alternative wake detector using pmset log.

    This detector polls the power management log instead of streaming,
    which may be more reliable in some scenarios but has higher latency.
    """

    def __init__(self, poll_interval: float = 60.0) -> None:
        """Initialize the simple wake detector.

        Args:
            poll_interval: Seconds between log checks
        """
        self._callbacks: list[Callable[[], None]] = []
        self._poll_interval = poll_interval
        self._running = False
        self._last_wake_time: str | None = None
        self._thread: threading.Thread | None = None

    def on_wake(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when the system wakes from sleep.

        Args:
            callback: Function to call on wake event
        """
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start monitoring for wake events (blocking)."""
        import time

        if self._running:
            raise RuntimeError("SimpleWakeDetector is already running")

        self._running = True
        logger.info("Starting simple wake detection...")

        try:
            while self._running:
                wake_time = self._check_recent_wake()
                if wake_time and wake_time != self._last_wake_time:
                    logger.info(f"Wake event detected at: {wake_time}")
                    self._last_wake_time = wake_time
                    self._trigger_callbacks()

                time.sleep(self._poll_interval)
        except Exception as e:
            logger.error(f"Error in simple wake detector: {e}")
            raise
        finally:
            self._running = False

    def start_async(self) -> None:
        """Start monitoring in a background thread."""
        if self._running:
            raise RuntimeError("SimpleWakeDetector is already running")

        self._thread = threading.Thread(target=self.start, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring for wake events."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval + 5)

    def _check_recent_wake(self) -> str | None:
        """Check if there was a recent wake event.

        Returns:
            Timestamp string of recent wake, or None if no recent wake
        """
        try:
            # Get recent power management events
            result = subprocess.run(
                ["pmset", "-g", "log", "|", "grep", "-i", "wake"],
                capture_output=True,
                text=True,
                shell=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split("\n")
                if lines:
                    # Return the most recent wake timestamp
                    return lines[-1].split()[0] if lines[-1] else None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Error checking wake events: {e}")

        return None

    def _trigger_callbacks(self) -> None:
        """Execute all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in wake callback: {e}")
