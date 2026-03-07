"""User permission prompts for LinkedIn outreach."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Literal

logger = logging.getLogger(__name__)


def show_permission_prompt(
    message: str = "LinkedIn Outreach: Your laptop just woke up. Would you like to send connection requests to 10 people on LinkedIn?",
    timeout: int = 60,
) -> bool:
    """Show a permission prompt asking user to approve LinkedIn outreach.

    Attempts to show a native macOS dialog using AppleScript. Falls back to
    a terminal prompt if that fails.

    Args:
        message: The message to display in the prompt
        timeout: Timeout in seconds (applies to terminal fallback only)

    Returns:
        True if user approved, False if declined or timed out
    """
    # Try native macOS dialog first
    if sys.platform == "darwin":
        result = _show_macos_dialog(message)
        if result is not None:
            return result

    # Fallback to terminal prompt
    logger.info("Falling back to terminal prompt")
    return _show_terminal_prompt(message, timeout)


def _show_macos_dialog(message: str) -> bool | None:
    """Show a native macOS dialog using AppleScript.

    Args:
        message: The message to display

    Returns:
        True if user clicked "Yes, proceed"
        False if user clicked "No, skip"
        None if dialog failed to show
    """
    try:
        # Create AppleScript for a dialog with custom buttons
        applescript = f'''
        display dialog "{message}" ¬
            buttons {{"No, skip", "Yes, proceed"}} ¬
            default button "Yes, proceed" ¬
            with title "LinkedIn Outreach" ¬
            with icon note ¬
            giving up after 60
        '''

        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            text=True,
            timeout=65,  # Slightly longer than dialog timeout
        )

        # Check which button was clicked
        if result.returncode == 0:
            # User clicked a button
            if "Yes, proceed" in result.stdout:
                logger.info("User approved outreach via macOS dialog")
                return True
            else:
                logger.info("User declined outreach via macOS dialog")
                return False
        else:
            # Dialog was cancelled or gave up (timeout)
            logger.info("macOS dialog was cancelled or timed out")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("macOS dialog timed out")
        return False
    except FileNotFoundError:
        logger.warning("osascript not found, cannot show macOS dialog")
        return None
    except Exception as e:
        logger.warning(f"Failed to show macOS dialog: {e}")
        return None


def _show_terminal_prompt(message: str, timeout: int) -> bool:
    """Show a permission prompt in the terminal using rich.

    Args:
        message: The message to display
        timeout: Timeout in seconds

    Returns:
        True if user approved, False otherwise
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Confirm

        console = Console()

        # Show a styled panel with the message
        panel = Panel(
            message,
            title="🔔 LinkedIn Outreach",
            border_style="blue",
            padding=(1, 2),
        )
        console.print()
        console.print(panel)
        console.print()

        # Use rich's Confirm prompt (note: doesn't support timeout natively)
        # For timeout support, we'd need to use threading or signal handling
        try:
            result = Confirm.ask(
                "[bold cyan]Proceed with outreach?[/bold cyan]",
                default=False,
            )
            logger.info(f"User {'approved' if result else 'declined'} via terminal prompt")
            return result
        except KeyboardInterrupt:
            logger.info("User cancelled prompt with Ctrl+C")
            return False

    except ImportError:
        logger.warning("rich library not available, using basic input")
        return _show_basic_prompt(message)


def _show_basic_prompt(message: str) -> bool:
    """Show a basic terminal prompt without rich.

    Args:
        message: The message to display

    Returns:
        True if user approved, False otherwise
    """
    try:
        print()
        print("=" * 70)
        print("LinkedIn Outreach")
        print("=" * 70)
        print(message)
        print()

        response = input("Proceed? (y/N): ").strip().lower()
        result = response in ("y", "yes")

        logger.info(f"User {'approved' if result else 'declined'} via basic prompt")
        return result

    except (EOFError, KeyboardInterrupt):
        logger.info("User cancelled basic prompt")
        return False


def show_notification(
    title: str,
    message: str,
    subtitle: str | None = None,
) -> None:
    """Show a macOS notification.

    Args:
        title: Notification title
        message: Notification message
        subtitle: Optional subtitle
    """
    if sys.platform != "darwin":
        logger.debug("Notifications only supported on macOS")
        return

    try:
        # Build AppleScript for notification
        subtitle_part = f'subtitle "{subtitle}"' if subtitle else ""

        applescript = f'''
        display notification "{message}" ¬
            with title "{title}" ¬
            {subtitle_part}
        '''

        subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            timeout=5,
        )

        logger.debug(f"Showed notification: {title}")

    except Exception as e:
        logger.warning(f"Failed to show notification: {e}")


def show_outreach_summary(
    num_sent: int,
    num_failed: int,
    duration_seconds: float,
) -> None:
    """Show a summary notification after outreach completes.

    Args:
        num_sent: Number of successful connection requests
        num_failed: Number of failed attempts
        duration_seconds: Time taken for outreach
    """
    if num_sent == 0 and num_failed == 0:
        return

    total = num_sent + num_failed
    success_rate = (num_sent / total * 100) if total > 0 else 0

    message = f"Sent {num_sent}/{total} requests ({success_rate:.0f}% success) in {duration_seconds:.1f}s"

    show_notification(
        title="LinkedIn Outreach Complete",
        message=message,
        subtitle=f"{num_failed} failed" if num_failed > 0 else None,
    )
