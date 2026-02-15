"""External model runner for non-Claude models (e.g. Gemini)."""

from __future__ import annotations

import logging
import os

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ExternalModelRunner:
    """Runs tasks against external model APIs. Text-in/text-out only."""

    def __init__(self, model_string: str):
        # Format: "external:provider:model_name"
        parts = model_string.split(":", 2)
        if len(parts) != 3 or parts[0] != "external":
            raise ValueError(
                f"Invalid external model format: {model_string!r}. "
                "Expected 'external:<provider>:<model>'"
            )
        self.provider = parts[1]
        self.model_name = parts[2]

    async def run(self, prompt: str, system_prompt: str = "") -> str:
        """Send prompt to external model and return response text."""
        if self.provider == "gemini":
            return await self._run_gemini(prompt, system_prompt)
        raise ValueError(f"Unsupported external provider: {self.provider!r}")

    async def _run_gemini(self, prompt: str, system_prompt: str = "") -> str:
        """Call Google Gemini generateContent API via httpx."""
        if httpx is None:
            raise RuntimeError(
                "httpx is required for external models. "
                "Install with: pip install meta-agent[external]"
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={api_key}"
        )

        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System Instructions]\n{system_prompt}"}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}],
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}],
        })

        payload = {"contents": contents}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            logger.error("Unexpected Gemini response: %s", data)
            raise RuntimeError(f"Failed to parse Gemini response: {e}") from e
