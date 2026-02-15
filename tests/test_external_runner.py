"""Tests for the external model runner (Gemini adapter)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meta_agent.external_runner import ExternalModelRunner


def test_parse_valid_model_string():
    runner = ExternalModelRunner("external:gemini:gemini-2.0-flash")
    assert runner.provider == "gemini"
    assert runner.model_name == "gemini-2.0-flash"


def test_parse_model_with_dots():
    runner = ExternalModelRunner("external:gemini:gemini-2.5-pro")
    assert runner.provider == "gemini"
    assert runner.model_name == "gemini-2.5-pro"


def test_parse_invalid_format():
    with pytest.raises(ValueError, match="Invalid external model format"):
        ExternalModelRunner("gemini:gemini-2.0-flash")


def test_parse_invalid_prefix():
    with pytest.raises(ValueError, match="Invalid external model format"):
        ExternalModelRunner("internal:gemini:gemini-2.0-flash")


@pytest.mark.asyncio
async def test_unsupported_provider():
    runner = ExternalModelRunner("external:openai:gpt-4")
    with pytest.raises(ValueError, match="Unsupported external provider"):
        await runner.run("Hello")


@pytest.mark.asyncio
async def test_gemini_missing_api_key():
    runner = ExternalModelRunner("external:gemini:gemini-2.0-flash")
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            await runner.run("Hello")


@pytest.mark.asyncio
async def test_gemini_successful_call():
    import meta_agent.external_runner as mod

    runner = ExternalModelRunner("external:gemini:gemini-2.0-flash")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                    "role": "model",
                }
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

    original_httpx = mod.httpx
    try:
        mod.httpx = mock_httpx
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            result = await runner.run("Hello", system_prompt="Be helpful")
    finally:
        mod.httpx = original_httpx

    assert result == "Hello from Gemini!"
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "gemini-2.0-flash" in call_args[0][0]
    assert "test-key" in call_args[0][0]


@pytest.mark.asyncio
async def test_gemini_malformed_response():
    import meta_agent.external_runner as mod

    runner = ExternalModelRunner("external:gemini:gemini-2.0-flash")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"candidates": []}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

    original_httpx = mod.httpx
    try:
        mod.httpx = mock_httpx
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with pytest.raises(RuntimeError, match="Failed to parse Gemini response"):
                await runner.run("Hello")
    finally:
        mod.httpx = original_httpx
