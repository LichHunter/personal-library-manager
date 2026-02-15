"""Unit tests for the shared LLM provider module.

All tests are fully mocked — no real API calls or auth files needed.
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures — fake auth file
# ---------------------------------------------------------------------------

FAKE_AUTH_ANTHROPIC = {
    "anthropic": {
        "type": "oauth",
        "access": "fake-access-token-anthropic",
        "refresh": "fake-refresh-token-anthropic",
        "expires": int(time.time() * 1000) + 3_600_000,  # 1 hour from now
    }
}

FAKE_AUTH_GOOGLE = {
    "google": {
        "type": "oauth",
        "access": "fake-access-token-google",
        "refresh": "fake-refresh-token|project-123|managed-456",
        "expires": int(time.time() * 1000) + 3_600_000,
    }
}

FAKE_AUTH_BOTH = {**FAKE_AUTH_ANTHROPIC, **FAKE_AUTH_GOOGLE}


@pytest.fixture()
def fake_auth_file(tmp_path: Path) -> Path:
    """Create a temporary auth.json with both providers."""
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(FAKE_AUTH_BOTH))
    return auth_path


@pytest.fixture()
def fake_anthropic_auth_file(tmp_path: Path) -> Path:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(FAKE_AUTH_ANTHROPIC))
    return auth_path


@pytest.fixture()
def fake_google_auth_file(tmp_path: Path) -> Path:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps(FAKE_AUTH_GOOGLE))
    return auth_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_httpx_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a fake httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.text = text or json.dumps(json_data or {})
    resp.headers = headers or {}
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


# ===================================================================
# 1. BASE MODULE — routing & call_llm
# ===================================================================


class TestModelRouting:
    """Verify model names route to the correct provider."""

    def test_gemini_models_detected(self):
        from shared.llm.base import _is_gemini_model

        assert _is_gemini_model("gemini-2.5-pro") is True
        assert _is_gemini_model("gemini-2.5-flash") is True
        assert _is_gemini_model("gemini-2.0-flash") is True
        assert _is_gemini_model("gemma-3-27b") is True
        assert _is_gemini_model("Gemini-Pro") is True  # case-insensitive

    def test_anthropic_models_detected(self):
        from shared.llm.base import _is_gemini_model

        assert _is_gemini_model("sonnet") is False
        assert _is_gemini_model("haiku") is False
        assert _is_gemini_model("opus") is False
        assert _is_gemini_model("claude-sonnet-4-5-20250929") is False
        assert _is_gemini_model("claude-haiku-4-5") is False

    def test_get_provider_returns_anthropic(self, fake_auth_file: Path):
        from shared.llm.base import _provider_cache

        _provider_cache.clear()  # reset singleton

        with patch(
            "shared.llm.anthropic_provider.AnthropicProvider.__init__",
            return_value=None,
        ):
            from shared.llm.base import get_provider

            provider = get_provider("sonnet")
            assert provider.name == "anthropic"

    def test_get_provider_returns_gemini(self, fake_auth_file: Path):
        from shared.llm.base import _provider_cache

        _provider_cache.clear()

        with patch(
            "shared.llm.gemini_provider.GeminiProvider.__init__",
            return_value=None,
        ):
            from shared.llm.base import get_provider

            provider = get_provider("gemini-2.5-flash")
            assert provider.name == "gemini"

    def test_provider_is_cached(self, fake_auth_file: Path):
        from shared.llm.base import _provider_cache

        _provider_cache.clear()

        with patch(
            "shared.llm.anthropic_provider.AnthropicProvider.__init__",
            return_value=None,
        ):
            from shared.llm.base import get_provider

            p1 = get_provider("sonnet")
            p2 = get_provider("haiku")
            assert p1 is p2  # same provider instance


# ===================================================================
# 2. ANTHROPIC PROVIDER
# ===================================================================


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_load_auth_success(self, fake_anthropic_auth_file: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(auth_path=str(fake_anthropic_auth_file))
        assert provider.name == "anthropic"
        assert provider._auth_data["access"] == "fake-access-token-anthropic"

    def test_load_auth_missing_file(self, tmp_path: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        with pytest.raises(FileNotFoundError, match="OpenCode auth file not found"):
            AnthropicProvider(auth_path=str(tmp_path / "nonexistent.json"))

    def test_load_auth_no_anthropic_key(self, tmp_path: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({"google": {}}))
        with pytest.raises(ValueError, match="Anthropic credentials not found"):
            AnthropicProvider(auth_path=str(auth_path))

    def test_load_auth_wrong_type(self, tmp_path: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        auth_path = tmp_path / "auth.json"
        auth_path.write_text(
            json.dumps({"anthropic": {"type": "api_key", "key": "sk-xxx"}})
        )
        with pytest.raises(ValueError, match="not OAuth type"):
            AnthropicProvider(auth_path=str(auth_path))

    def test_model_aliases(self):
        from shared.llm.anthropic_provider import AnthropicProvider

        assert AnthropicProvider._resolve_model("haiku") == "claude-3-5-haiku-latest"
        assert AnthropicProvider._resolve_model("sonnet") == "claude-sonnet-4-5-20250929"
        assert AnthropicProvider._resolve_model("opus") == "claude-opus-4-5-20251101"
        # Pass-through for full model IDs
        assert (
            AnthropicProvider._resolve_model("claude-sonnet-4-5-20250929")
            == "claude-sonnet-4-5-20250929"
        )

    def test_generate_success(self, fake_anthropic_auth_file: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(auth_path=str(fake_anthropic_auth_file))

        api_response = _make_httpx_response(
            json_data={
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "end_turn",
            }
        )

        with patch("httpx.post", return_value=api_response):
            result = provider.generate("Say hello", model="sonnet")
            assert result == "Hello from Claude!"

    def test_generate_returns_empty_on_error(self, fake_anthropic_auth_file: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(auth_path=str(fake_anthropic_auth_file))

        error_response = _make_httpx_response(
            status_code=400,
            text='{"error": "bad request"}',
        )

        with patch("httpx.post", return_value=error_response):
            result = provider.generate("Bad prompt", model="sonnet")
            assert result == ""

    def test_generate_retries_on_429(self, fake_anthropic_auth_file: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(auth_path=str(fake_anthropic_auth_file))

        rate_limit_response = _make_httpx_response(
            status_code=429,
            text="rate limited",
            headers={"retry-after": "0.01"},
        )
        success_response = _make_httpx_response(
            json_data={
                "content": [{"type": "text", "text": "Recovered!"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        with patch("httpx.post", side_effect=[rate_limit_response, success_response]):
            with patch("time.sleep"):  # skip actual sleeping
                result = provider.generate("Retry me", model="sonnet")
                assert result == "Recovered!"

    def test_generate_thinking_budget(self, fake_anthropic_auth_file: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(auth_path=str(fake_anthropic_auth_file))

        api_response = _make_httpx_response(
            json_data={
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Thought result"},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        )

        with patch("httpx.post", return_value=api_response) as mock_post:
            result = provider.generate(
                "Think hard", model="sonnet", thinking_budget=5000
            )
            assert result == "Thought result"

            # Verify thinking config was sent in request body
            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs["json"]
            assert body["thinking"]["type"] == "enabled"
            assert body["thinking"]["budget_tokens"] == 5000
            assert "temperature" not in body  # must NOT be set with thinking

    def test_token_refresh_called_when_expired(self, tmp_path: Path):
        from shared.llm.anthropic_provider import AnthropicProvider

        # Create auth with expired token
        auth_data = {
            "anthropic": {
                "type": "oauth",
                "access": "expired-token",
                "refresh": "refresh-token",
                "expires": int(time.time() * 1000) - 60_000,  # expired
            }
        }
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps(auth_data))

        provider = AnthropicProvider(auth_path=str(auth_path))

        # Mock the refresh call
        refresh_response = _make_httpx_response(
            json_data={
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
            }
        )
        api_response = _make_httpx_response(
            json_data={
                "content": [{"type": "text", "text": "After refresh"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        with patch("httpx.post", side_effect=[refresh_response, api_response]):
            result = provider.generate("Test refresh", model="sonnet")
            assert result == "After refresh"

        # Verify auth file was updated
        saved = json.loads(auth_path.read_text())
        assert saved["anthropic"]["access"] == "new-access-token"


# ===================================================================
# 3. GEMINI PROVIDER
# ===================================================================


class TestGeminiProvider:
    """Tests for GeminiProvider."""

    def test_load_auth_success(self, fake_google_auth_file: Path):
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))
        assert provider.name == "gemini"
        assert provider._auth_data["access"] == "fake-access-token-google"
        assert provider._project_id == "project-123"

    def test_load_auth_missing_file(self, tmp_path: Path):
        from shared.llm.gemini_provider import GeminiProvider

        with pytest.raises(FileNotFoundError, match="OpenCode auth file not found"):
            GeminiProvider(auth_path=str(tmp_path / "nonexistent.json"))

    def test_load_auth_no_google_key(self, tmp_path: Path):
        from shared.llm.gemini_provider import GeminiProvider

        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({"anthropic": {}}))
        with pytest.raises(ValueError, match="Google/Gemini credentials not found"):
            GeminiProvider(auth_path=str(auth_path))

    def test_load_auth_wrong_type(self, tmp_path: Path):
        from shared.llm.gemini_provider import GeminiProvider

        auth_path = tmp_path / "auth.json"
        auth_path.write_text(
            json.dumps({"google": {"type": "api_key", "key": "AIza..."}})
        )
        with pytest.raises(ValueError, match="not OAuth type"):
            GeminiProvider(auth_path=str(auth_path))

    def test_model_aliases(self):
        from shared.llm.gemini_provider import GeminiProvider

        # Convenience aliases → canonical short names
        assert GeminiProvider._resolve_model("gemini-pro") == "gemini-2.5-pro"
        assert GeminiProvider._resolve_model("gemini-flash") == "gemini-2.5-flash"
        assert GeminiProvider._resolve_model("gemini-3-pro") == "gemini-3-pro-preview"
        assert GeminiProvider._resolve_model("gemini-3-flash") == "gemini-3-flash-preview"
        # Short names pass through unchanged (Code Assist API uses them directly)
        assert GeminiProvider._resolve_model("gemini-2.5-pro") == "gemini-2.5-pro"
        assert GeminiProvider._resolve_model("gemini-2.5-flash") == "gemini-2.5-flash"
        assert GeminiProvider._resolve_model("gemini-2.0-flash") == "gemini-2.0-flash"
        assert GeminiProvider._resolve_model("gemini-3-pro-preview") == "gemini-3-pro-preview"
        assert GeminiProvider._resolve_model("gemini-3-flash-preview") == "gemini-3-flash-preview"
        assert GeminiProvider._resolve_model("gemini-2.5-flash-lite") == "gemini-2.5-flash-lite"

    def test_parse_refresh_parts(self):
        from shared.llm.gemini_provider import _parse_refresh_parts

        parts = _parse_refresh_parts("token123|project-abc|managed-xyz")
        assert parts["refresh_token"] == "token123"
        assert parts["project_id"] == "project-abc"
        assert parts["managed_project_id"] == "managed-xyz"

        # Minimal: just token
        parts = _parse_refresh_parts("token123")
        assert parts["refresh_token"] == "token123"
        assert parts["project_id"] == ""
        assert parts["managed_project_id"] == ""

    def test_format_refresh_parts(self):
        from shared.llm.gemini_provider import _format_refresh_parts

        assert _format_refresh_parts("tok") == "tok"
        assert _format_refresh_parts("tok", "proj") == "tok|proj|"
        assert _format_refresh_parts("tok", "proj", "mgd") == "tok|proj|mgd"

    def test_generate_success(self, fake_google_auth_file: Path):
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        api_response = _make_httpx_response(
            json_data={
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Hello from Gemini!"}],
                                "role": "model",
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                    },
                }
            }
        )

        with patch("httpx.post", return_value=api_response):
            result = provider.generate("Say hello", model="gemini-2.5-flash")
            assert result == "Hello from Gemini!"

    def test_generate_unwraps_code_assist_envelope(self, fake_google_auth_file: Path):
        """Verify response.response is unwrapped correctly."""
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        # Code Assist wraps response in {response: {...}}
        api_response = _make_httpx_response(
            json_data={
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Unwrapped!"}],
                                "role": "model",
                            }
                        }
                    ],
                    "usageMetadata": {},
                }
            }
        )

        with patch("httpx.post", return_value=api_response):
            result = provider.generate("Test", model="gemini-2.5-flash")
            assert result == "Unwrapped!"

    def test_generate_handles_raw_response(self, fake_google_auth_file: Path):
        """Verify we handle non-wrapped responses too (direct Gemini API)."""
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        api_response = _make_httpx_response(
            json_data={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Direct response!"}],
                            "role": "model",
                        }
                    }
                ],
                "usageMetadata": {},
            }
        )

        with patch("httpx.post", return_value=api_response):
            result = provider.generate("Test", model="gemini-2.5-flash")
            assert result == "Direct response!"

    def test_generate_returns_empty_on_error(self, fake_google_auth_file: Path):
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        error_response = _make_httpx_response(
            status_code=400, text='{"error": "bad request"}'
        )

        with patch("httpx.post", return_value=error_response):
            result = provider.generate("Bad prompt", model="gemini-2.5-flash")
            assert result == ""

    def test_generate_retries_on_503(self, fake_google_auth_file: Path):
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        error_resp = _make_httpx_response(
            status_code=503, text="service unavailable"
        )
        success_resp = _make_httpx_response(
            json_data={
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "Recovered!"}]}}
                    ],
                    "usageMetadata": {},
                }
            }
        )

        with patch("httpx.post", side_effect=[error_resp, success_resp]):
            with patch("time.sleep"):
                result = provider.generate("Retry me", model="gemini-2.5-flash")
                assert result == "Recovered!"

    def test_generate_no_retry_on_quota_exhausted(
        self, fake_google_auth_file: Path
    ):
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        quota_resp = _make_httpx_response(
            status_code=429,
            json_data={
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "reason": "QUOTA_EXHAUSTED",
                        }
                    ]
                }
            },
        )

        with patch("httpx.post", return_value=quota_resp) as mock_post:
            result = provider.generate("Quota test", model="gemini-2.5-flash")
            assert result == ""
            # Should NOT retry — only 1 call
            assert mock_post.call_count == 1

    def test_generate_request_body_structure(self, fake_google_auth_file: Path):
        """Verify the Code Assist envelope is built correctly."""
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        api_response = _make_httpx_response(
            json_data={
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "OK"}]}}
                    ],
                    "usageMetadata": {},
                }
            }
        )

        with patch("httpx.post", return_value=api_response) as mock_post:
            provider.generate(
                "Test prompt",
                model="gemini-2.5-flash",
                max_tokens=1000,
                temperature=0.5,
            )

            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs["json"]

            # Code Assist envelope
            assert "project" in body
            assert body["project"] == "project-123"
            assert "model" in body
            assert "request" in body

            # GenerateContent request inside envelope
            req = body["request"]
            assert req["contents"][0]["parts"][0]["text"] == "Test prompt"
            assert req["generationConfig"]["maxOutputTokens"] == 1000
            assert req["generationConfig"]["temperature"] == 0.5

    def test_generate_thinking_config_for_2_5(self, fake_google_auth_file: Path):
        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        api_response = _make_httpx_response(
            json_data={
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "Thought result"}]}}
                    ],
                    "usageMetadata": {},
                }
            }
        )

        with patch("httpx.post", return_value=api_response) as mock_post:
            provider.generate(
                "Think hard",
                model="gemini-2.5-pro",
                thinking_budget=8000,
            )

            body = mock_post.call_args.kwargs["json"]
            thinking_cfg = body["request"]["generationConfig"].get("thinkingConfig")
            assert thinking_cfg is not None
            assert thinking_cfg["thinkingBudget"] == 8000

    def test_project_id_from_env_var(self, tmp_path: Path):
        from shared.llm.gemini_provider import GeminiProvider

        auth_data = {
            "google": {
                "type": "oauth",
                "access": "tok",
                "refresh": "refresh-tok",  # no project_id in parts
                "expires": int(time.time() * 1000) + 3_600_000,
            }
        }
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps(auth_data))

        with patch.dict("os.environ", {"OPENCODE_GEMINI_PROJECT_ID": "env-project"}):
            provider = GeminiProvider(auth_path=str(auth_path))
            assert provider._project_id == "env-project"

    def test_token_refresh_called_when_expired(self, tmp_path: Path):
        from shared.llm.gemini_provider import GeminiProvider

        auth_data = {
            "google": {
                "type": "oauth",
                "access": "expired-token",
                "refresh": "refresh-tok|proj-123|",
                "expires": int(time.time() * 1000) - 60_000,  # expired
            }
        }
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps(auth_data))

        provider = GeminiProvider(auth_path=str(auth_path))

        refresh_response = _make_httpx_response(
            json_data={
                "access_token": "new-gemini-access",
                "expires_in": 3600,
            }
        )
        api_response = _make_httpx_response(
            json_data={
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "After refresh"}]}}
                    ],
                    "usageMetadata": {},
                }
            }
        )

        with patch.dict(
            "os.environ",
            {"GEMINI_CLIENT_ID": "test-id", "GEMINI_CLIENT_SECRET": "test-secret"},
        ):
            with patch("httpx.post", side_effect=[refresh_response, api_response]):
                result = provider.generate("Test refresh", model="gemini-2.5-flash")
                assert result == "After refresh"

        saved = json.loads(auth_path.read_text())
        assert saved["google"]["access"] == "new-gemini-access"


# ===================================================================
# 4. INTEGRATION — call_llm routes correctly
# ===================================================================


class TestCallLlmIntegration:
    """Test that call_llm correctly routes to the right provider."""

    def test_call_llm_anthropic(self, fake_auth_file: Path):
        from shared.llm.base import _provider_cache, call_llm

        _provider_cache.clear()

        api_response = _make_httpx_response(
            json_data={
                "content": [{"type": "text", "text": "From Anthropic"}],
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }
        )

        with patch(
            "shared.llm.anthropic_provider.AnthropicProvider.__init__",
            lambda self, **kw: self.__dict__.update(
                auth_path=Path(fake_auth_file),
                _auth_data=FAKE_AUTH_ANTHROPIC["anthropic"],
            ),
        ):
            with patch("httpx.post", return_value=api_response):
                result = call_llm("Hello", model="sonnet")
                assert result == "From Anthropic"

    def test_call_llm_gemini(self, fake_auth_file: Path):
        from shared.llm.base import _provider_cache, call_llm

        _provider_cache.clear()

        api_response = _make_httpx_response(
            json_data={
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "From Gemini"}]}}
                    ],
                    "usageMetadata": {},
                }
            }
        )

        with patch(
            "shared.llm.gemini_provider.GeminiProvider.__init__",
            lambda self, **kw: self.__dict__.update(
                auth_path=Path(fake_auth_file),
                _auth_data=FAKE_AUTH_GOOGLE["google"],
                _auth_key="google",
                _project_id="test-project",
            ),
        ):
            with patch("httpx.post", return_value=api_response):
                result = call_llm("Hello", model="gemini-2.5-flash")
                assert result == "From Gemini"


# ===================================================================
# 5. EDGE CASES
# ===================================================================


class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_backoff_calculation_with_retry_after(self):
        from shared.llm.anthropic_provider import AnthropicProvider

        # Should respect retry-after but cap at MAX_BACKOFF
        assert AnthropicProvider._calculate_backoff(0, 5.0) == 5.0
        assert AnthropicProvider._calculate_backoff(0, 200.0) == 120.0  # capped

    def test_backoff_calculation_exponential(self):
        from shared.llm.anthropic_provider import AnthropicProvider

        b0 = AnthropicProvider._calculate_backoff(0, None)
        b1 = AnthropicProvider._calculate_backoff(1, None)
        b2 = AnthropicProvider._calculate_backoff(2, None)

        # Exponential growth (with some jitter)
        assert b0 < b1 < b2
        # Base values: 1.0, 2.0, 4.0 (plus up to 10% jitter)
        assert 1.0 <= b0 <= 1.1
        assert 2.0 <= b1 <= 2.2
        assert 4.0 <= b2 <= 4.4

    def test_parse_retry_after_header(self):
        from shared.llm.anthropic_provider import AnthropicProvider

        resp = MagicMock()
        resp.headers = {"retry-after": "2.5"}
        assert AnthropicProvider._parse_retry_after(resp) == 2.5

        resp.headers = {}
        assert AnthropicProvider._parse_retry_after(resp) is None

        resp.headers = {"retry-after": "not-a-number"}
        assert AnthropicProvider._parse_retry_after(resp) is None

    def test_gemini_extract_text_multi_part(self):
        from shared.llm.gemini_provider import GeminiProvider

        data = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Part 1"},
                                {"text": "Part 2"},
                                {"thought": True, "text": "Thinking..."},  # skipped
                            ],
                            "role": "model",
                        }
                    }
                ]
            }
        }
        # All text parts joined (thinking parts have text too but we extract all)
        result = GeminiProvider._extract_text(data)
        assert "Part 1" in result
        assert "Part 2" in result

    def test_gemini_extract_text_empty_candidates(self):
        from shared.llm.gemini_provider import GeminiProvider

        assert GeminiProvider._extract_text({"candidates": []}) == ""
        assert GeminiProvider._extract_text({"response": {"candidates": []}}) == ""
        assert GeminiProvider._extract_text({}) == ""

    def test_anthropic_generate_handles_timeout(self, fake_anthropic_auth_file: Path):
        import httpx

        from shared.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(auth_path=str(fake_anthropic_auth_file))

        with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with patch("time.sleep"):
                result = provider.generate("Timeout test", model="sonnet")
                assert result == ""

    def test_gemini_generate_handles_timeout(self, fake_google_auth_file: Path):
        import httpx

        from shared.llm.gemini_provider import GeminiProvider

        provider = GeminiProvider(auth_path=str(fake_google_auth_file))

        with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with patch("time.sleep"):
                result = provider.generate("Timeout test", model="gemini-2.5-flash")
                assert result == ""
