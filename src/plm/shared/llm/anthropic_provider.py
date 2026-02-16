"""Anthropic (Claude) LLM provider with API key or OpenCode OAuth support.

Authentication supports two modes:
1. API Key: Set ANTHROPIC_API_KEY environment variable for direct authentication
2. OpenCode OAuth: Tokens read from ~/.local/share/opencode/auth.json under "anthropic" key
   - Customize auth path with OPENCODE_AUTH_PATH environment variable

This is a direct port of the per-POC ``utils/llm_provider.py`` module,
adapted to use stdlib ``logging`` instead of the custom ``BenchmarkLogger``
so that the module has zero internal dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from .base import LLMProvider

logger = logging.getLogger("plm.shared.llm.anthropic")

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 8
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 120.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
JITTER_FACTOR = 0.1
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}

# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    "claude-haiku": "claude-3-5-haiku-latest",
    "claude-haiku-4-5": "claude-3-5-haiku-latest",
    "claude-3-5-haiku": "claude-3-5-haiku-latest",
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-opus": "claude-opus-4-5-20251101",
    "opus": "claude-opus-4-5-20251101",
    "haiku": "claude-3-5-haiku-latest",
    "sonnet": "claude-sonnet-4-5-20250929",
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with API key or OpenCode OAuth support.

    Supports two authentication modes:
    - API Key: Uses ANTHROPIC_API_KEY environment variable
    - OpenCode OAuth: Reads tokens from ~/.local/share/opencode/auth.json
      (customize path with OPENCODE_AUTH_PATH env var)
    
    Automatically refreshes expired OAuth access tokens.
    """

    CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    TOKEN_ENDPOINT = "https://console.anthropic.com/v1/oauth/token"
    API_ENDPOINT = "https://api.anthropic.com/v1/messages"

    def __init__(self, auth_path: str | None = None) -> None:
        # Check for API key authentication first
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            self._use_api_key = True
            self._api_key = api_key
            self._auth_data: dict[str, Any] = {}
            logger.debug("Using ANTHROPIC_API_KEY for authentication")
            return
        
        # Fall back to OpenCode OAuth
        self._use_api_key = False
        if auth_path is None:
            auth_path = os.environ.get("OPENCODE_AUTH_PATH")
            if auth_path is None:
                auth_path = "~/.local/share/opencode/auth.json"
        self.auth_path = Path(os.path.expanduser(auth_path))
        self._auth_data: dict[str, Any] = {}
        self._load_auth()

    # -- auth ---------------------------------------------------------------

    def _load_auth(self) -> None:
        logger.debug("Loading auth from %s", self.auth_path)

        if not self.auth_path.exists():
            raise FileNotFoundError(
                f"OpenCode auth file not found: {self.auth_path}\n"
                "Run 'opencode auth' to authenticate with Anthropic."
            )

        with open(self.auth_path) as f:
            data = json.load(f)

        if "anthropic" not in data:
            raise ValueError(
                "Anthropic credentials not found in OpenCode auth file.\n"
                "Run 'opencode auth' and connect to Anthropic."
            )

        self._auth_data = data["anthropic"]

        if self._auth_data.get("type") != "oauth":
            raise ValueError(
                "Anthropic auth is not OAuth type.  This provider requires OAuth tokens.\n"
                "Authenticate with Claude Pro/Max subscription via OpenCode."
            )

        expires_in = (
            (self._auth_data.get("expires", 0) - time.time() * 1000) / 1000 / 60
        )
        logger.debug("Auth loaded, token expires in %.1f min", expires_in)

    def _save_auth(self) -> None:
        logger.debug("Saving updated auth to %s", self.auth_path)
        with open(self.auth_path) as f:
            data = json.load(f)

        data["anthropic"] = self._auth_data

        with open(self.auth_path, "w") as f:
            json.dump(data, f, indent=2)

    def _refresh_token_if_needed(self) -> None:
        import httpx

        expires = self._auth_data.get("expires", 0)
        now_ms = time.time() * 1000

        if expires > (now_ms + 60_000):
            return

        expires_ago = (now_ms - expires) / 1000
        logger.info("Token expired %.1fs ago, refreshing...", expires_ago)

        start = time.time()
        response = httpx.post(
            self.TOKEN_ENDPOINT,
            json={
                "grant_type": "refresh_token",
                "refresh_token": self._auth_data["refresh"],
                "client_id": self.CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        elapsed = time.time() - start

        if not response.is_success:
            logger.error(
                "Token refresh FAILED: %d %s", response.status_code, response.text
            )
            raise RuntimeError(
                f"Token refresh failed: {response.status_code} {response.text}"
            )

        token_data = response.json()

        self._auth_data["access"] = token_data["access_token"]
        self._auth_data["refresh"] = token_data["refresh_token"]
        self._auth_data["expires"] = int(
            time.time() * 1000 + token_data["expires_in"] * 1000
        )

        self._save_auth()
        logger.info(
            "Token refreshed in %.2fs, valid for %ds",
            elapsed,
            token_data["expires_in"],
        )

    # -- model resolution ---------------------------------------------------

    @property
    def name(self) -> str:
        return "anthropic"

    @staticmethod
    def _resolve_model(model: str) -> str:
        resolved = MODEL_MAP.get(model, model)
        if resolved != model:
            logger.debug("Model alias: %s -> %s", model, resolved)
        return resolved

    # -- retry helpers ------------------------------------------------------

    @staticmethod
    def _calculate_backoff(attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, DEFAULT_MAX_BACKOFF)
        backoff = DEFAULT_INITIAL_BACKOFF * (DEFAULT_BACKOFF_MULTIPLIER ** attempt)
        backoff = min(backoff, DEFAULT_MAX_BACKOFF)
        jitter = backoff * JITTER_FACTOR * random.random()
        return backoff + jitter

    @staticmethod
    def _parse_retry_after(response: Any) -> float | None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return None

    # -- main generate ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str = "claude-haiku-4-5",
        timeout: int = 90,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
    ) -> str:
        """Generate text using Anthropic Messages API.

        Args:
            prompt: User prompt text.
            model: Claude model name or alias.
            timeout: Request timeout in seconds.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            thinking_budget: Extended thinking budget (optional).

        Returns:
            Generated text. Empty string on failure.
        """
        import httpx

        resolved_model = self._resolve_model(model)

        thinking_label = f" thinking={thinking_budget}" if thinking_budget else ""
        logger.debug(
            "[anthropic] model=%s prompt_len=%d timeout=%ds%s",
            resolved_model,
            len(prompt),
            timeout,
            thinking_label,
        )

        start_time = time.time()

        request_body: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                }
            ],
            "messages": [{"role": "user", "content": prompt}],
        }

        if thinking_budget:
            request_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            request_body["max_tokens"] = max(max_tokens, thinking_budget + max_tokens)
            # temperature must NOT be set when thinking is enabled
        else:
            request_body["temperature"] = temperature

        headers = {
            "anthropic-beta": (
                "claude-code-20250219,oauth-2025-04-20,"
                "interleaved-thinking-2025-05-14,"
                "fine-grained-tool-streaming-2025-05-14"
            ),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        if self._use_api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            headers["Authorization"] = f"Bearer {self._auth_data['access']}"

        for attempt in range(DEFAULT_MAX_RETRIES):
            try:
                if not self._use_api_key:
                    self._refresh_token_if_needed()
                    headers["Authorization"] = f"Bearer {self._auth_data['access']}"

                response = httpx.post(
                    self.API_ENDPOINT,
                    json=request_body,
                    headers=headers,
                    timeout=timeout,
                )
                elapsed = time.time() - start_time

                if response.status_code in RETRYABLE_STATUS_CODES:
                    retry_after = self._parse_retry_after(response)
                    backoff = self._calculate_backoff(attempt, retry_after)
                    logger.info(
                        "[anthropic] RETRY %d | attempt=%d/%d | model=%s | "
                        "wait=%.1fs | elapsed=%.1fs",
                        response.status_code,
                        attempt + 1,
                        DEFAULT_MAX_RETRIES,
                        resolved_model,
                        backoff,
                        elapsed,
                    )
                    time.sleep(backoff)
                    continue

                if not response.is_success:
                    error_text = response.text[:300]
                    logger.error(
                        "[anthropic] FAILED %d | model=%s | elapsed=%.1fs | %s",
                        response.status_code,
                        resolved_model,
                        elapsed,
                        error_text,
                    )
                    return ""

                data = response.json()

                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                logger.debug(
                    "[anthropic] OK | model=%s | in=%d out=%d | %.1fs",
                    resolved_model,
                    input_tokens,
                    output_tokens,
                    elapsed,
                )

                content = data.get("content", [])
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if block.get("type") == "text"
                ]

                if text_parts:
                    text = "\n".join(text_parts).strip()
                    if attempt > 0:
                        logger.info(
                            "[anthropic] RECOVERED after %d retries | "
                            "model=%s | total=%.1fs",
                            attempt,
                            resolved_model,
                            elapsed,
                        )
                    return text

                logger.warning(
                    "[anthropic] Unexpected response: %s", json.dumps(data)[:500]
                )
                return ""

            except httpx.TimeoutException:
                elapsed = time.time() - start_time
                if attempt < DEFAULT_MAX_RETRIES - 1:
                    backoff = self._calculate_backoff(attempt, None)
                    logger.info(
                        "[anthropic] RETRY timeout | attempt=%d/%d | model=%s | "
                        "wait=%.1fs | elapsed=%.1fs",
                        attempt + 1,
                        DEFAULT_MAX_RETRIES,
                        resolved_model,
                        backoff,
                        elapsed,
                    )
                    time.sleep(backoff)
                    continue
                logger.error(
                    "[anthropic] FAILED timeout after %d attempts | model=%s | "
                    "elapsed=%.1fs",
                    DEFAULT_MAX_RETRIES,
                    resolved_model,
                    elapsed,
                )
                return ""

            except (
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                ConnectionError,
                OSError,
            ) as e:
                elapsed = time.time() - start_time
                if attempt < DEFAULT_MAX_RETRIES - 1:
                    backoff = self._calculate_backoff(attempt, None)
                    logger.info(
                        "[anthropic] RETRY connection error | attempt=%d/%d | "
                        "model=%s | error=%s: %s | wait=%.1fs",
                        attempt + 1,
                        DEFAULT_MAX_RETRIES,
                        resolved_model,
                        type(e).__name__,
                        e,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                logger.error(
                    "[anthropic] FAILED connection after %d attempts | model=%s | "
                    "error=%s: %s",
                    DEFAULT_MAX_RETRIES,
                    resolved_model,
                    type(e).__name__,
                    e,
                )
                return ""

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    "[anthropic] FAILED unexpected | model=%s | error=%s: %s | "
                    "elapsed=%.1fs",
                    resolved_model,
                    type(e).__name__,
                    e,
                    elapsed,
                )
                return ""

        logger.error(
            "[anthropic] EXHAUSTED %d retries | model=%s",
            DEFAULT_MAX_RETRIES,
            resolved_model,
        )
        return ""
