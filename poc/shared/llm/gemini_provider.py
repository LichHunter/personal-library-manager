"""Google Gemini LLM provider via OpenCode OAuth (Gemini Code Assist).

Authentication uses the same OpenCode CLI OAuth flow as Anthropic, but
against Google's OAuth 2.0 endpoint.  The provider reads tokens from
``~/.local/share/opencode/auth.json`` under the ``"google"`` key.

The first time you authenticate, OpenCode opens a browser for Google OAuth
consent and stores a refresh token.  This provider automatically refreshes
the access token when it expires.

Gemini Code Assist API:
    - Endpoint: ``https://cloudcode-pa.googleapis.com``
    - Auth: OAuth 2.0 Bearer token (from Google)
    - Request format: ``{project, model, request: <GenerateContent body>}``
    - Streaming: SSE on ``/v1internal:streamGenerateContent?alt=sse``

References:
    https://github.com/jenslys/opencode-gemini-auth
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from .base import LLMProvider

logger = logging.getLogger("shared.llm.gemini")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OAuth client credentials for Gemini Code Assist token refresh.
# These are public *application* credentials (not user secrets), published
# in google-gemini/gemini-cli and jenslys/opencode-gemini-auth.
# Set via environment variables:
#   export GEMINI_CLIENT_ID="..."
#   export GEMINI_CLIENT_SECRET="..."
# See README.md for values.


def _require_env(name: str) -> str:
    """Return env var or raise with helpful message."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set.\n"
            f"These are public OAuth client credentials for Gemini Code Assist.\n"
            f"See: https://github.com/jenslys/opencode-gemini-auth\n"
            f"     https://github.com/google-gemini/gemini-cli"
        )
    return value

GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GEMINI_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

CODE_ASSIST_HEADERS = {
    "User-Agent": "google-api-nodejs-client/9.15.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": (
        "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,"
        "pluginType=GEMINI"
    ),
}

# ---------------------------------------------------------------------------
# Retry configuration (matches AnthropicProvider)
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 8
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 120.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
JITTER_FACTOR = 0.1
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

# Note: The Code Assist API uses short model names directly (NOT the
# preview-dated versions like "gemini-2.5-flash-preview-05-20" which 404,
# and NOT the "models/" prefix which also 404).
# Gemma models are NOT available via Code Assist — only via the standard
# Gemini API (generativelanguage.googleapis.com).
#
# Verified working on Code Assist (Feb 2026):
#   gemini-2.0-flash, gemini-2.5-flash-lite, gemini-2.5-flash,
#   gemini-2.5-pro, gemini-3-flash-preview, gemini-3-pro-preview
#
# NOT working: gemini-2.0-flash-lite, gemini-1.5-*, models/* prefix,
#   preview-dated versions, -latest aliases, gemma-* (all 404)
MODEL_MAP: dict[str, str] = {
    # Convenience aliases → canonical short names
    "gemini-pro": "gemini-2.5-pro",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
}

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _parse_refresh_parts(refresh: str) -> dict[str, str]:
    """Parse pipe-delimited refresh token: ``token|projectId|managedProjectId``."""
    parts = (refresh or "").split("|")
    return {
        "refresh_token": parts[0] if len(parts) > 0 else "",
        "project_id": parts[1] if len(parts) > 1 and parts[1] else "",
        "managed_project_id": parts[2] if len(parts) > 2 and parts[2] else "",
    }


def _format_refresh_parts(
    refresh_token: str,
    project_id: str = "",
    managed_project_id: str = "",
) -> str:
    """Encode refresh parts back to pipe-delimited string."""
    if not project_id and not managed_project_id:
        return refresh_token
    return f"{refresh_token}|{project_id}|{managed_project_id}"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GeminiProvider(LLMProvider):
    """Google Gemini provider via OpenCode OAuth + Gemini Code Assist API.

    Reads OAuth tokens from ``~/.local/share/opencode/auth.json`` under the
    ``"google"`` (or ``"gemini"``) key.  Automatically refreshes expired
    access tokens using Google's OAuth endpoint.

    The provider wraps each request in the Gemini Code Assist envelope
    (``{project, model, request}``) and unwraps the response.
    """

    def __init__(self, auth_path: str | None = None) -> None:
        if auth_path is None:
            auth_path = os.path.expanduser("~/.local/share/opencode/auth.json")
        self.auth_path = Path(auth_path)
        self._auth_data: dict[str, Any] = {}
        self._project_id: str = ""
        self._load_auth()

    # -- auth ---------------------------------------------------------------

    def _load_auth(self) -> None:
        logger.debug("Loading auth from %s", self.auth_path)

        if not self.auth_path.exists():
            raise FileNotFoundError(
                f"OpenCode auth file not found: {self.auth_path}\n"
                "Run 'opencode auth' to authenticate with Google."
            )

        with open(self.auth_path) as f:
            data = json.load(f)

        # OpenCode stores Google creds under "google" key
        auth_key = "google" if "google" in data else "gemini"
        if auth_key not in data:
            raise ValueError(
                "Google/Gemini credentials not found in OpenCode auth file.\n"
                "Run 'opencode auth' and connect to Google."
            )

        self._auth_data = data[auth_key]
        self._auth_key = auth_key

        if self._auth_data.get("type") != "oauth":
            raise ValueError(
                "Google auth is not OAuth type.  This provider requires OAuth tokens.\n"
                "Authenticate via OpenCode with Google account."
            )

        # Extract project_id from refresh token parts
        parts = _parse_refresh_parts(self._auth_data.get("refresh", ""))
        self._project_id = (
            os.environ.get("OPENCODE_GEMINI_PROJECT_ID", "").strip()
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
            or parts["project_id"]
            or parts["managed_project_id"]
        )

        expires_in = (
            (self._auth_data.get("expires", 0) - time.time() * 1000) / 1000 / 60
        )
        logger.debug(
            "Auth loaded (key=%s), token expires in %.1f min, project=%s",
            auth_key,
            expires_in,
            self._project_id or "(none — will resolve on first call)",
        )

    def _save_auth(self) -> None:
        logger.debug("Saving updated auth to %s", self.auth_path)
        with open(self.auth_path) as f:
            data = json.load(f)

        data[self._auth_key] = self._auth_data

        with open(self.auth_path, "w") as f:
            json.dump(data, f, indent=2)

    def _refresh_token_if_needed(self) -> None:
        import httpx

        expires = self._auth_data.get("expires", 0)
        now_ms = time.time() * 1000

        # 60-second buffer (matches opencode-gemini-auth)
        if expires > (now_ms + 60_000):
            return

        expires_ago = (now_ms - expires) / 1000
        logger.info("Token expired %.1fs ago, refreshing...", expires_ago)

        parts = _parse_refresh_parts(self._auth_data.get("refresh", ""))
        if not parts["refresh_token"]:
            raise RuntimeError("No refresh token available for Gemini auth")

        start = time.time()
        response = httpx.post(
            GOOGLE_TOKEN_ENDPOINT,
            data={
                "grant_type": "refresh_token",
                "refresh_token": parts["refresh_token"],
                "client_id": _require_env("GEMINI_CLIENT_ID"),
                "client_secret": _require_env("GEMINI_CLIENT_SECRET"),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        elapsed = time.time() - start

        if not response.is_success:
            logger.error(
                "Token refresh FAILED: %d %s", response.status_code, response.text
            )
            raise RuntimeError(
                f"Gemini token refresh failed: {response.status_code} {response.text}"
            )

        token_data = response.json()

        self._auth_data["access"] = token_data["access_token"]
        # Google may or may not return a new refresh token
        if "refresh_token" in token_data:
            new_refresh = _format_refresh_parts(
                token_data["refresh_token"],
                parts["project_id"],
                parts["managed_project_id"],
            )
            self._auth_data["refresh"] = new_refresh
        self._auth_data["expires"] = int(
            time.time() * 1000 + token_data["expires_in"] * 1000
        )

        self._save_auth()
        logger.info(
            "Token refreshed in %.2fs, valid for %ds",
            elapsed,
            token_data["expires_in"],
        )

    def _resolve_project(self) -> None:
        """Resolve project ID via loadCodeAssist if not already known."""
        import httpx

        if self._project_id:
            return

        logger.info("No project ID cached — calling loadCodeAssist...")
        self._refresh_token_if_needed()

        response = httpx.post(
            f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            json={},
            headers={
                "Authorization": f"Bearer {self._auth_data['access']}",
                "Content-Type": "application/json",
                **CODE_ASSIST_HEADERS,
            },
            timeout=30,
        )

        if not response.is_success:
            raise RuntimeError(
                f"loadCodeAssist failed: {response.status_code} {response.text}\n"
                "Set OPENCODE_GEMINI_PROJECT_ID or GOOGLE_CLOUD_PROJECT env var."
            )

        payload = response.json()
        logger.debug("loadCodeAssist response: %s", json.dumps(payload)[:500])

        # Extract project from response
        project_id = (
            payload.get("projectId")
            or payload.get("managedProjectId")
            or payload.get("project", "")
        )

        if not project_id:
            # Try onboarding to free tier
            logger.info("No project found, attempting free-tier onboard...")
            onboard_resp = httpx.post(
                f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                json={"tierId": "free-tier"},
                headers={
                    "Authorization": f"Bearer {self._auth_data['access']}",
                    "Content-Type": "application/json",
                    **CODE_ASSIST_HEADERS,
                },
                timeout=30,
            )
            if onboard_resp.is_success:
                onboard_data = onboard_resp.json()
                project_id = onboard_data.get("projectId", "")

        if not project_id:
            raise RuntimeError(
                "Could not resolve Gemini project ID.\n"
                "Set OPENCODE_GEMINI_PROJECT_ID or GOOGLE_CLOUD_PROJECT env var."
            )

        self._project_id = project_id

        # Persist project_id in refresh token
        parts = _parse_refresh_parts(self._auth_data.get("refresh", ""))
        self._auth_data["refresh"] = _format_refresh_parts(
            parts["refresh_token"],
            project_id,
            parts["managed_project_id"],
        )
        self._save_auth()
        logger.info("Resolved project ID: %s", project_id)

    # -- model resolution ---------------------------------------------------

    @property
    def name(self) -> str:
        return "gemini"

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

    # -- request/response ---------------------------------------------------

    def _build_request_body(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        thinking_budget: int | None,
    ) -> dict[str, Any]:
        """Build Gemini Code Assist request envelope."""
        generate_content_request: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        # Extended thinking for Gemini 2.5+ models
        if thinking_budget and ("2.5" in model or "2-5" in model):
            generate_content_request["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
            }

        return {
            "project": self._project_id,
            "model": model,
            "request": generate_content_request,
        }

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Extract text from Gemini response (handles Code Assist wrapper)."""
        # Unwrap Code Assist envelope if present
        if "response" in data:
            data = data["response"]

        candidates = data.get("candidates", [])
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            # Skip thinking parts (thought=True)

        return "\n".join(text_parts).strip()

    # -- main generate ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        timeout: int = 90,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
    ) -> str:
        """Generate text using Gemini Code Assist API.

        Args:
            prompt: User prompt text.
            model: Gemini model name or alias.
            timeout: Request timeout in seconds.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            thinking_budget: Extended thinking budget (Gemini 2.5+ only).

        Returns:
            Generated text. Empty string on failure.
        """
        import httpx

        resolved_model = self._resolve_model(model)
        self._resolve_project()

        thinking_label = f" thinking={thinking_budget}" if thinking_budget else ""
        logger.debug(
            "[gemini] model=%s prompt_len=%d timeout=%ds%s",
            resolved_model,
            len(prompt),
            timeout,
            thinking_label,
        )

        start_time = time.time()

        request_body = self._build_request_body(
            prompt, resolved_model, max_tokens, temperature, thinking_budget,
        )

        url = (
            f"{GEMINI_CODE_ASSIST_ENDPOINT}"
            f"/v1internal:generateContent"
        )

        headers = {
            "Authorization": f"Bearer {self._auth_data['access']}",
            "Content-Type": "application/json",
            **CODE_ASSIST_HEADERS,
        }

        for attempt in range(DEFAULT_MAX_RETRIES):
            try:
                self._refresh_token_if_needed()
                headers["Authorization"] = f"Bearer {self._auth_data['access']}"

                response = httpx.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=timeout,
                )
                elapsed = time.time() - start_time

                if response.status_code in RETRYABLE_STATUS_CODES:
                    retry_after = self._parse_retry_after(response)

                    # Check for terminal QUOTA_EXHAUSTED
                    if response.status_code == 429:
                        try:
                            err_data = response.json()
                            details = err_data.get("error", {}).get("details", [])
                            for detail in details:
                                if (
                                    isinstance(detail, dict)
                                    and detail.get("reason") == "QUOTA_EXHAUSTED"
                                ):
                                    logger.error(
                                        "[gemini] QUOTA_EXHAUSTED (terminal) | "
                                        "model=%s | elapsed=%.1fs",
                                        resolved_model,
                                        elapsed,
                                    )
                                    return ""
                        except (json.JSONDecodeError, AttributeError):
                            pass

                    backoff = self._calculate_backoff(attempt, retry_after)
                    logger.info(
                        "[gemini] RETRY %d | attempt=%d/%d | model=%s | "
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
                        "[gemini] FAILED %d | model=%s | elapsed=%.1fs | %s",
                        response.status_code,
                        resolved_model,
                        elapsed,
                        error_text,
                    )
                    return ""

                data = response.json()

                # Extract usage metadata
                usage = data.get("response", data).get("usageMetadata", {})
                input_tokens = usage.get("promptTokenCount", 0)
                output_tokens = usage.get("candidatesTokenCount", 0)
                logger.debug(
                    "[gemini] OK | model=%s | in=%d out=%d | %.1fs",
                    resolved_model,
                    input_tokens,
                    output_tokens,
                    elapsed,
                )

                text = self._extract_text(data)

                if text:
                    if attempt > 0:
                        logger.info(
                            "[gemini] RECOVERED after %d retries | "
                            "model=%s | total=%.1fs",
                            attempt,
                            resolved_model,
                            elapsed,
                        )
                    return text

                logger.warning(
                    "[gemini] Empty response: %s", json.dumps(data)[:500]
                )
                return ""

            except httpx.TimeoutException:
                elapsed = time.time() - start_time
                if attempt < DEFAULT_MAX_RETRIES - 1:
                    backoff = self._calculate_backoff(attempt, None)
                    logger.info(
                        "[gemini] RETRY timeout | attempt=%d/%d | model=%s | "
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
                    "[gemini] FAILED timeout after %d attempts | model=%s | "
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
                        "[gemini] RETRY connection error | attempt=%d/%d | "
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
                    "[gemini] FAILED connection after %d attempts | model=%s | "
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
                    "[gemini] FAILED unexpected | model=%s | error=%s: %s | "
                    "elapsed=%.1fs",
                    resolved_model,
                    type(e).__name__,
                    e,
                    elapsed,
                )
                return ""

        logger.error(
            "[gemini] EXHAUSTED %d retries | model=%s",
            DEFAULT_MAX_RETRIES,
            resolved_model,
        )
        return ""
