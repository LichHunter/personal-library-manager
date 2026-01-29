"""LLM Provider interface for enrichment."""

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .logger import get_logger


class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str, model: str, timeout: int = 90) -> str:
        pass


class AnthropicProvider(LLMProvider):
    CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    TOKEN_ENDPOINT = "https://console.anthropic.com/v1/oauth/token"
    API_ENDPOINT = "https://api.anthropic.com/v1/messages"

    MODEL_MAP = {
        "claude-haiku": "claude-3-5-haiku-latest",
        "claude-haiku-4-5": "claude-3-5-haiku-latest",
        "claude-3-5-haiku": "claude-3-5-haiku-latest",
        "claude-sonnet": "claude-sonnet-4-20250514",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "haiku": "claude-3-5-haiku-latest",
        "sonnet": "claude-sonnet-4-20250514",
    }

    def __init__(self, auth_path: Optional[str] = None):
        if auth_path is None:
            auth_path = os.path.expanduser("~/.local/share/opencode/auth.json")
        self.auth_path = Path(auth_path)
        self._auth_data: dict = {}
        self._load_auth()

    def _load_auth(self) -> None:
        log = get_logger()
        log.debug(f"[anthropic] Loading auth from {self.auth_path}")

        if not self.auth_path.exists():
            raise FileNotFoundError(
                f"OpenCode auth file not found: {self.auth_path}\n"
                "Please run 'opencode auth' to authenticate with Anthropic."
            )

        with open(self.auth_path) as f:
            data = json.load(f)

        if "anthropic" not in data:
            raise ValueError(
                "Anthropic credentials not found in OpenCode auth file.\n"
                "Please run 'opencode auth' and connect to Anthropic."
            )

        self._auth_data = data["anthropic"]

        if self._auth_data.get("type") != "oauth":
            raise ValueError(
                "Anthropic auth is not OAuth type. This provider requires OAuth tokens.\n"
                "Please authenticate with Claude Pro/Max subscription via OpenCode."
            )

        expires_in = (
            (self._auth_data.get("expires", 0) - time.time() * 1000) / 1000 / 60
        )
        log.debug(f"[anthropic] Auth loaded, token expires in {expires_in:.1f} minutes")

    def _save_auth(self) -> None:
        log = get_logger()
        log.debug(f"[anthropic] Saving updated auth to {self.auth_path}")

        with open(self.auth_path) as f:
            data = json.load(f)

        data["anthropic"] = self._auth_data

        with open(self.auth_path, "w") as f:
            json.dump(data, f, indent=2)

    def _refresh_token_if_needed(self) -> None:
        import httpx

        log = get_logger()

        expires = self._auth_data.get("expires", 0)
        now_ms = time.time() * 1000

        if expires > (now_ms + 60000):
            return

        expires_ago = (now_ms - expires) / 1000
        log.info(f"[anthropic] Token expired {expires_ago:.1f}s ago, refreshing...")

        start_time = time.time()
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
        elapsed = time.time() - start_time

        if not response.is_success:
            log.error(
                f"[anthropic] Token refresh FAILED: {response.status_code} {response.text}"
            )
            raise RuntimeError(
                f"Token refresh failed: {response.status_code} {response.text}"
            )

        token_data = response.json()

        self._auth_data["access"] = token_data["access_token"]
        self._auth_data["refresh"] = token_data["refresh_token"]
        self._auth_data["expires"] = (
            int(time.time() * 1000) + token_data["expires_in"] * 1000
        )

        self._save_auth()
        log.info(
            f"[anthropic] Token refreshed in {elapsed:.2f}s, valid for {token_data['expires_in']}s"
        )

    @property
    def name(self) -> str:
        return "anthropic"

    def _resolve_model(self, model: str) -> str:
        resolved = self.MODEL_MAP.get(model, model)
        if resolved != model:
            log = get_logger()
            log.debug(f"[anthropic] Model alias: {model} -> {resolved}")
        return resolved

    def generate(
        self, prompt: str, model: str = "claude-haiku-4-5", timeout: int = 90
    ) -> str:
        import httpx

        log = get_logger()

        resolved_model = self._resolve_model(model)

        log.debug(
            f"[anthropic] model={resolved_model} prompt_len={len(prompt)} timeout={timeout}s"
        )
        log.debug(
            f"[anthropic] PROMPT: {prompt[:500]}{'...' if len(prompt) > 500 else ''}"
        )

        start_time = time.time()

        try:
            self._refresh_token_if_needed()

            request_body = {
                "model": resolved_model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }

            headers = {
                "Authorization": f"Bearer {self._auth_data['access']}",
                "anthropic-beta": "oauth-2025-04-20",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": "claude-cli/2.1.2 (external, cli)",
            }

            log.debug(f"[anthropic] REQUEST: POST {self.API_ENDPOINT}")
            log.debug(
                f"[anthropic] REQUEST body: model={resolved_model} max_tokens=1024"
            )

            response = httpx.post(
                self.API_ENDPOINT,
                json=request_body,
                headers=headers,
                timeout=timeout,
            )
            elapsed = time.time() - start_time

            log.debug(
                f"[anthropic] RESPONSE status={response.status_code} elapsed={elapsed:.2f}s"
            )

            if not response.is_success:
                error_text = response.text[:500]
                log.error(f"[anthropic] API ERROR: {response.status_code} {error_text}")
                return ""

            data = response.json()

            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            log.debug(
                f"[anthropic] USAGE: input_tokens={input_tokens} output_tokens={output_tokens}"
            )

            content = data.get("content", [])
            stop_reason = data.get("stop_reason", "unknown")

            if content and content[0].get("type") == "text":
                text = content[0].get("text", "").strip()
                log.debug(
                    f"[anthropic] SUCCESS stop_reason={stop_reason} response_len={len(text)}"
                )
                log.debug(
                    f"[anthropic] RESPONSE: {text[:500]}{'...' if len(text) > 500 else ''}"
                )
                return text

            log.warn(
                f"[anthropic] Unexpected response format: {json.dumps(data)[:500]}"
            )
            return ""

        except Exception as e:
            elapsed = time.time() - start_time
            log.error(f"[anthropic] ERROR after {elapsed:.2f}s: {e}")
            return ""


_provider_cache: dict[str, LLMProvider] = {}


def get_provider(model: str) -> LLMProvider:
    """Get LLM provider for the given model (Anthropic only)."""
    log = get_logger()

    # Always use Anthropic provider
    provider_type = "anthropic"

    if provider_type not in _provider_cache:
        log.debug(f"[provider] Creating {provider_type} provider for model={model}")
        _provider_cache[provider_type] = AnthropicProvider()

    return _provider_cache[provider_type]


def call_llm(prompt: str, model: str = "claude-haiku-4-5", timeout: int = 90) -> str:
    provider = get_provider(model)
    return provider.generate(prompt, model, timeout)
