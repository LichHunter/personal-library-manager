"""Ollama LLM provider for local model inference.

Supports local Ollama server with automatic model downloading.
Default endpoint: http://localhost:11434

Environment variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)

Models tested in POCs (in order of speed/quality tradeoff):
    - llama3.2:1b   (fastest, lower quality)
    - llama3.2:3b   (recommended balance)
    - llama3.1:8b   (better quality, slower)
    - mistral:7b    (alternative architecture)
    - qwen2.5:7b    (strong reasoning)

Example:
    >>> from plm.shared.llm import call_llm
    >>> call_llm("Hello", model="ollama/llama3.2:3b")
    >>> call_llm("Hello", model="llama3.2")  # shorthand
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any

from .base import LLMProvider

logger = logging.getLogger("plm.shared.llm.ollama")

# ---------------------------------------------------------------------------
# Retry configuration (matches other providers)
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 60.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
JITTER_FACTOR = 0.1
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# ---------------------------------------------------------------------------
# Model aliases and defaults
# ---------------------------------------------------------------------------

# Models tested in POCs - map shorthand to full Ollama model tags
MODEL_MAP: dict[str, str] = {
    # LLaMA 3.2 family
    "llama3.2": "llama3.2:3b",
    "llama3.2-1b": "llama3.2:1b",
    "llama3.2-3b": "llama3.2:3b",
    "llama": "llama3.2:3b",
    # LLaMA 3.1 family
    "llama3.1": "llama3.1:8b",
    "llama3.1-8b": "llama3.1:8b",
    "llama3": "llama3.1:8b",
    "llama3-8b": "llama3:8b",
    # Mistral
    "mistral": "mistral:7b",
    "mistral-7b": "mistral:7b",
    # Qwen
    "qwen": "qwen2.5:7b",
    "qwen2.5": "qwen2.5:7b",
    "qwen2.5-7b": "qwen2.5:7b",
    # DeepSeek
    "deepseek": "deepseek-r1:8b",
    "deepseek-r1": "deepseek-r1:8b",
    # Phi
    "phi": "phi4:14b",
    "phi4": "phi4:14b",
    # Gemma (local via Ollama)
    "gemma2": "gemma2:9b",
    "gemma2-9b": "gemma2:9b",
}

# Default model for Ollama (fast, good quality)
DEFAULT_MODEL = "llama3.2:3b"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference.

    Supports automatic model downloading when a model is not found locally.
    Communicates with Ollama server via REST API.

    Features:
        - Auto-download models on first use
        - Retry logic with exponential backoff
        - Connection error handling
        - Model alias resolution
    """

    def __init__(self, host: str | None = None) -> None:
        """Initialize OllamaProvider.

        Args:
            host: Ollama server URL. Defaults to OLLAMA_HOST env var
                  or http://localhost:11434.
        """
        self.host = (
            host
            or os.environ.get("OLLAMA_HOST", "").strip()
            or "http://localhost:11434"
        )
        self.host = self.host.rstrip("/")
        logger.debug("OllamaProvider initialized with host=%s", self.host)

    @property
    def name(self) -> str:
        return "ollama"

    @staticmethod
    def _resolve_model(model: str) -> str:
        """Resolve model alias to full Ollama model tag."""
        # Strip "ollama/" prefix if present
        if model.startswith("ollama/"):
            model = model[7:]

        resolved = MODEL_MAP.get(model, model)
        if resolved != model:
            logger.debug("Model alias: %s -> %s", model, resolved)
        return resolved

    @staticmethod
    def _calculate_backoff(attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, DEFAULT_MAX_BACKOFF)
        backoff = DEFAULT_INITIAL_BACKOFF * (DEFAULT_BACKOFF_MULTIPLIER ** attempt)
        backoff = min(backoff, DEFAULT_MAX_BACKOFF)
        jitter = backoff * JITTER_FACTOR * random.random()
        return backoff + jitter

    def _check_model_exists(self, model: str) -> bool:
        """Check if model is available locally."""
        import httpx

        try:
            response = httpx.get(
                f"{self.host}/api/tags",
                timeout=10,
            )
            if not response.is_success:
                return False

            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check exact match or match without tag
            model_base = model.split(":")[0]
            for name in model_names:
                if name == model or name.startswith(f"{model_base}:"):
                    return True
            return False

        except Exception as e:
            logger.warning("Failed to check model list: %s", e)
            return False

    def _pull_model(self, model: str) -> bool:
        """Download model from Ollama registry.

        Args:
            model: Model name/tag to download.

        Returns:
            True if download succeeded, False otherwise.
        """
        import httpx

        logger.info("[ollama] Model %s not found locally, downloading...", model)

        try:
            # Ollama pull is a streaming endpoint
            with httpx.stream(
                "POST",
                f"{self.host}/api/pull",
                json={"name": model},
                timeout=None,  # Downloads can take a long time
            ) as response:
                if not response.is_success:
                    logger.error(
                        "[ollama] Failed to start download: %d %s",
                        response.status_code,
                        response.text,
                    )
                    return False

                # Stream progress updates
                last_status = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if status != last_status:
                            logger.info("[ollama] Pull %s: %s", model, status)
                            last_status = status
                        if data.get("error"):
                            logger.error("[ollama] Pull error: %s", data["error"])
                            return False
                    except json.JSONDecodeError:
                        continue

            logger.info("[ollama] Successfully downloaded %s", model)
            return True

        except httpx.TimeoutException:
            logger.error("[ollama] Download timed out for %s", model)
            return False
        except Exception as e:
            logger.error("[ollama] Download failed for %s: %s", model, e)
            return False

    def _ensure_model(self, model: str) -> bool:
        """Ensure model is available, downloading if necessary.

        Args:
            model: Model name/tag.

        Returns:
            True if model is available, False otherwise.
        """
        if self._check_model_exists(model):
            return True

        return self._pull_model(model)

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        timeout: int = 90,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
    ) -> str:
        """Generate text using Ollama API.

        Args:
            prompt: User prompt text.
            model: Ollama model name or alias.
            timeout: Request timeout in seconds.
            max_tokens: Maximum output tokens (num_predict in Ollama).
            temperature: Sampling temperature.
            thinking_budget: Ignored (Ollama doesn't support extended thinking).

        Returns:
            Generated text. Empty string on failure.
        """
        import httpx

        resolved_model = self._resolve_model(model)

        logger.debug(
            "[ollama] model=%s prompt_len=%d timeout=%ds",
            resolved_model,
            len(prompt),
            timeout,
        )

        # Ensure model is available (download if needed)
        if not self._ensure_model(resolved_model):
            logger.error("[ollama] Model %s not available", resolved_model)
            return ""

        start_time = time.time()

        request_body: dict[str, Any] = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        url = f"{self.host}/api/generate"

        for attempt in range(DEFAULT_MAX_RETRIES):
            try:
                response = httpx.post(
                    url,
                    json=request_body,
                    timeout=timeout,
                )
                elapsed = time.time() - start_time

                if response.status_code in RETRYABLE_STATUS_CODES:
                    backoff = self._calculate_backoff(attempt, None)
                    logger.info(
                        "[ollama] RETRY %d | attempt=%d/%d | model=%s | "
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
                        "[ollama] FAILED %d | model=%s | elapsed=%.1fs | %s",
                        response.status_code,
                        resolved_model,
                        elapsed,
                        error_text,
                    )
                    return ""

                data = response.json()

                # Extract response text
                text = data.get("response", "").strip()

                # Log metrics if available
                eval_count = data.get("eval_count", 0)
                prompt_eval_count = data.get("prompt_eval_count", 0)
                logger.debug(
                    "[ollama] OK | model=%s | in=%d out=%d | %.1fs",
                    resolved_model,
                    prompt_eval_count,
                    eval_count,
                    elapsed,
                )

                if text:
                    if attempt > 0:
                        logger.info(
                            "[ollama] RECOVERED after %d retries | "
                            "model=%s | total=%.1fs",
                            attempt,
                            resolved_model,
                            elapsed,
                        )
                    return text

                logger.warning(
                    "[ollama] Empty response: %s", json.dumps(data)[:500]
                )
                return ""

            except httpx.TimeoutException:
                elapsed = time.time() - start_time
                if attempt < DEFAULT_MAX_RETRIES - 1:
                    backoff = self._calculate_backoff(attempt, None)
                    logger.info(
                        "[ollama] RETRY timeout | attempt=%d/%d | model=%s | "
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
                    "[ollama] FAILED timeout after %d attempts | model=%s | "
                    "elapsed=%.1fs",
                    DEFAULT_MAX_RETRIES,
                    resolved_model,
                    elapsed,
                )
                return ""

            except httpx.ConnectError as e:
                elapsed = time.time() - start_time
                logger.error(
                    "[ollama] Connection failed | model=%s | error=%s | "
                    "Is Ollama running? Try: ollama serve",
                    resolved_model,
                    e,
                )
                return ""

            except (
                httpx.RemoteProtocolError,
                ConnectionError,
                OSError,
            ) as e:
                elapsed = time.time() - start_time
                if attempt < DEFAULT_MAX_RETRIES - 1:
                    backoff = self._calculate_backoff(attempt, None)
                    logger.info(
                        "[ollama] RETRY connection error | attempt=%d/%d | "
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
                    "[ollama] FAILED connection after %d attempts | model=%s | "
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
                    "[ollama] FAILED unexpected | model=%s | error=%s: %s | "
                    "elapsed=%.1fs",
                    resolved_model,
                    type(e).__name__,
                    e,
                    elapsed,
                )
                return ""

        logger.error(
            "[ollama] EXHAUSTED %d retries | model=%s",
            DEFAULT_MAX_RETRIES,
            resolved_model,
        )
        return ""


def list_models(host: str | None = None) -> list[str]:
    """List available models on the Ollama server.

    Args:
        host: Ollama server URL. Defaults to OLLAMA_HOST or localhost:11434.

    Returns:
        List of model names, or empty list on error.
    """
    import httpx

    host = (
        host
        or os.environ.get("OLLAMA_HOST", "").strip()
        or "http://localhost:11434"
    ).rstrip("/")

    try:
        response = httpx.get(f"{host}/api/tags", timeout=10)
        if response.is_success:
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception as e:
        logger.warning("Failed to list models: %s", e)

    return []
