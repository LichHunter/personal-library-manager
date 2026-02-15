"""Base LLM provider interface and convenience functions.

This module defines the abstract LLMProvider interface and the call_llm()
convenience function that routes to the correct provider based on model name.
"""

import logging
import re
from abc import ABC, abstractmethod

logger = logging.getLogger("shared.llm")

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for LLM providers (Anthropic, Gemini, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'anthropic', 'gemini')."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        timeout: int = 90,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: User prompt text.
            model: Model name or alias (e.g. 'sonnet', 'gemini-2.5-pro').
            timeout: Request timeout in seconds.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature (0.0–1.0).
            thinking_budget: Optional extended-thinking token budget
                (Anthropic-specific; ignored by providers that don't support it).

        Returns:
            Generated text. Empty string on failure (never raises).
        """
        ...


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_provider_cache: dict[str, LLMProvider] = {}

# Patterns that identify a Gemini model name
_GEMINI_PATTERN = re.compile(
    r"^(gemini|gemma)", re.IGNORECASE
)


def _is_gemini_model(model: str) -> bool:
    """Return True if *model* should be routed to the Gemini provider."""
    return bool(_GEMINI_PATTERN.match(model))


def get_provider(model: str) -> LLMProvider:
    """Return (cached) provider for *model*.

    Routing logic:
        - Model names starting with ``gemini`` or ``gemma`` → GeminiProvider
        - Everything else → AnthropicProvider
    """
    if _is_gemini_model(model):
        key = "gemini"
        if key not in _provider_cache:
            from .gemini_provider import GeminiProvider
            logger.debug("Creating GeminiProvider for model=%s", model)
            _provider_cache[key] = GeminiProvider()
        return _provider_cache[key]

    key = "anthropic"
    if key not in _provider_cache:
        from .anthropic_provider import AnthropicProvider
        logger.debug("Creating AnthropicProvider for model=%s", model)
        _provider_cache[key] = AnthropicProvider()
    return _provider_cache[key]


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def call_llm(
    prompt: str,
    model: str = "claude-haiku-4-5",
    timeout: int = 90,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    thinking_budget: int | None = None,
) -> str:
    """Call an LLM with automatic provider routing.

    Routes to Anthropic or Gemini based on *model* name.  Signature is
    intentionally identical to the per-POC ``call_llm`` so it can be used
    as a drop-in replacement.

    Examples::

        # Anthropic (default)
        call_llm("Hello", model="sonnet")

        # Gemini
        call_llm("Hello", model="gemini-2.5-pro")
    """
    provider = get_provider(model)
    return provider.generate(
        prompt,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_budget=thinking_budget,
    )
