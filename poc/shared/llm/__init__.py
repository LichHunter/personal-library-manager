"""Shared LLM provider interface for all POCs.

Supports Anthropic (Claude) and Google Gemini via OpenCode OAuth tokens.

Usage:
    from shared.llm import call_llm

    # Anthropic (default)
    response = call_llm("Extract terms from: kubectl get pods", model="sonnet")

    # Gemini
    response = call_llm("Extract terms from: kubectl get pods", model="gemini-2.5-pro")

    # Or import specific providers
    from shared.llm import AnthropicProvider, GeminiProvider
"""

from .base import LLMProvider, call_llm, get_provider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider

__all__ = [
    "LLMProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "call_llm",
    "get_provider",
]
