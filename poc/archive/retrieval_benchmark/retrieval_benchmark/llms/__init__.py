"""LLM implementations."""

from typing import Optional

from ..config.schema import LLMConfig
from ..core.protocols import LLM
from .ollama import OllamaLLM


def create_llm(config: LLMConfig) -> LLM:
    """Factory function to create an LLM from config."""
    if config.provider == "ollama":
        return OllamaLLM(
            model=config.model,
            base_url=config.base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")


__all__ = ["create_llm", "OllamaLLM"]
