"""Ollama LLM implementation."""

import logging
from typing import Optional

import ollama

from ..core.protocols import LLM

logger = logging.getLogger(__name__)


class OllamaLLM(LLM):
    """LLM using local Ollama server."""
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: Optional[str] = None,
    ):
        self._model = model
        self._client = ollama.Client(host=base_url) if base_url else ollama.Client()
        logger.info(f"Initialized Ollama LLM: {model}")
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> str:
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return response["response"]
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text."""
        prompt = f"""Summarize the following text in {max_length} words or less. 
Focus on the key facts and main ideas. Be concise and factual.

Text:
{text}

Summary:"""
        
        return self.generate(prompt, max_tokens=max_length * 2, temperature=0.1)
