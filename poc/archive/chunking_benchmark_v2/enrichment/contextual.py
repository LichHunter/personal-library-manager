"""Contextual prefix enricher following Anthropic's pattern."""

import re
from typing import Optional

from . import Enricher, EnrichmentResult, call_llm

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
MIN_WORDS_FOR_LLM = 30

CONTEXT_PROMPT = """Given this document context and chunk, write a brief 1-2 sentence prefix that situates the chunk within the document.

Document: {doc_title}
Section: {section}

Chunk:
{content}

Write a prefix like "This chunk is from the [section] section of [document], discussing [main topic]...":"""


def _calculate_code_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    return code_chars / len(text)


def _word_count_without_code(text: str) -> int:
    text_without_code = CODE_BLOCK_PATTERN.sub(" ", text)
    return len(text_without_code.split())


class ContextualEnricher(Enricher):
    def __init__(self, model: str = "llama3.2:3b", code_ratio_threshold: float = 0.5):
        super().__init__(model)
        self.code_ratio_threshold = code_ratio_threshold

    @property
    def enrichment_type(self) -> str:
        return "contextual"

    def enrich(self, content: str, context: Optional[dict] = None) -> EnrichmentResult:
        if not content.strip():
            return EnrichmentResult(
                original_content=content,
                enhanced_content=content,
                enrichment_type=self.enrichment_type,
            )

        context = context or {}
        doc_title = context.get("doc_title", "Unknown Document")
        section = context.get("section", "Unknown Section")

        code_ratio = _calculate_code_ratio(content)
        word_count = _word_count_without_code(content)
        use_llm = (
            code_ratio < self.code_ratio_threshold and word_count >= MIN_WORDS_FOR_LLM
        )

        if use_llm:
            prefix = self._generate_prefix(content, doc_title, section)
        else:
            prefix = ""

        if prefix:
            enhanced = f"{prefix}\n\n{content}"
        else:
            enhanced = f"From {section} in {doc_title}:\n\n{content}"

        return EnrichmentResult(
            original_content=content,
            enhanced_content=enhanced,
            enrichment_type=self.enrichment_type,
            contextual_prefix=prefix or f"From {section} in {doc_title}",
            metadata={
                "code_ratio": round(code_ratio, 2),
                "used_llm": use_llm,
                "word_count": word_count,
            },
        )

    def _generate_prefix(self, content: str, doc_title: str, section: str) -> str:
        prompt = CONTEXT_PROMPT.format(
            doc_title=doc_title, section=section, content=content[:1000]
        )
        response = call_llm(prompt, self.model, timeout=60)

        prefix = response.strip()
        if len(prefix) > 200:
            prefix = prefix[:200].rsplit(" ", 1)[0] + "..."

        return prefix
