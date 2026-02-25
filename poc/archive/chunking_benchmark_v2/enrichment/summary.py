"""Summary enricher."""

import re
from typing import Optional

from . import Enricher, EnrichmentResult, call_llm

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")

SUMMARY_PROMPT = """Summarize the following text in 1-2 sentences (max 50 words). Focus on key facts.

Text:
{content}

Summary:"""


def _calculate_code_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    return code_chars / len(text)


def _word_count_without_code(text: str) -> int:
    text_without_code = CODE_BLOCK_PATTERN.sub(" ", text)
    return len(text_without_code.split())


class SummaryEnricher(Enricher):
    def __init__(
        self,
        model: str = "llama3.2:3b",
        min_words_to_summarize: int = 30,
        code_ratio_threshold: float = 0.5,
    ):
        super().__init__(model)
        self.min_words_to_summarize = min_words_to_summarize
        self.code_ratio_threshold = code_ratio_threshold

    @property
    def enrichment_type(self) -> str:
        return "summary"

    def enrich(self, content: str, context: Optional[dict] = None) -> EnrichmentResult:
        if not content.strip():
            return EnrichmentResult(
                original_content=content,
                enhanced_content=content,
                enrichment_type=self.enrichment_type,
                summary="",
            )

        code_ratio = _calculate_code_ratio(content)
        word_count = _word_count_without_code(content)
        use_llm = (
            code_ratio < self.code_ratio_threshold
            and word_count >= self.min_words_to_summarize
        )

        if not use_llm:
            return EnrichmentResult(
                original_content=content,
                enhanced_content=content,
                enrichment_type=self.enrichment_type,
                summary="",
                metadata={
                    "code_ratio": round(code_ratio, 2),
                    "used_llm": False,
                    "word_count": word_count,
                },
            )

        summary = self._generate_summary(content)

        if summary:
            enhanced = f"Summary: {summary}\n\n{content}"
        else:
            enhanced = content

        return EnrichmentResult(
            original_content=content,
            enhanced_content=enhanced,
            enrichment_type=self.enrichment_type,
            summary=summary,
            metadata={
                "code_ratio": round(code_ratio, 2),
                "used_llm": True,
                "word_count": word_count,
            },
        )

    def _generate_summary(self, content: str) -> str:
        prompt = SUMMARY_PROMPT.format(content=content[:2000])
        response = call_llm(prompt, self.model, timeout=60)

        summary = response.strip()
        words = summary.split()
        if len(words) > 50:
            summary = " ".join(words[:50]) + "..."

        return summary
