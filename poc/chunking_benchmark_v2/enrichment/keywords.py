"""Keyword extraction enricher."""

from typing import Optional
from collections import Counter
import re

from . import Enricher, EnrichmentResult, call_llm

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
MIN_WORDS_FOR_LLM = 30

KEYWORD_PROMPT = """Extract 5-10 important keywords from this text. Return only the keywords, one per line.

Text:
{content}

Keywords:"""


def _calculate_code_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    return code_chars / len(text)


def _word_count_without_code(text: str) -> int:
    text_without_code = CODE_BLOCK_PATTERN.sub(" ", text)
    return len(text_without_code.split())


class KeywordEnricher(Enricher):
    def __init__(self, model: str = "llama3.2:3b", code_ratio_threshold: float = 0.5):
        super().__init__(model)
        self.code_ratio_threshold = code_ratio_threshold

    @property
    def enrichment_type(self) -> str:
        return "keywords"

    def enrich(self, content: str, context: Optional[dict] = None) -> EnrichmentResult:
        if not content.strip():
            return EnrichmentResult(
                original_content=content,
                enhanced_content=content,
                enrichment_type=self.enrichment_type,
                keywords=[],
            )

        code_ratio = _calculate_code_ratio(content)
        word_count = _word_count_without_code(content)
        use_llm = (
            code_ratio < self.code_ratio_threshold and word_count >= MIN_WORDS_FOR_LLM
        )

        if use_llm:
            keywords = self._extract_keywords_llm(content)
        else:
            keywords = []

        if not keywords:
            keywords = self._extract_keywords_tfidf(content)

        keywords = keywords[:10]
        keyword_text = ", ".join(keywords)
        enhanced = f"{keyword_text}\n\n{content}"

        return EnrichmentResult(
            original_content=content,
            enhanced_content=enhanced,
            enrichment_type=self.enrichment_type,
            keywords=keywords,
            metadata={
                "code_ratio": round(code_ratio, 2),
                "used_llm": use_llm,
                "word_count": word_count,
            },
        )

    def _extract_keywords_llm(self, content: str) -> list[str]:
        prompt = KEYWORD_PROMPT.format(content=content[:1500])
        response = call_llm(prompt, self.model)

        keywords = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\d\-\.\*]+\s*", "", line)
            if line and len(line) < 50:
                keywords.append(line)

        return keywords

    def _extract_keywords_tfidf(self, content: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
        stopwords = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "her",
            "was",
            "one",
            "our",
            "out",
            "has",
            "have",
            "been",
            "this",
            "that",
            "with",
            "from",
            "they",
            "will",
            "would",
            "there",
            "their",
            "what",
            "about",
            "which",
            "when",
            "make",
            "like",
            "into",
            "just",
            "over",
            "such",
            "than",
        }
        words = [w for w in words if w not in stopwords]
        counts = Counter(words)
        return [word for word, _ in counts.most_common(10)]
