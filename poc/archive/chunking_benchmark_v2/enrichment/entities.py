"""Entity extraction enricher for technical documentation."""

import re
from typing import Optional

from . import Enricher, EnrichmentResult, call_llm

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
MIN_WORDS_FOR_LLM = 30

ENTITY_PROMPT = """Extract technical entities from this documentation text. List each type on a separate line.

Text:
{content}

API endpoints (paths starting with /):
Functions/methods (function names):
Config keys (UPPERCASE_NAMES):
Technical terms:"""


def _calculate_code_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    return code_chars / len(text)


def _word_count_without_code(text: str) -> int:
    text_without_code = CODE_BLOCK_PATTERN.sub(" ", text)
    return len(text_without_code.split())


class EntityEnricher(Enricher):
    def __init__(self, model: str = "llama3.2:3b", code_ratio_threshold: float = 0.5):
        super().__init__(model)
        self.code_ratio_threshold = code_ratio_threshold

    @property
    def enrichment_type(self) -> str:
        return "entities"

    def enrich(self, content: str, context: Optional[dict] = None) -> EnrichmentResult:
        if not content.strip():
            return EnrichmentResult(
                original_content=content,
                enhanced_content=content,
                enrichment_type=self.enrichment_type,
                entities={},
            )

        code_ratio = _calculate_code_ratio(content)
        word_count = _word_count_without_code(content)
        use_llm = (
            code_ratio < self.code_ratio_threshold and word_count >= MIN_WORDS_FOR_LLM
        )

        if use_llm:
            entities = self._extract_entities_llm(content)
        else:
            entities = {}

        if not any(entities.values()):
            entities = self._extract_entities_regex(content)

        entity_parts = []
        for category, items in entities.items():
            if items:
                entity_parts.append(f"{category}: {', '.join(items[:5])}")

        if entity_parts:
            entity_text = " | ".join(entity_parts)
            enhanced = f"[{entity_text}]\n\n{content}"
        else:
            enhanced = content

        return EnrichmentResult(
            original_content=content,
            enhanced_content=enhanced,
            enrichment_type=self.enrichment_type,
            entities=entities,
            metadata={
                "code_ratio": round(code_ratio, 2),
                "used_llm": use_llm,
                "word_count": word_count,
            },
        )

    def _extract_entities_llm(self, content: str) -> dict:
        prompt = ENTITY_PROMPT.format(content=content[:1500])
        response = call_llm(prompt, self.model)

        entities = {"apis": [], "functions": [], "configs": [], "terms": []}
        current_category = None

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if "api" in lower or "endpoint" in lower:
                current_category = "apis"
            elif "function" in lower or "method" in lower:
                current_category = "functions"
            elif "config" in lower or "key" in lower:
                current_category = "configs"
            elif "term" in lower:
                current_category = "terms"
            elif current_category and not line.endswith(":"):
                items = re.split(r"[,\n]", line)
                for item in items:
                    item = item.strip().strip("-•*").strip()
                    if item and len(item) < 50:
                        entities[current_category].append(item)

        return entities

    def _extract_entities_regex(self, content: str) -> dict:
        entities = {
            "apis": list(set(re.findall(r"/[a-z][a-z0-9_/\-{}]*", content, re.I)))[:5],
            "functions": list(set(re.findall(r"\b[a-z_][a-z0-9_]*\s*\(", content)))[:5],
            "configs": list(set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", content)))[:5],
            "terms": [],
        }
        entities["functions"] = [f.rstrip("( ") for f in entities["functions"]]
        return entities
