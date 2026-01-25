"""Disk cache for enrichment results to avoid regeneration."""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from . import EnrichmentResult


class EnrichmentCache:
    """Persists enrichment results to disk."""

    def __init__(self, cache_dir: str = "enrichment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, content: str, enrichment_type: str, model: str) -> str:
        data = f"{content}|{enrichment_type}|{model}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(
        self, content: str, enrichment_type: str, model: str
    ) -> Optional[EnrichmentResult]:
        key = self._make_key(content, enrichment_type, model)
        path = self._cache_path(key)

        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return EnrichmentResult(
                original_content=data["original_content"],
                enhanced_content=data["enhanced_content"],
                enrichment_type=data["enrichment_type"],
                metadata=data.get("metadata", {}),
                keywords=data.get("keywords", []),
                questions=data.get("questions", []),
                summary=data.get("summary", ""),
                entities=data.get("entities", {}),
                contextual_prefix=data.get("contextual_prefix", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def put(
        self, content: str, enrichment_type: str, model: str, result: EnrichmentResult
    ) -> None:
        key = self._make_key(content, enrichment_type, model)
        path = self._cache_path(key)

        data = {
            "original_content": result.original_content,
            "enhanced_content": result.enhanced_content,
            "enrichment_type": result.enrichment_type,
            "metadata": result.metadata,
            "keywords": result.keywords,
            "questions": result.questions,
            "summary": result.summary,
            "entities": result.entities,
            "contextual_prefix": result.contextual_prefix,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def clear(self) -> int:
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count
