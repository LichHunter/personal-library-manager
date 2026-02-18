"""GLiNER zero-shot entity extraction wrapper.

Provides a clean interface to GLiNER's predict_entities with:
- Lazy singleton model loading (not module-level, for testability)
- Dependency injection via optional model parameter
- Truncation detection (fail loudly if GLiNER truncates input)
- Entity deduplication for overlap handling
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gliner import GLiNER

# Entity labels for software/technology domain extraction
ENTITY_LABELS: list[str] = [
    "library",
    "framework",
    "programming language",
    "software tool",
    "API",
    "database",
    "protocol",
    "technology",
]

# Default confidence threshold for entity extraction
DEFAULT_THRESHOLD = 0.3

# Model identifier
MODEL_NAME = "urchade/gliner_medium-v2.1"


class TruncationError(Exception):
    """Raised when GLiNER truncates input, meaning tokens were lost."""


@dataclass(frozen=True)
class ExtractedEntity:
    """A single entity extracted by GLiNER."""

    text: str
    label: str
    score: float
    start: int
    end: int


# Lazy singleton — avoids loading torch + model at import time.
_model: GLiNER | None = None


def get_model() -> GLiNER:
    """Get or create the singleton GLiNER model instance.

    Loads the model on first call. Subsequent calls return the cached
    instance. This avoids expensive model loading at import time and
    enables tests to run without touching the model.
    """
    global _model
    if _model is None:
        from gliner import GLiNER

        _model = GLiNER.from_pretrained(MODEL_NAME)
    return _model


def reset_model() -> None:
    """Reset the singleton model (for testing)."""
    global _model
    _model = None


def extract_entities(
    text: str,
    labels: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    model: GLiNER | None = None,
) -> list[ExtractedEntity]:
    """Extract named entities from text using GLiNER zero-shot.

    Args:
        text: Input text to extract entities from. Must be pre-chunked
              to stay within GLiNER's token limits.
        labels: Entity type labels to search for. Defaults to ENTITY_LABELS.
        threshold: Minimum confidence score (0-1). Default 0.3.
        model: Optional GLiNER model instance (for dependency injection).
               If None, uses the lazy singleton.

    Returns:
        Deduplicated list of ExtractedEntity, sorted by position.

    Raises:
        TruncationError: If GLiNER emits a truncation warning, meaning
                         input was too long and tokens were silently dropped.
    """
    if not text or not text.strip():
        return []

    if labels is None:
        labels = ENTITY_LABELS

    if model is None:
        model = get_model()

    # Catch GLiNER truncation warnings — these indicate silent token loss.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        raw_entities = model.predict_entities(text, labels, threshold=threshold)

    # Check for truncation warnings
    for w in caught_warnings:
        msg = str(w.message).lower()
        if "truncat" in msg or "max_len" in msg:
            raise TruncationError(
                f"GLiNER truncated input ({len(text)} chars). "
                f"Warning: {w.message}"
            )

    # Convert to our dataclass and deduplicate
    entities: list[ExtractedEntity] = []
    seen: set[tuple[str, int, int]] = set()

    for ent in raw_entities:
        key = (ent["text"], ent["start"], ent["end"])
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            ExtractedEntity(
                text=ent["text"],
                label=ent["label"],
                score=ent["score"],
                start=ent["start"],
                end=ent["end"],
            )
        )

    # Sort by position for consistent output
    entities.sort(key=lambda e: (e.start, e.end))
    return entities
