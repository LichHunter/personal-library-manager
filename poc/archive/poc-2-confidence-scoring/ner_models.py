#!/usr/bin/env python3
"""
NER model adapters for the evaluation framework.

Each adapter wraps a different NER model behind the same interface:
  .name -> str
  .extract(text) -> list[ExtractedEntity]

All models extract technical entities and return confidence scores.
"""

from __future__ import annotations

import re
from eval_framework import ExtractedEntity


class HeuristicNER:
    """Baseline: regex-based fast extraction (same as src/plm/extraction/fast/heuristic.py)."""

    CAMEL_CASE = re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b')
    ALL_CAPS = re.compile(r'\b[A-Z][A-Z_]{2,}\b')
    DOT_PATH = re.compile(r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b')
    BACKTICK = re.compile(r'`([^`]+)`')
    FUNCTION_CALL = re.compile(r'\b[a-zA-Z_]\w*\(\)')

    @property
    def name(self) -> str:
        return "heuristic-regex"

    def extract(self, text: str) -> list[ExtractedEntity]:
        candidates: set[str] = set()
        for pattern in [self.CAMEL_CASE, self.ALL_CAPS, self.DOT_PATH, self.FUNCTION_CALL]:
            for m in pattern.finditer(text):
                candidates.add(m.group())
        for m in self.BACKTICK.finditer(text):
            candidates.add(m.group(1))
        return [ExtractedEntity(text=t, confidence=1.0, label="TECH") for t in candidates]


class SpacyTransformerNER:
    """spaCy en_core_web_trf — general NER with transformer backbone."""

    def __init__(self, model_name: str = "en_core_web_trf"):
        import spacy
        self._nlp = spacy.load(model_name)
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"spacy-{self._model_name}"

    def extract(self, text: str) -> list[ExtractedEntity]:
        doc = self._nlp(text)
        seen: set[str] = set()
        entities = []
        for ent in doc.ents:
            if ent.text in seen:
                continue
            seen.add(ent.text)
            # spaCy doesn't expose per-entity confidence natively;
            # use the max token probability from the underlying model as proxy
            conf = self._get_confidence(doc, ent)
            entities.append(ExtractedEntity(text=ent.text, confidence=conf, label=ent.label_))
        return entities

    def _get_confidence(self, doc, ent) -> float:
        """Extract confidence from spaCy's internal scores if available."""
        try:
            scores = doc.cats
            if scores:
                return max(scores.values())
        except (AttributeError, ValueError):
            pass
        # Fallback: spaCy doesn't provide per-entity scores
        # Use 0.5 as a neutral confidence (we document this limitation)
        return 0.5


class FlairNER:
    """Flair NER — has native per-entity confidence scores."""

    def __init__(self, model_name: str = "flair/ner-english"):
        from flair.nn import Classifier
        self._tagger = Classifier.load(model_name)
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"flair-{self._model_name.replace('/', '-')}"

    def extract(self, text: str) -> list[ExtractedEntity]:
        from flair.data import Sentence
        sentence = Sentence(text)
        self._tagger.predict(sentence)
        seen: set[str] = set()
        entities = []
        for span in sentence.get_spans("ner"):
            if span.text in seen:
                continue
            seen.add(span.text)
            entities.append(
                ExtractedEntity(text=span.text, confidence=span.score, label=span.tag)
            )
        return entities


class TransformersNER:
    """HuggingFace transformers pipeline NER — works for BERT, DeBERTa, etc."""

    def __init__(self, model_id: str, label: str | None = None):
        from transformers import pipeline
        self._pipe = pipeline(
            "ner",
            model=model_id,
            aggregation_strategy="simple",
            device="cpu",
        )
        self._label = label or model_id.split("/")[-1]

    @property
    def name(self) -> str:
        return self._label

    def extract(self, text: str) -> list[ExtractedEntity]:
        # transformers pipeline may fail on very long texts; truncate
        truncated = text[:512] if len(text) > 512 else text
        raw = self._pipe(truncated)
        seen: set[str] = set()
        entities = []
        for ent in raw:
            word = ent["word"].strip()
            if not word or word in seen:
                continue
            seen.add(word)
            entities.append(
                ExtractedEntity(
                    text=word,
                    confidence=float(ent["score"]),
                    label=ent["entity_group"],
                )
            )
        return entities


class SpanMarkerNER:
    """SpanMarker — span-based NER with native confidence scores."""

    def __init__(self, model_id: str = "tomaarsen/span-marker-roberta-large-ontonotes5"):
        from span_marker import SpanMarkerModel
        self._model = SpanMarkerModel.from_pretrained(model_id)
        self._model_id = model_id

    @property
    def name(self) -> str:
        return f"span-marker-{self._model_id.split('/')[-1]}"

    def extract(self, text: str) -> list[ExtractedEntity]:
        raw = self._model.predict(text[:512] if len(text) > 512 else text)
        seen: set[str] = set()
        entities = []
        for ent in raw:
            span = ent["span"]
            if span in seen:
                continue
            seen.add(span)
            entities.append(
                ExtractedEntity(
                    text=span,
                    confidence=float(ent["score"]),
                    label=ent["label"],
                )
            )
        return entities


class GLiNERModel:
    """GLiNER — zero-shot NER with custom entity labels."""

    def __init__(
        self,
        model_id: str = "urchade/gliner_medium-v2.1",
        labels: list[str] | None = None,
    ):
        from gliner import GLiNER
        self._model = GLiNER.from_pretrained(model_id)
        self._labels = labels or [
            "library", "framework", "programming language",
            "software tool", "data structure", "file format",
            "protocol", "API", "class", "function",
            "operating system", "database", "technology",
        ]
        self._model_id = model_id

    @property
    def name(self) -> str:
        return f"gliner-{self._model_id.split('/')[-1]}"

    def extract(self, text: str) -> list[ExtractedEntity]:
        truncated = text[:512] if len(text) > 512 else text
        raw = self._model.predict_entities(truncated, self._labels, threshold=0.3)
        seen: set[str] = set()
        entities = []
        for ent in raw:
            span = ent["text"]
            if span in seen:
                continue
            seen.add(span)
            entities.append(
                ExtractedEntity(
                    text=span,
                    confidence=float(ent["score"]),
                    label=ent["label"],
                )
            )
        return entities


class NuNERZero:
    """NuNER Zero — zero-shot NER based on numind/NuNER_Zero."""

    def __init__(self, model_id: str = "numind/NuNER_Zero"):
        from gliner import GLiNER
        self._model = GLiNER.from_pretrained(model_id)
        self._labels = [
            "library", "framework", "programming language",
            "software", "data structure", "file type",
            "protocol", "class name", "function name",
            "operating system", "database", "technology",
        ]

    @property
    def name(self) -> str:
        return "nuner-zero"

    def extract(self, text: str) -> list[ExtractedEntity]:
        truncated = text[:512] if len(text) > 512 else text
        raw = self._model.predict_entities(truncated, self._labels, threshold=0.3)
        seen: set[str] = set()
        entities = []
        for ent in raw:
            span = ent["text"]
            if span in seen:
                continue
            seen.add(span)
            entities.append(
                ExtractedEntity(
                    text=span,
                    confidence=float(ent["score"]),
                    label=ent["label"],
                )
            )
        return entities


def get_all_models() -> dict[str, callable]:
    """Registry of all available model constructors.

    Returns dict of name -> factory function.
    Call factory() to instantiate (downloads model on first use).
    """
    return {
        "heuristic": lambda: HeuristicNER(),
        "spacy-trf": lambda: SpacyTransformerNER("en_core_web_trf"),
        "flair": lambda: FlairNER("flair/ner-english"),
        "flair-large": lambda: FlairNER("flair/ner-english-large"),
        "bert-base-ner": lambda: TransformersNER("dslim/bert-base-NER", "bert-base-NER"),
        "bert-large-ner": lambda: TransformersNER("dslim/bert-large-NER", "bert-large-NER"),
        "deberta-v3-ner": lambda: TransformersNER(
            "chandc/deberta-v3-large-finetuned-ner", "deberta-v3-large-NER"
        ),
        "span-marker": lambda: SpanMarkerNER(
            "tomaarsen/span-marker-roberta-large-ontonotes5"
        ),
        "gliner": lambda: GLiNERModel("urchade/gliner_medium-v2.1"),
        "nuner-zero": lambda: NuNERZero(),
    }
