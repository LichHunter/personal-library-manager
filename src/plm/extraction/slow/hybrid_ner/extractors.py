"""Stage 1: High-recall candidate generation.

Multiple extraction strategies (LLM-based and heuristic) for maximum recall.
"""

import json
import re
import time

from ..retrieval_ner import safe_retrieve, parse_entity_response
from plm.shared.llm import call_llm
from .parsing import _parse_terms_json
from .prompts import EXHAUSTIVE_PROMPT, HAIKU_SIMPLE_PROMPT, RETRIEVAL_PROMPT_TEMPLATE
from .constants import _ALLCAPS_EXCLUDE


def _extract_exhaustive_sonnet(doc: dict) -> tuple[list[str], float]:
    """Exhaustive extraction with Sonnet using taxonomy-driven prompt."""
    prompt = EXHAUSTIVE_PROMPT.format(content=doc["text"][:5000])
    t0 = time.time()
    response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)
    elapsed = time.time() - t0
    return _parse_terms_json(response), elapsed


def _extract_haiku_simple(doc: dict) -> tuple[list[str], float]:
    """Simpler extraction with Haiku for diversity."""
    prompt = HAIKU_SIMPLE_PROMPT.format(content=doc["text"][:5000])
    t0 = time.time()
    response = call_llm(prompt, model="haiku", max_tokens=2000, temperature=0.0)
    elapsed = time.time() - t0
    return _parse_terms_json(response), elapsed


def _extract_retrieval_fixed(
    doc: dict,
    train_docs: list[dict],
    index,  # faiss.Index
    model,  # SentenceTransformer
    k: int = 5,
) -> tuple[list[str], float]:
    """Retrieval-augmented few-shot extraction with FIXED prompt."""
    t0 = time.time()
    similar_docs = safe_retrieve(doc, train_docs, index, model, k)

    # Build examples block
    examples_parts: list[str] = []
    for i, sdoc in enumerate(similar_docs, 1):
        text_preview = sdoc["text"][:800]
        terms = json.dumps(sdoc["gt_terms"], ensure_ascii=False)
        examples_parts.append(
            f"--- Example {i} ---\n"
            f"TEXT: {text_preview}\n"
            f"ENTITIES: {terms}"
        )
    examples_block = "\n\n".join(examples_parts)

    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        examples_block=examples_block,
        text=doc["text"][:5000],
    )
    response = call_llm(prompt, model="sonnet", max_tokens=2000, temperature=0.0)
    elapsed = time.time() - t0

    return parse_entity_response(response), elapsed


def _extract_seeds(doc: dict, seeds: list[str]) -> list[str]:
    """Regex-match auto-seed terms against document text."""
    text_lower = doc["text"].lower()
    found: list[str] = []

    for seed in seeds:
        seed_lower = seed.lower()
        # Word boundary matching
        pattern = rf"\b{re.escape(seed_lower)}\b"
        if re.search(pattern, text_lower):
            # Find the actual case from text
            match = re.search(
                rf"\b{re.escape(seed_lower)}\b", doc["text"], re.IGNORECASE
            )
            if match:
                found.append(match.group())
            else:
                found.append(seed)

    return found


def _extract_heuristic(doc: dict) -> list[str]:
    """Extract software entities using structural/heuristic patterns.

    Finds CamelCase identifiers, dot-separated paths, parenthesized calls,
    ALL_CAPS acronyms, and backtick-wrapped terms. Zero cost, no LLM calls.
    Used as +1 vote source alongside LLM extractors.
    """
    text = doc["text"]
    found: set[str] = set()

    # 1. CamelCase identifiers (e.g., ListView, NSMutableArray, getElementById)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b", text):
        term = m.group(1)
        if len(term) >= 4:
            found.add(term)

    # 2. lowerCamelCase (e.g., getElementById, setChoiceMode)
    for m in re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text):
        term = m.group(1)
        if len(term) >= 4:
            found.add(term)

    # 3. Dot-separated identifiers (e.g., R.layout.file.xml, System.out)
    for m in re.finditer(r"\b([\w]+(?:\.[\w]+){1,5})\b", text):
        term = m.group(1)
        if "." in term and not re.match(r"^\d+\.\d+(\.\d+)*$", term):
            parts = term.split(".")
            if all(len(p) >= 1 for p in parts) and len(term) >= 4:
                found.add(term)

    # 4. Function calls with parentheses (e.g., recv(), querySelector())
    for m in re.finditer(r"\b([\w.]+\(\))", text):
        term = m.group(1)
        if len(term) >= 4:
            found.add(term)

    # 5. ALL_CAPS acronyms (e.g., JSON, HTML, CPU, API)
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]{1,15})\b", text):
        term = m.group(1)
        if len(term) >= 2 and term not in _ALLCAPS_EXCLUDE:
            found.add(term)

    # 6. Backtick-wrapped terms (in case text preserves them)
    for m in re.finditer(r"`([^`]{2,50})`", text):
        term = m.group(1).strip()
        if term and len(term) >= 2:
            found.add(term)

    # 7. CSS class selectors (e.g., .long, .container, .my-class)
    for m in re.finditer(r"(?<!\w)(\.[a-zA-Z][a-zA-Z0-9_-]*)\b", text):
        term = m.group(1)
        # Skip common sentence-ending periods or abbreviations
        if len(term) >= 3:  # ".x" too short, ".long" is fine
            found.add(term)

    # 8. Brace-expansion paths (e.g., /usr/bin/{erb,gem,irb,rdoc,ri,ruby,testrb})
    for m in re.finditer(r"(/[^\s]*\{[^}]+\}[^\s]*)", text):
        term = m.group(1).strip()
        if term:
            found.add(term)

    # 9. Unix-style file paths (e.g., /usr/bin/ruby, /System/Library/Frameworks)
    for m in re.finditer(r"(/(?:[\w.+-]+/)+[\w.+-]*)", text):
        term = m.group(1)
        if len(term) >= 4 and not re.match(r"^/\d+(/\d+)*$", term):  # skip pure numeric paths
            found.add(term)

    return list(found)


def _extract_haiku_fewshot(
    doc: dict,
    train_docs: list[dict],
    index,  # faiss.Index
    model,  # SentenceTransformer
    k: int = 5,
) -> tuple[list[str], float]:
    """Retrieval-augmented few-shot extraction with Haiku (cheaper than Sonnet)."""
    t0 = time.time()
    similar_docs = safe_retrieve(doc, train_docs, index, model, k)

    # Build examples block (same as retrieval_fixed)
    examples_parts: list[str] = []
    for i, sdoc in enumerate(similar_docs, 1):
        text_preview = sdoc["text"][:800]
        terms = json.dumps(sdoc["gt_terms"], ensure_ascii=False)
        examples_parts.append(
            f"--- Example {i} ---\n"
            f"TEXT: {text_preview}\n"
            f"ENTITIES: {terms}"
        )
    examples_block = "\n\n".join(examples_parts)

    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        examples_block=examples_block,
        text=doc["text"][:5000],
    )
    response = call_llm(prompt, model="haiku", max_tokens=2000, temperature=0.0)
    elapsed = time.time() - t0

    return parse_entity_response(response), elapsed


def _extract_haiku_taxonomy(
    doc: dict, llm_model: str = "haiku",
) -> tuple[list[str], float]:
    prompt = EXHAUSTIVE_PROMPT.format(content=doc["text"][:5000])
    t0 = time.time()
    response = call_llm(prompt, model=llm_model, max_tokens=3000, temperature=0.0)
    elapsed = time.time() - t0
    return _parse_terms_json(response), elapsed
