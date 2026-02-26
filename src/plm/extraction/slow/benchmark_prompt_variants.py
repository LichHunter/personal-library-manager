#!/usr/bin/env python3
"""Candidate-verify extraction prompt variant.

Implements the candidate-verify strategy: generate candidates heuristically,
then ask LLM to classify each. Prevents forgetting entities when they're
explicitly listed as candidates.

Used by: hybrid_ner/pipeline.py (extract_candidate_verify stage)
"""

import json
import re as _re
import time

from .hybrid_ner import _parse_terms_json
from plm.shared.llm import call_llm


# ============================================================================
# PROMPT VARIANTS REGISTRY
# ============================================================================

PROMPT_VARIANTS: dict[str, dict] = {}


def register_variant(
    name: str,
    prompt_template: str,
    model: str = "haiku",
    needs_retrieval: bool = False,
    description: str = "",
    thinking_budget: int | None = None,
    custom_runner: "callable | None" = None,
) -> None:
    PROMPT_VARIANTS[name] = {
        "prompt_template": prompt_template,
        "model": model,
        "needs_retrieval": needs_retrieval,
        "description": description,
        "thinking_budget": thinking_budget,
        "custom_runner": custom_runner,
    }

# ---------------------------------------------------------------------------
# candidate_verify: Enhanced candidate generation + aggressive classifier
# Adds bigrams, hyphenated compounds, bare numbers, all-lowercase tokens
# ---------------------------------------------------------------------------


def _candidate_verify_runner(doc, train_docs, retrieval_index, retrieval_model):
    """Enhanced candidate generation with bigrams and aggressive classification."""
    text = doc["text"][:5000]
    candidates: set[str] = set()

    words = text.split()
    for w in words:
        clean = w.strip(".,;:!?()[]{}\"'")
        if clean and len(clean) >= 1:
            candidates.add(clean)

    for i in range(len(words) - 1):
        w1 = words[i].strip(".,;:!?()[]{}\"'")
        w2 = words[i + 1].strip(".,;:!?()[]{}\"'")
        if w1 and w2:
            candidates.add(f"{w1} {w2}")

    for m in _re.finditer(r"\b(\w+(?:-\w+)+)\b", text):
        candidates.add(m.group(1))

    for m in _re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([a-z]{3,30})\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([\w]+(?:\.[\w]+){1,5})\b", text):
        term = m.group(1)
        if "." in term:
            candidates.add(term)
    for m in _re.finditer(r"\b([\w.]+\(\))", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([A-Z][A-Z0-9_]{1,15})\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"`([^`]{2,50})`", text):
        candidates.add(m.group(1).strip())
    for m in _re.finditer(r"(?<!\w)(\.[a-zA-Z][a-zA-Z0-9_-]*)\b", text):
        if len(m.group(1)) >= 3:
            candidates.add(m.group(1))
    for m in _re.finditer(r"(/[^\s]*\{[^}]+\}[^\s]*)", text):
        candidates.add(m.group(1).strip())
    for m in _re.finditer(r"(/(?:[\w.+-]+/)+[\w.+-]*)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"([A-Z]:[\\/][^\s]+)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\d+)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\d+\.\d+(?:\.\d+)*)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"(\w+\|[\w|]*)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\w+\([^)]{1,30}\))\s*;?", text):
        full = m.group(0).strip()
        candidates.add(m.group(1))
        if full.endswith(";"):
            candidates.add(full)
    for m in _re.finditer(r'"(\s*[A-Za-z]\s*)"', text):
        candidates.add(m.group(0))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because", "but",
        "and", "or", "if", "while", "that", "this", "these", "those", "it",
        "its", "he", "she", "they", "them", "we", "you", "me", "my", "your",
        "his", "her", "our", "their", "what", "which", "who", "whom",
        "am", "about", "up", "down", "don't", "doesn't", "didn't", "won't",
        "wouldn't", "shouldn't", "couldn't", "can't", "I'm", "I've", "I'll",
    }
    candidates = {c for c in candidates if c not in stopwords and len(c) >= 1}

    candidate_list = sorted(candidates)

    BATCH_SIZE = 80
    all_entities: list[str] = []

    for i in range(0, len(candidate_list), BATCH_SIZE):
        batch = candidate_list[i : i + BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {term}" for j, term in enumerate(batch))

        prompt = f"""\
Given this StackOverflow text, classify which of the candidate terms below are \
software named entities per the StackOverflow NER annotation guidelines.

ENTITY TYPES (28 categories): Algorithm, Application, Class, Code_Block, \
Data_Structure, Data_Type, Device, Error_Name, File_Name, File_Type, Function, \
HTML_XML_Tag, Keyboard_IP, Language, Library, License, Operating_System, \
Organization, Output_Block, User_Interface_Element, User_Name, Value, Variable, \
Version, Website

CRITICAL — COMMON ENGLISH WORDS ARE ENTITIES in this dataset:
- "page", "list", "set", "tree", "cursor", "global", "setup", "arrow" → YES, entities
- "string", "table", "button", "server", "keyboard", "exception" → YES, entities
- "session", "phone", "image", "row", "column", "private", "long" → YES, entities
- Bare version numbers like "14", "2010", "3.0" → YES, entities (Version)
- Hyphenated compounds: "jQuery-generated", "cross-browser" → YES, entities
- Multi-word names: "visual editor", "command line" → YES, entities
- Example/demo content shown in posts: "ABCDEF|", sample data → YES, entities (Output_Block/Value)
- Terms with special chars: "this[int]", "keydown/keyup/keypress" → YES, entities
- Function calls with args: "free(pt) ;", "size(y, 1)", "buttons(new/edit)" → YES, entities
- Quoted drive letters or single-char refs: '" C "', '" D "' → YES, entities

When a candidate appears in technical context, INCLUDE it. The downstream pipeline \
will handle false positives. Missing entities is MUCH WORSE than including extras.

TEXT:
{text}

CANDIDATE TERMS:
{numbered}

Return ALL terms that could be entities. Be AGGRESSIVE — include borderline cases. \
Output JSON: {{"entities": ["term1", "term2", ...]}}
"""
        response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)
        batch_terms = _parse_terms_json(response)
        all_entities.extend(batch_terms)

    return all_entities


register_variant(
    "candidate_verify",
    model="sonnet",
    description="Enhanced candidates (bigrams, bare numbers, hyphenated) + aggressive classifier",
    prompt_template="(uses custom_runner)",
    custom_runner=_candidate_verify_runner,
)




# ============================================================================
# EXECUTION
# ============================================================================

def run_prompt_variant(
    variant_name: str,
    doc: dict,
    train_docs: list[dict] | None = None,
    retrieval_index=None,
    retrieval_model=None,
) -> tuple[list[str], float]:
    """Run a prompt variant on a single doc and return (terms, elapsed)."""
    variant = PROMPT_VARIANTS[variant_name]

    if variant.get("custom_runner"):
        t0 = time.time()
        terms = variant["custom_runner"](doc, train_docs, retrieval_index, retrieval_model)
        elapsed = time.time() - t0
        return terms, elapsed

    template = variant["prompt_template"]
    model = variant["model"]
    thinking_budget = variant.get("thinking_budget")

    t0 = time.time()

    if variant["needs_retrieval"]:
        from .retrieval_ner import safe_retrieve
        similar_docs = safe_retrieve(doc, train_docs, retrieval_index, retrieval_model, k=5)

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

        prompt = template.format(
            examples_block=examples_block,
            text=doc["text"][:5000],
        )
    else:
        prompt = template.format(content=doc["text"][:5000])

    response = call_llm(
        prompt,
        model=model,
        max_tokens=3000,
        temperature=0.0,
        thinking_budget=thinking_budget,
        timeout=180 if thinking_budget else 90,
    )
    elapsed = time.time() - t0

    terms = _parse_terms_json(response)
    return terms, elapsed



