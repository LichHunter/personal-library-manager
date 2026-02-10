#!/usr/bin/env python3
"""Auto-generate vocabulary lists from labeled training data.

Replaces manually curated CONTEXT_VALIDATION_BYPASS, MUST_EXTRACT_SEEDS,
and GT_NEGATIVE_TERMS with data-driven equivalents derived from 741 train docs.

Usage:
    python generate_vocab.py  # generates artifacts/auto_vocab.json
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_train_docs() -> list[dict]:
    with open(ARTIFACTS_DIR / "train_documents.json") as f:
        return json.load(f)


def compute_entity_stats(train_docs: list[dict]) -> dict:
    """Compute entity frequency and text presence stats from training data.

    Returns dict with:
        entity_doc_freq: Counter of term -> number of docs where it's a GT entity
        text_doc_freq: Counter of term -> number of docs where it appears in text
        entity_types: dict of term -> set of entity types it was annotated with
    """
    entity_doc_freq: Counter = Counter()
    text_doc_freq: Counter = Counter()
    entity_types: dict[str, set[str]] = defaultdict(set)

    for doc in train_docs:
        text_lower = doc["text"].lower()

        # Count entity frequency (doc-level deduped)
        seen_entities: set[str] = set()
        for i, term in enumerate(doc["gt_terms"]):
            t_lower = term.lower().strip()
            if t_lower not in seen_entities:
                entity_doc_freq[t_lower] += 1
                seen_entities.add(t_lower)
            if i < len(doc.get("entity_types", [])):
                entity_types[t_lower].add(doc["entity_types"][i])

        # Count text presence (word-boundary matching for single words,
        # substring matching for multi-word)
        words_in_text = set(re.findall(r"\b[\w#+.]+\b", text_lower))
        for word in words_in_text:
            text_doc_freq[word] += 1

        # Also check for multi-word entity terms in text
        for term in seen_entities:
            if " " in term and term in text_lower:
                text_doc_freq[term] += 1

    return {
        "entity_doc_freq": entity_doc_freq,
        "text_doc_freq": text_doc_freq,
        "entity_types": entity_types,
    }


def generate_bypass_list(
    stats: dict,
    min_entity_docs: int = 2,
    min_entity_ratio: float = 0.55,
) -> list[str]:
    """Generate CONTEXT_VALIDATION_BYPASS: terms that should always be treated as entities.

    These are terms with high entity-to-text ratio in training data.
    When they appear in a document, they're entities often enough that
    LLM context validation would incorrectly reject them.

    Lower thresholds than Oracle suggested to maximize recovery of rare-but-valid terms.
    """
    entity_freq = stats["entity_doc_freq"]
    text_freq = stats["text_doc_freq"]

    bypass: list[str] = []

    for term, entity_count in entity_freq.items():
        text_count = text_freq.get(term, entity_count)
        if text_count == 0:
            continue

        ratio = entity_count / text_count

        if entity_count >= min_entity_docs and ratio >= min_entity_ratio:
            bypass.append(term)

    return sorted(bypass)


def generate_seeds_list(
    stats: dict,
    min_entity_docs: int = 2,
    min_entity_ratio: float = 0.50,
) -> list[str]:
    """Generate MUST_EXTRACT_SEEDS: terms to force-check against document text.

    These are terms that LLMs systematically under-extract because they look
    like common English words. If found in text but not extracted, they get
    added to candidates.

    Focus on single-word terms that are common enough to matter.
    """
    entity_freq = stats["entity_doc_freq"]
    text_freq = stats["text_doc_freq"]

    seeds: list[str] = []

    for term, entity_count in entity_freq.items():
        text_count = text_freq.get(term, entity_count)
        if text_count == 0:
            continue

        ratio = entity_count / text_count

        # Focus on terms that appear enough to be worth seeding
        if entity_count >= min_entity_docs and ratio >= min_entity_ratio:
            # Only single words or very short compound terms for regex matching
            if len(term.split()) <= 2:
                seeds.append(term)

    return sorted(seeds)


KNOWN_NON_ENTITY_TECH_WORDS = {
    "code", "method", "function", "class", "object", "property",
    "type", "element", "variable", "variables", "module", "tag",
    "event", "handler", "model", "view", "service", "controller",
    "template", "plugin", "library", "framework", "package",
    "protocol", "database", "field", "fields", "target", "level",
    "gallery", "namespace", "action", "index", "value", "section",
    "print", "meta", "endpoint", "instance", "header", "footer",
    "toolbar", "dialog", "node", "child", "parent", "root",
    "listener", "callback", "promise", "response", "body",
    "path", "query", "token", "hash", "flag", "option",
    "state", "context", "provider", "consumer", "adapter",
    "wrapper", "factory", "proxy", "observer", "iterator",
    "stream", "buffer", "socket", "pipe", "channel",
    "extension", "driver", "engine", "runtime",
    "preview", "id", "ajax",
    "http", "https", "rest", "soap", "tcp", "ftp", "smtp",
    "ssl", "tls", "ssh", "dns", "cdn",
    "msvc", "gcc", "clang",
    "oop", "crud", "dsl",
    "boost", "microsoft",
}


def generate_negative_list(
    stats: dict,
    train_docs: list[dict],
    min_text_docs: int = 3,
    min_text_docs_known: int = 1,
    max_entity_ratio: float = 0.05,
) -> list[str]:
    """Generate GT_NEGATIVE_TERMS: terms that should always be rejected.

    These are plausible-looking tech terms that appear in text frequently
    but are NEVER (or almost never) annotated as entities in training data.

    Two tiers:
    - Known tech vocabulary: min_text_docs_known (1), entity_count=0
    - Auto-discovered acronyms: min_text_docs (3), entity_count=0
    - Borderline: entity_ratio < max_entity_ratio for rare 1-doc entities
    """
    entity_freq = stats["entity_doc_freq"]
    text_freq = stats["text_doc_freq"]

    negatives: list[str] = []
    tech_looking_words = set()

    for doc in train_docs:
        for match in re.finditer(r"\b[A-Z]{2,6}\b", doc["text"]):
            tech_looking_words.add(match.group().lower())
        text_lower = doc["text"].lower()
        for word in re.findall(r"\b[a-z]{3,15}\b", text_lower):
            if word in KNOWN_NON_ENTITY_TECH_WORDS:
                tech_looking_words.add(word)

    for word in tech_looking_words:
        text_count = text_freq.get(word, 0)
        entity_count = entity_freq.get(word, 0)

        is_known = word in KNOWN_NON_ENTITY_TECH_WORDS
        min_docs = min_text_docs_known if is_known else min_text_docs

        if entity_count == 0 and text_count >= min_docs:
            negatives.append(word)
            continue

        if (
            entity_count > 0
            and text_count >= min_text_docs
            and entity_count / text_count < max_entity_ratio
        ):
            negatives.append(word)

    return sorted(negatives)


def main() -> None:
    print("Loading training data...")
    train_docs = load_train_docs()
    print(f"  {len(train_docs)} train documents")

    print("\nComputing entity statistics...")
    stats = compute_entity_stats(train_docs)
    print(f"  {len(stats['entity_doc_freq'])} unique entity terms")
    print(f"  {len(stats['text_doc_freq'])} unique text tokens")

    print("\nGenerating CONTEXT_VALIDATION_BYPASS...")
    bypass = generate_bypass_list(stats)
    print(f"  {len(bypass)} terms")

    print("\nGenerating MUST_EXTRACT_SEEDS...")
    seeds = generate_seeds_list(stats)
    print(f"  {len(seeds)} terms")

    print("\nGenerating GT_NEGATIVE_TERMS...")
    negatives = generate_negative_list(stats, train_docs)
    print(f"  {len(negatives)} terms")

    # Save
    vocab = {
        "bypass": bypass,
        "seeds": seeds,
        "negatives": negatives,
        "stats": {
            "total_train_docs": len(train_docs),
            "unique_entity_terms": len(stats["entity_doc_freq"]),
            "bypass_count": len(bypass),
            "seeds_count": len(seeds),
            "negatives_count": len(negatives),
        },
    }

    out_path = ARTIFACTS_DIR / "auto_vocab.json"
    with open(out_path, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Print samples
    print("\n=== BYPASS (sample) ===")
    for t in bypass[:30]:
        ef = stats["entity_doc_freq"].get(t, 0)
        tf = stats["text_doc_freq"].get(t, ef)
        ratio = ef / tf if tf > 0 else 0
        print(f"  {t:25s} entity={ef:3d} text={tf:3d} ratio={ratio:.2f}")

    print("\n=== SEEDS (sample) ===")
    for t in seeds[:30]:
        ef = stats["entity_doc_freq"].get(t, 0)
        tf = stats["text_doc_freq"].get(t, ef)
        ratio = ef / tf if tf > 0 else 0
        print(f"  {t:25s} entity={ef:3d} text={tf:3d} ratio={ratio:.2f}")

    print("\n=== NEGATIVES (sample) ===")
    for t in negatives[:30]:
        tf = stats["text_doc_freq"].get(t, 0)
        print(f"  {t:25s} text={tf:3d}")


if __name__ == "__main__":
    main()
