#!/usr/bin/env python3
"""Analyze training data to find TRUE negatives — words that appear in many
documents but are NEVER annotated as entities.

This is the principled approach: only words with 0% entity rate across
a high number of documents belong on a negatives list.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

TRAIN_PATH = Path("artifacts/train_documents.json")


def main() -> None:
    with open(TRAIN_PATH) as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} training documents\n")

    # For each word: track how many docs it appears in, and how many it's entity in
    word_doc_count: dict[str, int] = defaultdict(int)      # docs where word appears in text
    word_entity_count: dict[str, int] = defaultdict(int)    # docs where word is GT entity

    for doc in docs:
        text_lower = doc["text"].lower()
        gt_lower = set(t.lower().strip() for t in doc["gt_terms"])

        # All words appearing in text
        words_in_text = set(re.findall(r"\b[\w#+.]+\b", text_lower))

        for w in words_in_text:
            if len(w) < 2:
                continue
            word_doc_count[w] += 1
            if w in gt_lower:
                word_entity_count[w] += 1

    # Find true negatives: entity_count = 0, doc_count >= threshold
    print("=" * 80)
    print("TRUE NEGATIVES: entity_ratio = 0.00 (never annotated as entity)")
    print("=" * 80)

    for min_docs in [50, 30, 20, 10]:
        true_negs = []
        for word, doc_count in word_doc_count.items():
            if doc_count >= min_docs and word_entity_count.get(word, 0) == 0:
                true_negs.append((word, doc_count))

        true_negs.sort(key=lambda x: -x[1])

        print(f"\n--- Appearing in >= {min_docs} docs, NEVER entity (top 60) ---")
        print(f"Total: {len(true_negs)} words")
        for word, count in true_negs[:60]:
            print(f"  {word:30s}  appears in {count:4d} docs, entity in 0")

    # Now check my CURRENT negatives list against reality
    print("\n" + "=" * 80)
    print("AUDIT: Current DOMAIN_INDEPENDENT_NEGATIVES vs training data")
    print("=" * 80)

    current_negatives = {
        "name", "value", "error", "line", "model", "handler", "service",
        "layout", "resources", "configuration", "selector", "instance",
        "interface", "module", "variable", "variables", "method", "function",
        "class", "object", "property", "type", "element", "tag", "event",
        "controller", "template", "plugin", "library", "framework", "package",
        "protocol", "database", "field", "fields", "target", "level",
        "namespace", "action", "index", "section", "endpoint", "header",
        "footer", "toolbar", "dialog", "node", "child", "parent", "root",
        "listener", "callback", "promise", "response", "body", "path",
        "query", "token", "hash", "flag", "option", "state", "context",
        "provider", "consumer", "adapter", "wrapper", "factory", "proxy",
        "observer", "iterator", "stream", "buffer", "socket", "pipe",
        "channel", "extension", "driver", "engine", "runtime", "preview",
        "code", "file", "print", "meta",
        "main", "src", "bin", "lib", "config", "test", "build",
        "h", "m", "x", "id",
        "tall", "gain", "thumb", "handle", "seed", "seeds",
        "cascade", "specificity", "siblings",
        "classpath", "constant", "distribution", "filesystem",
        "repository", "command",
    }

    safe = []
    unsafe = []
    unknown = []

    for word in sorted(current_negatives):
        doc_count = word_doc_count.get(word, 0)
        entity_count = word_entity_count.get(word, 0)

        if doc_count == 0:
            unknown.append((word, 0, 0))
        elif entity_count == 0:
            safe.append((word, doc_count, 0))
        else:
            ratio = entity_count / doc_count * 100
            unsafe.append((word, doc_count, entity_count, ratio))

    print(f"\n  SAFE (entity_ratio = 0.00, confirmed never-entity): {len(safe)}")
    for word, dc, ec in sorted(safe, key=lambda x: -x[1]):
        print(f"    {word:25s}  {dc:4d} docs, 0 entities  ✓")

    print(f"\n  UNSAFE (entity_ratio > 0, IS sometimes entity): {len(unsafe)}")
    for word, dc, ec, ratio in sorted(unsafe, key=lambda x: -x[3]):
        print(f"    {word:25s}  {dc:4d} docs, {ec:3d} entities ({ratio:.1f}%)  ✗")

    print(f"\n  UNKNOWN (not found in training data): {len(unknown)}")
    for word, _, _ in sorted(unknown):
        print(f"    {word:25s}  not in training data  ?")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Current negatives list: {len(current_negatives)} terms")
    print(f"  Confirmed safe (0% entity rate): {len(safe)} ({len(safe)/len(current_negatives)*100:.0f}%)")
    print(f"  UNSAFE (>0% entity rate): {len(unsafe)} ({len(unsafe)/len(current_negatives)*100:.0f}%)")
    print(f"  Unknown (no training data): {len(unknown)} ({len(unknown)/len(current_negatives)*100:.0f}%)")
    print(f"\n  --> {len(unsafe)} terms in the negatives list are WRONG.")
    print(f"  --> They are sometimes entities and should be removed.")


if __name__ == "__main__":
    main()
