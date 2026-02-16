#!/usr/bin/env python3
"""Build a tech-domain negatives list using ALL 1,237 SO NER documents.

Uses train + dev + test splits (1,237 docs total) as ground truth.
A word is a safe tech-domain negative IFF:
  - entity_ratio = 0.00 across ALL docs where it appears
  - appears in >= N docs (configurable threshold)
  - is not a pure stop word (could plausibly be extracted by NER)

Filters out obvious English stop words that no NER extractor would output.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

ARTIFACTS = Path("artifacts")

# Stop words that no NER extractor would ever extract as candidates
STOP_WORDS = {
    "the", "to", "is", "and", "in", "this", "it", "of", "that", "you",
    "have", "for", "but", "can", "with", "be", "not", "on", "do", "if",
    "my", "an", "as", "from", "like", "are", "your", "or", "there",
    "when", "want", "how", "what", "am", "will", "which", "need", "then",
    "any", "by", "all", "here", "would", "also", "one", "some", "should",
    "just", "only", "at", "does", "me", "has", "was", "been", "were",
    "being", "had", "did", "its", "they", "them", "their", "we", "our",
    "he", "she", "his", "her", "who", "whom", "whose", "where", "why",
    "each", "every", "both", "few", "more", "most", "other", "than",
    "too", "very", "much", "many", "so", "no", "nor", "yet", "about",
    "after", "before", "during", "between", "through", "into", "out",
    "up", "down", "over", "under", "again", "further", "above", "below",
    "using", "use", "used", "trying", "try", "tried", "get", "got",
    "getting", "set", "setting", "make", "made", "making", "work",
    "working", "worked", "works", "know", "think", "see", "look",
    "looking", "find", "found", "help", "run", "running", "create",
    "creating", "created", "add", "added", "adding", "change", "changed",
    "call", "called", "calling", "calls", "take", "takes", "give",
    "gives", "given", "seem", "seems", "going", "come", "comes",
    "pass", "passed", "passing", "keep", "write", "writing", "read",
    "reading", "show", "shows", "start", "end", "tell", "says",
    "define", "defined", "check", "checked", "return", "returns",
    "returned", "remove", "specify", "update", "move",
    "example", "thing", "things", "problem", "issue", "issues",
    "question", "answer", "solution", "idea", "case", "cases",
    "time", "times", "way", "ways", "part", "parts", "point",
    "result", "results", "reason", "order", "number", "place",
    "fact", "something", "anything", "nothing", "everything",
    "everyone", "someone", "lot", "bit", "kind", "stuff",
    "step", "steps", "piece", "side", "top", "bottom", "left", "right",
    "new", "first", "last", "different", "same", "other", "possible",
    "sure", "able", "good", "better", "best", "bad", "worse", "worst",
    "simple", "easy", "hard", "wrong", "true", "false",
    "actually", "basically", "probably", "maybe", "really", "already",
    "still", "even", "however", "though", "always", "never", "often",
    "well", "instead", "else", "since", "unless",
    "thanks", "please", "sorry", "hope", "edit", "note", "yes", "ok",
    "okay", "hi", "hello", "follow", "following", "could", "these",
    "doing", "might", "without", "may", "while", "such", "looks",
    "anyone", "put", "etc", "specific", "having", "inside", "once",
    "based", "multiple", "cannot", "those", "done", "correct",
    "send", "understand", "simply", "similar", "say", "e.g",
    "thank", "currently", "around", "far", "own", "current",
    "must", "single", "means", "within", "gets", "solve",
    "appreciated", "via", "either", "world", "second", "fine",
    "now", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "information", "click",
}


def main() -> None:
    # Load ALL splits
    all_docs = []
    for split_name in ["train_documents", "dev_documents", "test_documents"]:
        path = ARTIFACTS / f"{split_name}.json"
        with open(path) as f:
            docs = json.load(f)
        all_docs.extend(docs)
        print(f"  Loaded {len(docs)} from {split_name}")
    print(f"  TOTAL: {len(all_docs)} documents\n")

    # Count word appearances and entity annotations
    word_doc_count: dict[str, int] = defaultdict(int)
    word_entity_count: dict[str, int] = defaultdict(int)

    for doc in all_docs:
        text_lower = doc["text"].lower()
        gt_lower = set(t.lower().strip() for t in doc["gt_terms"])
        words_in_text = set(re.findall(r"\b[\w#+.]+\b", text_lower))

        for w in words_in_text:
            if len(w) < 2:
                continue
            word_doc_count[w] += 1
            if w in gt_lower:
                word_entity_count[w] += 1

    # ── Tier 1: 0% entity rate, >= 20 docs, not stop word ──
    # These are technical words that appear frequently in SO posts
    # but are NEVER annotated as software entities
    tiers = {
        "TIER_1 (>=20 docs)": 20,
        "TIER_2 (>=10 docs)": 10,
        "TIER_3 (>=5 docs)": 5,
    }

    for tier_name, min_docs in tiers.items():
        candidates = []
        for word, doc_count in word_doc_count.items():
            if doc_count < min_docs:
                continue
            if word_entity_count.get(word, 0) > 0:
                continue
            if word.lower() in STOP_WORDS:
                continue
            if len(word) <= 2:
                continue
            # Filter out pure numbers
            if word.isdigit():
                continue
            candidates.append((word, doc_count))

        candidates.sort(key=lambda x: -x[1])

        print(f"{'='*80}")
        print(f"{tier_name}: 0% entity rate in {len(all_docs)} tech docs, not stop word")
        print(f"Total: {len(candidates)} words")
        print(f"{'='*80}")

        for word, count in candidates[:100]:
            print(f"  {word:30s}  {count:4d} docs, 0 entities")

    # ── Build the final safe list ──
    # Use tier 1 (>=20 docs) for the high-confidence list
    # and tier 2 (>=10 docs) for extended list
    safe_20 = []
    safe_10 = []
    safe_5 = []

    for word, doc_count in word_doc_count.items():
        if word_entity_count.get(word, 0) > 0:
            continue
        if word.lower() in STOP_WORDS:
            continue
        if len(word) <= 2:
            continue
        if word.isdigit():
            continue

        if doc_count >= 20:
            safe_20.append(word)
        if doc_count >= 10:
            safe_10.append(word)
        if doc_count >= 5:
            safe_5.append(word)

    # ── Now show the UNSAFE words that were in our old negatives list ──
    old_negatives_that_were_unsafe = []
    for word in sorted(word_doc_count.keys()):
        ent = word_entity_count.get(word, 0)
        dc = word_doc_count.get(word, 0)
        if ent > 0 and dc >= 3:
            rate = ent / dc * 100
            if rate <= 30:  # Only show "surprising" ones — low rate but nonzero
                old_negatives_that_were_unsafe.append((word, dc, ent, rate))

    # ── Print words near the boundary (1-5% entity rate, 10+ docs) ──
    print(f"\n{'='*80}")
    print(f"BORDERLINE: 0 < entity_rate <= 5%, >= 10 docs")
    print(f"These are words that are ALMOST always generic but occasionally entities")
    print(f"{'='*80}")

    borderline = []
    for word, doc_count in word_doc_count.items():
        if doc_count < 10:
            continue
        ent = word_entity_count.get(word, 0)
        if ent == 0:
            continue
        rate = ent / doc_count * 100
        if rate <= 5:
            borderline.append((word, doc_count, ent, rate))

    borderline.sort(key=lambda x: x[3])
    for word, dc, ec, rate in borderline:
        print(f"  {word:30s}  {dc:4d} docs, {ec:3d} entities ({rate:.1f}%)")

    # ── Save results ──
    result = {
        "dataset_size": len(all_docs),
        "domain": "tech/programming (StackOverflow NER)",
        "tier_1_min_docs": 20,
        "tier_2_min_docs": 10,
        "tier_3_min_docs": 5,
        "tier_1_words": sorted(safe_20),
        "tier_2_words": sorted(safe_10),
        "tier_3_words": sorted(safe_5),
        "details": {},
    }

    # Save details for all tier 2 words
    for word in safe_10:
        result["details"][word] = {
            "doc_count": word_doc_count[word],
            "entity_count": 0,
            "entity_rate": 0.0,
        }

    out_path = ARTIFACTS / "tech_domain_negatives.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Total documents:  {len(all_docs)}")
    print(f"  Tier 1 (≥20 docs, 0% entity rate): {len(safe_20)} words")
    print(f"  Tier 2 (≥10 docs, 0% entity rate): {len(safe_10)} words")
    print(f"  Tier 3 (≥5 docs, 0% entity rate):  {len(safe_5)} words")
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
