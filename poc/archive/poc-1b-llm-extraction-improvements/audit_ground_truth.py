#!/usr/bin/env python3
"""Audit ground truth to understand recall ceiling.

For each GT term, determine:
1. Exact match in text?
2. Fuzzy match (minor variations)?
3. Only implied/inferred?

This tells us the theoretical maximum recall achievable with grounding.
"""

import json
import re
from pathlib import Path
from collections import Counter

from rapidfuzz import fuzz

POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"

print("Ground Truth Audit", flush=True)
print("=" * 70, flush=True)


def find_term_in_text(term: str, text: str) -> dict:
    """Check how a term appears in text."""
    text_lower = text.lower()
    term_lower = term.lower()

    result = {
        "term": term,
        "exact_match": False,
        "case_insensitive": False,
        "fuzzy_match": False,
        "partial_match": False,
        "implied": False,
        "match_type": "NOT_FOUND",
        "best_match": None,
    }

    # 1. Exact match
    if term in text:
        result["exact_match"] = True
        result["case_insensitive"] = True
        result["match_type"] = "EXACT"
        result["best_match"] = term
        return result

    # 2. Case-insensitive match
    if term_lower in text_lower:
        result["case_insensitive"] = True
        result["match_type"] = "CASE_INSENSITIVE"
        # Find actual occurrence
        idx = text_lower.find(term_lower)
        result["best_match"] = text[idx : idx + len(term)]
        return result

    # 3. Fuzzy match - handle CamelCase split, underscores, hyphens
    variations = [
        term_lower,
        term_lower.replace("_", " "),
        term_lower.replace("-", " "),
        re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower(),  # CamelCase to spaces
        term_lower.replace(" ", ""),  # Remove spaces
    ]

    for var in variations:
        if var in text_lower:
            result["fuzzy_match"] = True
            result["match_type"] = "FUZZY"
            result["best_match"] = var
            return result

    # 4. Partial match - at least 80% of tokens present
    term_tokens = set(term_lower.split())
    if len(term_tokens) > 1:
        text_tokens = set(text_lower.split())
        overlap = term_tokens & text_tokens
        if len(overlap) >= len(term_tokens) * 0.8:
            result["partial_match"] = True
            result["match_type"] = "PARTIAL"
            result["best_match"] = " ".join(overlap)
            return result

    # 5. High fuzzy ratio somewhere in text
    # Check sliding windows
    words = text_lower.split()
    term_word_count = len(term_lower.split())

    for i in range(len(words) - term_word_count + 1):
        window = " ".join(words[i : i + term_word_count])
        ratio = fuzz.ratio(term_lower, window)
        if ratio >= 85:
            result["fuzzy_match"] = True
            result["match_type"] = "HIGH_FUZZY"
            result["best_match"] = window
            return result

    # 6. Implied - not found
    result["implied"] = True
    result["match_type"] = "IMPLIED"
    return result


def audit_ground_truth():
    """Audit all ground truth terms."""
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    all_results = []
    type_counts = Counter()
    tier_type_counts = {1: Counter(), 2: Counter(), 3: Counter()}

    print(
        f"\nAnalyzing {gt['total_chunks']} chunks, {gt['total_terms']} terms...\n",
        flush=True,
    )

    implied_terms = []

    for chunk in gt["chunks"]:
        text = chunk["text"]

        for term_info in chunk["terms"]:
            term = term_info["term"]
            tier = term_info.get("tier", 1)

            result = find_term_in_text(term, text)
            result["chunk_id"] = chunk["chunk_id"]
            result["tier"] = tier

            all_results.append(result)
            type_counts[result["match_type"]] += 1
            tier_type_counts[tier][result["match_type"]] += 1

            if result["match_type"] == "IMPLIED":
                implied_terms.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "term": term,
                        "tier": tier,
                        "text_preview": text[:200] + "...",
                    }
                )

    # Summary
    total = len(all_results)

    print("=" * 70, flush=True)
    print("GROUND TRUTH AUDIT RESULTS", flush=True)
    print("=" * 70, flush=True)

    print(f"\nTotal terms: {total}", flush=True)
    print(f"\nMatch type distribution:", flush=True)

    recoverable = 0
    for match_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        mark = ""
        if match_type in [
            "EXACT",
            "CASE_INSENSITIVE",
            "FUZZY",
            "HIGH_FUZZY",
            "PARTIAL",
        ]:
            recoverable += count
            mark = " (recoverable)"
        print(f"  {match_type:<20}: {count:>4} ({pct:>5.1f}%){mark}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"THEORETICAL RECALL CEILING: {recoverable / total * 100:.1f}%", flush=True)
    print(f"(Terms with grounding in source text)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Tier breakdown
    print(f"\nRecoverable by Tier:", flush=True)
    for tier in [1, 2, 3]:
        tier_total = sum(tier_type_counts[tier].values())
        if tier_total > 0:
            tier_recoverable = sum(
                tier_type_counts[tier][t]
                for t in ["EXACT", "CASE_INSENSITIVE", "FUZZY", "HIGH_FUZZY", "PARTIAL"]
            )
            print(
                f"  Tier {tier}: {tier_recoverable}/{tier_total} ({tier_recoverable / tier_total * 100:.1f}%)",
                flush=True,
            )

    # Show implied terms
    print(f"\n{'=' * 70}", flush=True)
    print(f"IMPLIED TERMS (not found in text):", flush=True)
    print(f"{'=' * 70}", flush=True)

    for item in implied_terms[:20]:  # Show first 20
        print(
            f"  [{item['chunk_id']}] Tier {item['tier']}: '{item['term']}'", flush=True
        )

    if len(implied_terms) > 20:
        print(f"  ... and {len(implied_terms) - 20} more", flush=True)

    # Save full audit
    audit_path = Path(__file__).parent / "artifacts" / "gt_audit.json"
    audit_path.parent.mkdir(exist_ok=True)

    with open(audit_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "total_terms": total,
                    "type_counts": dict(type_counts),
                    "recoverable_count": recoverable,
                    "recoverable_pct": recoverable / total * 100,
                    "implied_count": len(implied_terms),
                },
                "implied_terms": implied_terms,
                "all_results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nFull audit saved to: {audit_path}", flush=True)

    return {
        "total": total,
        "recoverable": recoverable,
        "ceiling": recoverable / total * 100,
        "implied_terms": implied_terms,
    }


if __name__ == "__main__":
    audit_ground_truth()
