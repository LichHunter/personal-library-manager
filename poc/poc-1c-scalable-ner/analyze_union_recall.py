#!/usr/bin/env python3
"""Analyze union recall across multiple extractors.

Runs 3 LLM extractors on test docs, collects raw extracted terms,
and computes:
- Individual recall per extractor
- Union recall (ceiling)
- Unique contributions per extractor
- Uncovered GT terms (what NO extractor catches)
- Overlap analysis

Usage:
    python analyze_union_recall.py --n-docs 10 --seed 42
    python analyze_union_recall.py --n-docs 10 --seed 42 --taxonomy-model sonnet
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from parse_so_ner import select_documents
from scoring import v3_match, many_to_many_score
from benchmark_prompt_variants import (
    PROMPT_VARIANTS,
    run_prompt_variant,
    load_json,
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


# ============================================================================
# EXTRACTORS TO ANALYZE
# ============================================================================

# Default extractors matching the actual pipeline configuration
DEFAULT_EXTRACTORS = [
    "taxonomy_v0",    # Sonnet/Haiku taxonomy (exhaustive, structured)
    "fewshot_v0",     # Haiku fewshot (retrieval-augmented, learns from examples)
    "haiku_simple_v0",  # Haiku simple (aggressive, common-word specialist)
]


def find_covering_extractors(term: str, extractor_terms: dict[str, list[str]]) -> list[str]:
    """Find which extractors extracted a term that matches the given GT term."""
    covering = []
    for ext_name, terms in extractor_terms.items():
        for t in terms:
            if v3_match(t, term):
                covering.append(ext_name)
                break
    return covering


def analyze_union(
    docs: list[dict],
    extractor_names: list[str],
    train_docs: list[dict],
    retrieval_index,
    retrieval_model,
    taxonomy_model: str = "haiku",
) -> dict:
    """Run all extractors on docs and compute union recall analysis."""

    # Override taxonomy model if requested
    if taxonomy_model != "haiku" and "taxonomy_v0" in extractor_names:
        original_model = PROMPT_VARIANTS["taxonomy_v0"]["model"]
        PROMPT_VARIANTS["taxonomy_v0"]["model"] = taxonomy_model
        print(f"  [Note: taxonomy_v0 running on {taxonomy_model} instead of {original_model}]")

    all_doc_results = []
    # Aggregate counters
    total_gt = 0
    total_union_covered = 0
    per_extractor_covered = {name: 0 for name in extractor_names}
    per_extractor_unique = {name: 0 for name in extractor_names}
    uncovered_terms_all: list[str] = []
    unique_terms_all: dict[str, list[str]] = {name: [] for name in extractor_names}

    # Track overlap patterns
    coverage_patterns: Counter = Counter()  # frozenset of extractors -> count

    for doc_idx, doc in enumerate(docs):
        doc_id = doc["doc_id"]
        gt = doc["gt_terms"]
        total_gt += len(gt)

        print(f"\n{'='*70}")
        print(f"Doc [{doc_idx+1}/{len(docs)}]: {doc_id} (GT: {len(gt)} terms)")
        print(f"{'='*70}")

        # Run each extractor and collect raw terms
        extractor_terms: dict[str, list[str]] = {}
        extractor_times: dict[str, float] = {}

        for name in extractor_names:
            terms, elapsed = run_prompt_variant(
                name, doc, train_docs, retrieval_index, retrieval_model,
            )
            extractor_terms[name] = terms
            extractor_times[name] = elapsed

            # Individual scoring
            scores = many_to_many_score(terms, gt)
            per_extractor_covered[name] += scores["covered_gt"]

            print(
                f"  {name:25s}: P={scores['precision']*100:5.1f}%  "
                f"R={scores['recall']*100:5.1f}%  "
                f"({len(terms)} extracted, {elapsed:.1f}s)"
            )

        # Compute union: which GT terms are covered by ANY extractor?
        union_covered_gt: list[str] = []
        union_uncovered_gt: list[str] = []

        for gt_term in gt:
            covering = find_covering_extractors(gt_term, extractor_terms)
            if covering:
                union_covered_gt.append(gt_term)
                coverage_patterns[frozenset(covering)] += 1

                # Check uniqueness
                if len(covering) == 1:
                    per_extractor_unique[covering[0]] += 1
                    unique_terms_all[covering[0]].append(gt_term)
            else:
                union_uncovered_gt.append(gt_term)
                uncovered_terms_all.append(gt_term)

        total_union_covered += len(union_covered_gt)
        union_recall = len(union_covered_gt) / len(gt) if gt else 0

        print(f"\n  Union recall: {union_recall*100:.1f}% ({len(union_covered_gt)}/{len(gt)})")
        if union_uncovered_gt:
            print(f"  UNCOVERED: {union_uncovered_gt}")

        # Show unique contributions
        for name in extractor_names:
            doc_unique = [t for t in gt if len(find_covering_extractors(t, extractor_terms)) == 1
                         and find_covering_extractors(t, extractor_terms)[0] == name]
            if doc_unique:
                print(f"  Unique to {name}: {doc_unique}")

        all_doc_results.append({
            "doc_id": doc_id,
            "gt_count": len(gt),
            "union_covered": len(union_covered_gt),
            "union_uncovered": len(union_uncovered_gt),
            "union_recall": round(union_recall, 4),
            "uncovered_terms": union_uncovered_gt,
            "per_extractor": {
                name: {
                    "extracted_count": len(extractor_terms[name]),
                    "time": round(extractor_times[name], 1),
                }
                for name in extractor_names
            },
        })

    # ========================================================================
    # AGGREGATE RESULTS
    # ========================================================================
    print(f"\n\n{'='*70}")
    print("AGGREGATE UNION RECALL ANALYSIS")
    print(f"{'='*70}\n")

    union_recall_total = total_union_covered / total_gt if total_gt else 0
    print(f"Total GT terms: {total_gt}")
    print(f"Union covered:  {total_union_covered} ({union_recall_total*100:.1f}%)")
    print(f"Uncovered:      {total_gt - total_union_covered} ({(1 - union_recall_total)*100:.1f}%)")

    print(f"\n{'Extractor':25s} {'Covered':>8s} {'Recall':>8s} {'Unique':>8s} {'% of Union':>10s}")
    print("-" * 65)
    for name in extractor_names:
        r = per_extractor_covered[name] / total_gt if total_gt else 0
        u = per_extractor_unique[name]
        pct_union = per_extractor_covered[name] / total_union_covered * 100 if total_union_covered else 0
        print(f"{name:25s} {per_extractor_covered[name]:8d} {r*100:7.1f}% {u:8d} {pct_union:9.1f}%")

    print(f"\nUnique contributions (terms ONLY this extractor catches):")
    for name in extractor_names:
        u = per_extractor_unique[name]
        print(f"  {name}: {u} unique terms")
        if unique_terms_all[name]:
            for t in unique_terms_all[name][:20]:
                print(f"    - {t}")
            if len(unique_terms_all[name]) > 20:
                print(f"    ... and {len(unique_terms_all[name]) - 20} more")

    print(f"\nUNCOVERED terms (no extractor catches these):")
    uncovered_counts = Counter(uncovered_terms_all)
    for term, count in uncovered_counts.most_common(30):
        print(f"  [{count}x] {term}")
    if len(uncovered_counts) > 30:
        print(f"  ... and {len(uncovered_counts) - 30} more")

    # Categorize uncovered terms
    print(f"\nUncovered term categories:")
    categories: dict[str, list[str]] = defaultdict(list)
    for term in set(uncovered_terms_all):
        t = term.lower()
        if any(c in t for c in ['/', '\\', '..']) or t.startswith('.') or t.startswith('/'):
            categories["paths/files"].append(term)
        elif any(c.isdigit() for c in t) and '.' in t and len(t) < 10:
            categories["version numbers"].append(term)
        elif t in {"exception", "private", "ondemand", "global", "kernel", "configuration",
                    "phone", "cpu", "keyboard", "button", "pad", "column", "row", "long",
                    "padding", "borders", "calculator", "image", "table", "tables",
                    "slider", "scrollbar", "scrollbars", "symlinks", "bytearrays",
                    "session", "unix"}:
            categories["common words (GT says entity)"].append(term)
        elif t.isupper() or (len(t) <= 5 and t[0].isupper()):
            categories["short/acronym"].append(term)
        elif any(c.isupper() for c in t[1:]) and t[0].islower():
            categories["camelCase"].append(term)
        else:
            categories["other"].append(term)

    for cat, terms in sorted(categories.items()):
        print(f"  {cat}: {len(terms)} — {terms[:10]}")

    # Coverage pattern analysis
    print(f"\nCoverage patterns (which extractor combinations cover terms):")
    for pattern, count in coverage_patterns.most_common(15):
        names = sorted(pattern)
        short_names = [n.replace("_v0", "").replace("haiku_", "h_") for n in names]
        print(f"  {' + '.join(short_names):50s}: {count:4d} terms")

    # Restore taxonomy model
    if taxonomy_model != "haiku" and "taxonomy_v0" in extractor_names:
        PROMPT_VARIANTS["taxonomy_v0"]["model"] = "haiku"

    return {
        "config": {
            "extractors": extractor_names,
            "n_docs": len(docs),
            "taxonomy_model": taxonomy_model,
        },
        "aggregate": {
            "total_gt": total_gt,
            "union_covered": total_union_covered,
            "union_recall": round(union_recall_total, 4),
            "per_extractor_covered": per_extractor_covered,
            "per_extractor_unique": per_extractor_unique,
            "unique_terms": {k: v for k, v in unique_terms_all.items()},
            "uncovered_terms": list(set(uncovered_terms_all)),
        },
        "docs": all_doc_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze union recall across extractors")
    parser.add_argument("--n-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doc-ids", nargs="+", default=None)
    parser.add_argument(
        "--extractors", nargs="+", default=DEFAULT_EXTRACTORS,
        help="Extractor variant names (default: taxonomy_v0, fewshot_v0, haiku_simple_v0)",
    )
    parser.add_argument(
        "--taxonomy-model", default="haiku", choices=["haiku", "sonnet"],
        help="Model for taxonomy extractor (default: haiku, pipeline uses sonnet)",
    )
    parser.add_argument("--save", action="store_true", help="Save results to artifacts")
    args = parser.parse_args()

    for v in args.extractors:
        if v not in PROMPT_VARIANTS:
            parser.error(f"Unknown variant: {v}. Use benchmark_prompt_variants.py --list-variants")

    train_docs = load_json(ARTIFACTS_DIR / "train_documents.json")
    test_docs = load_json(ARTIFACTS_DIR / "test_documents.json")
    print(f"Loaded {len(train_docs)} train, {len(test_docs)} test docs")

    if args.doc_ids:
        selected = [d for d in test_docs if d["doc_id"] in args.doc_ids]
    else:
        selected = select_documents(test_docs, args.n_docs, seed=args.seed)

    # Build retrieval index if any extractor needs it
    retrieval_index = None
    retrieval_model = None
    if any(PROMPT_VARIANTS[v]["needs_retrieval"] for v in args.extractors):
        from retrieval_ner import build_retrieval_index
        retrieval_index, _, retrieval_model = build_retrieval_index(train_docs)

    print(f"\nAnalyzing union recall for {len(args.extractors)} extractors on {len(selected)} docs")
    print(f"Extractors: {args.extractors}")
    print(f"Taxonomy model: {args.taxonomy_model}")

    results = analyze_union(
        selected, args.extractors, train_docs,
        retrieval_index, retrieval_model,
        taxonomy_model=args.taxonomy_model,
    )

    if args.save:
        out_path = ARTIFACTS_DIR / "results" / "union_recall_analysis.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
