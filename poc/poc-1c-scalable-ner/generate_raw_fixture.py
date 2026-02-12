#!/usr/bin/env python3
"""Generate raw extraction fixture for validator tuning.

Runs candidate_verify_v1 + taxonomy_v0 (Sonnet) on selected test docs,
saves per-doc raw output without any validation/filtering.

Usage:
    python generate_raw_fixture.py --n-docs 10 --seed 42
"""

import argparse
import json
from pathlib import Path

from parse_so_ner import select_documents
from benchmark_prompt_variants import (
    PROMPT_VARIANTS,
    run_prompt_variant,
    load_json,
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def deduplicate_union(cv_terms: list[str], taxonomy_terms: list[str]) -> list[str]:
    seen: set[str] = set()
    union: list[str] = []
    for term in cv_terms + taxonomy_terms:
        key = term.lower().strip()
        if key not in seen:
            seen.add(key)
            union.append(term)
    return sorted(union, key=str.lower)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw extraction fixture")
    parser.add_argument("--n-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="artifacts/results/extraction_raw_10docs.json",
    )
    args = parser.parse_args()

    train_docs = load_json(ARTIFACTS_DIR / "train_documents.json")
    test_docs = load_json(ARTIFACTS_DIR / "test_documents.json")
    print(f"Loaded {len(train_docs)} train, {len(test_docs)} test docs")

    selected = select_documents(test_docs, args.n_docs, seed=args.seed)
    print(f"Selected {len(selected)} test docs (seed={args.seed})")

    PROMPT_VARIANTS["taxonomy_v0"]["model"] = "sonnet"

    results = []
    for i, doc in enumerate(selected):
        doc_id = doc["doc_id"]
        gt = doc["gt_terms"]
        print(f"\n[{i+1}/{len(selected)}] {doc_id} (GT={len(gt)})")

        cv_terms, cv_time = run_prompt_variant(
            "candidate_verify_v1", doc, train_docs, None, None,
        )
        print(f"  candidate_verify_v1: {len(cv_terms)} terms ({cv_time:.1f}s)")

        tax_terms, tax_time = run_prompt_variant(
            "taxonomy_v0", doc, train_docs, None, None,
        )
        print(f"  taxonomy_v0 (sonnet): {len(tax_terms)} terms ({tax_time:.1f}s)")

        raw_union = deduplicate_union(cv_terms, tax_terms)
        print(f"  union: {len(raw_union)} terms")

        results.append({
            "doc_id": doc_id,
            "text": doc["text"][:5000],
            "gt_terms": gt,
            "cv_terms": sorted(cv_terms),
            "taxonomy_terms": sorted(tax_terms),
            "raw_union": raw_union,
            "raw_union_count": len(raw_union),
            "gt_count": len(gt),
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_union = sum(r["raw_union_count"] for r in results)
    total_gt = sum(r["gt_count"] for r in results)
    print(f"\nSaved {len(results)} docs to {out_path}")
    print(f"Total: union={total_union}, gt={total_gt}")

    PROMPT_VARIANTS["taxonomy_v0"]["model"] = "haiku"


if __name__ == "__main__":
    main()
