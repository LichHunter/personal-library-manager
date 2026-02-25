#!/usr/bin/env python3
"""Apply pre-filter rules to raw extraction fixture and report results.

Rules:
1. Determiner-strip: remove "the X", "a X" etc. when bare X exists in union
2. Number-percent: remove "66.67 %" patterns

Usage:
    python run_prefilter.py
"""

import json
import re
from pathlib import Path
from scoring import v3_match

RAW_PATH = Path("artifacts/results/extraction_raw_10docs.json")

DETERMINER_RE = re.compile(
    r"^(a|an|the|some|any|my|this|that|these|those)\s+(.+)$", re.I
)
NUMBER_PERCENT_RE = re.compile(r"^\d+\.?\d*\s+%$")


def prefilter(raw_union: list[str]) -> tuple[list[str], list[str]]:
    """Apply pre-filter rules. Returns (kept, removed)."""
    union_lower = {t.lower().strip() for t in raw_union}

    kept = []
    removed = []

    for term in raw_union:
        # Rule 2: number-space-percent
        if NUMBER_PERCENT_RE.match(term):
            removed.append(term)
            continue

        # Rule 1: determiner-strip (only when bare form exists)
        m = DETERMINER_RE.match(term)
        if m:
            remainder = m.group(2)
            if remainder.lower().strip() in union_lower:
                removed.append(term)
                continue

        kept.append(term)

    return kept, removed


def main() -> None:
    with open(RAW_PATH) as f:
        docs = json.load(f)

    total_before = 0
    total_after = 0
    total_removed = 0
    total_gt = 0
    total_gt_covered = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        gt = doc["gt_terms"]
        raw = doc["raw_union"]
        total_gt += len(gt)
        total_before += len(raw)

        kept, removed = prefilter(raw)
        total_after += len(kept)
        total_removed += len(removed)

        # Check recall on kept terms
        covered = 0
        missed = []
        for g in gt:
            if any(v3_match(k, g) for k in kept):
                covered += 1
            else:
                missed.append(g)
        total_gt_covered += covered
        recall = covered / len(gt) * 100 if gt else 0

        # Classify removed as FP or TP-loss
        removed_fps = []
        removed_tps = []
        for r in removed:
            if any(v3_match(r, g) for g in gt):
                removed_tps.append(r)
            else:
                removed_fps.append(r)

        print(f"{doc_id}: {len(raw)} → {len(kept)} (removed {len(removed)}) | recall={recall:.1f}%")
        if removed:
            print(f"  Removed FPs ({len(removed_fps)}): {removed_fps}")
            if removed_tps:
                print(f"  ⚠ REMOVED TPs ({len(removed_tps)}): {removed_tps}")
        if missed:
            print(f"  MISSED GT: {missed}")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_before} → {total_after} terms (removed {total_removed})")
    print(f"Recall: {total_gt_covered}/{total_gt} = {total_gt_covered/total_gt*100:.1f}%")
    print(f"Avg per doc: {total_before/len(docs):.0f} → {total_after/len(docs):.0f}")

    # Now show FP counts before and after
    print(f"\n{'='*70}")
    print("FP ANALYSIS (before vs after pre-filter)")
    print(f"{'='*70}")

    total_fp_before = 0
    total_fp_after = 0

    for doc in docs:
        gt = doc["gt_terms"]
        raw = doc["raw_union"]
        kept, _ = prefilter(raw)

        fps_before = [t for t in raw if not any(v3_match(t, g) for g in gt)]
        fps_after = [t for t in kept if not any(v3_match(t, g) for g in gt)]
        total_fp_before += len(fps_before)
        total_fp_after += len(fps_after)

        print(f"  {doc['doc_id']}: FPs {len(fps_before)} → {len(fps_after)} (removed {len(fps_before)-len(fps_after)})")

    print(f"\n  Total FPs: {total_fp_before} → {total_fp_after} (removed {total_fp_before - total_fp_after})")

    # Full remaining FP list per doc
    print(f"\n{'='*70}")
    print("REMAINING FPs PER DOC (what Sonnet must handle)")
    print(f"{'='*70}")

    for doc in docs:
        gt = doc["gt_terms"]
        kept, _ = prefilter(doc["raw_union"])
        fps = sorted([t for t in kept if not any(v3_match(t, g) for g in gt)], key=str.lower)
        tps = sorted([t for t in kept if any(v3_match(t, g) for g in gt)], key=str.lower)
        print(f"\n  {doc['doc_id']}: {len(kept)} kept = {len(tps)} TPs + {len(fps)} FPs (GT={doc['gt_count']})")
        for fp in fps:
            print(f"    FP: '{fp}'")


if __name__ == "__main__":
    main()
