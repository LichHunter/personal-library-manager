#!/usr/bin/env python3
"""Rescore all D+v2 variants with improved metrics.

Two independent improvements applied:
  A. DEDUP-BEFORE-SCORE: Merge extracted terms that are obvious variants
     (singular/plural, exact normalized duplicates) before scoring.
     This prevents "controller" and "controllers" from both appearing
     and one counting as FP when they map to the same GT concept.

  B. IMPROVED MATCHING: Extend the match function to catch cases the
     current matcher misses, like:
       - "ownerReference" ↔ "owner references" (camelCase vs spaces)
       - "cgroups" ↔ "cgroup" (plural/singular)
       - Tighter partial_ratio matching for compound terms

Both improvements are applied independently AND together so we can
see the separate and combined effect.

This script uses ONLY the saved extracted_terms from each results file —
NO LLM calls. It's a pure re-scoring.
"""

import json
from pathlib import Path
from rapidfuzz import fuzz

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GT_V2_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth_v2.json"

RESULTS_FILES = {
    "D+v2": ARTIFACTS_DIR / "dplus_v2_results.json",
    "D+v2.1": ARTIFACTS_DIR / "dplus_v2_1_results.json",
    "D+v2.2": ARTIFACTS_DIR / "dplus_v2_2_results.json",
}

with open(GT_V2_PATH) as f:
    gt_data = json.load(f)

gt_by_chunk = {}
for chunk in gt_data["chunks"]:
    gt_by_chunk[chunk["chunk_id"]] = [t["term"] for t in chunk["terms"]]


# ============================================================================
# ORIGINAL MATCHING (from test_dplus_v2.py — kept identical)
# ============================================================================

def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def original_match(extracted: str, ground_truth: str) -> bool:
    """Original matching function from test_dplus_v2.py."""
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)
    if ext_norm == gt_norm:
        return True
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True
    if ext_norm.endswith("s") and ext_norm[:-1] == gt_norm:
        return True
    if gt_norm.endswith("s") and gt_norm[:-1] == ext_norm:
        return True
    return False


# ============================================================================
# IMPROVED MATCHING
# ============================================================================

def improved_match(extracted: str, ground_truth: str) -> bool:
    """Improved matching that catches more true matches.
    
    Additions over original:
    - CamelCase splitting (ownerReference → owner reference)
    - Bi-directional singular/plural (not just endswith 's')
    - partial_ratio for compound term overlap
    - Hyphen/underscore normalization for URL-style terms
    """
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)

    # 1. Exact normalized match
    if ext_norm == gt_norm:
        return True

    # 2. Full fuzzy match (same threshold as original)
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True

    # 3. Token overlap (same as original)
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True

    # 4. Singular/plural — both directions, more robust
    def depluralize(s):
        if s.endswith("ies") and len(s) > 4:
            return s[:-3] + "y"
        if s.endswith("es") and len(s) > 4:
            return s[:-2]
        if s.endswith("s") and len(s) > 3:
            return s[:-1]
        return s

    if depluralize(ext_norm) == depluralize(gt_norm):
        return True
    if depluralize(ext_norm) == gt_norm or ext_norm == depluralize(gt_norm):
        return True

    # 5. CamelCase splitting (e.g., ownerReference → owner reference)
    import re
    def camel_to_words(s):
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower().strip()
    
    ext_camel = camel_to_words(extracted)
    gt_camel = camel_to_words(ground_truth)
    
    if normalize_term(ext_camel) == normalize_term(gt_camel):
        return True
    if depluralize(normalize_term(ext_camel)) == depluralize(normalize_term(gt_camel)):
        return True

    # 6. Partial ratio for compound terms (one is substring of the other)
    # Only for terms with >= 4 chars to avoid tiny spurious matches
    if len(ext_norm) >= 4 and len(gt_norm) >= 4:
        if fuzz.partial_ratio(ext_norm, gt_norm) >= 90:
            # Additional check: the shorter term must be a significant portion of the longer
            shorter = min(ext_norm, gt_norm, key=len)
            longer = max(ext_norm, gt_norm, key=len)
            if len(shorter) / len(longer) >= 0.5:
                return True

    return False


# ============================================================================
# DEDUP-BEFORE-SCORE
# ============================================================================

def dedup_variants(terms: list[str]) -> list[str]:
    """Merge extracted terms that are obvious variants.
    
    Rules (conservative):
    - Exact normalized duplicates → keep first
    - Singular/plural of same base → keep the form that appears first
    - CamelCase vs spaced (ownerReference = owner reference) → keep first
    
    Does NOT merge:
    - Substrings (TLS vs TLS bootstrapping — both valid)
    - Different qualifiers (cgroup v1 vs cgroup v2)
    - Component words of compound terms (labels vs labels and selectors)
    """
    import re
    
    if not terms:
        return []
    
    def depluralize(s):
        if s.endswith("ies") and len(s) > 4:
            return s[:-3] + "y"
        if s.endswith("es") and len(s) > 4:
            return s[:-2]
        if s.endswith("s") and len(s) > 3:
            return s[:-1]
        return s

    def canonical(term):
        """Get a canonical form for dedup grouping."""
        n = normalize_term(term)
        # CamelCase → words
        n = re.sub(r"([a-z])([A-Z])", r"\1 \2", n).lower().strip()
        # Normalize spaces
        n = " ".join(n.split())
        # Depluralize
        n = depluralize(n)
        return n

    seen: dict[str, str] = {}  # canonical → first term
    result: list[str] = []
    
    for term in terms:
        c = canonical(term)
        if c not in seen:
            seen[c] = term
            result.append(term)
    
    return result


# ============================================================================
# SCORING
# ============================================================================

def score_chunk(
    extracted: list[str],
    gt_terms: list[str],
    match_fn,
) -> dict:
    """Score a chunk's extracted terms against GT."""
    matched_gt: set[int] = set()
    tp = 0
    tp_pairs: list[tuple[str, str]] = []
    fp_terms: list[str] = []

    for ext in extracted:
        found = False
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if match_fn(ext, gt):
                matched_gt.add(j)
                tp += 1
                tp_pairs.append((ext, gt))
                found = True
                break
        if not found:
            fp_terms.append(ext)

    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in matched_gt]

    fp = len(extracted) - tp
    fn = len(gt_terms) - tp
    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extracted_count": len(extracted),
        "gt_count": len(gt_terms),
        "fp_terms": fp_terms,
        "fn_terms": fn_terms,
        "tp_pairs": tp_pairs,
    }


def score_all_chunks(
    results_data: dict,
    match_fn,
    apply_dedup: bool = False,
) -> dict:
    """Score all chunks with given match function, optionally dedup first."""
    per_chunk: list[dict] = []
    total_extracted = 0
    total_deduped = 0

    for chunk_result in results_data["per_chunk_results"]:
        chunk_id = chunk_result["chunk_id"]
        extracted = chunk_result["extracted_terms"]
        gt_terms = gt_by_chunk.get(chunk_id, [])

        if apply_dedup:
            deduped = dedup_variants(extracted)
        else:
            deduped = extracted

        total_extracted += len(extracted)
        total_deduped += len(deduped)

        metrics = score_chunk(deduped, gt_terms, match_fn)
        metrics["chunk_id"] = chunk_id
        metrics["pre_dedup_count"] = len(extracted)
        per_chunk.append(metrics)

    # Aggregate (macro average over chunks)
    n = len(per_chunk)
    agg = {
        "precision": sum(c["precision"] for c in per_chunk) / n,
        "recall": sum(c["recall"] for c in per_chunk) / n,
        "hallucination": sum(c["hallucination"] for c in per_chunk) / n,
        "f1": sum(c["f1"] for c in per_chunk) / n,
        "total_extracted": total_extracted,
        "total_after_dedup": total_deduped,
        "total_tp": sum(c["tp"] for c in per_chunk),
        "total_fp": sum(c["fp"] for c in per_chunk),
        "total_fn": sum(c["fn"] for c in per_chunk),
    }

    return {"aggregate": agg, "per_chunk": per_chunk}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("RESCORE: All D+v2 variants with 4 scoring configurations")
    print("=" * 100)
    print()
    print("Configurations:")
    print("  A. Original match, no dedup        (baseline — same as benchmark)")
    print("  B. Original match, with dedup       (only dedup improvement)")
    print("  C. Improved match, no dedup         (only matching improvement)")
    print("  D. Improved match, with dedup       (both improvements)")
    print()

    configs = [
        ("A: orig match, no dedup",    original_match, False),
        ("B: orig match, + dedup",     original_match, True),
        ("C: improved match, no dedup", improved_match, False),
        ("D: improved match, + dedup",  improved_match, True),
    ]

    all_results = {}

    for strategy_name, results_path in RESULTS_FILES.items():
        with open(results_path) as f:
            results_data = json.load(f)

        all_results[strategy_name] = {}
        for config_name, match_fn, do_dedup in configs:
            scored = score_all_chunks(results_data, match_fn, do_dedup)
            all_results[strategy_name][config_name] = scored

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────
    print("=" * 100)
    print("AGGREGATE RESULTS (macro-averaged over 15 chunks)")
    print("=" * 100)
    print()
    
    header = f"{'Strategy':12s} {'Config':30s} {'P':>7s} {'R':>7s} {'H':>7s} {'F1':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Ext':>5s}"
    print(header)
    print("-" * len(header))

    for strategy_name in RESULTS_FILES:
        for config_name, _, _ in configs:
            agg = all_results[strategy_name][config_name]["aggregate"]
            print(
                f"{strategy_name:12s} {config_name:30s} "
                f"{agg['precision']:>6.1%} {agg['recall']:>6.1%} "
                f"{agg['hallucination']:>6.1%} {agg['f1']:>6.3f} "
                f"{agg['total_tp']:>5d} {agg['total_fp']:>5d} {agg['total_fn']:>5d} "
                f"{agg['total_after_dedup']:>5d}"
            )
        print()

    # ── DETAILED: Best config per strategy ────────────────────────────────
    print("=" * 100)
    print("BEST CONFIG PER STRATEGY (by F1)")
    print("=" * 100)
    print()

    for strategy_name in RESULTS_FILES:
        best_config = max(
            all_results[strategy_name].items(),
            key=lambda x: x[1]["aggregate"]["f1"]
        )
        config_name = best_config[0]
        agg = best_config[1]["aggregate"]
        print(f"  {strategy_name}: {config_name}")
        print(f"    P={agg['precision']:.1%}  R={agg['recall']:.1%}  H={agg['hallucination']:.1%}  F1={agg['f1']:.3f}")
        print(f"    TP={agg['total_tp']}  FP={agg['total_fp']}  FN={agg['total_fn']}  Extracted={agg['total_after_dedup']}")
        print()

    # ── DETAILED: D+v2.2 config D per-chunk breakdown ─────────────────────
    print("=" * 100)
    print("D+v2.2 with Config D (improved match + dedup) — PER CHUNK")
    print("=" * 100)
    print()

    best_v22 = all_results["D+v2.2"]["D: improved match, + dedup"]
    header = f"{'Chunk':55s} {'P':>6s} {'R':>6s} {'H':>6s} {'F1':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Ext':>4s}"
    print(header)
    print("-" * len(header))
    for c in best_v22["per_chunk"]:
        print(
            f"{c['chunk_id']:55s} {c['precision']:>5.1%} {c['recall']:>5.1%} "
            f"{c['hallucination']:>5.1%} {c['f1']:>5.3f} {c['tp']:>4d} {c['fp']:>4d} {c['fn']:>4d} "
            f"{c['extracted_count']:>4d}"
        )

    # ── REMAINING FPs for best config ─────────────────────────────────────
    print()
    print("=" * 100)
    print("D+v2.2 Config D — REMAINING FALSE POSITIVES")
    print("=" * 100)
    print()

    total_fps = 0
    for c in best_v22["per_chunk"]:
        if c["fp_terms"]:
            for fp in c["fp_terms"]:
                total_fps += 1
                print(f"  [{c['chunk_id']}] '{fp}'")
    print(f"\n  Total FPs: {total_fps}")

    # ── REMAINING FNs for best config ─────────────────────────────────────
    print()
    print("=" * 100)
    print("D+v2.2 Config D — REMAINING FALSE NEGATIVES")
    print("=" * 100)
    print()

    total_fns = 0
    for c in best_v22["per_chunk"]:
        if c["fn_terms"]:
            for fn in c["fn_terms"]:
                total_fns += 1
                print(f"  [{c['chunk_id']}] '{fn}'")
    print(f"\n  Total FNs: {total_fns}")

    # ── DEDUP IMPACT ─────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("DEDUP IMPACT (terms removed by dedup)")
    print("=" * 100)
    print()

    for strategy_name in RESULTS_FILES:
        with open(RESULTS_FILES[strategy_name]) as f:
            rd = json.load(f)
        total_orig = sum(len(c["extracted_terms"]) for c in rd["per_chunk_results"])
        total_deduped = 0
        for c in rd["per_chunk_results"]:
            total_deduped += len(dedup_variants(c["extracted_terms"]))
        print(f"  {strategy_name}: {total_orig} → {total_deduped} ({total_orig - total_deduped} removed)")

    # ── SAVE FULL RESCORE RESULTS ─────────────────────────────────────────
    output_path = ARTIFACTS_DIR / "rescore_results.json"
    
    # Serialize (strip non-JSON-safe parts)
    serializable = {}
    for strat, configs_data in all_results.items():
        serializable[strat] = {}
        for cfg, scored in configs_data.items():
            serializable[strat][cfg] = {
                "aggregate": scored["aggregate"],
                "per_chunk": [
                    {k: v for k, v in c.items() if k not in ("fp_terms", "fn_terms", "tp_pairs")}
                    for c in scored["per_chunk"]
                ],
            }
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Full results saved to {output_path}")


if __name__ == "__main__":
    main()
