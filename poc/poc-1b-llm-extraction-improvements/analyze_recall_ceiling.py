#!/usr/bin/env python3
"""Recall Ceiling Analysis: Trace every GT term through each pipeline stage.

For each GT term, classify its fate:
  1. NEVER_EXTRACTED — No extractor produced any term matching this GT term
  2. UNGROUNDED — Extracted but failed span grounding
  3. STRUCTURAL_FILTERED — Grounded but removed by structural filter
  4. SONNET_REJECTED — Grounded, passed structural, but Sonnet rejected (1-vote)
  5. DEDUP_MERGED — Extracted and approved but merged away by dedup
  6. COVERED — Present in final output

Also computes:
  - Extraction ceiling: % of GT terms matched by ANY raw extraction (before filtering)
  - Grounding ceiling: % of GT terms matched after grounding
  - Vote+structural ceiling: % after structural filter
  - Per-variant final coverage (using cached V3 sweep results)

Compares greedy_v3 vs m2o_v3 FN lists to explain the recall gap.

Usage:
    python analyze_recall_ceiling.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)

# ============================================================================
# PATHS
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GT_V2_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth_v2.json"
CACHE_PATH = ARTIFACTS_DIR / "v3_phase1_3_cache.json"
SWEEP_RESULTS_PATH = ARTIFACTS_DIR / "v3_sweep_results.json"

# ============================================================================
# MATCHING FUNCTIONS (copied from test_dplus_v3_sweep.py for consistency)
# ============================================================================


def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def depluralize(s: str) -> str:
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y"
    if s.endswith("es") and len(s) > 4:
        return s[:-2]
    if s.endswith("s") and len(s) > 3:
        return s[:-1]
    return s


def camel_to_words(s: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower().strip()


def v3_match(extracted: str, ground_truth: str) -> bool:
    """V3 matcher from test_dplus_v3_sweep.py."""
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

    if depluralize(ext_norm) == depluralize(gt_norm):
        return True
    if depluralize(ext_norm) == gt_norm or ext_norm == depluralize(gt_norm):
        return True

    ext_camel = normalize_term(camel_to_words(extracted))
    gt_camel = normalize_term(camel_to_words(ground_truth))
    if ext_camel == gt_camel:
        return True
    if depluralize(ext_camel) == depluralize(gt_camel):
        return True

    if len(ext_norm) >= 4 and len(gt_norm) >= 4:
        if fuzz.partial_ratio(ext_norm, gt_norm) >= 90:
            shorter = min(ext_norm, gt_norm, key=len)
            longer = max(ext_norm, gt_norm, key=len)
            if len(shorter) / len(longer) >= 0.5:
                return True

    shorter_term = min(ext_norm, gt_norm, key=len)
    longer_term = max(ext_norm, gt_norm, key=len)
    if 2 <= len(shorter_term) <= 5 and len(longer_term) > len(shorter_term):
        pattern = r"(?:^|\s)" + re.escape(shorter_term) + r"(?:\s|$)"
        if re.search(pattern, longer_term):
            return True
        shorter_dep = depluralize(shorter_term)
        if shorter_dep != shorter_term:
            pattern_dep = r"(?:^|\s)" + re.escape(shorter_dep) + r"(?:\s|$)"
            if re.search(pattern_dep, longer_term):
                return True

    return False


def find_matching_candidates(
    gt_term: str, candidates: dict[str, dict]
) -> list[dict]:
    """Find all candidates that match a GT term using v3_match."""
    matches = []
    for key, cand in candidates.items():
        if v3_match(cand["term"], gt_term):
            matches.append(cand)
    return matches


def find_matching_raw(gt_term: str, raw_terms: list[str]) -> list[str]:
    """Find all raw extracted terms that match a GT term."""
    return [t for t in raw_terms if v3_match(t, gt_term)]


# ============================================================================
# STAGE 1: RAW EXTRACTION CEILING
# ============================================================================


def analyze_extraction_ceiling(
    gt_data: dict, cache: dict
) -> dict:
    """For each GT term, check if ANY raw extraction matches it."""
    cache_by_id = {c["chunk_id"]: c for c in cache["chunks"]}

    total_gt = 0
    extracted_count = 0
    never_extracted: list[dict] = []
    extracted_details: list[dict] = []

    for chunk in gt_data["chunks"]:
        cid = chunk["chunk_id"]
        gt_terms = [t["term"] for t in chunk["terms"]]
        ccache = cache_by_id.get(cid)
        if not ccache:
            continue

        raw = ccache["raw_extractions"]
        all_raw = []
        for src_terms in raw.values():
            all_raw.extend(src_terms)

        for gt_term in gt_terms:
            total_gt += 1
            matches = find_matching_raw(gt_term, all_raw)
            if matches:
                extracted_count += 1
                # Which extractors found it?
                sources = set()
                for src_name, src_terms in raw.items():
                    for t in src_terms:
                        if v3_match(t, gt_term):
                            sources.add(src_name)
                            break
                extracted_details.append({
                    "chunk_id": cid,
                    "gt_term": gt_term,
                    "matched_by": list(sources),
                    "matched_terms": list(set(matches)),
                })
            else:
                never_extracted.append({
                    "chunk_id": cid,
                    "gt_term": gt_term,
                    "tier": next(
                        (t["tier"] for t in chunk["terms"] if t["term"] == gt_term),
                        None,
                    ),
                })

    return {
        "total_gt": total_gt,
        "extracted": extracted_count,
        "never_extracted_count": len(never_extracted),
        "extraction_ceiling": extracted_count / total_gt if total_gt else 0,
        "never_extracted": never_extracted,
        "extracted_details": extracted_details,
    }


# ============================================================================
# STAGE 2: PIPELINE STAGE CLASSIFICATION
# ============================================================================


def classify_gt_terms(
    gt_data: dict, cache: dict
) -> dict:
    """Classify every GT term by where it's lost in the pipeline."""
    cache_by_id = {c["chunk_id"]: c for c in cache["chunks"]}

    classifications: list[dict] = []
    stage_counts = Counter()

    for chunk in gt_data["chunks"]:
        cid = chunk["chunk_id"]
        gt_terms_data = chunk["terms"]
        ccache = cache_by_id.get(cid)
        if not ccache:
            continue

        raw = ccache["raw_extractions"]
        all_candidates = ccache["all_candidates"]
        auto_kept = ccache["auto_kept"]
        needs_review = ccache["needs_review"]
        ungrounded_list = ccache["ungrounded"]
        structural_removed = ccache["structural_removed"]

        for gt_entry in gt_terms_data:
            gt_term = gt_entry["term"]
            tier = gt_entry.get("tier", None)

            # Step 1: Was it extracted at all?
            all_raw = []
            for src_terms in raw.values():
                all_raw.extend(src_terms)
            raw_matches = find_matching_raw(gt_term, all_raw)

            if not raw_matches:
                stage_counts["NEVER_EXTRACTED"] += 1
                classifications.append({
                    "chunk_id": cid, "gt_term": gt_term, "tier": tier,
                    "stage": "NEVER_EXTRACTED",
                    "detail": "No extractor produced a matching term",
                })
                continue

            # Step 2: Was it in the candidate pool? (after dedup by normalized form)
            candidate_matches = find_matching_candidates(gt_term, all_candidates)
            if not candidate_matches:
                # Extracted but somehow not in candidates (shouldn't happen, but safety)
                stage_counts["NEVER_EXTRACTED"] += 1
                classifications.append({
                    "chunk_id": cid, "gt_term": gt_term, "tier": tier,
                    "stage": "NEVER_EXTRACTED",
                    "detail": f"Raw matches {raw_matches} but not in candidates",
                })
                continue

            # Step 3: Was it grounded?
            grounded_matches = [c for c in candidate_matches if c.get("is_grounded")]
            ungrounded_matches = [c for c in candidate_matches if not c.get("is_grounded")]

            if not grounded_matches:
                stage_counts["UNGROUNDED"] += 1
                classifications.append({
                    "chunk_id": cid, "gt_term": gt_term, "tier": tier,
                    "stage": "UNGROUNDED",
                    "detail": f"Extracted as {[c['term'] for c in candidate_matches]} but failed span grounding",
                    "candidate_terms": [c["term"] for c in candidate_matches],
                })
                continue

            # Step 4: Was it structural-filtered?
            non_structural = [c for c in grounded_matches if not c.get("structural_filtered")]
            structural_only = [c for c in grounded_matches if c.get("structural_filtered")]

            if not non_structural:
                stage_counts["STRUCTURAL_FILTERED"] += 1
                classifications.append({
                    "chunk_id": cid, "gt_term": gt_term, "tier": tier,
                    "stage": "STRUCTURAL_FILTERED",
                    "detail": f"Extracted as {[c['term'] for c in grounded_matches]} but structural-filtered",
                    "candidate_terms": [c["term"] for c in grounded_matches],
                })
                continue

            # Step 5: Check routing — was any match auto-kept or sent to review?
            auto_kept_matches = [
                c for c in non_structural
                if c.get("routing", "").startswith("auto_keep")
            ]
            review_matches = [
                c for c in non_structural
                if c.get("routing") == "sonnet_review"
            ]

            if auto_kept_matches:
                # At least one match was auto-kept (2+ votes)
                stage_counts["AUTO_KEPT"] += 1
                classifications.append({
                    "chunk_id": cid, "gt_term": gt_term, "tier": tier,
                    "stage": "AUTO_KEPT",
                    "detail": f"Auto-kept via {[c['term'] for c in auto_kept_matches]}",
                    "vote_counts": [c["vote_count"] for c in auto_kept_matches],
                })
                continue

            if review_matches:
                # Only 1-vote matches — goes to Sonnet review
                stage_counts["SONNET_REVIEW"] += 1
                classifications.append({
                    "chunk_id": cid, "gt_term": gt_term, "tier": tier,
                    "stage": "SONNET_REVIEW",
                    "detail": f"1-vote, sent to Sonnet review: {[c['term'] for c in review_matches]}",
                    "candidate_terms": [c["term"] for c in review_matches],
                })
                continue

            # Shouldn't reach here, but catch-all
            stage_counts["UNKNOWN"] += 1
            classifications.append({
                "chunk_id": cid, "gt_term": gt_term, "tier": tier,
                "stage": "UNKNOWN",
                "detail": f"Unclassified: matches={[c['term'] for c in non_structural]}",
            })

    return {
        "stage_counts": dict(stage_counts),
        "classifications": classifications,
    }


# ============================================================================
# STAGE 3: VARIANT-LEVEL ANALYSIS (using sweep results)
# ============================================================================


def analyze_variant_fns(
    gt_data: dict, sweep_results: dict, variant_name: str
) -> dict:
    """For a specific variant, classify each FN (m2o_v3) by root cause."""
    variant_data = sweep_results["results"].get(variant_name)
    if not variant_data:
        return {"error": f"Variant {variant_name} not found in sweep results"}

    gt_by_chunk = {
        c["chunk_id"]: [t["term"] for t in c["terms"]]
        for c in gt_data["chunks"]
    }

    per_chunk = variant_data["per_chunk_results"]

    # Recompute FN lists with m2o_v3
    all_fns: list[dict] = []
    all_fps: list[dict] = []

    for cr in per_chunk:
        cid = cr["chunk_id"]
        extracted = cr["extracted_terms"]
        gt_terms = gt_by_chunk.get(cid, [])

        # m2o scoring
        covered_gt: set[int] = set()
        unmatched: list[str] = []

        for ext in extracted:
            found_any = False
            for j, gt in enumerate(gt_terms):
                if v3_match(ext, gt):
                    covered_gt.add(j)
                    found_any = True
                    break
            if not found_any:
                unmatched.append(ext)

        fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in covered_gt]

        for fn in fn_terms:
            all_fns.append({"chunk_id": cid, "gt_term": fn})
        for fp in unmatched:
            all_fps.append({"chunk_id": cid, "extracted_term": fp})

    return {
        "variant": variant_name,
        "total_fns": len(all_fns),
        "total_fps": len(all_fps),
        "fn_list": all_fns,
        "fp_list": all_fps,
    }


def compare_greedy_vs_m2o(
    gt_data: dict, sweep_results: dict, variant_name: str
) -> dict:
    """Compare FN lists between greedy_v3 and m2o_v3 for a variant."""
    variant_data = sweep_results["results"].get(variant_name)
    if not variant_data:
        return {"error": f"Variant {variant_name} not found"}

    gt_by_chunk = {
        c["chunk_id"]: [t["term"] for t in c["terms"]]
        for c in gt_data["chunks"]
    }

    greedy_only_fns: list[dict] = []  # FN in greedy but NOT in m2o
    m2o_only_fns: list[dict] = []     # FN in m2o but NOT in greedy
    both_fns: list[dict] = []         # FN in both

    for cr in variant_data["per_chunk_results"]:
        cid = cr["chunk_id"]
        extracted = cr["extracted_terms"]
        gt_terms = gt_by_chunk.get(cid, [])

        # Greedy scoring
        g_matched_gt: set[int] = set()
        for ext in extracted:
            for j, gt in enumerate(gt_terms):
                if j in g_matched_gt:
                    continue
                if v3_match(ext, gt):
                    g_matched_gt.add(j)
                    break
        greedy_fn_idx = set(range(len(gt_terms))) - g_matched_gt

        # M2O scoring
        m_covered_gt: set[int] = set()
        for ext in extracted:
            for j, gt in enumerate(gt_terms):
                if v3_match(ext, gt):
                    m_covered_gt.add(j)
                    break
        m2o_fn_idx = set(range(len(gt_terms))) - m_covered_gt

        for j in range(len(gt_terms)):
            in_greedy = j in greedy_fn_idx
            in_m2o = j in m2o_fn_idx
            if in_greedy and in_m2o:
                both_fns.append({"chunk_id": cid, "gt_term": gt_terms[j]})
            elif in_greedy and not in_m2o:
                # FN in greedy but covered in m2o — means multiple extractions
                # of same term, one consumed by greedy but m2o counted coverage
                greedy_only_fns.append({"chunk_id": cid, "gt_term": gt_terms[j]})
            elif in_m2o and not in_greedy:
                # FN in m2o but NOT in greedy — a greedy collision artifact
                # where a different extraction consumed this GT slot in greedy
                m2o_only_fns.append({"chunk_id": cid, "gt_term": gt_terms[j]})

    return {
        "variant": variant_name,
        "greedy_v3_fn_count": len(greedy_only_fns) + len(both_fns),
        "m2o_v3_fn_count": len(m2o_only_fns) + len(both_fns),
        "both_fn": len(both_fns),
        "greedy_only_fn": len(greedy_only_fns),
        "m2o_only_fn": len(m2o_only_fns),
        "both_fn_list": both_fns,
        "greedy_only_fn_list": greedy_only_fns,
        "m2o_only_fn_list": m2o_only_fns,
    }


# ============================================================================
# CROSS-REFERENCE: merge pipeline stage + variant FNs
# ============================================================================


def cross_reference_fns(
    classifications: list[dict],
    variant_fns: list[dict],
) -> dict:
    """For each variant FN, look up its pipeline stage classification."""
    # Build lookup: (chunk_id, gt_term_normalized) -> classification
    class_lookup: dict[tuple[str, str], dict] = {}
    for c in classifications:
        key = (c["chunk_id"], normalize_term(c["gt_term"]))
        class_lookup[key] = c

    fn_by_stage = Counter()
    fn_details: list[dict] = []

    for fn in variant_fns:
        key = (fn["chunk_id"], normalize_term(fn["gt_term"]))
        cls = class_lookup.get(key, {})
        stage = cls.get("stage", "UNKNOWN")
        fn_by_stage[stage] += 1
        fn_details.append({
            "chunk_id": fn["chunk_id"],
            "gt_term": fn["gt_term"],
            "stage": stage,
            "detail": cls.get("detail", "No classification found"),
        })

    return {
        "fn_by_stage": dict(fn_by_stage),
        "fn_details": fn_details,
    }


# ============================================================================
# EXTRACTOR CONTRIBUTION ANALYSIS
# ============================================================================


def analyze_extractor_contributions(
    gt_data: dict, cache: dict
) -> dict:
    """Analyze each extractor's unique contribution to GT coverage."""
    cache_by_id = {c["chunk_id"]: c for c in cache["chunks"]}

    extractor_names = ["sonnet_exhaustive", "haiku_exhaustive", "haiku_simple"]
    unique_coverage: dict[str, int] = {e: 0 for e in extractor_names}
    total_coverage: dict[str, int] = {e: 0 for e in extractor_names}
    union_coverage = 0
    total_gt = 0

    for chunk in gt_data["chunks"]:
        cid = chunk["chunk_id"]
        gt_terms = [t["term"] for t in chunk["terms"]]
        ccache = cache_by_id.get(cid)
        if not ccache:
            continue

        raw = ccache["raw_extractions"]
        total_gt += len(gt_terms)

        for gt_term in gt_terms:
            covered_by: set[str] = set()
            for ext_name in extractor_names:
                ext_terms = raw.get(ext_name, [])
                if find_matching_raw(gt_term, ext_terms):
                    covered_by.add(ext_name)
                    total_coverage[ext_name] += 1

            if covered_by:
                union_coverage += 1
                if len(covered_by) == 1:
                    unique_coverage[next(iter(covered_by))] += 1

    return {
        "total_gt": total_gt,
        "union_coverage": union_coverage,
        "union_recall": union_coverage / total_gt if total_gt else 0,
        "per_extractor": {
            e: {
                "total_matches": total_coverage[e],
                "unique_matches": unique_coverage[e],
                "recall": total_coverage[e] / total_gt if total_gt else 0,
            }
            for e in extractor_names
        },
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("RECALL CEILING ANALYSIS")
    print("=" * 80)

    # Load data
    with open(GT_V2_PATH) as f:
        gt_data = json.load(f)
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    sweep_results = None
    if SWEEP_RESULTS_PATH.exists():
        with open(SWEEP_RESULTS_PATH) as f:
            sweep_results = json.load(f)

    total_gt = sum(c["term_count"] for c in gt_data["chunks"])
    print(f"\nTotal GT terms: {total_gt} across {len(gt_data['chunks'])} chunks")

    # ── EXTRACTION CEILING ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 1: EXTRACTION CEILING (raw extractions, before any filtering)")
    print("=" * 80)

    ext_ceiling = analyze_extraction_ceiling(gt_data, cache)
    print(f"\n  Total GT terms:      {ext_ceiling['total_gt']}")
    print(f"  Extracted (matched): {ext_ceiling['extracted']}")
    print(f"  Never extracted:     {ext_ceiling['never_extracted_count']}")
    print(f"  EXTRACTION CEILING:  {ext_ceiling['extraction_ceiling']:.1%}")

    if ext_ceiling["never_extracted"]:
        print(f"\n  --- Never-extracted GT terms ({ext_ceiling['never_extracted_count']}) ---")
        by_chunk = defaultdict(list)
        for ne in ext_ceiling["never_extracted"]:
            by_chunk[ne["chunk_id"]].append(ne)
        for cid, terms in sorted(by_chunk.items()):
            for t in terms:
                tier_str = f" (tier {t['tier']})" if t["tier"] else ""
                print(f"    [{cid}] '{t['gt_term']}'{tier_str}")

    # ── EXTRACTOR CONTRIBUTIONS ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 2: PER-EXTRACTOR CONTRIBUTION")
    print("=" * 80)

    ext_contrib = analyze_extractor_contributions(gt_data, cache)
    print(f"\n  Union recall (any extractor): {ext_contrib['union_recall']:.1%} ({ext_contrib['union_coverage']}/{ext_contrib['total_gt']})")
    for ename, edata in ext_contrib["per_extractor"].items():
        print(
            f"  {ename:25s}: recall={edata['recall']:.1%} "
            f"({edata['total_matches']}/{ext_contrib['total_gt']}), "
            f"unique={edata['unique_matches']}"
        )

    # ── PIPELINE STAGE CLASSIFICATION ───────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 3: PIPELINE STAGE CLASSIFICATION (where GT terms are lost)")
    print("=" * 80)

    pipeline = classify_gt_terms(gt_data, cache)
    counts = pipeline["stage_counts"]

    print(f"\n  Pipeline stage breakdown (all {total_gt} GT terms):")
    stage_order = [
        "NEVER_EXTRACTED", "UNGROUNDED", "STRUCTURAL_FILTERED",
        "SONNET_REVIEW", "AUTO_KEPT", "UNKNOWN",
    ]
    for stage in stage_order:
        c = counts.get(stage, 0)
        pct = c / total_gt * 100 if total_gt else 0
        print(f"    {stage:25s}: {c:4d}  ({pct:5.1f}%)")

    # Cumulative ceiling at each stage
    print(f"\n  Cumulative recall ceiling at each stage:")
    cum = total_gt
    cum -= counts.get("NEVER_EXTRACTED", 0)
    print(f"    After extraction:     {cum}/{total_gt} = {cum/total_gt:.1%}")
    cum -= counts.get("UNGROUNDED", 0)
    print(f"    After grounding:      {cum}/{total_gt} = {cum/total_gt:.1%}")
    cum -= counts.get("STRUCTURAL_FILTERED", 0)
    print(f"    After structural:     {cum}/{total_gt} = {cum/total_gt:.1%}")

    # AUTO_KEPT are guaranteed to survive; SONNET_REVIEW depends on variant
    auto_kept = counts.get("AUTO_KEPT", 0)
    sonnet_review = counts.get("SONNET_REVIEW", 0)
    print(f"    Auto-kept (2+ vote):  {auto_kept}/{total_gt} = {auto_kept/total_gt:.1%}")
    print(f"    Sonnet review (1v):   {sonnet_review}/{total_gt} = {sonnet_review/total_gt:.1%}")
    print(f"    MAX possible recall:  {(auto_kept + sonnet_review)}/{total_gt} = {(auto_kept + sonnet_review)/total_gt:.1%}")
    print(f"    (if Sonnet approves ALL 1-vote GT terms)")

    # Show NEVER_EXTRACTED details by tier
    never_ext = [c for c in pipeline["classifications"] if c["stage"] == "NEVER_EXTRACTED"]
    if never_ext:
        tier_counts = Counter(c.get("tier") for c in never_ext)
        print(f"\n  Never-extracted by tier:")
        for tier in sorted(tier_counts.keys(), key=lambda x: x or 99):
            print(f"    Tier {tier}: {tier_counts[tier]}")

    # Show UNGROUNDED details
    ungrounded = [c for c in pipeline["classifications"] if c["stage"] == "UNGROUNDED"]
    if ungrounded:
        print(f"\n  --- Ungrounded GT terms ({len(ungrounded)}) ---")
        for u in ungrounded:
            print(f"    [{u['chunk_id']}] '{u['gt_term']}' — {u['detail']}")

    # Show STRUCTURAL_FILTERED details
    structural = [c for c in pipeline["classifications"] if c["stage"] == "STRUCTURAL_FILTERED"]
    if structural:
        print(f"\n  --- Structural-filtered GT terms ({len(structural)}) ---")
        for s in structural:
            print(f"    [{s['chunk_id']}] '{s['gt_term']}' — {s['detail']}")

    # ── VARIANT FN ANALYSIS ─────────────────────────────────────────────
    if sweep_results:
        for vname in ["V_B", "V_BASELINE"]:
            print(f"\n{'=' * 80}")
            print(f"SECTION 4: VARIANT {vname} — FN ROOT CAUSE ANALYSIS (m2o_v3)")
            print("=" * 80)

            vfn = analyze_variant_fns(gt_data, sweep_results, vname)
            if "error" in vfn:
                print(f"  {vfn['error']}")
                continue

            print(f"\n  Total FPs: {vfn['total_fps']}")
            print(f"  Total FNs: {vfn['total_fns']}")

            # Cross-reference FNs with pipeline stages
            xref = cross_reference_fns(pipeline["classifications"], vfn["fn_list"])
            print(f"\n  FN breakdown by pipeline stage:")
            for stage in stage_order:
                c = xref["fn_by_stage"].get(stage, 0)
                pct = c / vfn["total_fns"] * 100 if vfn["total_fns"] else 0
                if c > 0:
                    print(f"    {stage:25s}: {c:4d}  ({pct:5.1f}%)")

            # Show FN details grouped by stage
            for stage in stage_order:
                stage_fns = [f for f in xref["fn_details"] if f["stage"] == stage]
                if stage_fns:
                    print(f"\n    --- {stage} ({len(stage_fns)}) ---")
                    for f in stage_fns:
                        print(f"      [{f['chunk_id']}] '{f['gt_term']}' — {f['detail']}")

            # FP details
            if vfn["fp_list"]:
                print(f"\n    --- FP terms ({vfn['total_fps']}) ---")
                for fp in vfn["fp_list"]:
                    print(f"      [{fp['chunk_id']}] '{fp['extracted_term']}'")

    # ── GREEDY vs M2O COMPARISON ────────────────────────────────────────
    if sweep_results:
        print(f"\n{'=' * 80}")
        print("SECTION 5: GREEDY_V3 vs M2O_V3 FN COMPARISON (V_B)")
        print("=" * 80)

        cmp = compare_greedy_vs_m2o(gt_data, sweep_results, "V_B")
        if "error" not in cmp:
            print(f"\n  greedy_v3 FN count: {cmp['greedy_v3_fn_count']}")
            print(f"  m2o_v3 FN count:    {cmp['m2o_v3_fn_count']}")
            print(f"  FN in BOTH:         {cmp['both_fn']}")
            print(f"  FN only in greedy:  {cmp['greedy_only_fn']} (these are 'covered' by m2o but missed by greedy)")
            print(f"  FN only in m2o:     {cmp['m2o_only_fn']} (these are 'covered' by greedy via collision)")

            if cmp["m2o_only_fn_list"]:
                print(f"\n  --- m2o-only FNs (covered by greedy collision, NOT real coverage) ---")
                for fn in cmp["m2o_only_fn_list"]:
                    print(f"    [{fn['chunk_id']}] '{fn['gt_term']}'")

            if cmp["greedy_only_fn_list"]:
                print(f"\n  --- greedy-only FNs (m2o covers them via multi-extraction) ---")
                for fn in cmp["greedy_only_fn_list"]:
                    print(f"    [{fn['chunk_id']}] '{fn['gt_term']}'")

    # ── SAVE RESULTS ────────────────────────────────────────────────────
    output = {
        "extraction_ceiling": {
            "total_gt": ext_ceiling["total_gt"],
            "extracted": ext_ceiling["extracted"],
            "never_extracted_count": ext_ceiling["never_extracted_count"],
            "ceiling_pct": ext_ceiling["extraction_ceiling"],
            "never_extracted": ext_ceiling["never_extracted"],
        },
        "extractor_contributions": ext_contrib,
        "pipeline_stages": {
            "stage_counts": pipeline["stage_counts"],
            "classifications": pipeline["classifications"],
        },
    }

    if sweep_results:
        for vname in ["V_B", "V_BASELINE"]:
            vfn = analyze_variant_fns(gt_data, sweep_results, vname)
            if "error" not in vfn:
                xref = cross_reference_fns(pipeline["classifications"], vfn["fn_list"])
                output[f"variant_{vname}_analysis"] = {
                    "total_fns": vfn["total_fns"],
                    "total_fps": vfn["total_fps"],
                    "fn_by_stage": xref["fn_by_stage"],
                    "fn_details": xref["fn_details"],
                    "fp_list": vfn["fp_list"],
                }

        cmp = compare_greedy_vs_m2o(gt_data, sweep_results, "V_B")
        if "error" not in cmp:
            output["greedy_vs_m2o_V_B"] = cmp

    out_path = ARTIFACTS_DIR / "recall_ceiling_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
