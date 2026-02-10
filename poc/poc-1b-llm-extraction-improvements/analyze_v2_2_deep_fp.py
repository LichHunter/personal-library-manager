#!/usr/bin/env python3
"""Deep analysis of D+v2.2 (with improved matching) remaining 48 FPs.

For each FP, outputs:
  - The term, chunk, vote count, source extractors
  - Sonnet reasoning (if 1-vote)
  - All GT terms for that chunk
  - Nearest GT match and score
  - WHY it's FP (classification)
"""

import json
import re
from pathlib import Path
from collections import Counter
from rapidfuzz import fuzz

AUDIT_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_2_audit.json"
RESULTS_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_2_results.json"
GT_PATH = Path(__file__).parent / "artifacts" / "small_chunk_ground_truth_v2.json"
RESCORE_PATH = Path(__file__).parent / "artifacts" / "rescore_results.json"

with open(AUDIT_PATH) as f:
    audit_data = json.load(f)
with open(RESULTS_PATH) as f:
    results_data = json.load(f)
with open(GT_PATH) as f:
    gt_data = json.load(f)
with open(RESCORE_PATH) as f:
    rescore_data = json.load(f)

def normalize_term(term):
    return term.lower().strip().replace("-", " ").replace("_", " ")

def depluralize(s):
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y"
    if s.endswith("es") and len(s) > 4:
        return s[:-2]
    if s.endswith("s") and len(s) > 3:
        return s[:-1]
    return s

def camel_to_words(s):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower().strip()

def improved_match(extracted, ground_truth):
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
    ext_camel = camel_to_words(extracted)
    gt_camel = camel_to_words(ground_truth)
    if normalize_term(ext_camel) == normalize_term(gt_camel):
        return True
    if depluralize(normalize_term(ext_camel)) == depluralize(normalize_term(gt_camel)):
        return True
    if len(ext_norm) >= 4 and len(gt_norm) >= 4:
        if fuzz.partial_ratio(ext_norm, gt_norm) >= 90:
            shorter = min(ext_norm, gt_norm, key=len)
            longer = max(ext_norm, gt_norm, key=len)
            if len(shorter) / len(longer) >= 0.5:
                return True
    return False

# Build GT lookup
gt_by_chunk = {}
gt_tiers_by_chunk = {}
for chunk in gt_data["chunks"]:
    gt_by_chunk[chunk["chunk_id"]] = [t["term"] for t in chunk["terms"]]
    gt_tiers_by_chunk[chunk["chunk_id"]] = {t["term"]: t["tier"] for t in chunk["terms"]}

# Get the FPs from improved-match rescore (Config C)
# We need to re-run the improved matching ourselves to get the FP list
# since rescore_results.json stripped fp_terms for serialization

audits = audit_data["audits"]

# Re-run improved matching per chunk
all_fps = []

for chunk_result in results_data["per_chunk_results"]:
    chunk_id = chunk_result["chunk_id"]
    extracted = chunk_result["extracted_terms"]
    gt_terms = gt_by_chunk.get(chunk_id, [])
    
    # Improved match scoring
    matched_gt = set()
    tp_pairs = []
    fp_terms_list = []
    
    for ext in extracted:
        found = False
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if improved_match(ext, gt):
                matched_gt.add(j)
                tp_pairs.append((ext, gt))
                found = True
                break
        if not found:
            fp_terms_list.append(ext)
    
    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in matched_gt]
    
    for fp_term in fp_terms_list:
        # Find audit entry
        fp_audit = next(
            (a for a in audits if a["chunk_id"] == chunk_id and a["term"] == fp_term and a["final_status"] == "KEPT"),
            None
        )
        
        # Find nearest GT match
        best_gt = None
        best_score = 0
        for gt in gt_terms:
            score = fuzz.ratio(normalize_term(fp_term), normalize_term(gt))
            # Also check containment
            fp_n = normalize_term(fp_term)
            gt_n = normalize_term(gt)
            if fp_n in gt_n or gt_n in fp_n:
                score = max(score, 75)
            if depluralize(fp_n) == depluralize(gt_n):
                score = max(score, 95)
            if score > best_score:
                best_score = score
                best_gt = gt
        
        # Check if this FP matches ANY already-matched GT (greedy collision)
        is_greedy_collision = False
        colliding_gt = None
        for ext_matched, gt_matched in tp_pairs:
            if improved_match(fp_term, gt_matched):
                is_greedy_collision = True
                colliding_gt = gt_matched
                break
        
        all_fps.append({
            "term": fp_term,
            "chunk_id": chunk_id,
            "votes": fp_audit["vote_count"] if fp_audit else "?",
            "sources": fp_audit["sources"] if fp_audit else [],
            "routing": fp_audit["routing"] if fp_audit else "?",
            "sonnet_decision": fp_audit["sonnet_decision"] if fp_audit else "?",
            "sonnet_reasoning": fp_audit["sonnet_reasoning"] if fp_audit else "",
            "nearest_gt": best_gt,
            "nearest_gt_score": best_score,
            "is_greedy_collision": is_greedy_collision,
            "colliding_gt": colliding_gt,
            "gt_terms": gt_terms,
        })

# ── CLASSIFICATION ─────────────────────────────────────────────────────
print("=" * 100)
print(f"D+v2.2 DEEP FP ANALYSIS (improved matching) — {len(all_fps)} FPs")
print("=" * 100)

# Classify each FP
categories = Counter()
categorized = {}

for fp in all_fps:
    term = fp["term"]
    t_norm = normalize_term(term)
    
    if fp["is_greedy_collision"]:
        cat = "GREEDY_COLLISION"
    elif fp["nearest_gt_score"] >= 75:
        cat = "NEAR_MISS_MATCH"
    elif t_norm in {"dchen1107", "liggitt", "v1.11"}:
        cat = "METADATA_NOISE"
    elif len(t_norm.split()) == 1 and len(t_norm) <= 4:
        cat = "SHORT_AMBIGUOUS"
    elif len(t_norm.split()) >= 3:
        cat = "OVER_SPECIFIC_PHRASE"
    elif fp["votes"] == 1:
        cat = "SONNET_OVER_APPROVE"
    else:
        cat = "MULTI_VOTE_NOISE"
    
    categories[cat] += 1
    fp["category"] = cat

print(f"\n{'Category':<25s} {'Count':>5s} {'%':>6s}")
print("-" * 40)
for cat, count in categories.most_common():
    print(f"  {cat:<23s} {count:>5d} {count/len(all_fps):>5.0%}")

# ── DETAILED BY CATEGORY ──────────────────────────────────────────────
for cat_name in ["GREEDY_COLLISION", "NEAR_MISS_MATCH", "SONNET_OVER_APPROVE", 
                  "MULTI_VOTE_NOISE", "METADATA_NOISE", "SHORT_AMBIGUOUS", "OVER_SPECIFIC_PHRASE"]:
    cat_fps = [fp for fp in all_fps if fp["category"] == cat_name]
    if not cat_fps:
        continue
    
    print(f"\n{'=' * 100}")
    print(f"{cat_name} ({len(cat_fps)} terms)")
    print("=" * 100)
    
    for fp in sorted(cat_fps, key=lambda x: (-x["votes"] if isinstance(x["votes"], int) else 0, x["chunk_id"])):
        collision_info = f" COLLIDES WITH GT:'{fp['colliding_gt']}'" if fp["is_greedy_collision"] else ""
        sonnet_info = f" Sonnet: {fp['sonnet_reasoning'][:60]}" if fp["votes"] == 1 and fp["sonnet_reasoning"] else ""
        print(f"  [{fp['votes']}v {fp['routing'][:10]:>10s}] '{fp['term']}' ({fp['chunk_id']})")
        print(f"    nearest GT: '{fp['nearest_gt']}' (score={fp['nearest_gt_score']:.0f}){collision_info}{sonnet_info}")

# ── ACTIONABILITY ANALYSIS ────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("ACTIONABILITY SUMMARY")
print("=" * 100)

greedy = sum(1 for fp in all_fps if fp["is_greedy_collision"])
near_miss = sum(1 for fp in all_fps if not fp["is_greedy_collision"] and fp["nearest_gt_score"] >= 75)
true_noise = sum(1 for fp in all_fps if not fp["is_greedy_collision"] and fp["nearest_gt_score"] < 75)

print(f"\n  Greedy collisions (same term matched by another extraction): {greedy}")
print(f"    → Fix: Better post-hoc dedup or many-to-one matching")
print(f"  Near-miss matches (score ≥75 but improved matcher missed): {near_miss}")
print(f"    → Fix: Relax matching further or improve the matcher")
print(f"  True noise (genuinely wrong terms): {true_noise}")
print(f"    → Fix: Better extraction/filtering")
print(f"\n  If greedy collisions were resolved: P ≈ {(250+greedy)/(298):.1%}")
print(f"  If greedy + near-miss resolved: P ≈ {(250+greedy+near_miss)/(298):.1%}")
print(f"  Hard floor (only true noise as FP): P ≈ {(298-true_noise)/(298):.1%}")

# ── FP BY VOTE COUNT ──────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("FP BREAKDOWN BY VOTE COUNT")
print("=" * 100)

for votes in [3, 2, 1]:
    v_fps = [fp for fp in all_fps if fp["votes"] == votes]
    v_greedy = [fp for fp in v_fps if fp["is_greedy_collision"]]
    v_near = [fp for fp in v_fps if not fp["is_greedy_collision"] and fp["nearest_gt_score"] >= 75]
    v_true = [fp for fp in v_fps if not fp["is_greedy_collision"] and fp["nearest_gt_score"] < 75]
    
    if v_fps:
        print(f"\n  {votes}-vote FPs: {len(v_fps)} total")
        print(f"    Greedy collisions: {len(v_greedy)}")
        print(f"    Near-miss matches: {len(v_near)}")
        print(f"    True noise: {len(v_true)}")
        if v_true:
            for fp in v_true:
                print(f"      '{fp['term']}' ({fp['chunk_id']})")

# ── FN ANALYSIS (for completeness) ───────────────────────────────────
print(f"\n{'=' * 100}")
print("FALSE NEGATIVES — WHERE RECALL IS LOST")
print("=" * 100)

all_fns = []
for chunk_result in results_data["per_chunk_results"]:
    chunk_id = chunk_result["chunk_id"]
    extracted = chunk_result["extracted_terms"]
    gt_terms = gt_by_chunk.get(chunk_id, [])
    
    matched_gt = set()
    for ext in extracted:
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if improved_match(ext, gt):
                matched_gt.add(j)
                break
    
    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in matched_gt]
    
    for fn in fn_terms:
        tier = gt_tiers_by_chunk.get(chunk_id, {}).get(fn, "?")
        # Check if extracted but rejected
        fn_audit = next(
            (a for a in audits 
             if a["chunk_id"] == chunk_id 
             and (normalize_term(a["term"]) == normalize_term(fn) or improved_match(a["term"], fn))
             and a["final_status"] != "KEPT"),
            None
        )
        
        if fn_audit:
            cause = fn_audit["final_status"]
            fn_votes = fn_audit["vote_count"]
        else:
            cause = "NEVER_EXTRACTED"
            fn_votes = 0
        
        all_fns.append({
            "term": fn,
            "chunk_id": chunk_id,
            "tier": tier,
            "cause": cause,
            "votes": fn_votes,
        })

fn_by_cause = Counter(fn["cause"] for fn in all_fns)
print(f"\n  Total FNs: {len(all_fns)}")
for cause, count in fn_by_cause.most_common():
    print(f"  {cause}: {count}")

print(f"\nFN details:")
for fn in sorted(all_fns, key=lambda x: (x["cause"], x["chunk_id"])):
    print(f"  [{fn['cause']:20s}] '{fn['term']}' (T{fn['tier']}, {fn['chunk_id']})")
