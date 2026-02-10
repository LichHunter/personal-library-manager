#!/usr/bin/env python3
"""Analyze matching failures — FPs that might actually be TPs with matching bugs."""

import json
from pathlib import Path
from rapidfuzz import fuzz

AUDIT_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_audit.json"
RESULTS_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_results.json"
GT_PATH = Path(__file__).parent / "artifacts" / "small_chunk_ground_truth_v2.json"

with open(AUDIT_PATH) as f:
    audit_data = json.load(f)
with open(RESULTS_PATH) as f:
    results_data = json.load(f)
with open(GT_PATH) as f:
    gt_data = json.load(f)

def normalize_term(term):
    return term.lower().strip().replace("-", " ").replace("_", " ")

def match_terms_fn(extracted, ground_truth):
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

# Build GT lookup by chunk
gt_by_chunk = {}
for chunk in gt_data["chunks"]:
    gt_by_chunk[chunk["chunk_id"]] = [t["term"] for t in chunk["terms"]]

# For each FP term, check if it SHOULD have matched any GT term in that chunk
# but didn't due to matching function limitations
audits = audit_data["audits"]
fp_audits = [a for a in audits if a["final_status"] == "KEPT" and not a["matched_gt"]]

print("=" * 80)
print("MATCHING ANALYSIS: FPs that might actually be TPs")
print("=" * 80)

potential_matches = []
true_fps = []

for a in fp_audits:
    chunk_id = a["chunk_id"]
    term = a["term"]
    gt_terms = gt_by_chunk.get(chunk_id, [])
    
    # Check against ALL GT terms (not just unmatched ones)
    best_match = None
    best_score = 0
    for gt in gt_terms:
        score = fuzz.ratio(normalize_term(term), normalize_term(gt))
        # Also check containment
        t_norm = normalize_term(term)
        g_norm = normalize_term(gt)
        
        # Check if term is a substring of a GT term or vice versa
        is_substring = t_norm in g_norm or g_norm in t_norm
        # Check singular/plural more aggressively
        is_plural_match = (
            (t_norm + "s" == g_norm) or 
            (g_norm + "s" == t_norm) or
            (t_norm.rstrip("s") == g_norm.rstrip("s") and len(t_norm) > 2)
        )
        
        effective_score = score
        if is_substring:
            effective_score = max(effective_score, 75)
        if is_plural_match:
            effective_score = max(effective_score, 90)
            
        if effective_score > best_score:
            best_score = effective_score
            best_match = gt
    
    if best_score >= 65:
        potential_matches.append({
            "term": term,
            "chunk_id": chunk_id,
            "best_gt_match": best_match,
            "score": best_score,
            "votes": a["vote_count"],
            "sources": a["sources"],
        })
    else:
        true_fps.append({
            "term": term,
            "chunk_id": chunk_id,
            "best_gt_match": best_match,
            "best_score": best_score,
            "votes": a["vote_count"],
            "sources": a["sources"],
        })

print(f"\nTotal FPs: {len(fp_audits)}")
print(f"Potential matching failures (score >= 65): {len(potential_matches)}")
print(f"True FPs (score < 65): {len(true_fps)}")

print(f"\n{'=' * 80}")
print("POTENTIAL MATCHING FAILURES (these might actually be TPs)")
print("=" * 80)

for pm in sorted(potential_matches, key=lambda x: -x["score"]):
    print(f"  '{pm['term']}' ↔ '{pm['best_gt_match']}' (score={pm['score']}, votes={pm['votes']}, chunk={pm['chunk_id']})")

print(f"\n{'=' * 80}")
print("TRUE FPs (no close GT match)")
print("=" * 80)

# Group by vote count
from collections import Counter
true_fp_by_vote = Counter(fp["votes"] for fp in true_fps)
print(f"\n  By vote count: {dict(sorted(true_fp_by_vote.items()))}")

for fp in sorted(true_fps, key=lambda x: (-x["votes"], x["chunk_id"])):
    gt_info = f" (nearest GT: '{fp['best_gt_match']}' score={fp['best_score']})" if fp["best_gt_match"] else ""
    print(f"  [{fp['votes']}v] '{fp['term']}' ({fp['chunk_id']}){gt_info}")

# Check how many FPs are near-duplicates of ALREADY-MATCHED GT terms
print(f"\n{'=' * 80}")
print("DUPLICATE ANALYSIS: FPs that are variants of already-matched terms")
print("=" * 80)

# Get all TP matches
tp_audits = [a for a in audits if a["final_status"] == "KEPT" and a["matched_gt"]]
tp_gt_terms_by_chunk = {}
for a in tp_audits:
    chunk = a["chunk_id"]
    if chunk not in tp_gt_terms_by_chunk:
        tp_gt_terms_by_chunk[chunk] = set()
    tp_gt_terms_by_chunk[chunk].add(normalize_term(a["matched_gt"]))

duplicate_fps = []
for fp in fp_audits:
    chunk = fp["chunk_id"]
    term = fp["term"]
    t_norm = normalize_term(term)
    matched_gts = tp_gt_terms_by_chunk.get(chunk, set())
    
    # Check if this FP is a variant of an already-matched GT term
    for matched_gt_norm in matched_gts:
        score = fuzz.ratio(t_norm, matched_gt_norm)
        is_substring = t_norm in matched_gt_norm or matched_gt_norm in t_norm
        is_plural = t_norm.rstrip("s") == matched_gt_norm.rstrip("s")
        
        if score >= 70 or is_substring or is_plural:
            duplicate_fps.append({
                "fp_term": term,
                "already_matched_gt": matched_gt_norm,
                "score": score,
                "chunk_id": chunk,
                "votes": fp["vote_count"],
            })
            break

print(f"\n  FPs that are variants of already-matched GT terms: {len(duplicate_fps)}")
for d in sorted(duplicate_fps, key=lambda x: -x["score"]):
    print(f"    '{d['fp_term']}' ≈ GT:'{d['already_matched_gt']}' (score={d['score']}, {d['votes']}v)")
