#!/usr/bin/env python3
"""Classify D+v2.2 FPs into true FPs vs GT-variant FPs (matching artifacts)."""

import json
from pathlib import Path
from rapidfuzz import fuzz

AUDIT_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_2_audit.json"
RESULTS_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_2_results.json"
GT_PATH = Path(__file__).parent / "artifacts" / "small_chunk_ground_truth_v2.json"

with open(AUDIT_PATH) as f:
    audit_data = json.load(f)
with open(RESULTS_PATH) as f:
    results_data = json.load(f)
with open(GT_PATH) as f:
    gt_data = json.load(f)

def normalize_term(term):
    return term.lower().strip().replace("-", " ").replace("_", " ")

# GT lookup by chunk
gt_by_chunk = {}
for chunk in gt_data["chunks"]:
    gt_by_chunk[chunk["chunk_id"]] = [t["term"] for t in chunk["terms"]]

# Classify each FP
true_fps = []
variant_fps = []  # FPs that are variants of GT terms (matching artifacts)

for chunk_result in results_data["per_chunk_results"]:
    chunk_id = chunk_result["chunk_id"]
    gt_terms = gt_by_chunk.get(chunk_id, [])
    
    for fp_term in chunk_result["fp_terms"]:
        fp_norm = normalize_term(fp_term)
        
        # Check against ALL GT terms (not just unmatched ones)
        best_match = None
        best_score = 0
        
        for gt in gt_terms:
            gt_norm = normalize_term(gt)
            
            # Exact match
            if fp_norm == gt_norm:
                best_match = gt
                best_score = 100
                break
            
            # Fuzzy
            score = fuzz.ratio(fp_norm, gt_norm)
            
            # Singular/plural
            if fp_norm.rstrip("s") == gt_norm.rstrip("s"):
                score = max(score, 95)
            
            # Substring (for compound terms)
            if fp_norm in gt_norm and len(fp_norm) >= 3:
                score = max(score, 80)
            if gt_norm in fp_norm and len(gt_norm) >= 3:
                score = max(score, 80)
            
            if score > best_score:
                best_score = score
                best_match = gt
        
        fp_audit = next(
            (a for a in audit_data["audits"] 
             if a["chunk_id"] == chunk_id and a["term"] == fp_term and a["final_status"] == "KEPT"),
            None
        )
        votes = fp_audit["vote_count"] if fp_audit else "?"
        
        entry = {
            "term": fp_term,
            "chunk_id": chunk_id,
            "votes": votes,
            "best_gt": best_match,
            "score": best_score,
        }
        
        if best_score >= 75:
            variant_fps.append(entry)
        else:
            true_fps.append(entry)

print("=" * 80)
print("D+v2.2 FP CLASSIFICATION")
print("=" * 80)
print(f"\nTotal FPs: {len(true_fps) + len(variant_fps)}")
print(f"  GT-variant FPs (matching artifacts, score>=75): {len(variant_fps)} ({len(variant_fps)/(len(true_fps)+len(variant_fps)):.0%})")
print(f"  True FPs (no GT match, score<75): {len(true_fps)} ({len(true_fps)/(len(true_fps)+len(variant_fps)):.0%})")

print(f"\n--- GT-VARIANT FPs (these are actually correct extractions) ---")
for fp in sorted(variant_fps, key=lambda x: -x["score"]):
    print(f"  '{fp['term']}' ↔ GT:'{fp['best_gt']}' (score={fp['score']:.0f}, {fp['votes']}v)")

print(f"\n--- TRUE FPs (genuine noise) ---")
for fp in sorted(true_fps, key=lambda x: (-x["votes"] if isinstance(x["votes"], int) else 0, x["chunk_id"])):
    print(f"  [{fp['votes']}v] '{fp['term']}' ({fp['chunk_id']}) nearest: '{fp['best_gt']}' (score={fp['score']:.0f})")

# Compute "honest" metrics if we don't penalize variant extractions
total_extracted = sum(len(c["extracted_terms"]) for c in results_data["per_chunk_results"])
total_tp = sum(c["metrics"]["tp"] for c in results_data["per_chunk_results"])
total_true_fp = len(true_fps)
total_variant_fp = len(variant_fps)

print(f"\n--- ADJUSTED METRICS (if variants not penalized) ---")
adjusted_tp = total_tp + total_variant_fp  # count variants as TP
adjusted_p = adjusted_tp / total_extracted
adjusted_r = adjusted_tp / 277  # may exceed GT count since multiple extractions map to same GT
print(f"  Standard:  P={total_tp/total_extracted:.1%}, R={total_tp/277:.1%}")
print(f"  Adjusted:  P={adjusted_p:.1%} (counting variants as TP)")
print(f"  True FP rate: {total_true_fp / total_extracted:.1%}")
