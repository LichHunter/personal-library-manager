#!/usr/bin/env python3
"""Analyze what improved dedup would do.

Two improvements:
1. Better dedup: merge terms that are singular/plural or sub/super strings
2. Better metrics: allow partial GT matching (don't count variants as FP)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz

RESULTS_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_results.json"
GT_PATH = Path(__file__).parent / "artifacts" / "small_chunk_ground_truth_v2.json"

with open(RESULTS_PATH) as f:
    results_data = json.load(f)
with open(GT_PATH) as f:
    gt_data = json.load(f)

def normalize_term(term):
    return term.lower().strip().replace("-", " ").replace("_", " ")

def is_variant(a, b):
    """Check if a is a variant of b (substring, plural, etc.)."""
    a_n = normalize_term(a)
    b_n = normalize_term(b)
    if a_n == b_n:
        return True
    # Singular/plural
    if a_n.rstrip("s") == b_n.rstrip("s"):
        return True
    # Substring (a contained in b or b contained in a)
    if a_n in b_n or b_n in a_n:
        return True
    # High fuzzy match
    if fuzz.ratio(a_n, b_n) >= 85:
        return True
    return False

def aggressive_dedup(terms):
    """Aggressively dedup terms — keep longest/most specific variant."""
    if not terms:
        return terms
    
    # Sort by length (longest first) — prefer more specific terms
    sorted_terms = sorted(terms, key=lambda t: -len(t))
    kept = []
    
    for term in sorted_terms:
        is_dup = False
        for existing in kept:
            if is_variant(term, existing):
                is_dup = True
                break
        if not is_dup:
            kept.append(term)
    
    return kept

def relaxed_match(extracted, gt_term):
    """More relaxed matching that allows partial overlap."""
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(gt_term)
    
    if ext_norm == gt_norm:
        return True
    
    # Fuzzy match
    if fuzz.ratio(ext_norm, gt_norm) >= 80:
        return True
    
    # Singular/plural
    if ext_norm.rstrip("s") == gt_norm.rstrip("s"):
        return True
    
    # Token overlap (80%+ of GT tokens appear in extracted)
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True
    
    # Substring matching for short terms
    if len(gt_norm) <= 15:
        if ext_norm in gt_norm or gt_norm in ext_norm:
            if len(min(ext_norm, gt_norm, key=len)) >= 3:  # avoid tiny matches
                return True
    
    return True if fuzz.partial_ratio(ext_norm, gt_norm) >= 90 and len(ext_norm) >= 4 else False

# Build GT lookup
gt_by_chunk = {}
for chunk in gt_data["chunks"]:
    gt_by_chunk[chunk["chunk_id"]] = [t["term"] for t in chunk["terms"]]

print("=" * 80)
print("IMPACT OF BETTER DEDUP + MATCHING")
print("=" * 80)

# Simulate per-chunk
total_original_extracted = 0
total_deduped_extracted = 0
total_original_tp = 0
total_deduped_tp = 0
total_gt = 0

for chunk_result in results_data["per_chunk_results"]:
    chunk_id = chunk_result["chunk_id"]
    extracted = chunk_result["extracted_terms"]
    gt_terms = gt_by_chunk.get(chunk_id, [])
    total_gt += len(gt_terms)
    
    # Original metrics
    total_original_extracted += len(extracted)
    total_original_tp += chunk_result["metrics"]["tp"]
    
    # Aggressive dedup
    deduped = aggressive_dedup(extracted)
    total_deduped_extracted += len(deduped)
    
    # Calculate with relaxed matching
    matched_gt = set()
    tp = 0
    fp_list = []
    
    for ext in deduped:
        found = False
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if relaxed_match(ext, gt):
                matched_gt.add(j)
                tp += 1
                found = True
                break
        if not found:
            fp_list.append(ext)
    
    total_deduped_tp += tp
    
    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in matched_gt]
    
    p = tp / len(deduped) if deduped else 0
    r = tp / len(gt_terms) if gt_terms else 0
    old_p = chunk_result["metrics"]["precision"]
    old_r = chunk_result["metrics"]["recall"]
    
    if abs(p - old_p) > 0.05 or abs(r - old_r) > 0.05:
        print(f"\n  [{chunk_id}]")
        print(f"    Original: {len(extracted)} terms, P={old_p:.1%}, R={old_r:.1%}")
        print(f"    Deduped:  {len(deduped)} terms, P={p:.1%}, R={r:.1%}")
        print(f"    Removed by dedup: {len(extracted) - len(deduped)}")
        if fp_list:
            print(f"    Remaining FPs: {fp_list[:10]}")

# Summary
print(f"\n{'=' * 80}")
print("SUMMARY")
print("=" * 80)

orig_p = total_original_tp / total_original_extracted
orig_r = total_original_tp / total_gt
dedup_p = total_deduped_tp / total_deduped_extracted
dedup_r = total_deduped_tp / total_gt

print(f"\n  Original D+v2:")
print(f"    {total_original_extracted} extracted, {total_original_tp} TP, P={orig_p:.1%}, R={orig_r:.1%}")
print(f"\n  With aggressive dedup + relaxed matching:")
print(f"    {total_deduped_extracted} extracted, {total_deduped_tp} TP, P={dedup_p:.1%}, R={dedup_r:.1%}")
print(f"    Removed: {total_original_extracted - total_deduped_extracted} duplicate terms")
print(f"    New TP found by relaxed matching: {total_deduped_tp - total_original_tp}")

# Now also calculate: what are the TRUE remaining FPs after all improvements?
print(f"\n{'=' * 80}")
print("REMAINING TRUE FPs (after dedup + relaxed matching)")
print("=" * 80)

all_remaining_fps = []
for chunk_result in results_data["per_chunk_results"]:
    chunk_id = chunk_result["chunk_id"]
    extracted = chunk_result["extracted_terms"]
    gt_terms = gt_by_chunk.get(chunk_id, [])
    
    deduped = aggressive_dedup(extracted)
    
    matched_gt = set()
    for ext in deduped:
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if relaxed_match(ext, gt):
                matched_gt.add(j)
                break
        else:
            all_remaining_fps.append((chunk_id, ext))

print(f"\n  Total remaining FPs: {len(all_remaining_fps)}")
for chunk_id, term in all_remaining_fps:
    print(f"    [{chunk_id}] '{term}'")
