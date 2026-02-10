#!/usr/bin/env python3
"""Quick analysis of D+v2.2 audit — FP rates by vote count and Sonnet analysis."""

import json
from pathlib import Path
from collections import Counter

AUDIT_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_2_audit.json"
RESULTS_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_2_results.json"

with open(AUDIT_PATH) as f:
    audit_data = json.load(f)
with open(RESULTS_PATH) as f:
    results_data = json.load(f)

audits = audit_data["audits"]

# FP rates by vote count
kept = [a for a in audits if a["final_status"] == "KEPT"]
for votes in [3, 2, 1]:
    v_kept = [a for a in kept if a["vote_count"] == votes]
    v_tp = [a for a in v_kept if a["matched_gt"]]
    v_fp = [a for a in v_kept if not a["matched_gt"]]
    if v_kept:
        print(f"{votes}-vote: {len(v_kept)} kept, {len(v_tp)} TP, {len(v_fp)} FP ({len(v_fp)/len(v_kept):.1%} FP rate)")

# Sonnet decisions breakdown
sonnet_reviewed = [a for a in audits if a["routing"] == "sonnet_review"]
sonnet_approved = [a for a in sonnet_reviewed if a["sonnet_decision"] == "APPROVE"]
sonnet_rejected = [a for a in sonnet_reviewed if a["sonnet_decision"] != "APPROVE"]

print(f"\nSonnet reviewed: {len(sonnet_reviewed)}")
print(f"  Approved: {len(sonnet_approved)} ({len(sonnet_approved)/len(sonnet_reviewed):.1%})")
print(f"  Rejected: {len(sonnet_rejected)} ({len(sonnet_rejected)/len(sonnet_reviewed):.1%})")

# Of approved, how many are TP vs FP?
approved_tp = [a for a in sonnet_approved if a["final_status"] == "KEPT" and a["matched_gt"]]
approved_fp = [a for a in sonnet_approved if a["final_status"] == "KEPT" and not a["matched_gt"]]
print(f"\n  Of approved+kept: {len(approved_tp)} TP, {len(approved_fp)} FP ({len(approved_fp)/(len(approved_tp)+len(approved_fp)):.1%} FP rate)")

# Of rejected, how many are false rejections (GT terms)?
all_fn = set()
for chunk_result in results_data["per_chunk_results"]:
    for fn in chunk_result["fn_terms"]:
        all_fn.add(fn.lower().strip())

false_rejections = [a for a in sonnet_rejected if a["normalized"] in all_fn]
print(f"\n  Sonnet false rejections: {len(false_rejections)}")
for a in false_rejections:
    print(f"    '{a['term']}' — {a['sonnet_reasoning'][:80]}")

# What-if: keep ALL Sonnet-approved terms (no dedup effect)?
# vs: reject ALL 1-vote terms?
total_kept = len(kept)
total_tp = len([a for a in kept if a["matched_gt"]])

no_1v_kept = len([a for a in kept if a["vote_count"] >= 2])
no_1v_tp = len([a for a in kept if a["vote_count"] >= 2 and a["matched_gt"]])

print(f"\nWhat-if scenarios:")
print(f"  Current: {total_kept} kept, {total_tp} TP, P={total_tp/total_kept:.1%}, R={total_tp/277:.1%}")
print(f"  No 1-vote: {no_1v_kept} kept, {no_1v_tp} TP, P={no_1v_tp/no_1v_kept:.1%}, R={no_1v_tp/277:.1%}")

# Dedup victims analysis
dedup_victims = [a for a in audits if a["final_status"] == "MERGED_DEDUP"]
print(f"\nDedup victims: {len(dedup_victims)}")
for a in dedup_victims:
    print(f"  '{a['term']}' (votes={a['vote_count']}, src={a['sources']})")

# Remaining FPs (the ones we need to kill)
all_fps = []
for chunk_result in results_data["per_chunk_results"]:
    chunk_id = chunk_result["chunk_id"]
    for fp in chunk_result["fp_terms"]:
        fp_audit = next(
            (a for a in audits if a["chunk_id"] == chunk_id and a["term"] == fp and a["final_status"] == "KEPT"),
            None
        )
        votes = fp_audit["vote_count"] if fp_audit else "?"
        all_fps.append({"term": fp, "chunk_id": chunk_id, "votes": votes})

print(f"\nRemaining FPs: {len(all_fps)}")
fp_by_vote = Counter(fp["votes"] for fp in all_fps)
print(f"  By vote count: {dict(sorted(fp_by_vote.items()))}")

# Show 1-vote FPs with Sonnet reasoning
print(f"\n1-vote FPs (Sonnet approved these incorrectly):")
for fp in all_fps:
    if fp["votes"] == 1:
        fp_a = next(
            (a for a in audits if a["chunk_id"] == fp["chunk_id"] and a["term"] == fp["term"] and a["final_status"] == "KEPT"),
            None
        )
        if fp_a:
            print(f"  '{fp['term']}' ({fp['chunk_id']}) — {fp_a['sonnet_reasoning'][:80]}")
