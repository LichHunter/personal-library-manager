#!/usr/bin/env python3
"""Analyze D+v2.1 recall loss — where did we lose terms?"""

import json
from pathlib import Path
from collections import Counter

AUDIT_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_1_audit.json"
RESULTS_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_1_results.json"
GT_PATH = Path(__file__).parent / "artifacts" / "small_chunk_ground_truth_v2.json"

with open(AUDIT_PATH) as f:
    audit_data = json.load(f)
with open(RESULTS_PATH) as f:
    results_data = json.load(f)
with open(GT_PATH) as f:
    gt_data = json.load(f)

audits = audit_data["audits"]

print("=" * 80)
print("D+v2.1 RECALL LOSS ANALYSIS")
print("=" * 80)

# Overall status distribution
print(f"\nStatus distribution:")
for status, count in sorted(audit_data["status_distribution"].items()):
    print(f"  {status}: {count}")

# 1. False Negatives: terms in GT not extracted
print(f"\n{'=' * 80}")
print("FALSE NEGATIVES BY CAUSE")
print("=" * 80)

all_fn_by_cause = Counter()
fn_details = []

for chunk_result in results_data["per_chunk_results"]:
    chunk_id = chunk_result["chunk_id"]
    fn_terms = chunk_result["fn_terms"]
    
    for fn in fn_terms:
        fn_lower = fn.lower().strip()
        # Find in audit trail
        fn_audit = None
        for a in audits:
            if a["chunk_id"] == chunk_id and a["normalized"] == fn_lower.replace("-", " ").replace("_", " "):
                fn_audit = a
                break
            # Try fuzzy
            if a["chunk_id"] == chunk_id and fn_lower in a["normalized"] or a["normalized"] in fn_lower:
                fn_audit = a
                break
        
        if fn_audit:
            cause = fn_audit["final_status"]
        else:
            cause = "NEVER_EXTRACTED"
        
        all_fn_by_cause[cause] += 1
        fn_details.append({
            "term": fn,
            "chunk_id": chunk_id,
            "cause": cause,
            "audit": fn_audit,
        })

print()
for cause, count in all_fn_by_cause.most_common():
    print(f"  {cause}: {count}")

total_fn = sum(all_fn_by_cause.values())
print(f"  TOTAL FN: {total_fn}")

# 2. Sonnet rejections that were GT terms
print(f"\n{'=' * 80}")
print("SONNET FALSE REJECTIONS (rejected GT terms)")
print("=" * 80)

sonnet_fn = [d for d in fn_details if d["cause"] == "REJECTED_SONNET"]
print(f"\n  Count: {len(sonnet_fn)}")
for d in sorted(sonnet_fn, key=lambda x: x["chunk_id"]):
    audit = d["audit"]
    reasoning = audit["sonnet_reasoning"][:100] if audit else "?"
    sources = audit["sources"] if audit else ["?"]
    print(f"  [{d['chunk_id']}] '{d['term']}' (src={sources}) — {reasoning}")

# 3. Dedup victims that were GT terms
print(f"\n{'=' * 80}")
print("DEDUP VICTIMS (merged GT terms)")
print("=" * 80)

dedup_fn = [d for d in fn_details if d["cause"] == "MERGED_DEDUP"]
print(f"\n  Count: {len(dedup_fn)}")
for d in sorted(dedup_fn, key=lambda x: x["chunk_id"]):
    audit = d["audit"]
    sources = audit["sources"] if audit else ["?"]
    votes = audit["vote_count"] if audit else 0
    print(f"  [{d['chunk_id']}] '{d['term']}' (votes={votes}, src={sources})")

# 4. Never extracted
print(f"\n{'=' * 80}")
print("NEVER EXTRACTED (no model found these)")
print("=" * 80)

never_fn = [d for d in fn_details if d["cause"] == "NEVER_EXTRACTED"]
print(f"\n  Count: {len(never_fn)}")
for d in sorted(never_fn, key=lambda x: x["chunk_id"]):
    # Get GT tier
    gt_tier = "?"
    for chunk in gt_data["chunks"]:
        if chunk["chunk_id"] == d["chunk_id"]:
            for t in chunk["terms"]:
                if t["term"] == d["term"]:
                    gt_tier = t["tier"]
                    break
            break
    print(f"  [{d['chunk_id']}] '{d['term']}' (T{gt_tier})")

# 5. Precision analysis — what FPs remain?
print(f"\n{'=' * 80}")
print("REMAINING FPs IN D+v2.1")
print("=" * 80)

all_fps = []
for chunk_result in results_data["per_chunk_results"]:
    chunk_id = chunk_result["chunk_id"]
    for fp in chunk_result["fp_terms"]:
        fp_audit = next(
            (a for a in audits if a["chunk_id"] == chunk_id and a["term"] == fp and a["final_status"] == "KEPT"),
            None
        )
        votes = fp_audit["vote_count"] if fp_audit else "?"
        sources = fp_audit["sources"] if fp_audit else ["?"]
        all_fps.append({"term": fp, "chunk_id": chunk_id, "votes": votes, "sources": sources})

print(f"\n  Total FPs: {len(all_fps)}")
fp_by_vote = Counter(fp["votes"] for fp in all_fps)
print(f"  By vote count: {dict(sorted(fp_by_vote.items()))}")
for fp in sorted(all_fps, key=lambda x: (-x["votes"] if isinstance(x["votes"], int) else 0, x["chunk_id"])):
    print(f"  [{fp['votes']}v] '{fp['term']}' ({fp['chunk_id']})")

# 6. Summary: Where to recover recall
print(f"\n{'=' * 80}")
print("RECOVERY OPPORTUNITY SUMMARY")
print("=" * 80)

print(f"\n  Total FN: {total_fn}")
print(f"  Recoverable from Sonnet rejections: {len(sonnet_fn)}")
print(f"  Recoverable from dedup: {len(dedup_fn)}")
print(f"  Not recoverable (never extracted): {len(never_fn)}")
print(f"  Max achievable recall: {(277 - len(never_fn)) / 277:.1%}")
