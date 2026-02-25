#!/usr/bin/env python3
"""Deep analysis of 1-vote terms from D+v2 audit data.

Categorizes FPs and TPs to understand failure modes and design fixes.
"""

import json
from pathlib import Path
from collections import Counter

AUDIT_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_audit.json"
RESULTS_PATH = Path(__file__).parent / "artifacts" / "dplus_v2_results.json"

with open(AUDIT_PATH) as f:
    audit_data = json.load(f)

with open(RESULTS_PATH) as f:
    results_data = json.load(f)

audits = audit_data["audits"]

# Separate 1-vote terms by outcome
one_vote_kept = [a for a in audits if a["vote_count"] == 1 and a["final_status"] == "KEPT"]
one_vote_rejected_sonnet = [a for a in audits if a["vote_count"] == 1 and a["final_status"] == "REJECTED_SONNET"]
one_vote_rejected_ungrounded = [a for a in audits if a["vote_count"] == 1 and a["final_status"] == "REJECTED_UNGROUNDED"]

# Split kept into TP and FP
one_vote_tp = [a for a in one_vote_kept if a["matched_gt"]]
one_vote_fp = [a for a in one_vote_kept if not a["matched_gt"]]

# Also check: any rejected terms that WERE in GT (false rejections)
# We need to cross-reference with FN lists from results
all_fn_terms = set()
for chunk_result in results_data["per_chunk_results"]:
    for fn in chunk_result["fn_terms"]:
        all_fn_terms.add(fn.lower().strip())

one_vote_false_reject = [
    a for a in one_vote_rejected_sonnet 
    if a["normalized"] in all_fn_terms or a["term"].lower().strip() in all_fn_terms
]

print("=" * 80)
print("1-VOTE TERM ANALYSIS (D+v2)")
print("=" * 80)
print()
print(f"Total 1-vote terms: {len(one_vote_kept) + len(one_vote_rejected_sonnet) + len(one_vote_rejected_ungrounded)}")
print(f"  KEPT (Sonnet approved): {len(one_vote_kept)}")
print(f"    TP (correct): {len(one_vote_tp)}")
print(f"    FP (incorrect): {len(one_vote_fp)}")
print(f"  REJECTED_SONNET: {len(one_vote_rejected_sonnet)}")
print(f"    False rejections (in GT): {len(one_vote_false_reject)}")
print(f"  REJECTED_UNGROUNDED: {len(one_vote_rejected_ungrounded)}")
print()

# === SOURCE ANALYSIS ===
print("=" * 80)
print("1-VOTE FP TERMS - BY SOURCE")
print("=" * 80)
fp_by_source = Counter()
for a in one_vote_fp:
    for src in a["sources"]:
        fp_by_source[src] += 1
print()
for src, count in fp_by_source.most_common():
    total_from_src = sum(1 for a in one_vote_kept if src in a["sources"])
    print(f"  {src}: {count} FP out of {total_from_src} kept ({count/total_from_src:.1%} FP rate)")

tp_by_source = Counter()
for a in one_vote_tp:
    for src in a["sources"]:
        tp_by_source[src] += 1
print()
print("1-VOTE TP TERMS - BY SOURCE")
for src, count in tp_by_source.most_common():
    print(f"  {src}: {count} TP")

# === TERM LENGTH ANALYSIS ===
print()
print("=" * 80)
print("TERM LENGTH ANALYSIS (word count)")
print("=" * 80)

def word_count(term):
    return len(term.split())

fp_wc = Counter(word_count(a["term"]) for a in one_vote_fp)
tp_wc = Counter(word_count(a["term"]) for a in one_vote_tp)

print()
print(f"{'Words':>6} | {'FP':>4} | {'TP':>4} | {'FP Rate':>8}")
print("-" * 35)
for wc in sorted(set(list(fp_wc.keys()) + list(tp_wc.keys()))):
    fp_n = fp_wc.get(wc, 0)
    tp_n = tp_wc.get(wc, 0)
    total = fp_n + tp_n
    rate = fp_n / total if total else 0
    print(f"{wc:>6} | {fp_n:>4} | {tp_n:>4} | {rate:>7.1%}")

# === CHAR LENGTH ANALYSIS ===
print()
print("CHAR LENGTH ANALYSIS")
fp_lens = [len(a["term"]) for a in one_vote_fp]
tp_lens = [len(a["term"]) for a in one_vote_tp]
print(f"  FP avg length: {sum(fp_lens)/len(fp_lens):.1f} chars, median: {sorted(fp_lens)[len(fp_lens)//2]}")
print(f"  TP avg length: {sum(tp_lens)/len(tp_lens):.1f} chars, median: {sorted(tp_lens)[len(tp_lens)//2]}")

# Short FPs (likely generic words)
short_fps = [a["term"] for a in one_vote_fp if len(a["term"]) <= 8]
print(f"\n  Short FPs (<=8 chars): {len(short_fps)}")
for t in sorted(short_fps):
    print(f"    '{t}'")

# === CATEGORIZE FP TERMS ===
print()
print("=" * 80)
print("ALL 1-VOTE FP TERMS (with source & Sonnet reasoning)")
print("=" * 80)

# Group by chunk for context
fp_by_chunk = {}
for a in one_vote_fp:
    chunk = a["chunk_id"]
    if chunk not in fp_by_chunk:
        fp_by_chunk[chunk] = []
    fp_by_chunk[chunk].append(a)

for chunk_id, fps in sorted(fp_by_chunk.items()):
    print(f"\n  [{chunk_id}] ({len(fps)} FPs)")
    for a in fps:
        src = a["sources"][0]
        reasoning = a["sonnet_reasoning"][:80] if a["sonnet_reasoning"] else "N/A"
        print(f"    '{a['term']}' (src={src}) — {reasoning}")

# === ALL 1-VOTE TP TERMS ===
print()
print("=" * 80)
print("ALL 1-VOTE TP TERMS (correctly kept)")
print("=" * 80)

tp_by_chunk = {}
for a in one_vote_tp:
    chunk = a["chunk_id"]
    if chunk not in tp_by_chunk:
        tp_by_chunk[chunk] = []
    tp_by_chunk[chunk].append(a)

for chunk_id, tps in sorted(tp_by_chunk.items()):
    print(f"\n  [{chunk_id}] ({len(tps)} TPs)")
    for a in tps:
        src = a["sources"][0]
        print(f"    '{a['term']}' → '{a['matched_gt']}' (src={src})")

# === FALSE REJECTIONS ===
print()
print("=" * 80)
print("FALSE REJECTIONS (1-vote, Sonnet rejected, but term IS in GT)")
print("=" * 80)
for a in one_vote_false_reject:
    print(f"  '{a['term']}' — Sonnet: {a['sonnet_reasoning']}")

# === SONNET APPROVAL PATTERNS ===
print()
print("=" * 80)
print("SONNET APPROVAL REASONING PATTERNS (for FPs)")
print("=" * 80)

# Extract key patterns from Sonnet reasoning
reasoning_words = Counter()
for a in one_vote_fp:
    reasoning = a.get("sonnet_reasoning", "").lower()
    if "technical" in reasoning:
        reasoning_words["mentions 'technical'"] += 1
    if "concept" in reasoning:
        reasoning_words["mentions 'concept'"] += 1
    if "infrastructure" in reasoning:
        reasoning_words["mentions 'infrastructure'"] += 1
    if "component" in reasoning:
        reasoning_words["mentions 'component'"] += 1
    if "learner" in reasoning or "learn" in reasoning:
        reasoning_words["mentions 'learn'"] += 1

print()
for pattern, count in reasoning_words.most_common():
    print(f"  {pattern}: {count}/{len(one_vote_fp)} ({count/len(one_vote_fp):.0%})")

# === 2-VOTE FP ANALYSIS (for comparison) ===
print()
print("=" * 80)
print("2-VOTE FP TERMS (for comparison)")
print("=" * 80)

two_vote_kept = [a for a in audits if a["vote_count"] == 2 and a["final_status"] == "KEPT"]
two_vote_fp = [a for a in two_vote_kept if not a["matched_gt"]]
two_vote_tp = [a for a in two_vote_kept if a["matched_gt"]]

print(f"\n  2-vote KEPT: {len(two_vote_kept)}, TP: {len(two_vote_tp)}, FP: {len(two_vote_fp)} ({len(two_vote_fp)/len(two_vote_kept):.1%})")
for a in two_vote_fp:
    print(f"    '{a['term']}' (sources: {a['sources']})")

# === 3-VOTE FP ANALYSIS ===
print()
print("=" * 80)
print("3-VOTE FP TERMS (for comparison)")
print("=" * 80)

three_vote_kept = [a for a in audits if a["vote_count"] == 3 and a["final_status"] == "KEPT"]
three_vote_fp = [a for a in three_vote_kept if not a["matched_gt"]]
three_vote_tp = [a for a in three_vote_kept if a["matched_gt"]]

print(f"\n  3-vote KEPT: {len(three_vote_kept)}, TP: {len(three_vote_tp)}, FP: {len(three_vote_fp)} ({len(three_vote_fp)/len(three_vote_kept):.1%})")
for a in three_vote_fp:
    print(f"    '{a['term']}' (sources: {a['sources']})")

# === WHAT-IF SCENARIOS ===
print()
print("=" * 80)
print("WHAT-IF SCENARIOS")
print("=" * 80)

# Scenario 1: Reject all 1-vote terms
total_kept = len([a for a in audits if a["final_status"] == "KEPT"])
total_tp = len([a for a in audits if a["final_status"] == "KEPT" and a["matched_gt"]])
total_fp = total_kept - total_tp

print(f"\n  Current D+v2: {total_kept} kept, {total_tp} TP, {total_fp} FP, P={total_tp/total_kept:.1%}")

# Without 1-vote
no_1v_kept = total_kept - len(one_vote_kept)
no_1v_tp = total_tp - len(one_vote_tp)
no_1v_fp = total_fp - len(one_vote_fp)
gt_total = 277
no_1v_p = no_1v_tp / no_1v_kept if no_1v_kept else 0
no_1v_r = no_1v_tp / gt_total
print(f"  Without 1-vote: {no_1v_kept} kept, {no_1v_tp} TP, {no_1v_fp} FP, P={no_1v_p:.1%}, R={no_1v_r:.1%}")
print(f"    Lost: {len(one_vote_tp)} TP, removed {len(one_vote_fp)} FP")

# Scenario 2: Only keep 1-vote from Haiku sources (lower FP rate)
one_vote_haiku = [a for a in one_vote_kept if a["sources"][0] in ("haiku_exhaustive", "haiku_simple")]
one_vote_sonnet_only = [a for a in one_vote_kept if a["sources"][0] == "sonnet_exhaustive"]

haiku_1v_tp = len([a for a in one_vote_haiku if a["matched_gt"]])
haiku_1v_fp = len([a for a in one_vote_haiku if not a["matched_gt"]])
sonnet_1v_tp = len([a for a in one_vote_sonnet_only if a["matched_gt"]])
sonnet_1v_fp = len([a for a in one_vote_sonnet_only if not a["matched_gt"]])

print(f"\n  1-vote from Haiku: {len(one_vote_haiku)} kept, TP={haiku_1v_tp}, FP={haiku_1v_fp}, FP rate={haiku_1v_fp/len(one_vote_haiku):.1%}" if one_vote_haiku else "  1-vote from Haiku: 0")
print(f"  1-vote from Sonnet: {len(one_vote_sonnet_only)} kept, TP={sonnet_1v_tp}, FP={sonnet_1v_fp}, FP rate={sonnet_1v_fp/len(one_vote_sonnet_only):.1%}" if one_vote_sonnet_only else "  1-vote from Sonnet: 0")

# Scenario 3: Keep 2+ vote, reject Sonnet 1-vote, keep Haiku 1-vote
s3_kept = no_1v_kept + len(one_vote_haiku)
s3_tp = no_1v_tp + haiku_1v_tp
s3_fp = no_1v_fp + haiku_1v_fp
s3_p = s3_tp / s3_kept if s3_kept else 0
s3_r = s3_tp / gt_total
print(f"\n  Keep 2+ vote + Haiku 1-vote only: {s3_kept} kept, TP={s3_tp}, FP={s3_fp}, P={s3_p:.1%}, R={s3_r:.1%}")

# Scenario 4: Stricter Sonnet approval (reject if term is single common English word)
import string
COMMON_WORDS = {
    "system", "systems", "set", "type", "types", "level", "levels", "group", "groups",
    "name", "names", "data", "state", "states", "run", "running", "use", "using",
    "process", "version", "versions", "change", "changes", "model", "address",
    "resource", "resources", "work", "works", "field", "fields", "key", "keys",
    "value", "values", "action", "actions", "rule", "rules", "mode", "modes",
    "role", "roles", "scope", "scopes", "target", "targets", "source", "sources",
    "object", "objects", "event", "events", "task", "tasks", "service", "services",
    "method", "methods", "function", "functions", "class", "classes",
    "default", "global", "local", "primary", "secondary", "standard",
    "access", "control", "manage", "create", "delete", "update", "list",
    "behaviors", "features", "decisions", "requirements", "options", "properties",
    "components", "operations", "applications", "mechanisms", "references",
    "definitions", "specifications", "configurations", "descriptions",
    "overview", "pattern", "patterns", "design", "architecture",
    "storage", "deletion", "health", "performance", "availability",
}

# Check how many FPs are common words
common_word_fps = [a for a in one_vote_fp if a["term"].lower().strip() in COMMON_WORDS]
non_common_fps = [a for a in one_vote_fp if a["term"].lower().strip() not in COMMON_WORDS]
common_word_tps = [a for a in one_vote_tp if a["term"].lower().strip() in COMMON_WORDS]

print(f"\n  Common-word filter on 1-vote:")
print(f"    Would catch {len(common_word_fps)} FPs")
print(f"    Would lose {len(common_word_tps)} TPs")
if common_word_tps:
    print(f"    TPs that would be lost:")
    for a in common_word_tps:
        print(f"      '{a['term']}' → '{a['matched_gt']}'")
print(f"    Non-common-word FPs remaining: {len(non_common_fps)}")
if non_common_fps:
    print(f"    Remaining FPs:")
    for a in non_common_fps:
        print(f"      '{a['term']}' (src={a['sources'][0]})")
