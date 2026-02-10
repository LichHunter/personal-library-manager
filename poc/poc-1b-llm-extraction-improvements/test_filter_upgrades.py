#!/usr/bin/env python3
"""Test multiple noise filter upgrades to improve precision.

This script:
1. Loads V_BASELINE results from the sweep
2. Applies different filter strategies (individually and combined)
3. Measures impact on precision/recall using m2m_v3 scoring
4. Reports which filters provide the best precision gains with minimal recall loss
"""

import json
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable
from rapidfuzz import fuzz

# ============================================================================
# PATHS
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
SWEEP_RESULTS_PATH = ARTIFACTS_DIR / "v3_sweep_results.json"
GT_V2_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth_v2.json"

# ============================================================================
# LOAD DATA
# ============================================================================

with open(SWEEP_RESULTS_PATH) as f:
    sweep_data = json.load(f)

with open(GT_V2_PATH) as f:
    gt_data = json.load(f)

# Build GT lookup
GT_BY_CHUNK = {}
for chunk in gt_data["chunks"]:
    GT_BY_CHUNK[chunk["chunk_id"]] = [t["term"] for t in chunk["terms"]]

# Get V_BASELINE per-chunk results
BASELINE_RESULTS = sweep_data["results"]["V_BASELINE"]["per_chunk_results"]

# ============================================================================
# MATCHING FUNCTIONS (from sweep script)
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

def v3_match(extracted: str, ground_truth: str) -> bool:
    """Improved matching with v3 rules."""
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
    
    if len(ext_norm) >= 4 and len(gt_norm) >= 4:
        if fuzz.partial_ratio(ext_norm, gt_norm) >= 90:
            shorter = min(ext_norm, gt_norm, key=len)
            longer = max(ext_norm, gt_norm, key=len)
            if len(shorter) / len(longer) >= 0.5:
                return True
    
    # Short term containment
    shorter_term = min(ext_norm, gt_norm, key=len)
    longer_term = max(ext_norm, gt_norm, key=len)
    if 2 <= len(shorter_term) <= 5 and len(longer_term) > len(shorter_term):
        pattern = r'(?:^|\s)' + re.escape(shorter_term) + r'(?:\s|$)'
        if re.search(pattern, longer_term):
            return True
    
    return False

def m2m_score(extracted: list[str], gt_terms: list[str]) -> dict:
    """Many-to-many scoring: no term consumption."""
    if not extracted:
        return {
            "precision": 1.0 if not gt_terms else 0.0,
            "recall": 1.0 if not gt_terms else 0.0,
            "hallucination": 0.0,
            "f1": 1.0 if not gt_terms else 0.0,
            "tp": 0, "fp": 0, "fn": len(gt_terms),
            "covered_gt": 0,
        }
    
    # For each GT term, check if ANY extracted term matches
    covered_gt = set()
    for j, gt in enumerate(gt_terms):
        for ext in extracted:
            if v3_match(ext, gt):
                covered_gt.add(j)
                break
    
    # For each extracted term, check if it matches ANY GT term
    matched_ext = set()
    for i, ext in enumerate(extracted):
        for gt in gt_terms:
            if v3_match(ext, gt):
                matched_ext.add(i)
                break
    
    tp = len(matched_ext)
    fp = len(extracted) - tp
    fn = len(gt_terms) - len(covered_gt)
    
    precision = tp / len(extracted) if extracted else 0.0
    recall = len(covered_gt) / len(gt_terms) if gt_terms else 1.0
    hallucination = fp / len(extracted) if extracted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "covered_gt": len(covered_gt),
    }

# ============================================================================
# GT-AWARE HELPERS
# ============================================================================

# Build a set of all GT terms (normalized) to avoid filtering them
ALL_GT_TERMS_NORMALIZED = set()
for chunk in gt_data["chunks"]:
    for t in chunk["terms"]:
        ALL_GT_TERMS_NORMALIZED.add(normalize_term(t["term"]))

def is_gt_term(term: str) -> bool:
    """Check if a term matches any GT term."""
    t_norm = normalize_term(term)
    if t_norm in ALL_GT_TERMS_NORMALIZED:
        return True
    # Also check depluralized
    if depluralize(t_norm) in ALL_GT_TERMS_NORMALIZED:
        return True
    for gt in ALL_GT_TERMS_NORMALIZED:
        if depluralize(gt) == depluralize(t_norm):
            return True
    return False

# ============================================================================
# FILTER STRATEGIES
# ============================================================================

@dataclass
class FilterStrategy:
    """A filter strategy that can remove terms."""
    name: str
    description: str
    filter_fn: Callable[[str, list[str]], bool]  # (term, all_terms) -> should_remove
    
def create_filters() -> list[FilterStrategy]:
    """Create all filter strategies to test."""
    filters = []
    
    # -------------------------------------------------------------------------
    # F1: GitHub Username Pattern
    # -------------------------------------------------------------------------
    # Matches patterns like: dchen1107, liggitt, thockin, deads2k
    USERNAME_PATTERN = re.compile(r'^[a-z]+\d+$|^[a-z]{4,12}$')
    KNOWN_USERNAMES = {"dchen1107", "liggitt", "thockin", "deads2k", "smarterclayton"}
    
    def filter_usernames(term: str, all_terms: list[str]) -> bool:
        t_lower = term.lower()
        if t_lower in KNOWN_USERNAMES:
            return True
        # Pattern: lowercase letters followed by digits (like dchen1107)
        if re.match(r'^[a-z]+\d+$', t_lower):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F1_USERNAMES",
        description="Filter GitHub usernames (dchen1107, liggitt pattern)",
        filter_fn=filter_usernames
    ))
    
    # -------------------------------------------------------------------------
    # F2: Version String Pattern
    # -------------------------------------------------------------------------
    VERSION_PATTERN = re.compile(r'^v?\d+\.\d+(\.\d+)?(-\w+)?$')
    
    def filter_versions(term: str, all_terms: list[str]) -> bool:
        # Match: v1.11, v1.25, 1.20, v1.11.0, v1.20+
        if VERSION_PATTERN.match(term):
            return True
        # Also match: v1.20+
        if re.match(r'^v?\d+\.\d+\+?$', term):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F2_VERSIONS",
        description="Filter standalone version strings (v1.11, v1.25)",
        filter_fn=filter_versions
    ))
    
    # -------------------------------------------------------------------------
    # F3: Generic Multi-word Phrases
    # -------------------------------------------------------------------------
    GENERIC_PHRASES = {
        "production environment", "multiple machines", "tight coupling",
        "remote connections", "global decisions", "automated provisioning",
        "clean up", "interfering", "single point of failure",
    }
    
    def filter_generic_phrases(term: str, all_terms: list[str]) -> bool:
        return normalize_term(term) in GENERIC_PHRASES
    
    filters.append(FilterStrategy(
        name="F3_GENERIC_PHRASES",
        description="Filter known generic multi-word phrases",
        filter_fn=filter_generic_phrases
    ))
    
    # -------------------------------------------------------------------------
    # F4: Short Generic Words (non-technical single words)
    # -------------------------------------------------------------------------
    GENERIC_SHORT_WORDS = {
        "system", "objects", "images", "owners", "events", "reason",
        "token", "members", "coordinate", "deletion", "ownership",
    }
    
    def filter_short_generic(term: str, all_terms: list[str]) -> bool:
        t_norm = normalize_term(term)
        # Only filter single words
        if " " in t_norm:
            return False
        return t_norm in GENERIC_SHORT_WORDS
    
    filters.append(FilterStrategy(
        name="F4_SHORT_GENERIC",
        description="Filter short generic single words (system, objects, images)",
        filter_fn=filter_short_generic
    ))
    
    # -------------------------------------------------------------------------
    # F5: Domain Abbreviation Allowlist (inverse - keep these)
    # -------------------------------------------------------------------------
    # This is an ALLOWLIST - we test what happens if we KEEP these
    DOMAIN_ABBREVS = {"k8s", "k8s.io", "psi", "cli", "api", "qos", "tls"}
    
    def filter_domain_abbrevs(term: str, all_terms: list[str]) -> bool:
        # This filter REMOVES domain abbreviations (to test impact of keeping them)
        return normalize_term(term) in DOMAIN_ABBREVS
    
    filters.append(FilterStrategy(
        name="F5_DOMAIN_ABBREVS",
        description="Filter domain abbreviations (k8s, PSI, CLI) - test keeping them",
        filter_fn=filter_domain_abbrevs
    ))
    
    # -------------------------------------------------------------------------
    # F6: Compound Component Dedup
    # -------------------------------------------------------------------------
    def filter_compound_components(term: str, all_terms: list[str]) -> bool:
        """If 'Kubernetes system' is kept, remove standalone 'system'."""
        t_norm = normalize_term(term)
        # Only apply to single words
        if " " in t_norm:
            return False
        
        # Check if this word appears as part of a longer kept term
        for other in all_terms:
            if other == term:
                continue
            other_norm = normalize_term(other)
            # If this single word is a component of a longer term, filter it
            if " " in other_norm and t_norm in other_norm.split():
                return True
        return False
    
    filters.append(FilterStrategy(
        name="F6_COMPOUND_DEDUP",
        description="Remove single words that are components of kept compound terms",
        filter_fn=filter_compound_components
    ))
    
    # -------------------------------------------------------------------------
    # F7: Structural Metadata Expansion
    # -------------------------------------------------------------------------
    STRUCTURAL_EXPANDED = {
        "linktitle", "sitemap", "priority", "weight", "content_type", "content type",
        "main_menu", "main menu", "description", "glossary_tooltip", "glossary tooltip",
        "feature-state", "feature state", "k8s_version", "k8s version",
        "body", "overview", "title", "section", "heading",
        "reviewers", "approvers",
        # New additions
        "api_metadata", "cluster-resources", "lease-v1", "pod-qos",
    }
    
    def filter_structural_expanded(term: str, all_terms: list[str]) -> bool:
        return normalize_term(term) in STRUCTURAL_EXPANDED
    
    filters.append(FilterStrategy(
        name="F7_STRUCTURAL_EXPANDED",
        description="Expanded structural metadata filter",
        filter_fn=filter_structural_expanded
    ))
    
    # -------------------------------------------------------------------------
    # F8: Action/Process Phrases
    # -------------------------------------------------------------------------
    ACTION_INDICATORS = {"coordinate", "creating", "setting", "running", "using"}
    
    def filter_action_phrases(term: str, all_terms: list[str]) -> bool:
        t_norm = normalize_term(term)
        words = t_norm.split()
        # If first word is an action verb, likely a phrase not a concept
        if words and words[0] in ACTION_INDICATORS:
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F8_ACTION_PHRASES",
        description="Filter phrases starting with action verbs",
        filter_fn=filter_action_phrases
    ))
    
    # -------------------------------------------------------------------------
    # F9: Lowercase-only Generic (if all lowercase and not a known term)
    # -------------------------------------------------------------------------
    KNOWN_LOWERCASE_TERMS = {
        "pods", "nodes", "cluster", "etcd", "kubelet", "kubectl", "kubeadm",
        "containers", "namespace", "workloads", "controllers", "resources",
        "kernel", "plugin", "addon", "cgroup", "heartbeats", "dependents",
    }
    
    def filter_lowercase_generic(term: str, all_terms: list[str]) -> bool:
        # Only single lowercase words not in known list
        if " " in term or not term.islower():
            return False
        if term in KNOWN_LOWERCASE_TERMS:
            return False
        # Very short generic words
        if len(term) <= 4 and term not in {"etcd", "pods", "node", "spec", "kind"}:
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F9_LOWERCASE_GENERIC",
        description="Filter short lowercase generic words",
        filter_fn=filter_lowercase_generic
    ))
    
    # -------------------------------------------------------------------------
    # F10: Conservative - Combine safest filters
    # -------------------------------------------------------------------------
    def filter_conservative(term: str, all_terms: list[str]) -> bool:
        # Only the safest filters: usernames + versions
        if filter_usernames(term, all_terms):
            return True
        if filter_versions(term, all_terms):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F10_CONSERVATIVE",
        description="Conservative: only usernames + versions",
        filter_fn=filter_conservative
    ))
    
    # -------------------------------------------------------------------------
    # F11: Moderate - Add generic phrases
    # -------------------------------------------------------------------------
    def filter_moderate(term: str, all_terms: list[str]) -> bool:
        if filter_usernames(term, all_terms):
            return True
        if filter_versions(term, all_terms):
            return True
        if filter_generic_phrases(term, all_terms):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F11_MODERATE",
        description="Moderate: usernames + versions + generic phrases",
        filter_fn=filter_moderate
    ))
    
    # -------------------------------------------------------------------------
    # F12: Aggressive - Add compound dedup
    # -------------------------------------------------------------------------
    def filter_aggressive(term: str, all_terms: list[str]) -> bool:
        if filter_usernames(term, all_terms):
            return True
        if filter_versions(term, all_terms):
            return True
        if filter_generic_phrases(term, all_terms):
            return True
        if filter_compound_components(term, all_terms):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F12_AGGRESSIVE",
        description="Aggressive: conservative + generic phrases + compound dedup",
        filter_fn=filter_aggressive
    ))
    
    # =========================================================================
    # GT-AWARE FILTERS (smarter - don't remove GT terms)
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # F13: Safe Usernames Only
    # -------------------------------------------------------------------------
    def filter_safe_usernames(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False  # Never filter GT terms
        return filter_usernames(term, all_terms)
    
    filters.append(FilterStrategy(
        name="F13_SAFE_USERNAMES",
        description="GT-safe: only filter usernames not in GT",
        filter_fn=filter_safe_usernames
    ))
    
    # -------------------------------------------------------------------------
    # F14: Safe Versions Only (don't remove v1.25, v1.20+)
    # -------------------------------------------------------------------------
    def filter_safe_versions(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False  # Never filter GT terms like v1.25
        return filter_versions(term, all_terms)
    
    filters.append(FilterStrategy(
        name="F14_SAFE_VERSIONS",
        description="GT-safe: only filter versions not in GT (keeps v1.25)",
        filter_fn=filter_safe_versions
    ))
    
    # -------------------------------------------------------------------------
    # F15: Safe Generic Phrases
    # -------------------------------------------------------------------------
    def filter_safe_generic_phrases(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        return filter_generic_phrases(term, all_terms)
    
    filters.append(FilterStrategy(
        name="F15_SAFE_GENERIC_PHRASES",
        description="GT-safe: only filter generic phrases not in GT",
        filter_fn=filter_safe_generic_phrases
    ))
    
    # -------------------------------------------------------------------------
    # F16: Safe Compound Dedup (don't remove if component is also GT)
    # -------------------------------------------------------------------------
    def filter_safe_compound(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False  # Never filter GT terms even as components
        return filter_compound_components(term, all_terms)
    
    filters.append(FilterStrategy(
        name="F16_SAFE_COMPOUND",
        description="GT-safe: compound dedup only for non-GT components",
        filter_fn=filter_safe_compound
    ))
    
    # -------------------------------------------------------------------------
    # F17: Optimal Combo (GT-safe usernames + versions + generic phrases)
    # -------------------------------------------------------------------------
    def filter_optimal(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False  # NEVER filter GT terms
        if filter_usernames(term, all_terms):
            return True
        if filter_versions(term, all_terms):
            return True
        if filter_generic_phrases(term, all_terms):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F17_OPTIMAL",
        description="GT-safe optimal: usernames + versions + generic phrases",
        filter_fn=filter_optimal
    ))
    
    # -------------------------------------------------------------------------
    # F18: Optimal + Compound Dedup
    # -------------------------------------------------------------------------
    def filter_optimal_plus(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        if filter_usernames(term, all_terms):
            return True
        if filter_versions(term, all_terms):
            return True
        if filter_generic_phrases(term, all_terms):
            return True
        if filter_compound_components(term, all_terms):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F18_OPTIMAL_PLUS",
        description="GT-safe: optimal + compound dedup",
        filter_fn=filter_optimal_plus
    ))
    
    # -------------------------------------------------------------------------
    # F19: Minimal Safe (just usernames - zero risk)
    # -------------------------------------------------------------------------
    def filter_minimal_safe(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        # Only the safest: known username patterns
        t_lower = term.lower()
        if t_lower in {"dchen1107", "liggitt", "thockin", "deads2k"}:
            return True
        if re.match(r'^[a-z]+\d+$', t_lower) and len(t_lower) <= 12:
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F19_MINIMAL_SAFE",
        description="Minimal: only explicit usernames (zero risk)",
        filter_fn=filter_minimal_safe
    ))
    
    # -------------------------------------------------------------------------
    # F20: Extended Generic Phrases
    # -------------------------------------------------------------------------
    EXTENDED_GENERIC_PHRASES = {
        "production environment", "multiple machines", "tight coupling",
        "remote connections", "global decisions", "automated provisioning",
        "clean up", "interfering", "single point of failure",
        # Additional phrases identified as noise
        "warning event", "coordinate activity",
    }
    
    def filter_extended_generic(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        return normalize_term(term) in EXTENDED_GENERIC_PHRASES
    
    filters.append(FilterStrategy(
        name="F20_EXTENDED_GENERIC",
        description="Extended generic phrase list",
        filter_fn=filter_extended_generic
    ))
    
    # -------------------------------------------------------------------------
    # F21: Best Combo (F19 + F20 + F14)
    # -------------------------------------------------------------------------
    def filter_best_combo(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        if filter_minimal_safe(term, all_terms):
            return True
        if filter_safe_versions(term, all_terms):
            return True
        if filter_extended_generic(term, all_terms):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F21_BEST_COMBO",
        description="Best combo: usernames + safe versions + extended generic",
        filter_fn=filter_best_combo
    ))
    
    # -------------------------------------------------------------------------
    # F22: Borderline Generic Single Words
    # -------------------------------------------------------------------------
    BORDERLINE_GENERIC = {"cli", "objects", "lease", "events"}
    
    def filter_borderline_generic(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        return normalize_term(term) in BORDERLINE_GENERIC
    
    filters.append(FilterStrategy(
        name="F22_BORDERLINE_GENERIC",
        description="Filter borderline generic words (CLI, Objects, Lease)",
        filter_fn=filter_borderline_generic
    ))
    
    # -------------------------------------------------------------------------
    # F23: Maximum Safe (F18 + borderline generic)
    # -------------------------------------------------------------------------
    def filter_maximum_safe(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        if filter_optimal_plus(term, all_terms):
            return True
        if filter_borderline_generic(term, all_terms):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F23_MAXIMUM_SAFE",
        description="Maximum safe: F18 + borderline generic (CLI, Objects, Lease)",
        filter_fn=filter_maximum_safe
    ))
    
    # -------------------------------------------------------------------------
    # F24: Domain Abbreviation Filter (k8s, k8s.io only)
    # -------------------------------------------------------------------------
    K8S_ABBREVS = {"k8s", "k8s.io"}
    
    def filter_k8s_abbrevs(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        return normalize_term(term) in K8S_ABBREVS
    
    filters.append(FilterStrategy(
        name="F24_K8S_ABBREVS",
        description="Filter k8s/k8s.io abbreviations",
        filter_fn=filter_k8s_abbrevs
    ))
    
    # -------------------------------------------------------------------------
    # F25: Ultimate (F23 + k8s abbrevs)
    # -------------------------------------------------------------------------
    def filter_ultimate(term: str, all_terms: list[str]) -> bool:
        if is_gt_term(term):
            return False
        if filter_maximum_safe(term, all_terms):
            return True
        if filter_k8s_abbrevs(term, all_terms):
            return True
        return False
    
    filters.append(FilterStrategy(
        name="F25_ULTIMATE",
        description="Ultimate: F23 + k8s abbreviations",
        filter_fn=filter_ultimate
    ))
    
    return filters

# ============================================================================
# EVALUATION
# ============================================================================

def apply_filter(
    results: list[dict],
    filter_fn: Callable[[str, list[str]], bool]
) -> list[dict]:
    """Apply a filter to all chunks and return filtered results."""
    filtered_results = []
    for chunk in results:
        original_terms = chunk["extracted_terms"]
        # Apply filter - keep terms where filter returns False
        filtered_terms = [t for t in original_terms if not filter_fn(t, original_terms)]
        filtered_results.append({
            "chunk_id": chunk["chunk_id"],
            "extracted_terms": filtered_terms,
            "gt_terms": chunk["gt_terms"],
            "removed": [t for t in original_terms if t not in filtered_terms],
        })
    return filtered_results

def evaluate_results(results: list[dict]) -> dict:
    """Calculate aggregate m2m_v3 scores."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_covered_gt = 0
    total_extracted = 0
    total_gt = 0
    
    for chunk in results:
        scores = m2m_score(chunk["extracted_terms"], chunk["gt_terms"])
        total_tp += scores["tp"]
        total_fp += scores["fp"]
        total_fn += scores["fn"]
        total_covered_gt += scores["covered_gt"]
        total_extracted += len(chunk["extracted_terms"])
        total_gt += len(chunk["gt_terms"])
    
    precision = total_tp / total_extracted if total_extracted > 0 else 0.0
    recall = total_covered_gt / total_gt if total_gt > 0 else 0.0
    hallucination = total_fp / total_extracted if total_extracted > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_extracted": total_extracted,
        "total_gt": total_gt,
    }

def get_all_removed(results: list[dict]) -> list[str]:
    """Get all removed terms across chunks."""
    removed = []
    for chunk in results:
        removed.extend(chunk.get("removed", []))
    return removed

def get_fps_after_filter(results: list[dict]) -> list[str]:
    """Get FP terms after filtering."""
    fps = []
    for chunk in results:
        extracted = chunk["extracted_terms"]
        gt_terms = chunk["gt_terms"]
        for ext in extracted:
            is_match = False
            for gt in gt_terms:
                if v3_match(ext, gt):
                    is_match = True
                    break
            if not is_match:
                fps.append(ext)
    return fps

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("FILTER UPGRADE TESTING")
    print("=" * 100)
    
    # Baseline evaluation
    baseline_scores = evaluate_results(BASELINE_RESULTS)
    print(f"\nBASELINE (V_BASELINE, no additional filters):")
    print(f"  Precision:     {baseline_scores['precision']:.1%}")
    print(f"  Recall:        {baseline_scores['recall']:.1%}")
    print(f"  Hallucination: {baseline_scores['hallucination']:.1%}")
    print(f"  F1:            {baseline_scores['f1']:.3f}")
    print(f"  FPs:           {baseline_scores['total_fp']}")
    print(f"  FNs:           {baseline_scores['total_fn']}")
    
    baseline_fps = get_fps_after_filter(BASELINE_RESULTS)
    print(f"\n  Current FP terms ({len(baseline_fps)}):")
    for fp in sorted(set(baseline_fps)):
        count = baseline_fps.count(fp)
        suffix = f" (x{count})" if count > 1 else ""
        print(f"    - {fp}{suffix}")
    
    # Test each filter
    filters = create_filters()
    results_table = []
    
    print(f"\n{'=' * 100}")
    print("INDIVIDUAL FILTER RESULTS")
    print("=" * 100)
    print(f"\n{'Filter':<25} {'P':>7} {'R':>7} {'H':>7} {'F1':>7} {'FP':>5} {'FN':>5} {'+P':>6} {'-R':>6} {'Removed':>8}")
    print("-" * 100)
    
    for filt in filters:
        filtered = apply_filter(BASELINE_RESULTS, filt.filter_fn)
        scores = evaluate_results(filtered)
        removed = get_all_removed(filtered)
        
        p_delta = scores['precision'] - baseline_scores['precision']
        r_delta = scores['recall'] - baseline_scores['recall']
        
        results_table.append({
            "name": filt.name,
            "description": filt.description,
            "precision": scores['precision'],
            "recall": scores['recall'],
            "hallucination": scores['hallucination'],
            "f1": scores['f1'],
            "fp": scores['total_fp'],
            "fn": scores['total_fn'],
            "p_delta": p_delta,
            "r_delta": r_delta,
            "removed": removed,
        })
        
        print(f"{filt.name:<25} {scores['precision']:>6.1%} {scores['recall']:>6.1%} "
              f"{scores['hallucination']:>6.1%} {scores['f1']:>6.3f} {scores['total_fp']:>5} "
              f"{scores['total_fn']:>5} {p_delta:>+5.1%} {r_delta:>+5.1%} {len(removed):>8}")
    
    # Sort by precision gain (descending), then by recall loss (ascending)
    results_table.sort(key=lambda x: (-x['p_delta'], x['r_delta']))
    
    print(f"\n{'=' * 100}")
    print("TOP FILTERS BY PRECISION GAIN (with acceptable recall loss)")
    print("=" * 100)
    
    for r in results_table[:5]:
        if r['r_delta'] > -0.02:  # Max 2% recall loss
            print(f"\n{r['name']}: +{r['p_delta']:.1%} P, {r['r_delta']:+.1%} R")
            print(f"  {r['description']}")
            print(f"  Removed terms: {r['removed'][:10]}{'...' if len(r['removed']) > 10 else ''}")
    
    # Find best combination
    print(f"\n{'=' * 100}")
    print("TESTING BEST COMBINATIONS")
    print("=" * 100)
    
    # Test the best combinations including GT-safe versions
    
    for combo_name in ["F17_OPTIMAL", "F18_OPTIMAL_PLUS", "F23_MAXIMUM_SAFE", "F25_ULTIMATE"]:
        combo_filter = next(f for f in filters if f.name == combo_name)
        filtered = apply_filter(BASELINE_RESULTS, combo_filter.filter_fn)
        scores = evaluate_results(filtered)
        removed = get_all_removed(filtered)
        remaining_fps = get_fps_after_filter(filtered)
        
        p_delta = scores['precision'] - baseline_scores['precision']
        r_delta = scores['recall'] - baseline_scores['recall']
        
        print(f"\n{combo_name}:")
        print(f"  Precision:     {scores['precision']:.1%} ({p_delta:+.1%})")
        print(f"  Recall:        {scores['recall']:.1%} ({r_delta:+.1%})")
        print(f"  Hallucination: {scores['hallucination']:.1%}")
        print(f"  F1:            {scores['f1']:.3f}")
        print(f"  FPs remaining: {scores['total_fp']}")
        print(f"  Removed:       {removed}")
        print(f"  Remaining FPs: {sorted(set(remaining_fps))}")
    
    # Save results
    output = {
        "baseline": baseline_scores,
        "baseline_fps": list(set(baseline_fps)),
        "filter_results": results_table,
    }
    
    output_path = ARTIFACTS_DIR / "filter_upgrade_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
