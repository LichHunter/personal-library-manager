#!/usr/bin/env python3
"""Scoring functions ported from poc-1b test_dplus_v3_sweep.py.

Uses the same v3_match + many_to_many_score as iter 26 for fair comparison.
"""

import re

from rapidfuzz import fuzz


def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def verify_span(term: str, content: str) -> tuple[bool, str]:
    if not term or len(term) < 2:
        return False, "too_short"
    content_lower = content.lower()
    term_lower = term.lower().strip()
    if term_lower in content_lower:
        return True, "exact"
    term_norm = term_lower.replace("-", " ").replace("_", " ")
    content_norm = content_lower.replace("-", " ").replace("_", " ")
    if term_norm in content_norm:
        return True, "normalized"
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True, "camelcase"
    if term_lower.endswith("s") and len(term_lower) > 3 and term_lower[:-1] in content_lower:
        return True, "singular_of_plural"
    if not term_lower.endswith("s") and (term_lower + "s") in content_lower:
        return True, "plural_of_singular"
    if term_lower.endswith("es") and len(term_lower) > 4 and term_lower[:-2] in content_lower:
        return True, "singular_of_plural_es"
    return False, "none"


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
    """V3 matcher with prefix/suffix for short terms — same as iter 26."""
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


def many_to_many_score(
    extracted: list[str], gt_terms: list[str],
) -> dict:
    """V4 many-to-many scoring — same as iter 26."""
    if not extracted or not gt_terms:
        return {
            "precision": 0, "recall": 0, "hallucination": 1 if extracted else 0,
            "f1": 0, "tp": 0, "fp": len(extracted), "fn": len(gt_terms),
            "covered_gt": 0,
            "fp_terms": list(extracted), "fn_terms": list(gt_terms),
        }

    covered_gt: set[int] = set()
    for j, gt in enumerate(gt_terms):
        for ext in extracted:
            if v3_match(ext, gt):
                covered_gt.add(j)
                break

    unmatched: list[str] = []
    for ext in extracted:
        found = False
        for gt in gt_terms:
            if v3_match(ext, gt):
                found = True
                break
        if not found:
            unmatched.append(ext)

    tp = len(extracted) - len(unmatched)
    fp = len(unmatched)
    fn = len(gt_terms) - len(covered_gt)
    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in covered_gt]

    precision = tp / len(extracted) if extracted else 0
    recall = len(covered_gt) / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision, "recall": recall,
        "hallucination": hallucination, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "covered_gt": len(covered_gt),
        "fp_terms": unmatched, "fn_terms": fn_terms,
    }
