#!/usr/bin/env python3
"""Analyze raw extraction data to build evidence for pre-filter rules.

Reads extraction_raw_10docs.json and training data to provide concrete
evidence for each proposed filter rule.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

RAW_PATH = Path("artifacts/results/extraction_raw_10docs.json")
TRAIN_PATH = Path("artifacts/train_documents.json")


def load_data():
    with open(RAW_PATH) as f:
        docs = json.load(f)
    with open(TRAIN_PATH) as f:
        train = json.load(f)
    return docs, train


def analyze_bare_numbers(docs):
    """Find all bare number patterns in raw_union and check against GT."""
    print("=" * 80)
    print("ANALYSIS 1: BARE NUMBERS & VERSIONS IN RAW UNION vs GT")
    print("=" * 80)
    
    # Patterns to check
    pure_digit = re.compile(r"^\d+$")
    digit_x_digit = re.compile(r"^\d+\s*x\s*\d+$", re.I)
    digit_x = re.compile(r"^\d+\s+x$")
    version_like = re.compile(r"^\d+\.\d+(\.\d+)*$")
    
    for doc in docs:
        doc_id = doc["doc_id"]
        gt_lower = {t.lower() for t in doc["gt_terms"]}
        
        numbers_in_union = []
        for term in doc["raw_union"]:
            if (pure_digit.match(term) or digit_x_digit.match(term) or 
                digit_x.match(term) or version_like.match(term)):
                in_gt = "GT" if term.lower() in gt_lower else "FP"
                # Also fuzzy check - is any GT term a version that contains this?
                gt_contains = [g for g in doc["gt_terms"] if term.lower() in g.lower()]
                numbers_in_union.append((term, in_gt, gt_contains))
        
        if numbers_in_union:
            print(f"\n  {doc_id}:")
            for term, status, gt_contains in numbers_in_union:
                extra = f"  (contained in GT: {gt_contains})" if gt_contains and status == "FP" else ""
                print(f"    [{status}] '{term}'{extra}")


def analyze_determiner_prefixed(docs):
    """Find determiner-prefixed terms and check if bare form exists in union."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: DETERMINER-PREFIXED TERMS")
    print("=" * 80)
    
    det_pattern = re.compile(r"^(a|an|the|some|any|my|this|that|these|those)\s+(.+)$", re.I)
    
    for doc in docs:
        doc_id = doc["doc_id"]
        gt_lower = {t.lower() for t in doc["gt_terms"]}
        union_lower = {t.lower() for t in doc["raw_union"]}
        
        det_terms = []
        for term in doc["raw_union"]:
            m = det_pattern.match(term)
            if m:
                det = m.group(1)
                remainder = m.group(2)
                bare_exists = remainder.lower() in union_lower
                in_gt = term.lower() in gt_lower
                remainder_in_gt = remainder.lower() in gt_lower
                det_terms.append((term, det, remainder, bare_exists, in_gt, remainder_in_gt))
        
        if det_terms:
            print(f"\n  {doc_id}:")
            for term, det, remainder, bare_exists, in_gt, rem_gt in det_terms:
                gt_status = "GT" if in_gt else "FP"
                bare_note = f"bare '{remainder}' in union={'YES' if bare_exists else 'NO'}"
                rem_gt_note = f", bare in GT={'YES' if rem_gt else 'NO'}"
                print(f"    [{gt_status}] '{term}' → det='{det}' remainder='{remainder}' | {bare_note}{rem_gt_note}")


def analyze_so_metadata(docs):
    """Find SO metadata patterns across all docs."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: STACKOVERFLOW METADATA IN RAW UNION")
    print("=" * 80)
    
    # Known SO metadata patterns
    metadata_patterns = [
        re.compile(r"Answer_to_Question_ID", re.I),
        re.compile(r"Question_ID", re.I),
        re.compile(r"^Q\d{5,}$"),
        re.compile(r"^\d{6,}$"),  # Long numbers that could be SO IDs
    ]
    
    # Also check for patterns that look like SO data
    for doc in docs:
        doc_id = doc["doc_id"]
        gt_lower = {t.lower() for t in doc["gt_terms"]}
        
        metadata = []
        for term in doc["raw_union"]:
            for pat in metadata_patterns:
                if pat.search(term):
                    in_gt = "GT" if term.lower() in gt_lower else "FP"
                    metadata.append((term, in_gt, pat.pattern))
                    break
        
        if metadata:
            print(f"\n  {doc_id}:")
            for term, status, pattern in metadata:
                print(f"    [{status}] '{term}' (matched: {pattern})")
    
    # Also search for Answer_to_Question_ID in doc texts
    print(f"\n  --- Searching doc texts for 'Answer_to_Question_ID' ---")
    for doc in docs:
        if "Answer_to_Question_ID" in doc["text"]:
            print(f"    {doc['doc_id']}: FOUND in text")
        else:
            print(f"    {doc['doc_id']}: not in text")


def analyze_percentages(docs, train_docs):
    """Check percentage-like terms in training data GT."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: PERCENTAGE/DECIMAL TERMS IN RAW UNION + TRAINING DATA")
    print("=" * 80)
    
    pct_pattern = re.compile(r"^\d+\.?\d*\s*%$")
    decimal_pattern = re.compile(r"^\d+\.\d+$")
    
    # Check raw union
    print("\n  --- In 10-doc test fixture ---")
    for doc in docs:
        doc_id = doc["doc_id"]
        gt_lower = {t.lower() for t in doc["gt_terms"]}
        
        pct_terms = []
        for term in doc["raw_union"]:
            if pct_pattern.match(term) or decimal_pattern.match(term):
                in_gt = "GT" if term.lower() in gt_lower else "FP"
                pct_terms.append((term, in_gt))
        
        if pct_terms:
            print(f"\n    {doc_id}:")
            for term, status in pct_terms:
                print(f"      [{status}] '{term}'")
    
    # Check ALL training data GT terms for percentage/decimal patterns
    print("\n  --- Searching ALL 741 training docs GT for percentage/decimal patterns ---")
    pct_in_gt = []
    decimal_in_gt = []
    version_in_gt = []
    
    for doc in train_docs:
        for term in doc["gt_terms"]:
            if pct_pattern.match(term.strip()):
                pct_in_gt.append((doc["doc_id"], term))
            elif decimal_pattern.match(term.strip()):
                # Distinguish version-like from pure decimal
                if re.match(r"^\d+\.\d+\.\d+", term.strip()):
                    version_in_gt.append((doc["doc_id"], term))
                else:
                    decimal_in_gt.append((doc["doc_id"], term))
    
    print(f"\n    Terms matching X.Y% pattern in GT: {len(pct_in_gt)}")
    for doc_id, term in pct_in_gt[:20]:
        print(f"      {doc_id}: '{term}'")
    
    print(f"\n    Terms matching X.Y (bare decimal) pattern in GT: {len(decimal_in_gt)}")
    for doc_id, term in decimal_in_gt[:30]:
        print(f"      {doc_id}: '{term}'")
    
    print(f"\n    Terms matching X.Y.Z+ (version) pattern in GT: {len(version_in_gt)}")
    for doc_id, term in version_in_gt[:20]:
        print(f"      {doc_id}: '{term}'")


def analyze_adjective_noun(docs):
    """Analyze adjective+noun patterns - what's FP, what's GT, what's already filtered."""
    print("\n" + "=" * 80)
    print("ANALYSIS 5: ADJECTIVE+NOUN PATTERNS IN RAW UNION")
    print("=" * 80)
    
    # Current DESCRIPTIVE_ADJECTIVES from hybrid_ner.py
    current_adjectives = {
        "basic", "simple", "main", "original", "new", "old", "current",
        "default", "standard", "generic", "general", "common", "normal",
        "typical", "regular", "similar", "different", "various", "multiple",
        "specific", "particular", "individual", "single", "double", "triple",
        "primary", "secondary", "above", "below", "certain", "existing",
        "given", "known", "actual", "real", "corresponding", "respective",
        "relevant", "associated", "related", "entire", "whole", "full",
        "complete", "total", "initial", "final", "previous", "following",
        "additional", "extra", "separate", "proper", "correct", "wrong",
        "invalid", "bad", "good", "best", "first", "second", "third",
        "last", "latest", "same", "other", "inner", "outer", "hidden",
        "visible", "absolute", "relative", "vertical", "horizontal",
    }
    
    # Proposed additions
    proposed_adjectives = {
        "lower", "upper", "next", "max", "min", "only", "away",
        "custom", "native", "available", "expected", "possible",
    }
    
    for doc in docs:
        doc_id = doc["doc_id"]
        gt_lower = {t.lower() for t in doc["gt_terms"]}
        
        adj_noun = []
        for term in doc["raw_union"]:
            words = term.lower().split()
            if len(words) == 2:
                adj = words[0]
                noun = words[1]
                in_gt = term.lower() in gt_lower
                in_current = adj in current_adjectives
                in_proposed = adj in proposed_adjectives
                if in_current or in_proposed or not in_gt:
                    adj_noun.append((term, in_gt, in_current, in_proposed))
        
        if adj_noun:
            print(f"\n  {doc_id}:")
            for term, in_gt, in_cur, in_prop in adj_noun:
                gt_status = "GT" if in_gt else "FP"
                filter_status = "ALREADY FILTERED" if in_cur else ("PROPOSED FILTER" if in_prop else "NOT FILTERED")
                print(f"    [{gt_status}] '{term}' → adj filter: {filter_status}")


def analyze_trailing_suffixes(docs):
    """Analyze terms with trailing generic suffixes."""
    print("\n" + "=" * 80)
    print("ANALYSIS 6: TRAILING GENERIC SUFFIXES (of, documentation, repository, etc.)")
    print("=" * 80)
    
    suffix_pattern = re.compile(r"\s+(of|documentation|repository|session|level|values?|size)$", re.I)
    
    for doc in docs:
        doc_id = doc["doc_id"]
        gt_lower = {t.lower() for t in doc["gt_terms"]}
        union_lower = {t.lower() for t in doc["raw_union"]}
        
        suffix_terms = []
        for term in doc["raw_union"]:
            m = suffix_pattern.search(term)
            if m:
                stripped = term[:m.start()]
                in_gt = term.lower() in gt_lower
                stripped_in_union = stripped.lower() in union_lower
                stripped_in_gt = stripped.lower() in gt_lower
                suffix_terms.append((term, m.group().strip(), stripped, in_gt, stripped_in_union, stripped_in_gt))
        
        if suffix_terms:
            print(f"\n  {doc_id}:")
            for term, suffix, stripped, in_gt, strip_union, strip_gt in suffix_terms:
                gt_status = "GT" if in_gt else "FP"
                strip_note = f"stripped='{stripped}' in union={'YES' if strip_union else 'NO'}, in GT={'YES' if strip_gt else 'NO'}"
                print(f"    [{gt_status}] '{term}' suffix='{suffix}' | {strip_note}")


def main():
    docs, train = load_data()
    
    print(f"Loaded {len(docs)} test docs, {len(train)} training docs\n")
    
    # Summary stats
    total_union = sum(len(d["raw_union"]) for d in docs)
    total_gt = sum(len(d["gt_terms"]) for d in docs)
    print(f"Total raw_union terms: {total_union}")
    print(f"Total GT terms: {total_gt}")
    
    analyze_bare_numbers(docs)
    analyze_determiner_prefixed(docs)
    analyze_so_metadata(docs)
    analyze_percentages(docs, train)
    analyze_adjective_noun(docs)
    analyze_trailing_suffixes(docs)
    
    # Final: count all FPs in raw_union (terms not matching any GT)
    print("\n" + "=" * 80)
    print("ANALYSIS 7: FULL FP LIST PER DOC (raw_union terms not in GT)")
    print("=" * 80)
    
    from scoring import v3_match
    
    for doc in docs:
        doc_id = doc["doc_id"]
        gt = doc["gt_terms"]
        
        fps = []
        for term in doc["raw_union"]:
            matched = any(v3_match(term, g) for g in gt)
            if not matched:
                fps.append(term)
        
        print(f"\n  {doc_id}: {len(fps)} FPs out of {len(doc['raw_union'])} union terms (GT={len(gt)})")
        for fp in sorted(fps, key=str.lower):
            print(f"    FP: '{fp}'")


if __name__ == "__main__":
    main()
