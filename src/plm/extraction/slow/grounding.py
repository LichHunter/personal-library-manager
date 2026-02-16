"""Stage 2: Grounding and deduplication for slow extraction pipeline.

Verifies that extracted terms appear in source text and normalizes/deduplicates.
Ported from poc-1c-scalable-ner/scoring.py and hybrid_ner.py.
"""

import re


def normalize_term(term: str) -> str:
    """Normalize term for matching and deduplication.
    
    Converts to lowercase, strips whitespace, normalizes hyphens/underscores.
    
    Args:
        term: Raw term string
        
    Returns:
        Normalized term key
    """
    return term.lower().strip().replace("-", " ").replace("_", " ")


def verify_span(term: str, content: str) -> tuple[bool, str]:
    """Verify that a term appears in the document content.
    
    Checks multiple match strategies: exact, normalized, camelcase split,
    and singular/plural variants.
    
    Args:
        term: Candidate term to verify
        content: Document text to search in
        
    Returns:
        Tuple of (found: bool, match_type: str)
        match_type is one of: "too_short", "exact", "normalized", "camelcase",
        "singular_of_plural", "plural_of_singular", "singular_of_plural_es", "none"
    """
    if not term or len(term) < 2:
        return False, "too_short"
    
    content_lower = content.lower()
    term_lower = term.lower().strip()
    
    # Exact match
    if term_lower in content_lower:
        return True, "exact"
    
    # Normalized match (hyphens/underscores as spaces)
    term_norm = term_lower.replace("-", " ").replace("_", " ")
    content_norm = content_lower.replace("-", " ").replace("_", " ")
    if term_norm in content_norm:
        return True, "normalized"
    
    # CamelCase split match
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True, "camelcase"
    
    # Singular of plural
    if term_lower.endswith("s") and len(term_lower) > 3 and term_lower[:-1] in content_lower:
        return True, "singular_of_plural"
    
    # Plural of singular
    if not term_lower.endswith("s") and (term_lower + "s") in content_lower:
        return True, "plural_of_singular"
    
    # Singular of -es plural
    if term_lower.endswith("es") and len(term_lower) > 4 and term_lower[:-2] in content_lower:
        return True, "singular_of_plural_es"
    
    return False, "none"


def ground_candidates(
    candidates_by_source: dict[str, list[str]],
    content: str,
) -> dict[str, dict]:
    """Verify spans, normalize, and merge candidates from multiple sources.
    
    For each candidate term from each source:
    1. Verify it appears in the document (span verification)
    2. Normalize to create a dedup key
    3. Track which sources found each term
    4. Prefer capitalized/longer forms when merging
    
    Args:
        candidates_by_source: Dict mapping source name to list of terms
            e.g. {"llm": ["React", "react"], "heuristic": ["useState()"]}
        content: Document text for span verification
        
    Returns:
        Dict mapping normalized key to candidate info:
        {
            "normalized_key": {
                "term": "BestForm",
                "sources": {"llm", "heuristic"},
                "source_count": 2
            }
        }
    """
    merged: dict[str, dict] = {}
    
    for source_name, terms in candidates_by_source.items():
        seen_in_source: set[str] = set()
        
        for term in terms:
            key = normalize_term(term)
            
            # Skip duplicates within same source
            if key in seen_in_source:
                continue
            seen_in_source.add(key)
            
            # Span verification - term must appear in document
            ok, _ = verify_span(term, content)
            if not ok:
                continue
            
            # Initialize or update entry
            if key not in merged:
                merged[key] = {
                    "term": term,
                    "sources": set(),
                    "source_count": 0,
                }
            
            merged[key]["sources"].add(source_name)
            merged[key]["source_count"] = len(merged[key]["sources"])
            
            # Prefer capitalized forms over lowercase
            existing = merged[key]["term"]
            if term[0].isupper() and existing[0].islower():
                merged[key]["term"] = term
            # Prefer longer forms when same normalized key
            elif len(term) > len(existing) and term.lower() == existing.lower():
                merged[key]["term"] = term
    
    return merged


def deduplicate(terms: list[str]) -> list[str]:
    """Deduplicate a list of terms by normalized key.
    
    Keeps the first occurrence of each normalized term.
    
    Args:
        terms: List of terms to deduplicate
        
    Returns:
        Deduplicated list preserving order
    """
    seen: set[str] = set()
    result: list[str] = []
    
    for term in terms:
        key = normalize_term(term)
        if key not in seen:
            seen.add(key)
            result.append(term)
    
    return result
