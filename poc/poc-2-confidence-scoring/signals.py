"""
Confidence Signal Calculators for Extraction Quality Assessment

Phase 0: Core signal functions for evaluating extraction confidence.
"""


def known_term_ratio(terms: list[str], vocab: set[str]) -> float:
    """
    Calculate the ratio of extracted terms found in vocabulary.
    
    Args:
        terms: List of extracted terms
        vocab: Set of known vocabulary terms
        
    Returns:
        Float between 0.0 and 1.0 representing the ratio of known terms
        
    Examples:
        >>> known_term_ratio(['python', 'java'], {'python', 'c++'})
        0.5
        >>> known_term_ratio([], {'python'})
        0.0
    """
    if not terms:
        return 0.0
    known = sum(1 for t in terms if t.lower() in vocab)
    return known / len(terms)


def coverage_score(terms: list[str], text: str) -> float:
    """
    Calculate the percentage of text "covered" by extracted terms (character-based).
    
    Args:
        terms: List of extracted terms
        text: Original text from which terms were extracted
        
    Returns:
        Float between 0.0 and 1.0 representing the coverage ratio
        
    Examples:
        >>> coverage_score(['hello', 'world'], 'hello world')
        1.0
        >>> coverage_score(['hello'], 'hello world')
        0.5
        >>> coverage_score([], 'hello world')
        0.0
    """
    if not text:
        return 0.0
    covered_chars = sum(len(t) * text.lower().count(t.lower()) for t in terms)
    return min(covered_chars / len(text), 1.0)


def entity_density(terms: list[str], text: str) -> float:
    """
    Calculate entity density as terms per 100 tokens.
    
    Args:
        terms: List of extracted terms
        text: Original text from which terms were extracted
        
    Returns:
        Float representing the number of terms per 100 tokens
        
    Examples:
        >>> entity_density(['python'], 'python is great')
        33.33333333333333
        >>> entity_density([], 'python is great')
        0.0
    """
    tokens = len(text.split())
    if tokens == 0:
        return 0.0
    return (len(terms) / tokens) * 100


def section_type_mismatch(terms: list[str], section_type: str | None) -> float:
    """
    Calculate section type mismatch penalty.
    
    Stub implementation for Stack Overflow data (no sections).
    Returns 0.0 (no mismatch) for SO data; will be implemented for docs later.
    
    Args:
        terms: List of extracted terms
        section_type: Type of section (e.g., 'title', 'body', 'code')
        
    Returns:
        Float between 0.0 and 1.0 representing mismatch penalty
        
    Examples:
        >>> section_type_mismatch(['python'], None)
        0.0
        >>> section_type_mismatch(['python'], 'body')
        0.0
    """
    # Returns 0.0 (no mismatch) for SO data; implement for docs later
    return 0.0
