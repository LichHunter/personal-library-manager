"""
Confidence Signal Calculators for Extraction Quality Assessment

Phase 0: Core signal functions for evaluating extraction confidence.
Phase 4a: New signals for improved correlation with extraction quality.
"""

import re


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


# Phase 4a: New signals for improved correlation

# Patterns for technical naming conventions
_CAMEL_CASE = re.compile(r'^[a-z]+[A-Z][a-zA-Z]*$')  # camelCase
_PASCAL_CASE = re.compile(r'^[A-Z][a-z]+[A-Z][a-zA-Z]*$')  # PascalCase
_CONSTANT_CASE = re.compile(r'^[A-Z][A-Z0-9]*_[A-Z0-9_]+$')  # CONSTANT_CASE (requires underscore)
_SNAKE_CASE = re.compile(r'^[a-z][a-z0-9]*_[a-z0-9_]+$')  # snake_case
_DOT_NOTATION = re.compile(r'^[a-zA-Z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)+$')  # dot.notation
_FILE_PATH = re.compile(r'^[a-zA-Z0-9_/\\-]+\.[a-z]{1,5}$')  # file.ext or path/file.ext
_QUALIFIED_NAME = re.compile(r'^[A-Z][a-zA-Z]*(\.[A-Z][a-zA-Z]*)+$')  # Class.Method


def _is_technical_term(term: str) -> bool:
    """Check if term matches common technical naming patterns."""
    if len(term) < 2:
        return False
    return bool(
        _CAMEL_CASE.match(term)
        or _PASCAL_CASE.match(term)
        or _CONSTANT_CASE.match(term)
        or _SNAKE_CASE.match(term)
        or _DOT_NOTATION.match(term)
        or _FILE_PATH.match(term)
        or _QUALIFIED_NAME.match(term)
    )


def technical_pattern_ratio(terms: list[str]) -> float:
    """
    Calculate ratio of terms matching technical naming patterns.
    
    Patterns: CamelCase, PascalCase, CONSTANT_CASE, snake_case, dot.notation, file.ext
    
    Args:
        terms: List of extracted terms
        
    Returns:
        Float between 0.0 and 1.0 representing ratio of technical pattern matches
        
    Examples:
        >>> technical_pattern_ratio(['DirectoryStream', 'google.maps.Api'])
        1.0
        >>> technical_pattern_ratio(['hello', 'world'])
        0.0
        >>> technical_pattern_ratio([])
        0.0
    """
    if not terms:
        return 0.0
    technical_count = sum(1 for t in terms if _is_technical_term(t))
    return technical_count / len(terms)


def avg_term_length(terms: list[str]) -> float:
    """
    Calculate average character length of extracted terms.
    
    Technical terms typically 5-20 chars. Very short (1-3) or very long (>40)
    terms often indicate noise or extraction issues.
    
    Args:
        terms: List of extracted terms
        
    Returns:
        Float representing average term length (0.0 if no terms)
        
    Examples:
        >>> avg_term_length(['python', 'java'])
        5.5
        >>> avg_term_length(['DirectoryStream'])
        15.0
        >>> avg_term_length([])
        0.0
    """
    if not terms:
        return 0.0
    return sum(len(t) for t in terms) / len(terms)


def text_grounding_score(terms: list[str], text: str) -> float:
    """
    Calculate ratio of terms found verbatim in original text.
    
    Low grounding suggests hallucination - terms were not actually present
    in the source text. Case-insensitive matching.
    
    Args:
        terms: List of extracted terms
        text: Original text from which terms were extracted
        
    Returns:
        Float between 0.0 and 1.0 representing ratio of grounded terms
        
    Examples:
        >>> text_grounding_score(['python', 'java'], 'Python and Java are languages')
        1.0
        >>> text_grounding_score(['xyz'], 'Python is great')
        0.0
        >>> text_grounding_score([], 'any text')
        0.0
    """
    if not terms:
        return 0.0
    if not text:
        return 0.0
    text_lower = text.lower()
    grounded = sum(1 for t in terms if t.lower() in text_lower)
    return grounded / len(terms)
