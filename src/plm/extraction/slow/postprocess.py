"""Stage 5: Post-processing for slow extraction pipeline.

Expands partial spans to full entities, suppresses subspans,
and performs final deduplication.
Ported from poc-1c-scalable-ner/hybrid_ner.py.
"""

import re

from .grounding import normalize_term


def expand_spans(terms: list[str], text: str) -> list[str]:
    """Expand partial spans to full entity mentions.
    
    Tries to extend terms to include:
    - Parenthesized arguments: "func" -> "func(arg)"
    - Method prefixes: "map()" -> "Array.prototype.map()"
    - Version suffixes: "Python" -> "Python 3.8"
    - Slash-separated paths: "bin" -> "/usr/bin"
    - Multi-word entity names: "REST" -> "REST API"
    
    Args:
        terms: List of terms to expand
        text: Document text for context
        
    Returns:
        List of expanded terms
    """
    return [_try_expand_span(t, text) for t in terms]


def _try_expand_span(term: str, text: str) -> str:
    """Try to expand a single span to include more context."""
    t = term.strip()
    text_lower = text.lower()
    t_lower = t.lower()
    
    idx = text_lower.find(t_lower)
    if idx == -1:
        return term
    
    # Expand to include parenthesized content after term
    if not t.endswith(")"):
        remainder = text[idx + len(t):]
        if remainder.startswith("("):
            depth, end = 1, 1
            while depth > 0 and end < len(remainder):
                if remainder[end] == "(":
                    depth += 1
                elif remainder[end] == ")":
                    depth -= 1
                end += 1
            candidate = text[idx:idx + len(t) + end]
            if len(candidate) <= 80:
                return candidate
    
    # Expand to include method chain prefix (e.g., "map()" -> "Array.prototype.map()")
    if idx > 0 and t.endswith(")"):
        prefix_text = text[:idx]
        prefix_match = re.search(r'([\w:]+\.)+$', prefix_text)
        if prefix_match:
            candidate = prefix_match.group(0) + text[idx:idx + len(t)]
            if len(candidate) <= 80:
                return candidate
    
    # Expand to include version suffix (e.g., "Python" -> "Python 3.8 server")
    remainder = text[idx + len(t):]
    version_match = re.match(r'(\s+\d+[A-Za-z]*(?:\s+\w+)?)', remainder)
    if version_match:
        candidate = text[idx:idx + len(t)] + version_match.group(1)
        candidate_lower = candidate.lower()
        if candidate_lower.endswith(" server") or candidate_lower.endswith(" client"):
            return candidate
    
    # Expand error codes (e.g., "error 500" -> "server internal error 500")
    if re.match(r'(error|internal)\s+\d+', t, re.I):
        prefix_text = text[:idx]
        err_prefix = re.search(
            r'((?:server\s+)?(?:internal\s+)?(?:server\s+)?)$',
            prefix_text, re.I
        )
        if err_prefix and err_prefix.group(1).strip():
            candidate = err_prefix.group(1) + text[idx:idx + len(t)]
            if len(candidate) <= 80:
                return candidate
    
    # Expand to include slash-separated paths
    slash_match = re.search(
        rf'(\S+/)?{re.escape(t)}(/\S+)?',
        text, re.IGNORECASE
    )
    if slash_match and (slash_match.group(1) or slash_match.group(2)):
        candidate = slash_match.group(0)
        if len(candidate) <= 100 and "/" in candidate:
            return candidate
    
    # Expand multi-word entity names (e.g., "REST" -> "REST API")
    remainder = text[idx + len(t):idx + len(t) + 50]
    multiword_match = re.match(r'(\s+\w+){1,3}', remainder)
    if multiword_match and not t.endswith(")") and " " not in t:
        candidate = text[idx:idx + len(t)] + multiword_match.group(0)
        if re.search(r'\b(API|SDK|framework|library|protocol)\b', candidate, re.I):
            return candidate.strip()
    
    return term


def suppress_subspans(
    terms: list[str],
    protected: set[str] | None = None,
) -> list[str]:
    """Remove terms that are substrings of other extracted terms.
    
    Only suppresses when the shorter term is a non-word-boundary substring
    (e.g., "getInputSizes" inside "getInputSizes(ImageFormat...)") or when
    a single word appears inside a 3+-word compound. Preserves both forms
    for 2-word pairs like "Right" / "arrow right".
    
    Args:
        terms: List of extracted terms
        protected: Set of terms to never suppress (e.g., known common-word entities)
        
    Returns:
        List with subspans removed
    """
    if not terms:
        return terms
    
    _protected = protected or set()
    lower_terms = [(t, t.lower()) for t in terms]
    kept: list[str] = []
    
    for term, term_lower in lower_terms:
        # Very short terms always kept
        if len(term) <= 3:
            kept.append(term)
            continue
        
        # Protected terms never suppressed
        if term_lower in _protected:
            kept.append(term)
            continue
        
        is_subspan = False
        for other, other_lower in lower_terms:
            if term_lower == other_lower:
                continue
            if term_lower not in other_lower or len(term_lower) >= len(other_lower):
                continue
            
            shorter_words = term_lower.split()
            longer_words = other_lower.split()
            
            # Preserve both forms for 2-word pairs (e.g., "Right" / "arrow right")
            if len(shorter_words) == 1 and len(longer_words) == 2:
                continue
            
            is_subspan = True
            break
        
        if not is_subspan:
            kept.append(term)
    
    return kept


def final_dedup(terms: list[str]) -> list[str]:
    """Final deduplication by normalized key.
    
    Removes duplicate terms that normalize to the same key,
    keeping the first occurrence.
    
    Args:
        terms: List of terms (potentially with duplicates)
        
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


def is_embedded_in_path(term: str, text: str) -> bool:
    """Check if term only appears inside brace-expansion paths.
    
    Terms like "ruby" that only appear in paths like
    "/usr/bin/{erb,gem,irb,ruby,testrb}" should be filtered.
    
    Args:
        term: Term to check
        text: Document text
        
    Returns:
        True if term only appears in brace expansions
    """
    tl = term.lower()
    brace_pattern = r"\{[^}]*\b" + re.escape(tl) + r"\b[^}]*\}"
    
    if re.search(brace_pattern, text.lower()):
        # Check if term appears outside braces
        text_no_braces = re.sub(r"\{[^}]+\}", "", text.lower())
        if not re.search(rf"\b{re.escape(tl)}\b", text_no_braces):
            return True
    
    return False


def filter_path_embedded(terms: list[str], text: str) -> list[str]:
    """Remove terms that only appear inside brace-expansion paths.
    
    Args:
        terms: List of terms
        text: Document text
        
    Returns:
        Filtered list
    """
    return [t for t in terms if not is_embedded_in_path(t, text)]


def filter_urls(terms: list[str]) -> list[str]:
    """Remove terms that look like URLs.
    
    Args:
        terms: List of terms
        
    Returns:
        Filtered list without URL-like terms
    """
    url_re = re.compile(r"https?://\S+|^www\.\S+", re.I)
    return [t for t in terms if not url_re.search(t)]
