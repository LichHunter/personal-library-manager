"""Stage 3: Noise filtering for slow extraction pipeline.

Filters out stop words, generic phrases, and known false positives.
Ported from poc-1c-scalable-ner/hybrid_ner.py.
"""

import json
import re
from pathlib import Path


# Stop words that are never technical entities
PURE_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "so", "if", "than",
    "about", "up", "out", "no", "just", "also", "more",
    "some", "any", "all", "each", "every", "both",
}

# Gerunds that describe actions, not entities
ACTION_GERUNDS: set[str] = {
    "appending", "serializing", "deserializing", "floating", "wrapping",
    "loading", "downloading", "subscribing", "referencing", "toggling",
    "de-serialized", "cross-platform", "cross-compile",
}

# Adjectives that modify nouns but aren't entities themselves
DESCRIPTIVE_ADJECTIVES: set[str] = {
    "hidden", "visible", "vertical", "horizontal", "floating",
    "absolute", "relative", "nested", "multiple", "various",
    "specific", "general", "dynamic", "static", "custom",
    "native", "proper", "basic", "simple", "complex",
    "actual", "original", "current", "previous", "following",
    "hardware", "software", "entropy", "random", "external",
    "internal", "main", "local", "global", "primary", "secondary",
    "default", "existing", "standard", "typical", "generic",
    "certain", "optional", "required", "initial", "final",
    "entire", "single", "separate", "different", "additional",
}

# Category suffixes that make phrases too generic
CATEGORY_SUFFIXES: set[str] = {
    "items", "elements", "values", "settings", "parameters",
    "options", "properties", "fields", "catalog", "orientation",
    "behavior", "handling", "management", "compatibility",
    "content", "position", "libraries", "events", "factors",
    "pool", "level", "keys", "engine", "mode", "navigation",
    "access", "support", "system", "design", "architecture",
    "configuration", "implementation", "specification",
}


def load_negatives(vocab_path: Path | None = None) -> set[str]:
    """Load the negatives vocabulary from file.
    
    Negatives are terms that appear frequently in tech text but
    are never entities (safe to always reject).
    
    Args:
        vocab_path: Path to tech_domain_negatives.json.
            Defaults to data/vocabularies/tech_domain_negatives.json
            
    Returns:
        Set of negative terms (lowercase)
    """
    if vocab_path is None:
        # Default path relative to package
        vocab_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "data" / "vocabularies" / "tech_domain_negatives.json"
        )
    
    if not vocab_path.exists():
        return set()
    
    with open(vocab_path) as f:
        data = json.load(f)
    
    # Collect words from all safe_* lists
    negatives: set[str] = set()
    for key in ["safe_1000", "safe_500", "safe_200", "safe_100"]:
        for entry in data.get(key, []):
            negatives.add(entry["word"].lower())
    
    return negatives


def is_stop_word(term: str) -> bool:
    """Check if term is a pure stop word.
    
    Args:
        term: Term to check
        
    Returns:
        True if term is a stop word
    """
    return term.lower().strip() in PURE_STOP_WORDS


def is_generic_phrase(term: str) -> bool:
    """Check if term is a generic descriptive phrase.
    
    Generic phrases are:
    - Action gerunds (e.g., "serializing data")
    - Adjective + noun combos (e.g., "hidden fields")
    - Noun + category suffix (e.g., "configuration options")
    - 3+ word phrases without code markers
    
    Args:
        term: Term to check
        
    Returns:
        True if term is a generic phrase
    """
    t = term.lower().strip()
    
    # Action gerunds
    if t in ACTION_GERUNDS:
        return True
    
    words = t.split()
    
    # Two-word phrases with descriptive adjective
    if len(words) == 2:
        if words[0] in DESCRIPTIVE_ADJECTIVES:
            return True
        if words[1] in CATEGORY_SUFFIXES:
            return True
    
    # 3+ word phrases without code markers
    if len(words) >= 3:
        # Check for code markers that indicate real entities
        if not re.search(r"[A-Z]", term[1:]):  # No camelCase
            if not re.search(r"[()._::<>\[\]]", term):  # No code punctuation
                return True
    
    return False


def filter_noise(
    terms: list[str],
    negatives: set[str] | None = None,
    bypass: set[str] | None = None,
    doc_text: str | None = None,
) -> list[str]:
    """Filter out noise terms from candidate list.
    
    Removes:
    - Empty or single-char terms
    - Pure numbers
    - Stop words
    - Action gerunds
    - URLs
    - Known negatives (from vocabulary)
    - Generic phrases (unless in bypass set)
    - Orphan version numbers
    
    Args:
        terms: List of candidate terms
        negatives: Set of known negative terms (loaded from vocab)
        bypass: Set of terms to never filter (known good entities)
        doc_text: Document text (for context-aware filtering)
        
    Returns:
        Filtered list of terms
    """
    if negatives is None:
        negatives = set()
    if bypass is None:
        bypass = set()
    
    result: list[str] = []
    
    for term in terms:
        t = term.strip()
        tl = t.lower()
        
        # Basic rejection rules
        if not t or len(t) <= 1:
            continue
        if re.match(r"^\d+$", t):  # Pure numbers
            continue
        if tl in PURE_STOP_WORDS:
            continue
        if tl in ACTION_GERUNDS:
            continue
        
        # URL rejection
        if re.match(r"^https?://", t) or re.match(r"^www\.", t):
            continue
        if re.search(r"https?://\S+", t):
            continue
        
        # Known negatives
        if tl in negatives:
            continue
        
        # Structural patterns are always kept (code markers)
        if _has_code_markers(t):
            result.append(term)
            continue
        
        # Bypass set overrides other filters
        if tl in bypass:
            result.append(term)
            continue
        
        # Jira-style ticket IDs
        if re.match(r"^[A-Z]+-\d+$", t):
            continue
        
        # Generic phrase check
        if is_generic_phrase(t):
            continue
        
        # Orphan version numbers (context-dependent)
        if doc_text and _is_orphan_version(t, doc_text):
            continue
        
        result.append(term)
    
    return result


def _has_code_markers(term: str) -> bool:
    """Check if term has structural markers indicating code entity.
    
    Code markers include:
    - Parentheses, brackets, dots, colons
    - CamelCase
    - ALL_CAPS (2+ chars)
    - Keyboard key patterns
    """
    # Punctuation markers
    if re.search(r"[().\[\]_::<>+]", term):
        return True
    
    # CamelCase
    if re.search(r"[a-z][A-Z]", term):
        return True
    
    # ALL_CAPS
    if re.match(r"^[A-Z][A-Z0-9_]+$", term) and len(term) >= 2:
        return True
    
    # Keyboard key patterns
    if re.match(
        r"^(Left|Right|Up|Down|Ctrl|Alt|Shift|Tab|Enter|Esc|PgUp|PgDn|PageUp|PageDown|Home|End|F\d+)"
        r"(\s+(arrow|key))?$",
        term,
        re.I,
    ):
        return True
    
    # Arrow key combinations
    if re.match(
        r"^(arrow|page|up|down|left|right)"
        r"(\s+(up|down|left|right|arrow|key|keys))"
        r"(\s+(up|down|left|right|arrow|key|keys))?$",
        term,
        re.I,
    ):
        return True
    
    if re.match(r"^(up/down|left/right)\s+(arrow|key|keys)$", term, re.I):
        return True
    
    return False


def _is_orphan_version(term: str, doc_text: str) -> bool:
    """Check if a version number lacks context (orphan).
    
    Version numbers like "1.2" or "2.0" should have context
    like "Python 1.2" or "v1.2" to be valid entities.
    """
    if not re.match(r"^\d+\.\d+(\.\d+)*$", term):
        return False
    
    # Check for context pattern
    ctx_pattern = (
        rf"(?:[A-Za-z][\w.-]*\s+){re.escape(term)}"
        rf"|(?:v|version\s+){re.escape(term)}"
    )
    return not re.search(ctx_pattern, doc_text, re.I)
