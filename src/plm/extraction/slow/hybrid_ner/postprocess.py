"""Stage 5: Span expansion + subspan suppression."""

import re


def _try_expand_span(term: str, text: str) -> str:
    t = term.strip()
    text_lower = text.lower()
    t_lower = t.lower()

    idx = text_lower.find(t_lower)
    if idx == -1:
        return term

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

    if idx > 0 and t.endswith(")"):
        prefix_text = text[:idx]
        prefix_match = re.search(r'([\w:]+\.)+$', prefix_text)
        if prefix_match:
            candidate = prefix_match.group(0) + text[idx:idx + len(t)]
            if len(candidate) <= 80:
                return candidate

    remainder = text[idx + len(t):]
    version_match = re.match(r'(\s+\d+[A-Za-z]*(?:\s+\w+)?)', remainder)
    if version_match:
        candidate = text[idx:idx + len(t)] + version_match.group(1)
        candidate_lower = candidate.lower()
        if candidate_lower.endswith(" server") or candidate_lower.endswith(" client"):
            return candidate

    if re.match(r'(error|internal)\s+\d+', t, re.I):
        prefix_text = text[:idx]
        err_prefix = re.search(r'((?:server\s+)?(?:internal\s+)?(?:server\s+)?)$', prefix_text, re.I)
        if err_prefix and err_prefix.group(1).strip():
            candidate = err_prefix.group(1) + text[idx:idx + len(t)]
            if len(candidate) <= 80:
                return candidate

    slash_match = re.search(
        rf'(\S+/)?{re.escape(t)}(/\S+)?',
        text, re.IGNORECASE
    )
    if slash_match and (slash_match.group(1) or slash_match.group(2)):
        candidate = slash_match.group(0)
        if len(candidate) <= 100 and "/" in candidate:
            return candidate

    remainder = text[idx + len(t):idx + len(t) + 50]
    multiword_match = re.match(r'(\s+\w+){1,3}', remainder)
    if multiword_match and not t.endswith(")") and " " not in t:
        candidate = text[idx:idx + len(t)] + multiword_match.group(0)
        if re.search(r'\b(API|SDK|framework|library|protocol)\b', candidate, re.I):
            return candidate.strip()

    return term


def _is_embedded_in_path(term: str, doc_text: str) -> bool:
    tl = term.lower()
    brace_pattern = r"\{[^}]*\b" + re.escape(tl) + r"\b[^}]*\}"
    if re.search(brace_pattern, doc_text.lower()):
        text_no_braces = re.sub(r"\{[^}]+\}", "", doc_text.lower())
        if not re.search(rf"\b{re.escape(tl)}\b", text_no_braces):
            return True
    return False


def _expand_spans(terms: list[str], doc_text: str) -> list[str]:
    return [_try_expand_span(t, doc_text) for t in terms]


def _suppress_subspans(
    extracted: list[str],
    protected_seeds: set[str] | None = None,
) -> list[str]:
    """Remove terms that are substrings of other extracted terms.

    Only suppresses when the shorter term is a non-word-boundary substring
    (e.g., "getInputSizes" inside "getInputSizes(ImageFormat...)") or when
    a single word appears inside a 3+-word compound. Preserves both forms
    for 2-word pairs like "Right" / "arrow right".

    Terms in *protected_seeds* are never suppressed (they are legitimate
    common-word entities like "image" or "keyboard").
    """
    if not extracted:
        return extracted

    _protected = protected_seeds or set()
    lower_terms = [(t, t.lower()) for t in extracted]
    kept: list[str] = []

    for term, term_lower in lower_terms:
        if len(term) <= 3:
            kept.append(term)
            continue

        # Never suppress protected seed terms
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

            if len(shorter_words) == 1 and len(longer_words) == 2:
                continue

            is_subspan = True
            break

        if not is_subspan:
            kept.append(term)

    return kept
