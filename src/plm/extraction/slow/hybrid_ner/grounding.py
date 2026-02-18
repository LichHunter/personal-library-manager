"""Stage 2: Grounding (span verification) + deduplication."""

from ..scoring import normalize_term, verify_span


def _ground_and_dedup(
    candidates_by_source: dict[str, list[str]],
    doc_text: str,
) -> dict[str, dict]:
    """Verify spans, normalize, and deduplicate candidates from all sources.

    Returns: {normalized_key: {"term": best_form, "sources": set, "source_count": int}}
    """
    merged: dict[str, dict] = {}

    for source_name, terms in candidates_by_source.items():
        seen_in_source: set[str] = set()
        for term in terms:
            key = normalize_term(term)
            if key in seen_in_source:
                continue
            seen_in_source.add(key)

            # Span verification
            ok, _ = verify_span(term, doc_text)
            if not ok:
                continue

            if key not in merged:
                merged[key] = {
                    "term": term,
                    "sources": set(),
                    "source_count": 0,
                }
            merged[key]["sources"].add(source_name)
            merged[key]["source_count"] = len(merged[key]["sources"])

            # Prefer capitalized forms
            existing = merged[key]["term"]
            if term[0].isupper() and existing[0].islower():
                merged[key]["term"] = term
            elif len(term) > len(existing) and term.lower() == existing.lower():
                merged[key]["term"] = term

    return merged
