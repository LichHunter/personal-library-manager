"""LLM response parsing utilities."""

import json
import re


def _parse_terms_json(response: str) -> list[str]:
    """Parse LLM response that returns {"terms": [...]} or [...] format."""
    response = response.strip()

    # Try {"terms": [...]}
    obj_match = re.search(r"\{.*\}", response, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict) and "terms" in parsed:
                terms = parsed["terms"]
                if isinstance(terms, list):
                    return [str(t).strip() for t in terms if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass

    # Try bare [...]
    arr_match = re.search(r"\[.*\]", response, re.DOTALL)
    if arr_match:
        try:
            terms = json.loads(arr_match.group())
            if isinstance(terms, list):
                return [str(t).strip() for t in terms if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: line-by-line
    entities: list[str] = []
    for line in response.splitlines():
        line = line.strip().strip("-*•").strip().strip('"').strip("'").strip(",").strip()
        if line and len(line) >= 2 and not line.startswith("{") and not line.startswith("["):
            entities.append(line)
    return entities
