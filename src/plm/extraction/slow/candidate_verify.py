"""Stage 1 part: Heuristic candidate extraction and LLM verification.

Extracts candidates using structural patterns (zero-cost heuristics),
then optionally verifies with LLM classification.
Ported from poc-1c-scalable-ner/hybrid_ner.py.
"""

import json
import re

from plm.shared.llm import call_llm


# ALL_CAPS terms to exclude (common non-entity acronyms)
_ALLCAPS_EXCLUDE: set[str] = {
    "THE", "AND", "BUT", "NOT", "FOR", "ARE", "WAS", "HAS", "HAD",
    "SO", "IT", "IS", "OR", "MY", "IN", "TO", "OF", "AT", "BY",
    "ON", "AN", "AS", "IF", "NO", "DO", "UP", "BE", "AM", "HE",
    "OUTPUT", "UPDATE", "EDIT", "OK", "ERROR", "WARNING", "INFO",
    "DEBUG", "TRUE", "FALSE", "NULL", "NONE", "TODO", "FIXME",
    "NOTE", "CODE", "LEVEL", "NEW", "END", "SET", "GET", "ADD",
    "PUT", "DELETE", "POST", "THEN", "ELSE", "CASE", "WHEN",
    "WITH", "FROM", "INTO", "LIKE", "WHERE", "ORDER", "GROUP",
}


# Prompt for LLM candidate review
_REVIEW_PROMPT = """\
You are reviewing candidate software named entities extracted from a StackOverflow \
post. Each candidate was found by only ONE extractor (low confidence).

ENTITY DEFINITION:
A term is an ENTITY if it names a SPECIFIC technical thing: library, class, \
function, language, data type (string, int, boolean, long, private), data \
structure (array, list, table, row, column, image), UI element (button, page, \
form, keyboard, pad), file format, error name (exception), device, website, \
application (server, console, browser), keyboard key, version, constant, enum.

COMMON WORDS ARE ENTITIES when naming specific technical things in context.

NOT entities: descriptive phrases, process descriptions, adjectives, generic \
vocabulary without a specific referent.

TEXT:
{content}

CANDIDATES:
{terms_json}

APPROVE if the term names a specific technical thing per the definition above. \
REJECT if it is a descriptive phrase, process description, adjective, or generic \
vocabulary. When in doubt about a recognized tech term, APPROVE.

Output JSON: {{"terms": [{{"term": "...", "decision": "APPROVE|REJECT"}}]}}
"""


def extract_candidates_heuristic(text: str) -> list[str]:
    """Extract software entity candidates using structural/heuristic patterns.
    
    Finds code-like patterns with zero LLM cost:
    - CamelCase identifiers (e.g., ListView, NSMutableArray)
    - lowerCamelCase (e.g., getElementById, setChoiceMode)
    - Dot-separated identifiers (e.g., R.layout.file.xml)
    - Function calls (e.g., recv(), querySelector())
    - ALL_CAPS acronyms (e.g., JSON, HTML, CPU)
    - Backtick-wrapped terms
    - CSS class selectors (e.g., .long, .container)
    - Brace-expansion paths (e.g., /usr/bin/{erb,gem,ruby})
    - Unix file paths (e.g., /usr/bin/ruby)
    
    Args:
        text: Document text to extract from
        
    Returns:
        List of candidate terms (may have duplicates)
    """
    found: set[str] = set()
    
    # 1. CamelCase identifiers (e.g., ListView, NSMutableArray, getElementById)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b", text):
        term = m.group(1)
        if len(term) >= 4:
            found.add(term)
    
    # 2. lowerCamelCase (e.g., getElementById, setChoiceMode)
    for m in re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text):
        term = m.group(1)
        if len(term) >= 4:
            found.add(term)
    
    # 3. Dot-separated identifiers (e.g., R.layout.file.xml, System.out)
    for m in re.finditer(r"\b([\w]+(?:\.[\w]+){1,5})\b", text):
        term = m.group(1)
        if "." in term and not re.match(r"^\d+\.\d+(\.\d+)*$", term):
            parts = term.split(".")
            if all(len(p) >= 1 for p in parts) and len(term) >= 4:
                found.add(term)
    
    # 4. Function calls with parentheses (e.g., recv(), querySelector())
    for m in re.finditer(r"\b([\w.]+\(\))", text):
        term = m.group(1)
        if len(term) >= 4:
            found.add(term)
    
    # 5. ALL_CAPS acronyms (e.g., JSON, HTML, CPU, API)
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]{1,15})\b", text):
        term = m.group(1)
        if len(term) >= 2 and term not in _ALLCAPS_EXCLUDE:
            found.add(term)
    
    # 6. Backtick-wrapped terms (in case text preserves them)
    for m in re.finditer(r"`([^`]{2,50})`", text):
        term = m.group(1).strip()
        if term and len(term) >= 2:
            found.add(term)
    
    # 7. CSS class selectors (e.g., .long, .container, .my-class)
    for m in re.finditer(r"(?<!\w)(\.[a-zA-Z][a-zA-Z0-9_-]*)\b", text):
        term = m.group(1)
        # Skip common sentence-ending periods
        if len(term) >= 3:
            found.add(term)
    
    # 8. Brace-expansion paths (e.g., /usr/bin/{erb,gem,irb,rdoc,ri,ruby,testrb})
    for m in re.finditer(r"(/[^\s]*\{[^}]+\}[^\s]*)", text):
        term = m.group(1).strip()
        if term:
            found.add(term)
    
    # 9. Unix-style file paths (e.g., /usr/bin/ruby, /System/Library/Frameworks)
    for m in re.finditer(r"(/(?:[\w.+-]+/)+[\w.+-]*)", text):
        term = m.group(1)
        # Skip pure numeric paths
        if len(term) >= 4 and not re.match(r"^/\d+(/\d+)*$", term):
            found.add(term)
    
    return list(found)


def classify_candidates_llm(
    candidates: list[str],
    text: str,
    model: str = "sonnet",
) -> list[str]:
    """Classify candidates using LLM to filter false positives.
    
    Sends candidates to LLM for APPROVE/REJECT classification.
    Returns only approved terms.
    
    Args:
        candidates: List of candidate terms to classify
        text: Document text for context
        model: LLM model to use (default: "sonnet")
        
    Returns:
        List of approved terms
    """
    if not candidates:
        return []
    
    terms_json = json.dumps(candidates, indent=2)
    prompt = _REVIEW_PROMPT.format(
        content=text[:3000],
        terms_json=terms_json,
    )
    
    response = call_llm(prompt, model=model, max_tokens=3000, temperature=0.0)
    
    # Parse response
    decisions: dict[str, str] = {}
    try:
        json_match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            for item in parsed.get("terms", []):
                decisions[item["term"]] = item.get("decision", "APPROVE")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    
    # Return approved terms
    approved = []
    for term in candidates:
        decision = decisions.get(term, "REJECT")  # Default to REJECT if not found
        if decision == "APPROVE":
            approved.append(term)
    
    return approved
