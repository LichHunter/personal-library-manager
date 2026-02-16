"""Stage 4: Context validation for slow extraction pipeline.

Validates ambiguous common words using LLM context classification.
Routes single lowercase words to Sonnet for ENTITY/GENERIC classification.
Ported from poc-1c-scalable-ner/hybrid_ner.py.
"""

import json
import re

from plm.shared.llm import call_llm


# Context validation prompt
_CONTEXT_VALIDATION_PROMPT = """\
You are classifying candidate software named entities from a StackOverflow post.

ENTITY DEFINITION:
A term is an ENTITY if it names a SPECIFIC technical thing: library, class, \
function, language, data type (string, int, boolean, long, private), data \
structure (array, list, table, row, column, image), UI element (button, page, \
form, keyboard), file format, error name (exception), device, website, \
application (server, console, browser), keyboard key, version.

COMMON WORDS ARE ENTITIES when naming specific technical things:
  "the string is empty" -> string = ENTITY (Data_Type)
  "stored in a table" -> table = ENTITY (Data_Structure)
  "check the model properties" -> model = GENERIC, properties = GENERIC

NOT entities: descriptive phrases, process descriptions, adjectives, generic \
vocabulary without a specific referent.

TEXT:
{content}

TERMS TO CLASSIFY:
{terms_json}

Output JSON: {{"terms": [{{"term": "...", "decision": "ENTITY|GENERIC"}}]}}
"""


def build_validation_prompt(terms: list[str], text: str) -> str:
    """Build the context validation prompt.
    
    Creates a prompt for LLM to classify ambiguous terms as
    ENTITY or GENERIC based on their usage context.
    
    Args:
        terms: List of terms to classify
        text: Document text (truncated to 3000 chars in prompt)
        
    Returns:
        Formatted prompt string
    """
    return _CONTEXT_VALIDATION_PROMPT.format(
        content=text[:3000],
        terms_json=json.dumps(terms),
    )


def validate_terms(
    terms: list[str],
    text: str,
    bypass: set[str] | None = None,
    model: str = "sonnet",
) -> list[str]:
    """Validate ambiguous common words using LLM context classification.
    
    Routes single lowercase words (without code markers, not in bypass)
    to LLM for ENTITY/GENERIC classification. Terms with structural
    markers (camelCase, dots, parentheses, etc.) pass through automatically.
    
    Args:
        terms: List of candidate terms to validate
        text: Document text for context
        bypass: Set of terms to skip validation (known good entities)
        model: LLM model to use (default: "sonnet")
        
    Returns:
        Filtered list of validated terms
    """
    if bypass is None:
        bypass = set()
    
    # Split into terms that need validation vs safe terms
    needs_check = [t for t in terms if _needs_context_validation(t, bypass)]
    safe_terms = [t for t in terms if not _needs_context_validation(t, bypass)]
    
    if not needs_check:
        return terms
    
    prompt = build_validation_prompt(needs_check, text)
    response = call_llm(prompt, model=model, max_tokens=2000, temperature=0.0)
    
    # Parse response
    decisions: dict[str, str] = {}
    try:
        json_match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            for item in parsed.get("terms", []):
                decisions[item["term"]] = item.get("decision", "ENTITY")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    
    # Keep terms classified as ENTITY
    kept = []
    for t in needs_check:
        decision = decisions.get(t, "ENTITY")  # Default to ENTITY if not found
        if decision == "ENTITY":
            kept.append(t)
    
    return safe_terms + kept


def _needs_context_validation(term: str, bypass: set[str]) -> bool:
    """Check if a term needs LLM context validation.
    
    Routes single lowercase words (without code markers, not in bypass)
    to Sonnet for ENTITY/GENERIC classification.
    
    Args:
        term: Term to check
        bypass: Set of bypass terms (skip validation)
        
    Returns:
        True if term needs validation
    """
    # Structural terms don't need validation
    if re.search(r"[().\[\]_::<>]", term):
        return False
    
    # CamelCase doesn't need validation
    if re.search(r"[a-z][A-Z]", term):
        return False
    
    # ALL_CAPS doesn't need validation
    if re.match(r"^[A-Z][A-Z0-9_]+$", term) and len(term) >= 2:
        return False
    
    # Multi-word terms don't go through this
    if " " in term or "/" in term:
        return False
    
    # Bypass terms skip validation
    if term.lower() in bypass:
        return False
    
    # All single lowercase alpha words >= 3 chars need validation
    if term.islower() and term.isalpha() and len(term) >= 3:
        return True
    
    return False
