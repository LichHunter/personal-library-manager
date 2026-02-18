"""Stage 4: Context validation and term-retrieval validation."""

import json
import re
from collections import defaultdict

from plm.shared.llm import call_llm
from .config import StrategyConfig
from .prompts import (
    CONTEXT_VALIDATION_PROMPT,
    TERM_RETRIEVAL_PROMPT,
    TERM_RETRIEVAL_PROMPT_WITH_REASONING,
    TERM_RETRIEVAL_REVIEW_PROMPT,
)


# ============================================================================
# CONTEXT VALIDATION
# ============================================================================

def _needs_context_validation(term: str, bypass_set: set[str]) -> bool:
    """Check if a term needs LLM context validation.

    Routes single lowercase words (without code markers, not in bypass)
    to Sonnet for ENTITY/GENERIC classification.
    """
    # Structural terms don't need validation
    if re.search(r"[().\[\]_::<>]", term):
        return False
    if re.search(r"[a-z][A-Z]", term):
        return False
    if re.match(r"^[A-Z][A-Z0-9_]+$", term) and len(term) >= 2:
        return False
    # Multi-word terms don't go through this
    if " " in term or "/" in term:
        return False
    # Bypass terms skip validation
    if term.lower() in bypass_set:
        return False
    # All single lowercase alpha words >= 3 chars → needs validation
    if term.islower() and term.isalpha() and len(term) >= 3:
        return True
    return False


def _run_context_validation(
    terms: list[str],
    doc_text: str,
    bypass_set: set[str],
) -> list[str]:
    """Filter ambiguous common words using LLM context classification."""
    needs_check = [t for t in terms if _needs_context_validation(t, bypass_set)]
    safe_terms = [t for t in terms if not _needs_context_validation(t, bypass_set)]

    if not needs_check:
        return terms

    prompt = CONTEXT_VALIDATION_PROMPT.format(
        content=doc_text[:3000],
        terms_json=json.dumps(needs_check),
    )

    response = call_llm(prompt, model="sonnet", max_tokens=2000, temperature=0.0)

    decisions: dict[str, str] = {}
    try:
        text = response.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            for item in parsed.get("terms", []):
                decisions[item["term"]] = item.get("decision", "ENTITY")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    kept = []
    for t in needs_check:
        decision = decisions.get(t, "ENTITY")  # Default to ENTITY if not found
        if decision == "ENTITY":
            kept.append(t)

    return safe_terms + kept


# ============================================================================
# TERM-RETRIEVAL CONTEXT VALIDATION
# ============================================================================

def build_term_index(train_docs: list[dict]) -> dict[str, dict]:
    """Build a term → training evidence index.

    For each term that appears in training data (as entity or in text),
    stores entity_ratio and up to 3 positive + 3 negative context snippets.
    """
    term_entity_docs: dict[str, list[dict]] = defaultdict(list)
    term_generic_docs: dict[str, list[dict]] = defaultdict(list)

    for doc in train_docs:
        text_lower = doc["text"].lower()
        gt_lower = set(t.lower().strip() for t in doc["gt_terms"])

        seen: set[str] = set()
        for t in doc["gt_terms"]:
            tl = t.lower().strip()
            if tl in seen or len(tl) < 2:
                continue
            seen.add(tl)
            snippet = _extract_snippet(text_lower, tl, doc["text"])
            term_entity_docs[tl].append({
                "doc_id": doc["doc_id"],
                "snippet": snippet,
            })

        words_in_text = set(re.findall(r"\b[\w#+.]+\b", text_lower))
        for w in words_in_text:
            if w not in gt_lower and len(w) >= 2 and w not in seen:
                if len(term_generic_docs[w]) < 5:
                    snippet = _extract_snippet(text_lower, w, doc["text"])
                    term_generic_docs[w].append({
                        "doc_id": doc["doc_id"],
                        "snippet": snippet,
                    })

    index: dict[str, dict] = {}
    all_terms = set(term_entity_docs.keys()) | set(term_generic_docs.keys())
    for t in all_terms:
        pos = term_entity_docs.get(t, [])
        neg = term_generic_docs.get(t, [])
        total = len(pos) + len(neg)
        index[t] = {
            "entity_ratio": len(pos) / total if total > 0 else 0.5,
            "entity_count": len(pos),
            "generic_count": len(neg),
            "positive_examples": pos[:3],
            "negative_examples": neg[:3],
        }

    return index


def _extract_snippet(text_lower: str, term_lower: str, original_text: str) -> str:
    idx = text_lower.find(term_lower)
    if idx == -1:
        return original_text[:150]
    start = max(0, idx - 60)
    end = min(len(original_text), idx + len(term_lower) + 60)
    snippet = original_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(original_text):
        snippet = snippet + "..."
    return snippet


def _build_candidate_block(
    term: str,
    term_index: dict[str, dict],
    n_positive: int = 2,
    n_negative: int = 2,
) -> str:
    tl = term.lower().strip()
    info = term_index.get(tl)

    if not info or (info["entity_count"] == 0 and info["generic_count"] == 0):
        return (
            f'Term: "{term}"\n'
            f"  Training data: NO DATA (unseen term)\n"
        )

    lines = [
        f'Term: "{term}" '
        f'(entity in {info["entity_count"]} train docs, '
        f'generic in {info["generic_count"]} train docs, '
        f'ratio={info["entity_ratio"]:.0%})'
    ]

    for ex in info["positive_examples"][:n_positive]:
        lines.append(f'  [ENTITY example]: "{ex["snippet"]}"')
    for ex in info["negative_examples"][:n_negative]:
        lines.append(f'  [GENERIC example]: "{ex["snippet"]}"')

    return "\n".join(lines)


def _needs_term_retrieval_validation(
    term: str, bypass_set: set[str] | None = None,
) -> bool:
    if bypass_set and term.lower() in bypass_set:
        return False
    if re.search(r"[().\[\]_::<>]", term):
        return False
    if re.search(r"[a-z][A-Z]", term):
        return False
    if re.match(r"^[A-Z][A-Z0-9_]+$", term) and len(term) >= 2:
        return False
    if re.match(
        r"^(Left|Right|Up|Down|Ctrl|Alt|Shift|Tab|Enter|Esc|PgUp|PgDn|PageUp|PageDown|Home|End|F\d+)"
        r"(\s+(arrow|key))?$",
        term, re.I,
    ):
        return False
    if re.match(
        r"^(arrow|page|up|down|left|right)"
        r"(\s+(up|down|left|right|arrow|key|keys))"
        r"(\s+(up|down|left|right|arrow|key|keys))?$",
        term, re.I,
    ):
        return False
    if re.match(r"^(up/down|left/right)\s+(arrow|key|keys)$", term, re.I):
        return False
    return True


def _run_term_retrieval_validation(
    terms: list[str],
    doc_text: str,
    term_index: dict[str, dict],
    strategy: StrategyConfig | None = None,
    bypass_set: set[str] | None = None,
) -> list[str]:
    cfg = strategy or StrategyConfig()
    needs_check = [t for t in terms if _needs_term_retrieval_validation(t, bypass_set)]
    safe_terms = [t for t in terms if not _needs_term_retrieval_validation(t, bypass_set)]

    if not needs_check:
        return terms

    candidate_blocks = []
    for t in needs_check:
        candidate_blocks.append(_build_candidate_block(
            t, term_index,
            n_positive=cfg.contrastive_positive_snippets,
            n_negative=cfg.contrastive_negative_snippets,
        ))

    prompt_template = (
        TERM_RETRIEVAL_PROMPT_WITH_REASONING if cfg.contrastive_show_reasoning
        else TERM_RETRIEVAL_PROMPT
    )
    prompt = prompt_template.format(
        content=doc_text[:3000],
        candidates_block="\n\n".join(candidate_blocks),
    )

    max_tokens = 4000 if cfg.contrastive_show_reasoning else 3000
    response = call_llm(prompt, model="sonnet", max_tokens=max_tokens, temperature=0.0)

    decisions: dict[str, str] = {}
    try:
        text = response.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            for item in parsed.get("terms", []):
                decisions[item["term"]] = item.get("decision", "ENTITY")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    kept = []
    for t in needs_check:
        tl = t.lower().strip()
        decision = decisions.get(t, decisions.get(t.lower(), "ENTITY"))
        info = term_index.get(tl)

        if decision == "ENTITY":
            kept.append(t)
        elif info and info["entity_ratio"] >= cfg.safety_net_ratio:
            kept.append(t)

    return safe_terms + kept


def _run_term_retrieval_review(
    candidates: list[str],
    doc_text: str,
    term_index: dict[str, dict],
    strategy: StrategyConfig | None = None,
) -> dict[str, str]:
    cfg = strategy or StrategyConfig()
    if not candidates:
        return {}

    candidate_blocks = []
    for t in candidates:
        candidate_blocks.append(_build_candidate_block(
            t, term_index,
            n_positive=cfg.contrastive_positive_snippets,
            n_negative=cfg.contrastive_negative_snippets,
        ))

    prompt = TERM_RETRIEVAL_REVIEW_PROMPT.format(
        content=doc_text[:3000],
        candidates_block="\n\n".join(candidate_blocks),
    )

    response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)

    decisions: dict[str, str] = {}
    try:
        text = response.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            for item in parsed.get("terms", []):
                decisions[item["term"]] = item.get("decision", "APPROVE")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return decisions


# ============================================================================
# TECHNICAL CONTEXT CHECK
# ============================================================================

_TECHNICAL_CONTEXT_RE = re.compile(
    r"""(?ix)
      \b(?:install|import|require|using|library|framework|module|package|gem|
         plugin|extension|dependency|sdk|api|cli|command|tool|app|
         programming\s+language|written\s+in|built\s+with|powered\s+by|
         \.(?:js|py|rb|java|cs|cpp|go|rs|ts|swift|kt|ex|sh|yaml|yml|json|xml|
              html|css|scss|sass|less|sql|md|txt|cfg|ini|conf|config|
              dll|so|jar|whl|gem|egg|tar|gz|zip))\b
    """,
)


def _has_technical_context(term: str, doc_text: str) -> bool:
    """Check if *term* appears near technical indicators in *doc_text*."""
    tl = term.lower()
    for m in re.finditer(rf"\b{re.escape(tl)}\b", doc_text, re.IGNORECASE):
        start = max(0, m.start() - 120)
        end = min(len(doc_text), m.end() + 120)
        window = doc_text[start:end]
        if _TECHNICAL_CONTEXT_RE.search(window):
            return True
    return False
