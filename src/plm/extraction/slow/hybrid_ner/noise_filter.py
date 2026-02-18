"""Stage 3: Noise filter + confidence + tiered filtering."""

import json
import re

from plm.shared.llm import call_llm
from .config import StrategyConfig
from .constants import PURE_STOP_WORDS, ACTION_GERUNDS, DESCRIPTIVE_ADJECTIVES, CATEGORY_SUFFIXES
from .prompts import REVIEW_PROMPT


def _auto_keep_structural(term: str) -> bool:
    """Structural patterns that are ALWAYS kept (high confidence)."""
    if re.search(r"[().\[\]_::<>+]", term):
        return True
    if re.search(r"[a-z][A-Z]", term):
        return True
    if re.match(r"^[A-Z][A-Z0-9_]+$", term) and len(term) >= 2:
        return True
    if re.match(
        r"^(Left|Right|Up|Down|Ctrl|Alt|Shift|Tab|Enter|Esc|PgUp|PgDn|PageUp|PageDown|Home|End|F\d+)"
        r"(\s+(arrow|key))?$",
        term,
        re.I,
    ):
        return True
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


def _is_version_fragment(version: str, doc_text: str) -> bool:
    longer_pattern = re.escape(version) + r"\.\d+"
    if not re.search(longer_pattern, doc_text):
        return False
    standalone_pattern = re.escape(version) + r"(?!\.\d)"
    if re.search(standalone_pattern, doc_text):
        return False
    return True


def _is_orphan_version(term: str, doc_text: str) -> bool:
    ctx_pattern = rf"(?:[A-Za-z][\w.-]*\s+){re.escape(term)}|(?:v|version\s+){re.escape(term)}"
    return not re.search(ctx_pattern, doc_text, re.I)


def _smart_version_filter(term: str, doc_text: str) -> bool:
    if not re.match(r"^\d+\.\d+(\.\d+)*$", term):
        return False
    segments = term.count(".") + 1
    if segments >= 3:
        return False
    if segments == 2:
        if _is_version_fragment(term, doc_text):
            return True
        return _is_orphan_version(term, doc_text)
    return False


def _auto_reject_noise(
    term: str,
    negatives_set: set[str],
    bypass_set: set[str] | None = None,
    strategy: StrategyConfig | None = None,
    doc_text: str | None = None,
) -> bool:
    t = term.strip()
    cfg = strategy or StrategyConfig()

    if not t or len(t) <= 1:
        return True
    if re.match(r"^\d+$", t):
        return True
    if t.lower() in PURE_STOP_WORDS:
        return True
    if t.lower() in ACTION_GERUNDS:
        return True
    if re.match(r"^https?://", t) or re.match(r"^www\.", t):
        return True
    if re.search(r"https?://\S+", t):
        return True

    if cfg.smart_version_filter and doc_text:
        if _smart_version_filter(t, doc_text):
            return True
    elif cfg.reject_bare_version_numbers and re.match(r"^\d+\.\d+(\.\d+)*$", t):
        return True
    if cfg.reject_bare_numbers_with_dot and re.match(r"^\d+\.\d+$", t):
        return True

    if t.lower() in negatives_set:
        return True
    if _auto_keep_structural(t):
        return False
    if bypass_set and t.lower() in bypass_set:
        return False
    if re.match(r"^[A-Z]+-\d+$", t):
        return True

    words = t.lower().split()
    if len(words) == 2:
        if words[0] in DESCRIPTIVE_ADJECTIVES:
            return True
        if words[1] in CATEGORY_SUFFIXES:
            return True
    if len(words) >= 3 and not re.search(r"[A-Z]", t[1:]) and not re.search(
        r"[()._::<>\[\]]", t
    ):
        return True

    return False


def _run_sonnet_review(
    candidates: list[str], doc_text: str
) -> dict[str, str]:
    """Batch Sonnet review for single-vote candidates. Returns {term: APPROVE|REJECT}."""
    if not candidates:
        return {}

    terms_json = json.dumps(candidates, indent=2)
    prompt = REVIEW_PROMPT.format(
        content=doc_text[:3000], terms_json=terms_json
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
