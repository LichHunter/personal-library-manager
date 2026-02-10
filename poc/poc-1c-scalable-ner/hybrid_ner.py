#!/usr/bin/env python3
"""Hybrid multi-stage NER extraction pipeline.

Combines retrieval few-shot, exhaustive extraction, haiku extraction,
and auto-vocabulary seeding for high recall, then applies progressive
filtering (voting, noise filter, context validation) for high precision.

Target: 95% precision, 95% recall, <5% hallucination — with ZERO manual vocabulary.

Architecture:
  Stage 1: High-recall candidate generation (3 LLM extractors + auto-seeds)
  Stage 2: Grounding (span verification) + dedup
  Stage 3: Confidence scoring + tiered filtering (voting, noise, negatives)
  Stage 4: Context validation for ambiguous common words
  Stage 5: Final assembly
"""

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from retrieval_ner import safe_retrieve, parse_entity_response
from scoring import normalize_term, verify_span
from utils.llm_provider import call_llm


# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

@dataclass
class StrategyConfig:
    name: str = "baseline"

    high_confidence_min_sources: int = 3
    validate_min_sources: int = 2

    use_term_retrieval_for_review: bool = False
    review_default_decision: str = "REJECT"

    contrastive_positive_snippets: int = 2
    contrastive_negative_snippets: int = 2
    contrastive_show_reasoning: bool = False
    safety_net_ratio: float = 0.8

    reject_bare_version_numbers: bool = False
    reject_bare_numbers_with_dot: bool = False
    smart_version_filter: bool = False

    boost_common_word_seeds: bool = False
    common_word_seed_list: list[str] = field(default_factory=list)

    ratio_gated_review: bool = False
    ratio_auto_approve_threshold: float = 0.70
    ratio_auto_reject_threshold: float = 0.20

    seed_bypass_to_high_confidence: bool = False
    suppress_path_embedded: bool = False

    validate_high_confidence_too: bool = False


STRATEGY_PRESETS: dict[str, StrategyConfig] = {
    "baseline": StrategyConfig(name="baseline"),

    "strategy_a": StrategyConfig(
        name="strategy_a",
        high_confidence_min_sources=2,
        validate_min_sources=1,
    ),

    "strategy_b": StrategyConfig(
        name="strategy_b",
        use_term_retrieval_for_review=True,
    ),

    "strategy_c": StrategyConfig(
        name="strategy_c",
        contrastive_positive_snippets=3,
        contrastive_negative_snippets=3,
        contrastive_show_reasoning=True,
    ),

    "strategy_d": StrategyConfig(
        name="strategy_d",
        reject_bare_version_numbers=True,
        reject_bare_numbers_with_dot=True,
    ),

    "strategy_e": StrategyConfig(
        name="strategy_e",
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
        ],
    ),

    "strategy_v4": StrategyConfig(
        name="strategy_v4",
        smart_version_filter=True,
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
        ],
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.20,
    ),

    "strategy_v4_1": StrategyConfig(
        name="strategy_v4_1",
        smart_version_filter=True,
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
        ],
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
    ),

    "strategy_v4_2": StrategyConfig(
        name="strategy_v4_2",
        smart_version_filter=True,
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
        ],
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        seed_bypass_to_high_confidence=True,
        suppress_path_embedded=True,
    ),
}


def get_strategy_config(name: str) -> StrategyConfig:
    if name not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_PRESETS.keys())}")
    return STRATEGY_PRESETS[name]


# ============================================================================
# CONSTANTS
# ============================================================================

PURE_STOP_WORDS = {
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

ACTION_GERUNDS = {
    "appending", "serializing", "deserializing", "floating", "wrapping",
    "loading", "downloading", "subscribing", "referencing", "toggling",
    "de-serialized", "cross-platform", "cross-compile",
}

DESCRIPTIVE_ADJECTIVES = {
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

CATEGORY_SUFFIXES = {
    "items", "elements", "values", "settings", "parameters",
    "options", "properties", "fields", "catalog", "orientation",
    "behavior", "handling", "management", "compatibility",
    "content", "position", "libraries", "events", "factors",
    "pool", "level", "keys", "engine", "mode", "navigation",
    "access", "support", "system", "design", "architecture",
    "configuration", "implementation", "specification",
}


# ============================================================================
# PROMPTS
# ============================================================================

EXHAUSTIVE_PROMPT = """\
You are a Named Entity Recognition system for software text. Extract ALL SOFTWARE \
NAMED ENTITIES from this StackOverflow text. Be EXHAUSTIVE — missing an entity is \
WORSE than including a borderline one.

TEXT:
{content}

Extract instances of these entity types:

CODE ENTITIES:
- Library/Framework: jQuery, .NET, Prism, boost, libc++, React, NumPy, SOAP
- Library_Class: ArrayList, HttpClient, ListView, WebClient, IEnumerable, Session
- Library_Function: recv(), querySelector(), send(), map(), post()
- Language: Java, Python, C#, C++11, JavaScript, AS3, HTML, CSS, SQL

DATA:
- Data_Type: string, int, boolean, float, Byte[], var, private, long, bytearrays
- Data_Structure: array, HashMap, table, list, column, row, graph, image, container

INFRASTRUCTURE:
- Application: Visual Studio, Docker, Chrome, Silverlight, jsFiddle, Codepen, IDE
- Operating_System: Linux, Windows, macOS, Android, unix
- File_Type: JSON, XML, CSV, WSDL, xaml, jpg, pom.xml
- Version: v3.2, ES6, 2.18.3, Silverlight 4
- Website: GitHub, codeplex, codepen.io, W3C

INTERFACE:
- UI_Element: checkbox, button, slider, screen, page, form, trackbar, scrollbar, keyboard
- HTML_Tag: <div>, <input>, <table>, li

OTHER:
- Error_Name: NullPointerException, error 500, exception
- Device: iPhone, GPU, microphone, phone, camera, CPU
- Organization: Google, Microsoft, W3C

CRITICAL RULES:
1. COMMON WORDS ARE ENTITIES when they name specific things in context:
   "string" = Data_Type, "table" = Data_Structure, "image" = Data_Structure,
   "button" = UI_Element, "screen" = UI_Element, "page" = UI_Element,
   "row" = Data_Structure, "column" = Data_Structure, "exception" = Error_Name,
   "list" = Data_Structure, "server" = Application, "console" = Application,
   "form" = UI_Element, "keyboard" = Device, "browser" = Application,
   "array" = Data_Structure, "float" = Data_Type, "boolean" = Data_Type
2. Extract terms EXACTLY as they appear in the text
3. Be EXHAUSTIVE — scan every sentence for entities
4. DO NOT extract descriptive phrases: "vertical orientation", "hidden fields"
5. DO NOT extract action words: "appending", "serializing", "floating"

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

HAIKU_SIMPLE_PROMPT = """\
Extract ALL named software entities from this StackOverflow text. Be EXHAUSTIVE.

Include: library names, class names, function names, languages, data types \
(string, int, long, boolean, float), data structures (table, array, row, column, \
list, image), UI elements (button, screen, page, form, keyboard, checkbox), \
file types (JSON, XML, CSV), OS names (Linux, Windows, Android), error names \
(exception, NullPointerException), version numbers, devices (phone, CPU, GPU), \
websites (GitHub, Google).

IMPORTANT: Common words ARE entities when they name specific technical things. \
"table" = Data_Structure, "string" = Data_Type, "button" = UI_Element, \
"server" = Application, "exception" = Error_Name, "list" = Data_Structure, \
"image" = Data_Structure, "row" = Data_Structure, "console" = Application.

Do NOT extract descriptive phrases or generic concept descriptions.

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

RETRIEVAL_PROMPT_TEMPLATE = """\
You are extracting technical named entities from StackOverflow posts.

Here are examples of correct entity extraction from similar posts:

{examples_block}

---

Now extract ALL technical entities from this text, following the same annotation \
style as the examples above.

Rules:
- Extract ALL technical named entities following the same annotation style as the examples
- CONTEXT DETERMINES ENTITY STATUS: Common words like "list", "string", "table", \
"button", "image", "server", "row", "column", "exception" ARE entities when \
referring to specific technical concepts
- Extract specific technologies, libraries, frameworks, APIs, functions, classes, \
tools, platforms, data types, UI components, file formats, error types, and websites
- Only skip truly generic vocabulary with no technical referent (e.g., "the code", \
"my project", "this thing")
- Extract minimal spans with version numbers when present

TEXT: {text}

Return ONLY a JSON array of entity strings. No explanations.
ENTITIES:"""

REVIEW_PROMPT = """\
You are reviewing candidate software named entities extracted from a StackOverflow post. \
Each candidate was found by only ONE extractor (low confidence). Be BALANCED.

TEXT:
{content}

CANDIDATES:
{terms_json}

APPROVE if the term is a SPECIFIC named entity in software context:
- Library/framework/application name (jQuery, Docker, React, NumPy)
- Class/function/API name (ArrayList, recv(), WebClient)
- Language name (Java, C#, Python, SQL, HTML, CSS)
- Data type keyword (string, int, boolean, float, long)
- Data structure TYPE (array, HashMap, table, list, row, column, image)
- UI element TYPE (button, checkbox, slider, screen, page, keyboard)
- File type (JSON, XML, WSDL, jpg)
- Error name (NullPointerException, exception)
- Keyboard key or shortcut (arrow up, PgUp, Left, Tab, Ctrl+C)
- Version, OS/platform, Device, Website, Organization

REJECT if the term is:
- A descriptive phrase: "vertical orientation", "hardware level", "entropy pool"
- A category or concept: "gallery items", "module catalog"
- A process description: "loading on demand", "keyboard navigation"
- An adjective or generic modifier: "Deterministic", "cross-platform"

When in doubt about well-known tech terms, APPROVE. \
When in doubt about generic English phrases, REJECT.

Output JSON: {{"terms": [{{"term": "...", "decision": "APPROVE|REJECT"}}]}}
"""

CONTEXT_VALIDATION_PROMPT = """\
For each term below, determine if it names a SPECIFIC, WELL-KNOWN software entity \
or if it's used as generic English vocabulary in this context.

TEXT:
{content}

TERMS TO CLASSIFY:
{terms_json}

Answer "ENTITY" ONLY if the term is the PROPER NAME of:
- A specific programming language (Java, Python, C#)
- A specific library, framework, or application (jQuery, Docker, React)
- A specific class, function, or API name (ArrayList, querySelector)
- A recognized data type keyword (string, int, boolean, float, long)
- A recognized data structure TYPE (array, HashMap, table, list, queue, stack)
- A specific UI component TYPE (button, checkbox, slider, trackbar)
- A specific error/exception TYPE (NullPointerException, TypeError)
- A specific file format (JSON, XML, CSV)
- A specific OS, device, or website name (Linux, iPhone, GitHub)

Answer "GENERIC" for EVERYTHING ELSE, including:
- General programming vocabulary: object, class, function, method, property, \
collection, action, model, element, index, loop, field, instance, type, value, \
module, tag, node, endpoint, event, handler, service
- Descriptive nouns: title, popup, thumb, slide, demo, nav, database, credential

KEY: If the word is everyday programmer vocabulary you'd use in a sentence WITHOUT \
referring to a specific API/type, classify as GENERIC.
- "the ArrayList stores objects" → ArrayList=ENTITY
- "the string is empty" → string=ENTITY
- "check the model properties" → model=GENERIC, properties=GENERIC

Output JSON: {{"terms": [{{"term": "...", "decision": "ENTITY|GENERIC"}}]}}
"""


# ============================================================================
# PARSING
# ============================================================================

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


# ============================================================================
# STAGE 1: HIGH-RECALL CANDIDATE GENERATION
# ============================================================================

def _extract_exhaustive_sonnet(doc: dict) -> tuple[list[str], float]:
    """Exhaustive extraction with Sonnet using taxonomy-driven prompt."""
    prompt = EXHAUSTIVE_PROMPT.format(content=doc["text"][:5000])
    t0 = time.time()
    response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)
    elapsed = time.time() - t0
    return _parse_terms_json(response), elapsed


def _extract_haiku_simple(doc: dict) -> tuple[list[str], float]:
    """Simpler extraction with Haiku for diversity."""
    prompt = HAIKU_SIMPLE_PROMPT.format(content=doc["text"][:5000])
    t0 = time.time()
    response = call_llm(prompt, model="haiku", max_tokens=2000, temperature=0.0)
    elapsed = time.time() - t0
    return _parse_terms_json(response), elapsed


def _extract_retrieval_fixed(
    doc: dict,
    train_docs: list[dict],
    index,  # faiss.Index
    model,  # SentenceTransformer
    k: int = 5,
) -> tuple[list[str], float]:
    """Retrieval-augmented few-shot extraction with FIXED prompt."""
    t0 = time.time()
    similar_docs = safe_retrieve(doc, train_docs, index, model, k)

    # Build examples block
    examples_parts: list[str] = []
    for i, sdoc in enumerate(similar_docs, 1):
        text_preview = sdoc["text"][:800]
        terms = json.dumps(sdoc["gt_terms"], ensure_ascii=False)
        examples_parts.append(
            f"--- Example {i} ---\n"
            f"TEXT: {text_preview}\n"
            f"ENTITIES: {terms}"
        )
    examples_block = "\n\n".join(examples_parts)

    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        examples_block=examples_block,
        text=doc["text"][:5000],
    )
    response = call_llm(prompt, model="sonnet", max_tokens=2000, temperature=0.0)
    elapsed = time.time() - t0

    return parse_entity_response(response), elapsed


def _extract_seeds(doc: dict, seeds: list[str]) -> list[str]:
    """Regex-match auto-seed terms against document text."""
    text_lower = doc["text"].lower()
    found: list[str] = []

    for seed in seeds:
        seed_lower = seed.lower()
        # Word boundary matching
        pattern = rf"\b{re.escape(seed_lower)}\b"
        if re.search(pattern, text_lower):
            # Find the actual case from text
            match = re.search(
                rf"\b{re.escape(seed_lower)}\b", doc["text"], re.IGNORECASE
            )
            if match:
                found.append(match.group())
            else:
                found.append(seed)

    return found


# ============================================================================
# STAGE 2: GROUNDING + DEDUP
# ============================================================================

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


# ============================================================================
# STAGE 3: NOISE FILTER + CONFIDENCE + TIERED FILTERING
# ============================================================================

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
    if re.match(r"^https?://", t):
        return True

    if cfg.smart_version_filter and doc_text:
        if _smart_version_filter(t, doc_text):
            return True
    elif cfg.reject_bare_version_numbers and re.match(r"^\d+\.\d+(\.\d+)*$", t):
        return True
    if cfg.reject_bare_numbers_with_dot and re.match(r"^\d+\.\d+$", t):
        return True

    if _auto_keep_structural(t):
        return False
    if bypass_set and t.lower() in bypass_set:
        return False
    if t.lower() in negatives_set:
        return True
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


# ============================================================================
# STAGE 4: CONTEXT VALIDATION
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

TERM_RETRIEVAL_PROMPT = """\
You are classifying candidate software named entities. For each term, I show you \
how it was labeled in SIMILAR StackOverflow documents from our training data, plus \
the current document where you must decide.

CURRENT DOCUMENT:
{content}

CANDIDATES TO CLASSIFY:
{candidates_block}

For each term, classify as ENTITY or GENERIC based on context:
- ENTITY: The term names a specific software thing (language, library, class, \
data type, UI element, file format, error, device, OS, website)
- GENERIC: The term is general programming vocabulary, a description, or a \
common English word not referring to a specific named software thing

Use the training examples as calibration — they show how human annotators \
labeled the same term in similar contexts.

Output JSON: {{"terms": [{{"term": "...", "decision": "ENTITY|GENERIC"}}]}}
"""

TERM_RETRIEVAL_PROMPT_WITH_REASONING = """\
You are classifying candidate software named entities. For each term, I show you \
how it was labeled in SIMILAR StackOverflow documents from our training data, plus \
the current document where you must decide.

CURRENT DOCUMENT:
{content}

CANDIDATES TO CLASSIFY:
{candidates_block}

For each term, classify as ENTITY or GENERIC based on context:
- ENTITY: The term names a specific software thing (language, library, class, \
data type, UI element, file format, error, device, OS, website)
- GENERIC: The term is general programming vocabulary, a description, or a \
common English word not referring to a specific named software thing

Use the training examples as STRONG calibration. The entity_ratio tells you how \
often human annotators labeled this term as an entity vs generic. A term with \
entity_ratio=80%+ is almost always an entity. A term with entity_ratio<20% is \
almost always generic. For ratios between 20-80%, carefully examine how the term \
is used in the CURRENT DOCUMENT compared to the training snippets.

For each term, briefly explain your reasoning before deciding.

Output JSON: {{"terms": [{{"term": "...", "reasoning": "...", "decision": "ENTITY|GENERIC"}}]}}
"""

TERM_RETRIEVAL_REVIEW_PROMPT = """\
You are reviewing candidate software named entities that had LOW agreement among \
extractors (only 1 extractor found them). For each term, I show you training data \
evidence about how human annotators labeled this term, plus the current document.

CURRENT DOCUMENT:
{content}

LOW-CONFIDENCE CANDIDATES:
{candidates_block}

Decide APPROVE or REJECT for each:
- APPROVE if the term names a specific software entity AND the training data supports it
- REJECT if the term is generic vocabulary, a description, or training data shows it's rarely an entity

Pay special attention to entity_ratio: if <20%, strongly lean REJECT. If >60%, lean APPROVE. \
Between 20-60%, examine the current document context carefully.

Output JSON: {{"terms": [{{"term": "...", "decision": "APPROVE|REJECT"}}]}}
"""


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
# SPAN EXPANSION + SUBSPAN SUPPRESSION (ported from POC-1b)
# ============================================================================

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


def _suppress_subspans(extracted: list[str]) -> list[str]:
    """Remove terms that are substrings of other extracted terms.

    Only suppresses when the shorter term is a non-word-boundary substring
    (e.g., "getInputSizes" inside "getInputSizes(ImageFormat...)") or when
    a single word appears inside a 3+-word compound. Preserves both forms
    for 2-word pairs like "Right" / "arrow right".
    """
    if not extracted:
        return extracted

    lower_terms = [(t, t.lower()) for t in extracted]
    kept: list[str] = []

    for term, term_lower in lower_terms:
        if len(term) <= 3:
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


# ============================================================================
# STAGE 5: MAIN PIPELINE
# ============================================================================

def extract_hybrid(
    doc: dict,
    train_docs: list[dict],
    index,  # faiss.Index
    model,  # SentenceTransformer
    auto_vocab: dict,
    term_index: dict[str, dict] | None = None,
    strategy: StrategyConfig | None = None,
) -> list[str]:
    cfg = strategy or StrategyConfig()

    bypass_set = set(t.lower() for t in auto_vocab.get("bypass", []))
    seeds_list = list(auto_vocab.get("seeds", []))
    negatives_set = set(t.lower() for t in auto_vocab.get("negatives", []))

    if cfg.boost_common_word_seeds and cfg.common_word_seed_list:
        seeds_list = seeds_list + [s for s in cfg.common_word_seed_list if s not in seeds_list]
        bypass_set.update(s.lower() for s in cfg.common_word_seed_list)

    doc_text = doc["text"]

    retrieval_terms, _ = _extract_retrieval_fixed(doc, train_docs, index, model)
    exhaustive_terms, _ = _extract_exhaustive_sonnet(doc)
    haiku_terms, _ = _extract_haiku_simple(doc)
    seed_terms = _extract_seeds(doc, seeds_list)

    candidates_by_source = {
        "retrieval": retrieval_terms,
        "exhaustive": exhaustive_terms,
        "haiku": haiku_terms,
        "seeds": seed_terms,
    }

    grounded = _ground_and_dedup(candidates_by_source, doc_text)

    after_noise: dict[str, dict] = {}
    for key, cand in grounded.items():
        term = cand["term"]
        if _auto_reject_noise(term, negatives_set, bypass_set, strategy=cfg, doc_text=doc_text):
            continue
        after_noise[key] = cand

    high_confidence: list[str] = []
    needs_validation: list[str] = []
    needs_review: list[str] = []

    protected_seed_set: set[str] = set()
    if cfg.seed_bypass_to_high_confidence and cfg.common_word_seed_list:
        protected_seed_set = {s.lower() for s in cfg.common_word_seed_list}

    for key, cand in after_noise.items():
        term = cand["term"]
        source_count = cand["source_count"]

        if _auto_keep_structural(term):
            high_confidence.append(term)
            continue

        if cfg.seed_bypass_to_high_confidence and term.lower() in protected_seed_set and "seeds" in cand["sources"]:
            high_confidence.append(term)
            continue

        if source_count >= cfg.high_confidence_min_sources:
            high_confidence.append(term)
            continue

        if source_count >= cfg.validate_min_sources:
            needs_validation.append(term)
            continue

        needs_review.append(term)

    if needs_review:
        if cfg.ratio_gated_review and term_index:
            protected_terms = set(bypass_set)
            if cfg.boost_common_word_seeds and cfg.common_word_seed_list:
                protected_terms.update(s.lower() for s in cfg.common_word_seed_list)

            auto_approved: list[str] = []
            uncertain: list[str] = []
            for term in needs_review:
                info = term_index.get(term.lower())
                is_protected = term.lower() in protected_terms

                if is_protected:
                    uncertain.append(term)
                elif info:
                    ratio = info.get("entity_ratio", 0.5)
                    if ratio > cfg.ratio_auto_approve_threshold:
                        auto_approved.append(term)
                    elif ratio < cfg.ratio_auto_reject_threshold:
                        pass
                    else:
                        uncertain.append(term)
                else:
                    uncertain.append(term)

            review_decisions = _run_sonnet_review(uncertain, doc_text) if uncertain else {}
            for term in uncertain:
                decision = review_decisions.get(term, cfg.review_default_decision)
                if decision == "APPROVE":
                    needs_validation.append(term)
            needs_validation.extend(auto_approved)

        elif cfg.use_term_retrieval_for_review and term_index:
            review_decisions = _run_term_retrieval_review(
                needs_review, doc_text, term_index, strategy=cfg,
            )
            for term in needs_review:
                decision = review_decisions.get(term, cfg.review_default_decision)
                if decision == "APPROVE":
                    needs_validation.append(term)
        else:
            review_decisions = _run_sonnet_review(needs_review, doc_text)
            for term in needs_review:
                decision = review_decisions.get(term, cfg.review_default_decision)
                if decision == "APPROVE":
                    needs_validation.append(term)

    if cfg.validate_high_confidence_too and term_index:
        all_to_validate = high_confidence + needs_validation
        validated = _run_term_retrieval_validation(
            all_to_validate, doc_text, term_index, strategy=cfg,
            bypass_set=bypass_set,
        )
    elif term_index:
        validated_subset = _run_term_retrieval_validation(
            needs_validation, doc_text, term_index, strategy=cfg,
            bypass_set=bypass_set,
        )
        validated = high_confidence + validated_subset
    else:
        validated_subset = _run_context_validation(
            needs_validation, doc_text, bypass_set,
        )
        validated = high_confidence + validated_subset

    expanded = _expand_spans(validated, doc_text)

    if cfg.suppress_path_embedded:
        expanded = [t for t in expanded if not _is_embedded_in_path(t, doc_text)]

    suppressed = _suppress_subspans(expanded)

    seen: set[str] = set()
    final: list[str] = []
    for term in suppressed:
        key = normalize_term(term)
        if key not in seen:
            seen.add(key)
            final.append(term)

    return final
