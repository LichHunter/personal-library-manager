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

from .retrieval_ner import safe_retrieve, parse_entity_response
from .scoring import normalize_term, verify_span
from plm.shared.llm import call_llm


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
    seed_bypass_require_context: bool = False
    seed_bypass_min_sources_for_auto: int = 2
    suppress_path_embedded: bool = False

    validate_high_confidence_too: bool = False

    # v5: Haiku+heuristic extraction mode
    use_haiku_extraction: bool = False
    use_heuristic_extraction: bool = False
    log_low_confidence: bool = False
    high_entity_ratio_threshold: float = 0.8
    medium_entity_ratio_threshold: float = 0.5
    skip_validation_entity_ratio: float = 0.7

    # v5.1: Optimizations
    use_contextual_seeds: bool = False
    use_low_precision_filter: bool = False
    allcaps_require_corroboration: bool = False
    use_sonnet_taxonomy: bool = False

    # v5.2: entity_ratio as signal not gate
    route_single_vote_to_validation: bool = False
    # v5.3: minimum entity_ratio for single-vote routing to validation
    single_vote_min_entity_ratio: float = 0.0

    # v5.4: tighten validation routing
    structural_require_llm_vote: bool = False
    disable_seed_bypass: bool = False

    # v6: candidate_verify extraction mode
    use_candidate_verify: bool = False


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

    "strategy_v4_3": StrategyConfig(
        name="strategy_v4_3",
        smart_version_filter=True,
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
            "ruby", "symlinks",
        ],
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=2,
        suppress_path_embedded=True,
    ),

    "strategy_v5": StrategyConfig(
        name="strategy_v5",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=2,
    ),

    "strategy_v5_1": StrategyConfig(
        name="strategy_v5_1",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=1,
        use_contextual_seeds=True,
        use_low_precision_filter=True,
        allcaps_require_corroboration=True,
        use_sonnet_taxonomy=True,
    ),

    "strategy_v5_2": StrategyConfig(
        name="strategy_v5_2",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=1,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=False,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
    ),

    "strategy_v5_3": StrategyConfig(
        name="strategy_v5_3",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=1,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=False,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
    ),

    "strategy_v5_4": StrategyConfig(
        name="strategy_v5_4",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.95,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.90,
        safety_net_ratio=0.95,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=2,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=True,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
        structural_require_llm_vote=True,
        disable_seed_bypass=False,
    ),

    "strategy_v6": StrategyConfig(
        name="strategy_v6",
        use_candidate_verify=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.95,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.90,
        safety_net_ratio=0.95,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=2,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=True,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
        structural_require_llm_vote=True,
        disable_seed_bypass=False,
    ),

    "strategy_v5_3b": StrategyConfig(
        name="strategy_v5_3b",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=1,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=False,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
        single_vote_min_entity_ratio=0.01,
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

# ---------------------------------------------------------------------------
# SHARED ENTITY DEFINITION (single source of truth for all prompts)
# ---------------------------------------------------------------------------

SHARED_ENTITY_DEFINITION = """\
SOFTWARE NAMED ENTITY: A term that names a SPECIFIC, identifiable technical thing.

ENTITY TYPES:
- Code: Library, Framework, Class, Function, Method, API, Variable, Constant, Enum value
- Language: Programming language, Query language, Markup (Java, SQL, HTML, C++11)
- Data: Data_Type (string, int, boolean, float, long, private, var), \
Data_Structure (array, list, table, row, column, image, HashMap, container)
- Infrastructure: Application (server, console, browser, IDE), OS, Device, \
File_Type, Version, Website, Organization
- Interface: UI_Element (button, checkbox, form, screen, page, keyboard, \
trackbar, scrollbar, slider), HTML_Tag, Keyboard_Key (Left, Up, PgUp)
- Other: Error_Name (exception, NullPointerException), Algorithm

THE COMMON-WORD RULE:
Everyday English words ARE entities when they NAME specific technical things:
  "the string is empty" → string = Data_Type
  "click the button" → button = UI_Element
  "stored in a table" → table = Data_Structure
  "throws an exception" → exception = Error_Name
  "runs on the server" → server = Application
  "press the Left key" → Left = Keyboard_Key
  "the Session class" → Session = Class
  "a long value" → long = Data_Type
  "set the padding" → padding = UI property

NOT ENTITIES:
- Descriptive phrases: "vertical orientation", "hidden fields", "entropy pool"
- Process descriptions: "loading on demand", "serializing data"
- Adjectives/modifiers: "cross-platform", "deterministic", "floating"
- Generic vocabulary WITHOUT a specific referent: "check the code", "my project"\
"""

# ---------------------------------------------------------------------------
# EXTRACTION PROMPTS (optimize for RECALL — when in doubt, include)
# ---------------------------------------------------------------------------

EXHAUSTIVE_PROMPT = """\
You are a Named Entity Recognition system for StackOverflow technical text.

""" + SHARED_ENTITY_DEFINITION + """

EXTRACTION MODE: Be EXHAUSTIVE. Missing an entity is WORSE than including a \
borderline one. When in doubt, INCLUDE the term.

TAXONOMY WITH EXAMPLES:
- Library/Framework: jQuery, React, NumPy, .NET, boost, Prism, libc++, SOAP
- Library_Class: ArrayList, HttpClient, Session, WebClient, ListView, IEnumerable
- Library_Function: recv(), querySelector(), map(), send(), post()
- Language: Java, Python, C#, JavaScript, HTML, CSS, SQL, AS3, C++11
- Data_Type: string, int, boolean, float, long, var, private, Byte[], bytearrays
- Data_Structure: array, list, table, row, column, image, HashMap, graph, container
- Application: Visual Studio, Docker, Chrome, browser, server, console, IDE
- Operating_System: Linux, Windows, macOS, Android, unix
- Device: iPhone, GPU, phone, camera, keyboard, microphone, CPU
- File_Type: JSON, XML, CSV, WSDL, xaml, jpg, pom.xml
- UI_Element: button, checkbox, slider, screen, page, form, trackbar, scrollbar, pad
- HTML_Tag: <div>, <input>, li
- Error_Name: NullPointerException, exception, error 500
- Version: v3.2, ES6, 2.18.3, Silverlight 4
- Website/Org: GitHub, Google, Microsoft, W3C, codeplex, codepen.io
- Keyboard_Key: Left, Right, Up, Down, PgUp, PageDown, Tab, Ctrl+C

RULES:
1. Extract terms EXACTLY as they appear in the text
2. Scan EVERY sentence — do not skip any
3. Common words like "string", "table", "button", "server", "list", "image", \
"exception", "session", "key", "padding", "long", "private" ARE entities per \
the Common-Word Rule above
4. DO NOT extract descriptive phrases or action words

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

HAIKU_SIMPLE_PROMPT = """\
You are a SPECIALIST extractor for software entities that look like common English \
words. Other extractors handle named libraries/frameworks well. Your job is to \
catch the entities they MISS.

FOCUS ON THESE CATEGORIES (other extractors under-extract them):

1. COMMON WORDS THAT ARE ENTITIES in StackOverflow technical text:
   - Data types used as entities: string, int, boolean, float, long, private, var
   - Data structures: table, tables, array, list, row, column, image, container
   - UI elements: button, keyboard, screen, page, form, slider, scrollbar, pad, checkbox
   - Infrastructure: server, console, browser, exception, session, kernel, global
   - Devices: phone, camera, CPU, microphone
   - Other: configuration, calculator, borders, padding, symlinks, ondemand

2. FILE PATHS & DIRECTORY REFERENCES:
   - /usr/bin, /System/Library/..., src/main/resources/...
   - Relative paths: ../../path/to/file
   - Path patterns: /usr/bin/{{file1,file2,file3}}

3. UNUSUAL ENTITY FORMATS:
   - Compound key names: PgUp, PageDown, Ctrl+C, Left arrow
   - Function calls with args: getInputSizes(ImageFormat.YUV_420_888)
   - Size expressions: MAX SIZE(4608x3456)
   - Type notation: Byte[], bytearrays
   - Objective-C selectors: setCustomDoneTarget:action:, doneAction:
   - Words with punctuation: Pseudo-randomic, cross-platform (when a tool name)

4. VERSION NUMBERS (standalone or with context):
   - 1.9.3, 2.18.3, v3.2, 0.7.3, 13.10, 1.1.0

5. FILE EXTENSIONS: .h, .xib, .cs, .m, .long, .xaml

EXTRACTION RULES:
- Extract terms EXACTLY as they appear in the text
- Scan EVERY sentence — do not skip any
- When a common English word appears in a technical context, ALWAYS extract it
- Include file paths, version numbers, unusual formats
- DO NOT extract: generic CS vocabulary (code, method, function, class, property, \
  module, field, namespace — unless part of a specific name)
- DO NOT extract: descriptive phrases, process words, adjectives

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

RETRIEVAL_PROMPT_TEMPLATE = """\
You are extracting software named entities from StackOverflow text.

WHAT IS AN ENTITY:
Named libraries, classes, functions, APIs, tools, platforms, languages, \
data types (string, int, boolean, long), data structures (array, list, table, \
row, column, image), UI elements (button, screen, page, form, keyboard), \
file formats, error names, devices, websites, versions, keyboard keys.

COMMON WORDS ARE ENTITIES when they name specific technical things in context. \
Only skip truly generic vocabulary with no technical referent ("the code", \
"my project").

Here are examples of correct annotation from similar posts:

{examples_block}

---

Now extract ALL entities from this text. Follow the annotation style of the \
examples above. When in doubt, INCLUDE — missing entities is worse than \
including borderline ones.

TEXT: {text}

Return ONLY a JSON array of entity strings. No explanations.
ENTITIES:"""

# ---------------------------------------------------------------------------
# VALIDATION PROMPTS (same entity definition, entity_ratio as calibration)
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """\
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

CONTEXT_VALIDATION_PROMPT = """\
You are classifying candidate software named entities from a StackOverflow post.

ENTITY DEFINITION:
A term is an ENTITY if it names a SPECIFIC technical thing: library, class, \
function, language, data type (string, int, boolean, long, private), data \
structure (array, list, table, row, column, image), UI element (button, page, \
form, keyboard), file format, error name (exception), device, website, \
application (server, console, browser), keyboard key, version.

COMMON WORDS ARE ENTITIES when naming specific technical things:
  "the string is empty" → string = ENTITY (Data_Type)
  "stored in a table" → table = ENTITY (Data_Structure)
  "check the model properties" → model = GENERIC, properties = GENERIC

NOT entities: descriptive phrases, process descriptions, adjectives, generic \
vocabulary without a specific referent.

TEXT:
{content}

TERMS TO CLASSIFY:
{terms_json}

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


_ALLCAPS_EXCLUDE = {
    "THE", "AND", "BUT", "NOT", "FOR", "ARE", "WAS", "HAS", "HAD",
    "SO", "IT", "IS", "OR", "MY", "IN", "TO", "OF", "AT", "BY",
    "ON", "AN", "AS", "IF", "NO", "DO", "UP", "BE", "AM", "HE",
    "OUTPUT", "UPDATE", "EDIT", "OK", "ERROR", "WARNING", "INFO",
    "DEBUG", "TRUE", "FALSE", "NULL", "NONE", "TODO", "FIXME",
    "NOTE", "CODE", "LEVEL", "NEW", "END", "SET", "GET", "ADD",
    "PUT", "DELETE", "POST", "THEN", "ELSE", "CASE", "WHEN",
    "WITH", "FROM", "INTO", "LIKE", "WHERE", "ORDER", "GROUP",
}


def _extract_heuristic(doc: dict) -> list[str]:
    """Extract software entities using structural/heuristic patterns.

    Finds CamelCase identifiers, dot-separated paths, parenthesized calls,
    ALL_CAPS acronyms, and backtick-wrapped terms. Zero cost, no LLM calls.
    Used as +1 vote source alongside LLM extractors.
    """
    text = doc["text"]
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
        # Skip common sentence-ending periods or abbreviations
        if len(term) >= 3:  # ".x" too short, ".long" is fine
            found.add(term)

    # 8. Brace-expansion paths (e.g., /usr/bin/{erb,gem,irb,rdoc,ri,ruby,testrb})
    for m in re.finditer(r"(/[^\s]*\{[^}]+\}[^\s]*)", text):
        term = m.group(1).strip()
        if term:
            found.add(term)

    # 9. Unix-style file paths (e.g., /usr/bin/ruby, /System/Library/Frameworks)
    for m in re.finditer(r"(/(?:[\w.+-]+/)+[\w.+-]*)", text):
        term = m.group(1)
        if len(term) >= 4 and not re.match(r"^/\d+(/\d+)*$", term):  # skip pure numeric paths
            found.add(term)

    return list(found)


def _extract_haiku_fewshot(
    doc: dict,
    train_docs: list[dict],
    index,  # faiss.Index
    model,  # SentenceTransformer
    k: int = 5,
) -> tuple[list[str], float]:
    """Retrieval-augmented few-shot extraction with Haiku (cheaper than Sonnet)."""
    t0 = time.time()
    similar_docs = safe_retrieve(doc, train_docs, index, model, k)

    # Build examples block (same as retrieval_fixed)
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
    response = call_llm(prompt, model="haiku", max_tokens=2000, temperature=0.0)
    elapsed = time.time() - t0

    return parse_entity_response(response), elapsed


def _extract_haiku_taxonomy(
    doc: dict, llm_model: str = "haiku",
) -> tuple[list[str], float]:
    prompt = EXHAUSTIVE_PROMPT.format(content=doc["text"][:5000])
    t0 = time.time()
    response = call_llm(prompt, model=llm_model, max_tokens=3000, temperature=0.0)
    elapsed = time.time() - t0
    return _parse_terms_json(response), elapsed


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
You are reviewing LOW-CONFIDENCE candidates (found by only 1 extractor). For each \
term, I show training data evidence and the current document.

ENTITY DEFINITION:
A term is an ENTITY if it names a SPECIFIC technical thing: library, class, \
function, language, data type (string, int, boolean, long, private), data \
structure (array, list, table, row, column, image), UI element (button, page, \
form, keyboard, pad), file format, error name (exception), device, website, \
application (server, console, browser), keyboard key, version.

COMMON WORDS ARE ENTITIES when naming specific technical things in context.
NOT entities: descriptive phrases, process descriptions, adjectives, generic \
vocabulary without a specific referent.

CURRENT DOCUMENT:
{content}

LOW-CONFIDENCE CANDIDATES:
{candidates_block}

REVIEW GUIDELINES:
- entity_ratio is calibration: high (>70%) lean APPROVE, low (<30%) lean REJECT, \
middle = examine context
- A low ratio does NOT mean "always reject" — APPROVE if this document clearly \
uses the term as a named entity
- REJECT descriptive phrases, process descriptions, and adjectives regardless

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
            if not cfg.seed_bypass_require_context:
                high_confidence.append(term)
                continue
            if source_count >= cfg.seed_bypass_min_sources_for_auto:
                high_confidence.append(term)
                continue
            if _has_technical_context(term, doc_text):
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

    suppressed = _suppress_subspans(expanded, protected_seeds=protected_seed_set)

    seen: set[str] = set()
    final: list[str] = []
    for term in suppressed:
        key = normalize_term(term)
        if key not in seen:
            seen.add(key)
            final.append(term)

    return final


# ============================================================================
# LOW CONFIDENCE STATISTICS LOGGING
# ============================================================================

_LOW_CONF_STATS_PATH = Path(__file__).parent / "artifacts" / "results" / "low_confidence_stats.jsonl"


def _log_low_confidence_stats(
    term: str,
    doc_id: str,
    vote_count: int,
    entity_ratio: float,
    sources: list[str],
    in_gt: bool,
) -> None:
    _LOW_CONF_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "term": term,
        "doc_id": doc_id,
        "vote_count": vote_count,
        "entity_ratio": round(entity_ratio, 4),
        "sources": sources,
        "in_gt": in_gt,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(_LOW_CONF_STATS_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def clear_low_confidence_stats() -> None:
    if _LOW_CONF_STATS_PATH.exists():
        _LOW_CONF_STATS_PATH.unlink()


# ============================================================================
# STAGE 5b: V5 PIPELINE (3 Haiku + Heuristic + Sonnet validation)
# ============================================================================

def extract_hybrid_v5(
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
    contextual_seeds_set = set(
        t.lower() for t in auto_vocab.get("contextual_seeds", [])
    ) if cfg.use_contextual_seeds else set()
    low_precision_set = set(
        t.lower() for t in auto_vocab.get("low_precision", [])
    ) if cfg.use_low_precision_filter else set()

    doc_text = doc["text"]
    doc_id = doc.get("doc_id", "unknown")
    gt_terms_lower = {t.lower() for t in doc.get("gt_terms", [])}

    # --- Stage 1: Extraction ---
    if cfg.use_candidate_verify:
        from .benchmark_prompt_variants import run_prompt_variant
        cv_terms, _ = run_prompt_variant(
            "candidate_verify_v1", doc, train_docs, index, model,
        )
        taxonomy_model = "sonnet" if cfg.use_sonnet_taxonomy else "haiku"
        taxonomy_terms, _ = _extract_haiku_taxonomy(doc, llm_model=taxonomy_model)
        heuristic_terms = _extract_heuristic(doc) if cfg.use_heuristic_extraction else []
        seed_terms = _extract_seeds(doc, seeds_list)
        contextual_seed_terms = (
            _extract_seeds(doc, list(auto_vocab.get("contextual_seeds", [])))
            if cfg.use_contextual_seeds else []
        )

        taxonomy_source_name = "sonnet_taxonomy" if cfg.use_sonnet_taxonomy else "haiku_taxonomy"
        candidates_by_source = {
            "candidate_verify": cv_terms,
            taxonomy_source_name: taxonomy_terms,
            "heuristic": heuristic_terms,
            "seeds": seed_terms,
            "contextual_seeds": contextual_seed_terms,
        }
    else:
        haiku_fewshot_terms, _ = _extract_haiku_fewshot(doc, train_docs, index, model)
        taxonomy_model = "sonnet" if cfg.use_sonnet_taxonomy else "haiku"
        taxonomy_terms, _ = _extract_haiku_taxonomy(doc, llm_model=taxonomy_model)
        haiku_simple_terms, _ = _extract_haiku_simple(doc)
        heuristic_terms = _extract_heuristic(doc) if cfg.use_heuristic_extraction else []
        seed_terms = _extract_seeds(doc, seeds_list)
        contextual_seed_terms = (
            _extract_seeds(doc, list(auto_vocab.get("contextual_seeds", [])))
            if cfg.use_contextual_seeds else []
        )

        taxonomy_source_name = "sonnet_taxonomy" if cfg.use_sonnet_taxonomy else "haiku_taxonomy"
        candidates_by_source = {
            "haiku_fewshot": haiku_fewshot_terms,
            taxonomy_source_name: taxonomy_terms,
            "haiku_simple": haiku_simple_terms,
            "heuristic": heuristic_terms,
            "seeds": seed_terms,
            "contextual_seeds": contextual_seed_terms,
        }

    # --- Stage 2: Grounding + dedup ---
    grounded = _ground_and_dedup(candidates_by_source, doc_text)

    # --- Stage 3: Noise filter ---
    after_noise: dict[str, dict] = {}
    for key, cand in grounded.items():
        term = cand["term"]
        if _auto_reject_noise(term, negatives_set, bypass_set, strategy=cfg, doc_text=doc_text):
            continue
        after_noise[key] = cand

    if cfg.use_candidate_verify:
        llm_sources = {"candidate_verify", taxonomy_source_name}
    else:
        llm_sources = {"haiku_fewshot", taxonomy_source_name, "haiku_simple"}

    # --- Stage 4: Confidence tier routing ---
    high_confidence: list[str] = []
    needs_validation: list[str] = []
    low_confidence: list[dict] = []

    seeds_set = {s.lower() for s in seeds_list}

    for key, cand in after_noise.items():
        term = cand["term"]
        source_count = cand["source_count"]
        sources = list(cand["sources"])
        sources_set = cand["sources"]

        tl = term.lower().strip()
        info = term_index.get(tl) if term_index else None
        entity_ratio = info["entity_ratio"] if info else 0.5

        has_llm_vote = bool(sources_set & llm_sources)
        is_heuristic_only = sources_set == {"heuristic"} or sources_set <= {"heuristic", "seeds", "contextual_seeds"}

        # ALL_CAPS corroboration: heuristic-only ALL_CAPS need ≥1 LLM vote
        # Skip for terms validated by training data (seeds/bypass/high entity_ratio)
        if (
            cfg.allcaps_require_corroboration
            and re.match(r"^[A-Z][A-Z0-9_]+$", term)
            and not has_llm_vote
            and tl not in bypass_set
            and tl not in seeds_set
            and entity_ratio < cfg.high_entity_ratio_threshold
        ):
            low_confidence.append({
                "term": term,
                "vote_count": source_count,
                "entity_ratio": entity_ratio,
                "sources": sources,
            })
            continue

        # LOW_PRECISION filter: borderline generic terms need 3+ sources
        if (
            cfg.use_low_precision_filter
            and tl in low_precision_set
            and source_count < cfg.high_confidence_min_sources
        ):
            low_confidence.append({
                "term": term,
                "vote_count": source_count,
                "entity_ratio": entity_ratio,
                "sources": sources,
            })
            continue

        # HIGH: structural pattern keeps (with training evidence guard)
        if _auto_keep_structural(term):
            if entity_ratio == 0 and tl not in seeds_set and tl not in bypass_set:
                needs_validation.append(term)
            elif cfg.structural_require_llm_vote and not has_llm_vote:
                needs_validation.append(term)
            else:
                high_confidence.append(term)
            continue

        # HIGH: seed bypass (data-driven common words from training)
        if cfg.seed_bypass_to_high_confidence and not cfg.disable_seed_bypass and tl in seeds_set and "seeds" in sources_set:
            if not cfg.seed_bypass_require_context:
                high_confidence.append(term)
                continue
            if source_count >= cfg.seed_bypass_min_sources_for_auto:
                high_confidence.append(term)
                continue
            if _has_technical_context(term, doc_text):
                high_confidence.append(term)
                continue

        # HIGH: 3+ sources agree
        if source_count >= cfg.high_confidence_min_sources:
            high_confidence.append(term)
            continue

        # HIGH: training data strongly supports (entity_ratio >= 0.8)
        if entity_ratio >= cfg.high_entity_ratio_threshold:
            high_confidence.append(term)
            continue

        # MEDIUM: contextual seed + any LLM vote
        if cfg.use_contextual_seeds and tl in contextual_seeds_set and "contextual_seeds" in sources_set:
            needs_validation.append(term)
            continue

        # MEDIUM: 2 sources agree
        if source_count >= cfg.validate_min_sources:
            needs_validation.append(term)
            continue

        # MEDIUM: training data moderately supports
        if entity_ratio >= cfg.medium_entity_ratio_threshold:
            needs_validation.append(term)
            continue

        # v5.2: Route single-vote LLM terms to validation instead of LOW
        if cfg.route_single_vote_to_validation and has_llm_vote and source_count >= 1:
            if entity_ratio >= cfg.single_vote_min_entity_ratio:
                needs_validation.append(term)
                continue

        # LOW: everything else
        low_confidence.append({
            "term": term,
            "vote_count": source_count,
            "entity_ratio": entity_ratio,
            "sources": sources,
        })

    # --- Log LOW confidence stats ---
    if cfg.log_low_confidence and low_confidence:
        for lc in low_confidence:
            in_gt = lc["term"].lower() in gt_terms_lower
            _log_low_confidence_stats(
                term=lc["term"],
                doc_id=doc_id,
                vote_count=lc["vote_count"],
                entity_ratio=lc["entity_ratio"],
                sources=lc["sources"],
                in_gt=in_gt,
            )

    # --- Stage 4b: Validate MEDIUM confidence ---
    if needs_validation and term_index:
        skip_validation = []
        needs_sonnet = []
        for term in needs_validation:
            tl = term.lower().strip()
            info = term_index.get(tl)
            er = info["entity_ratio"] if info else 0.5
            if er >= cfg.skip_validation_entity_ratio:
                skip_validation.append(term)
            else:
                needs_sonnet.append(term)

        validated_sonnet = _run_term_retrieval_validation(
            needs_sonnet, doc_text, term_index, strategy=cfg,
            bypass_set=bypass_set,
        ) if needs_sonnet else []

        validated = high_confidence + skip_validation + validated_sonnet
    elif needs_validation:
        validated = high_confidence + needs_validation
    else:
        validated = high_confidence

    # --- Stage 5: Post-processing ---
    expanded = _expand_spans(validated, doc_text)

    if cfg.suppress_path_embedded:
        expanded = [t for t in expanded if not _is_embedded_in_path(t, doc_text)]

    _url_re = re.compile(r"https?://\S+|^www\.\S+", re.I)
    expanded = [t for t in expanded if not _url_re.search(t)]

    protected_seed_set = seeds_set | contextual_seeds_set | bypass_set
    suppressed = _suppress_subspans(expanded, protected_seeds=protected_seed_set)

    seen: set[str] = set()
    final: list[str] = []
    for term in suppressed:
        key = normalize_term(term)
        if key not in seen:
            seen.add(key)
            final.append(term)

    return final
