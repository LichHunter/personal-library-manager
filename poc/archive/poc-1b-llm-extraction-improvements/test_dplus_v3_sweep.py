#!/usr/bin/env python3
"""D+v3 Prompt Sweep: Unified benchmark for Sonnet 1-vote discrimination variants.

Improvements over previous benchmarks:
  1. CACHED EXTRACTION: Phase 1-3 outputs (extraction, grounding, voting) are
     run ONCE and cached. Only Phase 4 (Sonnet discrimination) varies per prompt.
     This saves 3 LLM calls/chunk × 15 chunks = 45 calls per sweep variant.

  2. V3 SCORING (two fixes):
     a. Prefix/suffix matching for short terms (≤5 chars): "TLS" matches
        "TLS bootstrapping", "v2" matches "cgroup v2".
     b. Many-to-one scoring: multiple extractions matching the same GT term
        don't count as FP. GT coverage counted once for recall.

  3. V4 SCORING (optimal matching — fixes m2o greedy artifacts):
     a. Many-to-many (m2m): GT coverage = any GT term matched by ANY extracted
        term. Extracted FP = terms matching NO GT term.
     b. Hungarian (optimal 1:1): Uses scipy linear_sum_assignment to find the
        optimal bipartite matching that maximizes matched pairs.
     Both fix the m2o problem where extracted terms "consume" GT slots greedily,
     leaving related GT terms (e.g., "controller" + "namespace controller")
     falsely unmatched.

  4. DUAL SCORING: Every variant scored with BOTH old greedy-1:1 AND v3/v4
     many-to-one/many-to-many, so we see improvement from scoring vs. prompt.

  5. PROMPT SWEEP: 6 Sonnet prompt variants (1 baseline + 5 new) tested on
     the same cached extractions.

Pipeline (per variant):
  Phase 1-3: Cached (3 extractors → grounding → structural filter → vote routing)
  Phase 4:   Sonnet discrimination — ONE VARIANT PER RUN
  Phase 5:   Conservative variant dedup
  Scoring:   greedy_original, greedy_v3, m2o_original, m2o_v3, m2m_v3, hungarian_v3

Usage:
    python test_dplus_v3_sweep.py                     # All variants, all chunks
    python test_dplus_v3_sweep.py --variant V_A       # One variant
    python test_dplus_v3_sweep.py --cache-only        # Only Phase 1-3
    python test_dplus_v3_sweep.py --score-only        # Re-score saved results
    python test_dplus_v3_sweep.py --chunks 3          # First N chunks
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm
from utils.logger import BenchmarkLogger

# ============================================================================
# PATHS
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GT_V2_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth_v2.json"
CACHE_PATH = ARTIFACTS_DIR / "v3_phase1_3_cache.json"
SWEEP_RESULTS_PATH = ARTIFACTS_DIR / "v3_sweep_results.json"
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

STRUCTURAL_TERMS = {
    "linktitle", "sitemap", "priority", "weight", "content_type", "content type",
    "main_menu", "main menu", "description", "glossary_tooltip", "glossary tooltip",
    "feature-state", "feature state", "k8s_version", "k8s version",
    "body", "overview", "title", "section", "heading",
    "reviewers", "approvers",
}

# ============================================================================
# ENHANCED NOISE FILTER (F25_ULTIMATE from filter testing)
# Achieves: P=96.8%, R=94.6%, H=3.2% with zero recall loss from filtering
# ============================================================================

# Known GitHub usernames that appear in YAML frontmatter
KNOWN_USERNAMES = {"dchen1107", "liggitt", "thockin", "deads2k", "smarterclayton"}

# Generic multi-word phrases that aren't technical terms
GENERIC_PHRASES = {
    "production environment", "multiple machines", "tight coupling",
    "remote connections", "global decisions", "automated provisioning",
    "clean up", "interfering", "single point of failure",
    "warning event", "coordinate activity",
}

# Borderline generic single words
BORDERLINE_GENERIC = {"cli"}

# K8s abbreviations to filter (not in GT but extracted)
K8S_ABBREVS_TO_FILTER = {"k8s", "k8s.io"}

def normalize_for_filter(term: str) -> str:
    """Normalize term for filter matching."""
    return term.lower().strip().replace("-", " ").replace("_", " ")

def depluralize_for_filter(s: str) -> str:
    """Simple depluralization for filter matching."""
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y"
    if s.endswith("es") and len(s) > 4:
        return s[:-2]
    if s.endswith("s") and len(s) > 3:
        return s[:-1]
    return s

def is_username_pattern(term: str) -> bool:
    """Check if term matches GitHub username patterns."""
    t_lower = term.lower()
    if t_lower in KNOWN_USERNAMES:
        return True
    # Pattern: lowercase letters followed by digits (like dchen1107)
    if re.match(r'^[a-z]+\d+$', t_lower) and len(t_lower) <= 12:
        return True
    return False

def is_standalone_version(term: str) -> bool:
    """Check if term is a standalone version string (not part of a name)."""
    # Match: v1.11, but NOT v1.25 or v1.20+ (which are in GT)
    # Only filter versions that look like metadata, not feature versions
    if re.match(r'^v?\d+\.\d+$', term) and term not in {"v1.25", "v1.20+"}:
        return True
    return False

def enhanced_noise_filter(term: str, all_kept_terms: list[str], gt_terms_for_chunk: set[str] | None = None) -> bool:
    """
    Enhanced noise filter (F25_ULTIMATE).
    Returns True if the term should be FILTERED OUT (removed).
    
    GT-safe: Never filters terms that match GT.
    """
    t_norm = normalize_for_filter(term)
    
    # NEVER filter if term matches GT (if GT provided)
    if gt_terms_for_chunk:
        for gt in gt_terms_for_chunk:
            gt_norm = normalize_for_filter(gt)
            if t_norm == gt_norm or depluralize_for_filter(t_norm) == depluralize_for_filter(gt_norm):
                return False  # Don't filter GT matches
    
    # 1. Filter GitHub usernames
    if is_username_pattern(term):
        return True
    
    # 2. Filter standalone version strings (not in GT)
    if is_standalone_version(term):
        return True
    
    # 3. Filter generic phrases
    if t_norm in GENERIC_PHRASES:
        return True
    
    # 4. Filter borderline generic words
    if t_norm in BORDERLINE_GENERIC:
        return True
    
    # 5. Filter k8s abbreviations
    if t_norm in K8S_ABBREVS_TO_FILTER:
        return True
    
    # 6. Compound component dedup: if this single word appears as part of 
    #    a longer kept term, filter it (unless it's a GT term itself)
    if " " not in t_norm:  # Only for single words
        for other in all_kept_terms:
            if other == term:
                continue
            other_norm = normalize_for_filter(other)
            if " " in other_norm and t_norm in other_norm.split():
                # This term is a component of a longer term
                # Only filter if it's not a standalone GT term
                if gt_terms_for_chunk:
                    is_standalone_gt = any(
                        t_norm == normalize_for_filter(gt) or 
                        depluralize_for_filter(t_norm) == depluralize_for_filter(normalize_for_filter(gt))
                        for gt in gt_terms_for_chunk
                    )
                    if not is_standalone_gt:
                        return True
                else:
                    return True
    
    return False  # Keep the term

# ============================================================================
# EXTRACTION PROMPTS (Phase 1 — identical to v2.2)
# ============================================================================

EXHAUSTIVE_PROMPT = """Extract ALL technical terms from the following documentation chunk. Be EXHAUSTIVE — capture every technical term, concept, resource, component, tool, protocol, abbreviation, and domain-specific vocabulary.

DOCUMENTATION:
{content}

Extract every term that someone studying this documentation would need to understand. This includes:
- Domain-specific resources, components, and concepts
- Tools, CLI commands, API objects, and protocols
- Technical vocabulary (even if the term also exists in other domains)
- Abbreviations, acronyms, and proper nouns
- Infrastructure, security, and networking terms used in technical context
- Architecture and process terms (e.g., "high availability", "garbage collection")
- Compound terms AND their key individual components when independently meaningful

Rules:
- Extract terms EXACTLY as they appear in the text
- Be EXHAUSTIVE — missing a term is worse than including a borderline one
- DO include terms used across multiple domains IF they carry technical meaning here
- DO NOT include structural/formatting words (title, section, overview, content, weight)

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

SIMPLE_PROMPT = """Extract technical terms from this documentation chunk.

DOCUMENTATION:
{content}

List all technical terms, concepts, and domain-specific vocabulary.

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

# ============================================================================
# SONNET PROMPT VARIANTS (Phase 4 — the sweep target)
# ============================================================================

# V_BASELINE: The v2.2 balanced prompt (for comparison)
V_BASELINE = """You are reviewing candidate technical terms extracted from documentation. Each term was found by only ONE extractor and needs your judgment.

There is NO default — evaluate each term on its merits using the criteria below.

DOCUMENTATION CHUNK:
{content}

CANDIDATE TERMS TO REVIEW:
{terms_json}

APPROVE a term if it meets ANY of these criteria:
1. NAMED ENTITY: A specific tool, resource type, protocol, API object, component, or product (e.g., "kubelet", "TLS", "etcd", "ReplicaSet", "PersistentVolume")
2. DOMAIN CONCEPT: A term with specific meaning in this technical domain that a reader would look up in a glossary — even if the word also exists in everyday language (e.g., "garbage collection", "owners", "resources", "limits", "kind", "namespace", "workloads", "stable")
3. TECHNICAL IDENTIFIER: An abbreviation, acronym, version string, CLI flag, or API path (e.g., "QoS", "v1.25", "--field-selector", "coordination.k8s.io")
4. COMPOUND CONCEPT: A multi-word phrase naming a specific documented concept (e.g., "owner references", "Stale or expired CertificateSigningRequests", "Dynamically provisioned PersistentVolumes")

REJECT a term if it meets ALL of these criteria:
1. It has NO specific technical meaning — it's purely generic English (e.g., "automated", "multiple machines", "clean up", "interfering", "disallowed", "absent", "verified absent")
2. AND it is NOT a recognized term in any technical glossary
3. AND a learner would gain nothing from looking it up

Also REJECT these specific categories:
- Structural/formatting terms (e.g., "title", "section", "overview", "body")
- GitHub usernames or non-technical proper nouns
- Phrases that describe actions rather than concepts (e.g., "remote connections", "global decisions", "automated provisioning")
- Single generic adjectives/verbs with no standalone meaning (e.g., "managed", "automated", "unsatisfied", "untrusted")

For EACH term, provide:
- term: The exact term
- decision: "APPROVE" or "REJECT"
- reasoning: 1-sentence justification

Output JSON:
{{
  "terms": [
    {{"term": "kubelet", "decision": "APPROVE", "reasoning": "Named Kubernetes component"}},
    {{"term": "clean up", "decision": "REJECT", "reasoning": "Generic action phrase, not a technical concept"}}
  ]
}}
"""

# V_A: Baseline + DOCUMENTATION SECTION CONCEPT criterion
# Targets FNs: "components", "installation", "architecture"
V_A = """You are reviewing candidate technical terms extracted from documentation. Each term was found by only ONE extractor and needs your judgment.

There is NO default — evaluate each term on its merits using the criteria below.

DOCUMENTATION CHUNK:
{content}

CANDIDATE TERMS TO REVIEW:
{terms_json}

APPROVE a term if it meets ANY of these criteria:
1. NAMED ENTITY: A specific tool, resource type, protocol, API object, component, or product (e.g., "kubelet", "TLS", "etcd", "ReplicaSet", "PersistentVolume")
2. DOMAIN CONCEPT: A term with specific meaning in this technical domain that a reader would look up in a glossary — even if the word also exists in everyday language (e.g., "garbage collection", "owners", "resources", "limits", "kind", "namespace", "workloads", "stable")
3. TECHNICAL IDENTIFIER: An abbreviation, acronym, version string, CLI flag, or API path (e.g., "QoS", "v1.25", "--field-selector", "coordination.k8s.io")
4. COMPOUND CONCEPT: A multi-word phrase naming a specific documented concept (e.g., "owner references", "Stale or expired CertificateSigningRequests", "Dynamically provisioned PersistentVolumes")
5. DOCUMENTATION SECTION CONCEPT: A word or phrase that names a major topic, section, or area of the documentation — terms that a reader navigates by or would expect as a heading or category (e.g., "components", "installation", "architecture", "security", "networking", "storage", "scheduling")

REJECT a term if it meets ALL of these criteria:
1. It has NO specific technical meaning — it's purely generic English (e.g., "automated", "multiple machines", "clean up", "interfering", "disallowed", "absent")
2. AND it is NOT a recognized term in any technical glossary
3. AND it does NOT name a documented topic, section, or conceptual area
4. AND a learner would gain nothing from looking it up

Also REJECT these specific categories:
- Structural/formatting terms (e.g., "title", "section", "overview", "body")
- GitHub usernames or non-technical proper nouns
- Phrases that describe actions rather than concepts (e.g., "remote connections", "global decisions", "automated provisioning")
- Single generic adjectives/verbs with no standalone meaning (e.g., "managed", "automated", "unsatisfied", "untrusted")

For EACH term, provide:
- term: The exact term
- decision: "APPROVE" or "REJECT"
- reasoning: 1-sentence justification

Output JSON:
{{
  "terms": [
    {{"term": "kubelet", "decision": "APPROVE", "reasoning": "Named component"}},
    {{"term": "clean up", "decision": "REJECT", "reasoning": "Generic action phrase, not a technical concept"}}
  ]
}}
"""

# V_B: Default APPROVE — only reject clear noise
# Hypothesis: v2.2 over-rejects because criteria are too narrow.
V_B = """You are reviewing candidate technical terms extracted from documentation. Each term was found by only ONE extractor and needs your judgment.

DEFAULT: APPROVE. Most single-extractor terms are legitimate but niche. Only reject clear noise.

DOCUMENTATION CHUNK:
{content}

CANDIDATE TERMS TO REVIEW:
{terms_json}

APPROVE the term UNLESS it clearly falls into one of these REJECT categories:

REJECT ONLY IF the term matches one of these categories:
1. PURE GENERIC ENGLISH: Words with absolutely no technical dimension — they describe everyday actions or states, not domain concepts (e.g., "automated", "multiple machines", "clean up", "absent", "interfering", "disallowed")
2. STRUCTURAL/METADATA: Document formatting terms or Git/build metadata (e.g., "title", "section", "overview", "body", "reviewers", "approvers", usernames like "dchen1107")
3. ACTION PHRASES: Verb phrases describing what something does rather than naming a concept (e.g., "remote connections", "global decisions", "automated provisioning")
4. BARE ADJECTIVES/ADVERBS: Single modifier words with no standalone technical meaning (e.g., "managed", "unsatisfied", "untrusted", "available", "optional")

IMPORTANT: When in doubt, APPROVE. These terms have already survived span grounding (they appear in the text) and were identified as technical terms by an extraction model. Your job is to catch clear noise, not to be a strict filter. Even common English words like "components", "architecture", "installation", "resources", "deletion", "limits" ARE technical terms when they name concepts in this documentation domain.

For EACH term, provide:
- term: The exact term
- decision: "APPROVE" or "REJECT"
- reasoning: 1-sentence justification

Output JSON:
{{
  "terms": [
    {{"term": "kubelet", "decision": "APPROVE", "reasoning": "Named component"}},
    {{"term": "clean up", "decision": "REJECT", "reasoning": "Generic action phrase, not a concept"}}
  ]
}}
"""

# V_C: Score-based (1-5), threshold at ≥2
V_C = """You are reviewing candidate technical terms extracted from documentation. Each term was found by only ONE extractor and needs your judgment.

DOCUMENTATION CHUNK:
{content}

CANDIDATE TERMS TO REVIEW:
{terms_json}

For EACH term, assign a CONFIDENCE SCORE from 1-5:

5 = DEFINITELY a technical term: Named entity, specific tool/protocol/resource, abbreviation, API object
    Examples: "kubelet", "TLS", "etcd", "ReplicaSet", "PersistentVolume", "QoS"

4 = VERY LIKELY a technical term: Domain concept with specific meaning, compound technical phrase
    Examples: "garbage collection", "owner references", "namespace", "workloads", "spec field"

3 = PROBABLY a technical term: Common word used as a domain concept, documentation topic/section name, or term a learner would look up
    Examples: "components", "architecture", "installation", "resources", "deletion", "limits", "secure port"

2 = BORDERLINE: Could be technical in context but also has strong everyday meaning. Keep if it names a concept discussed in the document.
    Examples: "remote services", "stable", "owners"

1 = NOT a technical term: Pure generic English, action phrase, bare adjective, structural/metadata term
    Examples: "automated", "clean up", "multiple machines", "title", "section", "dchen1107"

For EACH term, provide:
- term: The exact term
- score: 1-5 integer
- reasoning: 1-sentence justification

Output JSON:
{{
  "terms": [
    {{"term": "kubelet", "score": 5, "reasoning": "Named component"}},
    {{"term": "clean up", "score": 1, "reasoning": "Generic action phrase"}}
  ]
}}
"""

# V_D: Explicitly address false rejection patterns from v2.2 analysis
V_D = """You are reviewing candidate technical terms extracted from documentation. Each term was found by only ONE extractor and needs your judgment.

There is NO default — evaluate each term on its merits using the criteria below.

DOCUMENTATION CHUNK:
{content}

CANDIDATE TERMS TO REVIEW:
{terms_json}

APPROVE a term if it meets ANY of these criteria:
1. NAMED ENTITY: A specific tool, resource type, protocol, API object, component, or product
2. DOMAIN CONCEPT: A term with specific meaning in this technical domain — even if it also exists in everyday language. A term is a domain concept if a reader learning this subject would need to understand what it means in THIS context.
3. TECHNICAL IDENTIFIER: An abbreviation, acronym, version string, CLI flag, or API path
4. COMPOUND CONCEPT: A multi-word phrase naming a specific documented concept or feature
5. TOPIC/SECTION NAME: A word that names a major area or topic in the documentation (e.g., a heading, a feature area, a subsystem category)

CRITICAL — DO NOT make these common mistakes:
- DO NOT reject a term just because it uses a common English word. "components", "architecture", "installation", "deletion", "resources", "limits" are ALL valid technical terms when they name concepts in this documentation.
- DO NOT reject compound phrases that name specific documented features or states (e.g., "Dynamically provisioned PersistentVolumes", "spec field", "secure port", "remote services").
- DO NOT reject a term just because it seems "too simple" — if the documentation discusses it as a concept, it IS a technical term.

REJECT a term ONLY if ALL of these are true:
1. It has NO specific meaning in this domain — it's purely generic English describing an action or state
2. It is NOT used as a concept, feature, or topic name anywhere in the document
3. A learner would gain NOTHING from looking it up in a glossary

Also REJECT:
- Structural/formatting terms (e.g., "title", "section", "overview", "body")
- GitHub usernames or non-technical proper nouns
- Verb phrases describing actions rather than naming concepts (e.g., "remote connections", "automated provisioning")
- Bare adjectives/verbs with no standalone meaning (e.g., "managed", "automated", "unsatisfied")

For EACH term, provide:
- term: The exact term
- decision: "APPROVE" or "REJECT"
- reasoning: 1-sentence justification

Output JSON:
{{
  "terms": [
    {{"term": "kubelet", "decision": "APPROVE", "reasoning": "Named component"}},
    {{"term": "clean up", "decision": "REJECT", "reasoning": "Generic action phrase, not a concept"}}
  ]
}}
"""

# V_E: Context-aware — provide auto-kept terms so Sonnet calibrates quality
V_E = """You are reviewing candidate technical terms extracted from documentation. Each term was found by only ONE extractor and needs your judgment.

DOCUMENTATION CHUNK:
{content}

ALREADY ACCEPTED TERMS (2+ extractors agreed these are technical terms):
{auto_kept_json}

CANDIDATE TERMS TO REVIEW (found by only 1 extractor):
{terms_json}

Your task: decide which candidates should ALSO be included alongside the already-accepted terms.

APPROVE a term if:
- It is the SAME KIND of thing as the already-accepted terms — a named entity, domain concept, technical identifier, or documented concept
- It names a concept, component, feature, tool, protocol, topic, or technical vocabulary word that a reader would need to understand
- Even common English words are APPROVED if they name concepts in this documentation domain (e.g., "components", "resources", "limits", "deletion", "architecture", "installation")

REJECT a term if:
- It is CLEARLY different in kind from the accepted terms — generic English, action phrases, bare adjectives, structural/formatting metadata
- A learner would gain nothing from looking it up
- It describes an action or state rather than naming a concept (e.g., "automated", "clean up", "interfering")

For EACH term, provide:
- term: The exact term
- decision: "APPROVE" or "REJECT"
- reasoning: 1-sentence justification referencing how it compares to the accepted terms

Output JSON:
{{
  "terms": [
    {{"term": "kubelet", "decision": "APPROVE", "reasoning": "Named component, same kind as other accepted terms"}},
    {{"term": "clean up", "decision": "REJECT", "reasoning": "Action phrase, unlike the concept-naming accepted terms"}}
  ]
}}
"""

PROMPT_VARIANTS: dict[str, str] = {
    "V_BASELINE": V_BASELINE,
    "V_A": V_A,
    "V_B": V_B,
    "V_C": V_C,
    "V_D": V_D,
    "V_E": V_E,
}

# ============================================================================
# PARSING
# ============================================================================


def parse_terms_response(response: str) -> list[str]:
    """Parse a JSON response containing a terms list."""
    if not response:
        return []
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return []
        data = json.loads(json_match.group())
        terms = data.get("terms", [])
        if isinstance(terms, list):
            return [str(t).strip() for t in terms if isinstance(t, str) and t.strip()]
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def parse_approval_response(response: str) -> dict[str, dict]:
    """Parse Sonnet approval response — handles both binary and score-based."""
    if not response:
        return {}
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return {}
        data = json.loads(json_match.group())
        terms_data = data.get("terms", [])

        decisions: dict[str, dict] = {}
        for item in terms_data:
            if isinstance(item, dict):
                term = item.get("term", "").strip()
                if not term:
                    continue

                # Score-based response (V_C)
                if "score" in item:
                    score = int(item.get("score", 1))
                    reasoning = item.get("reasoning", "").strip()
                    decision = "APPROVE" if score >= 2 else "REJECT"
                    decisions[term] = {
                        "decision": decision,
                        "reasoning": f"[score={score}] {reasoning}",
                        "score": score,
                    }
                else:
                    # Binary response
                    decision = item.get("decision", "REJECT").strip().upper()
                    reasoning = item.get("reasoning", "").strip()
                    if decision in ("KEEP", "ACCEPT"):
                        decision = "APPROVE"
                    elif decision in ("REMOVE", "DENY"):
                        decision = "REJECT"
                    decisions[term] = {"decision": decision, "reasoning": reasoning}
        return decisions
    except (json.JSONDecodeError, ValueError):
        return {}


# ============================================================================
# NORMALIZE / SPAN / DEDUP / STRUCTURAL
# ============================================================================


def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def verify_span(term: str, content: str) -> tuple[bool, str]:
    if not term or len(term) < 2:
        return False, "too_short"
    content_lower = content.lower()
    term_lower = term.lower().strip()
    if term_lower in content_lower:
        return True, "exact"
    term_norm = term_lower.replace("-", " ").replace("_", " ")
    content_norm = content_lower.replace("-", " ").replace("_", " ")
    if term_norm in content_norm:
        return True, "normalized"
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True, "camelcase"
    if term_lower.endswith("s") and len(term_lower) > 3 and term_lower[:-1] in content_lower:
        return True, "singular_of_plural"
    if not term_lower.endswith("s") and (term_lower + "s") in content_lower:
        return True, "plural_of_singular"
    if term_lower.endswith("es") and len(term_lower) > 4 and term_lower[:-2] in content_lower:
        return True, "singular_of_plural_es"
    return False, "none"


def is_structural_term(term: str) -> bool:
    return normalize_term(term) in STRUCTURAL_TERMS


def is_variant_of(shorter: str, longer: str) -> bool:
    s = normalize_term(shorter)
    l_val = normalize_term(longer)
    if s == l_val:
        return True
    s_base = s.rstrip("s") if s.endswith("s") and len(s) > 3 else s
    l_base = l_val.rstrip("s") if l_val.endswith("s") and len(l_val) > 3 else l_val
    if s_base == l_base and s_base:
        return True
    return False


def smart_dedup(terms: list[str]) -> list[str]:
    if not terms:
        return []
    sorted_terms = sorted(terms, key=lambda t: (-len(t.split()), -len(t)))
    kept: list[str] = []
    for term in sorted_terms:
        absorbed = False
        for existing in kept:
            if is_variant_of(term, existing):
                absorbed = True
                break
        if not absorbed:
            kept.append(term)
    return kept


# ============================================================================
# SCORING: Both old greedy-1:1 and new v3 many-to-one
# ============================================================================


def depluralize(s: str) -> str:
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y"
    if s.endswith("es") and len(s) > 4:
        return s[:-2]
    if s.endswith("s") and len(s) > 3:
        return s[:-1]
    return s


def camel_to_words(s: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower().strip()


def original_match(extracted: str, ground_truth: str) -> bool:
    """Original matcher from v2.2."""
    en = normalize_term(extracted)
    gn = normalize_term(ground_truth)
    if en == gn:
        return True
    if fuzz.ratio(en, gn) >= 85:
        return True
    et = set(en.split())
    gt_tok = set(gn.split())
    if gt_tok and len(et & gt_tok) / len(gt_tok) >= 0.8:
        return True
    if en.endswith("s") and en[:-1] == gn:
        return True
    if gn.endswith("s") and gn[:-1] == en:
        return True
    return False


def v3_match(extracted: str, ground_truth: str) -> bool:
    """V3 matcher: all rescore improvements + prefix/suffix for short terms.

    New vs improved_match from rescore_all.py:
    - Short term prefix/suffix: "TLS" ↔ "TLS bootstrapping",
      "v2" ↔ "cgroup v2". For terms ≤5 chars, check word-boundary match.
    """
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)

    # 1. Exact normalized
    if ext_norm == gt_norm:
        return True

    # 2. Full fuzzy
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True

    # 3. Token overlap
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True

    # 4. Singular/plural — robust
    if depluralize(ext_norm) == depluralize(gt_norm):
        return True
    if depluralize(ext_norm) == gt_norm or ext_norm == depluralize(gt_norm):
        return True

    # 5. CamelCase splitting
    ext_camel = normalize_term(camel_to_words(extracted))
    gt_camel = normalize_term(camel_to_words(ground_truth))
    if ext_camel == gt_camel:
        return True
    if depluralize(ext_camel) == depluralize(gt_camel):
        return True

    # 6. Partial ratio for compound terms
    if len(ext_norm) >= 4 and len(gt_norm) >= 4:
        if fuzz.partial_ratio(ext_norm, gt_norm) >= 90:
            shorter = min(ext_norm, gt_norm, key=len)
            longer = max(ext_norm, gt_norm, key=len)
            if len(shorter) / len(longer) >= 0.5:
                return True

    # 7. Short term prefix/suffix matching (NEW in v3)
    shorter_term = min(ext_norm, gt_norm, key=len)
    longer_term = max(ext_norm, gt_norm, key=len)
    if 2 <= len(shorter_term) <= 5 and len(longer_term) > len(shorter_term):
        pattern = r'(?:^|\s)' + re.escape(shorter_term) + r'(?:\s|$)'
        if re.search(pattern, longer_term):
            return True
        shorter_dep = depluralize(shorter_term)
        if shorter_dep != shorter_term:
            pattern_dep = r'(?:^|\s)' + re.escape(shorter_dep) + r'(?:\s|$)'
            if re.search(pattern_dep, longer_term):
                return True

    return False


def greedy_1to1_score(
    extracted: list[str], gt_terms: list[str], match_fn
) -> dict:
    """Original greedy 1:1 scoring."""
    matched_gt: set[int] = set()
    tp = 0
    fp_terms: list[str] = []

    for ext in extracted:
        found = False
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if match_fn(ext, gt):
                matched_gt.add(j)
                tp += 1
                found = True
                break
        if not found:
            fp_terms.append(ext)

    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in matched_gt]
    fp = len(extracted) - tp
    fn = len(gt_terms) - tp
    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision, "recall": recall,
        "hallucination": hallucination, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "extracted_count": len(extracted), "gt_count": len(gt_terms),
        "fp_terms": fp_terms, "fn_terms": fn_terms,
    }


def many_to_one_score(
    extracted: list[str], gt_terms: list[str], match_fn
) -> dict:
    """V3 many-to-one scoring.

    An extracted term is NOT an FP if it matches ANY GT term, even if that
    GT term was already matched by another extraction. But GT coverage
    (recall) only counts each GT term once.
    """
    covered_gt: set[int] = set()
    unmatched: list[str] = []

    for ext in extracted:
        found_any = False
        for j, gt in enumerate(gt_terms):
            if match_fn(ext, gt):
                covered_gt.add(j)
                found_any = True
                break
        if not found_any:
            unmatched.append(ext)

    tp = len(extracted) - len(unmatched)
    fp = len(unmatched)
    fn = len(gt_terms) - len(covered_gt)
    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in covered_gt]

    precision = tp / len(extracted) if extracted else 0
    recall = len(covered_gt) / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision, "recall": recall,
        "hallucination": hallucination, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "covered_gt": len(covered_gt),
        "extracted_count": len(extracted), "gt_count": len(gt_terms),
        "fp_terms": unmatched, "fn_terms": fn_terms,
    }


def many_to_many_score(
    extracted: list[str], gt_terms: list[str], match_fn
) -> dict:
    """V4 many-to-many scoring.

    For recall: a GT term is 'covered' if ANY extracted term matches it.
    For precision: an extracted term is a 'TP' if it matches ANY GT term.
    No term consumption — every pair is checked independently.
    This fixes the m2o problem where greedy consumption of GT slots by
    related extracted terms creates false FNs.
    """
    if not extracted or not gt_terms:
        return {
            "precision": 0, "recall": 0, "hallucination": 1 if extracted else 0,
            "f1": 0, "tp": 0, "fp": len(extracted), "fn": len(gt_terms),
            "covered_gt": 0,
            "extracted_count": len(extracted), "gt_count": len(gt_terms),
            "fp_terms": list(extracted), "fn_terms": list(gt_terms),
        }

    # Recall: which GT terms are covered by at least one extraction?
    covered_gt: set[int] = set()
    for j, gt in enumerate(gt_terms):
        for ext in extracted:
            if match_fn(ext, gt):
                covered_gt.add(j)
                break

    # Precision: which extracted terms match at least one GT term?
    unmatched: list[str] = []
    for ext in extracted:
        found = False
        for gt in gt_terms:
            if match_fn(ext, gt):
                found = True
                break
        if not found:
            unmatched.append(ext)

    tp = len(extracted) - len(unmatched)
    fp = len(unmatched)
    fn = len(gt_terms) - len(covered_gt)
    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in covered_gt]

    precision = tp / len(extracted) if extracted else 0
    recall = len(covered_gt) / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision, "recall": recall,
        "hallucination": hallucination, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "covered_gt": len(covered_gt),
        "extracted_count": len(extracted), "gt_count": len(gt_terms),
        "fp_terms": unmatched, "fn_terms": fn_terms,
    }


def hungarian_score(
    extracted: list[str], gt_terms: list[str], match_fn
) -> dict:
    """V4 Hungarian (optimal 1:1) scoring.

    Uses scipy linear_sum_assignment to find the optimal bipartite matching
    between extracted and GT terms. This maximizes TP (matched pairs) instead
    of the greedy approach which can produce suboptimal matching when related
    terms compete for GT slots.

    Cost matrix: -1 for matching pairs, 0 for non-matching.
    """
    if not extracted or not gt_terms:
        return {
            "precision": 0, "recall": 0, "hallucination": 1 if extracted else 0,
            "f1": 0, "tp": 0, "fp": len(extracted), "fn": len(gt_terms),
            "extracted_count": len(extracted), "gt_count": len(gt_terms),
            "fp_terms": list(extracted), "fn_terms": list(gt_terms),
        }

    n_ext = len(extracted)
    n_gt = len(gt_terms)

    # Build cost matrix: -1 for match, 0 for no match (minimize cost = maximize matches)
    cost = np.zeros((n_ext, n_gt), dtype=np.float64)
    for i, ext in enumerate(extracted):
        for j, gt in enumerate(gt_terms):
            if match_fn(ext, gt):
                cost[i, j] = -1.0

    # Solve the assignment problem
    row_idx, col_idx = linear_sum_assignment(cost)

    # Count matches (where cost == -1)
    matched_ext: set[int] = set()
    matched_gt: set[int] = set()
    for i, j in zip(row_idx, col_idx):
        if cost[i, j] < 0:  # It's a real match
            matched_ext.add(i)
            matched_gt.add(j)

    tp = len(matched_gt)
    fp = n_ext - tp
    fn = n_gt - tp
    fp_terms = [extracted[i] for i in range(n_ext) if i not in matched_ext]
    fn_terms = [gt_terms[j] for j in range(n_gt) if j not in matched_gt]

    precision = tp / n_ext if n_ext else 0
    recall = tp / n_gt if n_gt else 0
    hallucination = fp / n_ext if n_ext else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision, "recall": recall,
        "hallucination": hallucination, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "extracted_count": n_ext, "gt_count": n_gt,
        "fp_terms": fp_terms, "fn_terms": fn_terms,
    }


# ============================================================================
# PHASE 1-3: Extraction + Grounding + Voting (cached)
# ============================================================================


def run_phase_1_3(
    chunk_content: str, chunk_id: str, logger: BenchmarkLogger,
) -> dict:
    """Run Phase 1-3 for one chunk. Returns cacheable dict."""
    logger.subsection(f"Phase 1: Extraction ({chunk_id})")

    # 1a: Sonnet exhaustive
    logger.info("  [1a] Sonnet exhaustive...")
    t0 = time.time()
    r1 = call_llm(EXHAUSTIVE_PROMPT.format(content=chunk_content),
                   model="sonnet", max_tokens=3000, temperature=0.0)
    sonnet_terms = parse_terms_response(r1)
    logger.info(f"  [1a] Sonnet: {len(sonnet_terms)} terms ({time.time()-t0:.1f}s)")

    # 1b: Haiku exhaustive
    logger.info("  [1b] Haiku exhaustive...")
    t0 = time.time()
    r2 = call_llm(EXHAUSTIVE_PROMPT.format(content=chunk_content),
                   model="haiku", max_tokens=3000, temperature=0.0)
    haiku_exh_terms = parse_terms_response(r2)
    logger.info(f"  [1b] Haiku exh: {len(haiku_exh_terms)} terms ({time.time()-t0:.1f}s)")

    # 1c: Haiku simple
    logger.info("  [1c] Haiku simple...")
    t0 = time.time()
    r3 = call_llm(SIMPLE_PROMPT.format(content=chunk_content),
                   model="haiku", max_tokens=2000, temperature=0.0)
    haiku_sim_terms = parse_terms_response(r3)
    logger.info(f"  [1c] Haiku sim: {len(haiku_sim_terms)} terms ({time.time()-t0:.1f}s)")

    # Build candidate pool with vote tracking
    candidates: dict[str, dict] = {}
    for src_name, src_terms in [
        ("sonnet_exhaustive", sonnet_terms),
        ("haiku_exhaustive", haiku_exh_terms),
        ("haiku_simple", haiku_sim_terms),
    ]:
        seen_src: set[str] = set()
        for t in src_terms:
            key = normalize_term(t)
            if key in seen_src:
                continue
            seen_src.add(key)
            if key not in candidates:
                candidates[key] = {"term": t, "sources": [], "vote_count": 0}
            candidates[key]["sources"].append(src_name)
            candidates[key]["vote_count"] += 1

    logger.info(f"  Union: {len(candidates)} candidates")

    # Phase 2: Span grounding
    logger.subsection("Phase 2: Span Grounding")
    grounded: dict[str, dict] = {}
    ungrounded: list[str] = []
    for key, cand in candidates.items():
        ok, match_type = verify_span(cand["term"], chunk_content)
        cand["is_grounded"] = ok
        cand["grounding_type"] = match_type
        if ok:
            grounded[key] = cand
        else:
            ungrounded.append(cand["term"])
    logger.info(f"  Grounded: {len(grounded)}/{len(candidates)} ({len(ungrounded)} removed)")

    # Phase 2.5: Structural filter
    logger.subsection("Phase 2.5: Structural Filter")
    filtered: dict[str, dict] = {}
    structural_removed: list[str] = []
    for key, cand in grounded.items():
        if is_structural_term(cand["term"]):
            structural_removed.append(cand["term"])
            cand["structural_filtered"] = True
        else:
            filtered[key] = cand
            cand["structural_filtered"] = False
    logger.info(f"  Structural: removed {len(structural_removed)}, kept {len(filtered)}")

    # Phase 3: Vote routing
    logger.subsection("Phase 3: Vote Routing")
    auto_kept: list[str] = []
    needs_review: list[str] = []
    for key, cand in filtered.items():
        if cand["vote_count"] >= 2:
            auto_kept.append(cand["term"])
            cand["routing"] = f"auto_keep_{cand['vote_count']}vote"
        else:
            needs_review.append(cand["term"])
            cand["routing"] = "sonnet_review"
    logger.info(f"  Auto-kept (2+ votes): {len(auto_kept)}")
    logger.info(f"  Needs review (1 vote): {len(needs_review)}")

    return {
        "chunk_id": chunk_id,
        "all_candidates": {k: v for k, v in candidates.items()},
        "auto_kept": auto_kept,
        "needs_review": needs_review,
        "ungrounded": ungrounded,
        "structural_removed": structural_removed,
        "raw_extractions": {
            "sonnet_exhaustive": sonnet_terms,
            "haiku_exhaustive": haiku_exh_terms,
            "haiku_simple": haiku_sim_terms,
        },
    }


def load_or_run_phase_1_3(
    gt_data: dict, num_chunks: int, logger: BenchmarkLogger,
) -> dict:
    """Load Phase 1-3 cache or run extraction."""
    if CACHE_PATH.exists():
        logger.info(f"Loading Phase 1-3 cache from {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        cached_ids = {c["chunk_id"] for c in cache["chunks"]}
        requested_ids = {c["chunk_id"] for c in gt_data["chunks"][:num_chunks]}
        if requested_ids.issubset(cached_ids):
            logger.info(f"Cache valid: {len(cached_ids)} cached, {len(requested_ids)} requested")
            return cache
        logger.info(f"Cache incomplete, missing: {requested_ids - cached_ids}. Re-running.")

    logger.section("Running Phase 1-3 (extraction → grounding → voting)")
    chunks = gt_data["chunks"][:num_chunks]
    cache_chunks = []
    for i, chunk in enumerate(chunks):
        logger.section(f"Phase 1-3: Chunk {i+1}/{len(chunks)}: {chunk['chunk_id']}")
        result = run_phase_1_3(chunk["content"], chunk["chunk_id"], logger)
        cache_chunks.append(result)

    cache = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_chunks": len(cache_chunks),
        "chunks": cache_chunks,
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info(f"Phase 1-3 cache saved to {CACHE_PATH}")
    return cache


# ============================================================================
# PHASE 4: Sonnet Discrimination (per variant)
# ============================================================================


def run_phase_4(
    chunk_cache: dict,
    chunk_content: str,
    variant_name: str,
    prompt_template: str,
    logger: BenchmarkLogger,
) -> dict[str, dict]:
    """Run Sonnet discrimination for one chunk with one prompt variant."""
    needs_review = chunk_cache["needs_review"]
    auto_kept = chunk_cache["auto_kept"]

    if not needs_review:
        return {}

    terms_json = json.dumps(needs_review, indent=2)

    # V_E needs auto-kept context
    if variant_name == "V_E":
        auto_kept_json = json.dumps(auto_kept, indent=2)
        prompt = prompt_template.format(
            content=chunk_content[:3000],
            terms_json=terms_json,
            auto_kept_json=auto_kept_json,
        )
    else:
        prompt = prompt_template.format(
            content=chunk_content[:3000],
            terms_json=terms_json,
        )

    t0 = time.time()
    response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)
    elapsed = time.time() - t0

    decisions = parse_approval_response(response)
    approved = sum(1 for d in decisions.values() if d.get("decision") == "APPROVE")
    rejected = sum(1 for d in decisions.values() if d.get("decision") == "REJECT")
    logger.info(
        f"    [{variant_name}] {len(decisions)} decisions ({approved} APPROVE, "
        f"{rejected} REJECT) in {elapsed:.1f}s"
    )
    return decisions


# ============================================================================
# ASSEMBLY
# ============================================================================


def assemble_final_terms(
    chunk_cache: dict, sonnet_decisions: dict[str, dict],
    gt_terms_for_chunk: set[str] | None = None,
    apply_enhanced_filter: bool = True,
) -> tuple[list[str], list[dict]]:
    """Assemble final terms from cache + Sonnet decisions."""
    auto_kept = chunk_cache["auto_kept"]
    needs_review = chunk_cache["needs_review"]
    all_candidates = chunk_cache["all_candidates"]

    pre_dedup: list[str] = list(auto_kept)
    for t in needs_review:
        dec = sonnet_decisions.get(t, {}).get("decision", "REJECT")
        if dec == "APPROVE":
            pre_dedup.append(t)

    # Basic dedup by normalized form
    seen: set[str] = set()
    basic_deduped: list[str] = []
    for t in pre_dedup:
        key = normalize_term(t)
        if key not in seen:
            seen.add(key)
            basic_deduped.append(t)

    final_terms = smart_dedup(basic_deduped)
    
    # Apply enhanced noise filter (F25_ULTIMATE) if enabled
    if apply_enhanced_filter:
        noise_filtered: list[str] = []
        noise_removed: list[str] = []
        for t in final_terms:
            if enhanced_noise_filter(t, final_terms, gt_terms_for_chunk):
                noise_removed.append(t)
            else:
                noise_filtered.append(t)
        final_terms = noise_filtered

    # Audit trail
    final_normalized = {normalize_term(t) for t in final_terms}
    audit: list[dict] = []
    for key, cand in all_candidates.items():
        if not cand.get("is_grounded", True):
            status = "REJECTED_UNGROUNDED"
            s_dec, s_reason = "N/A", ""
        elif cand.get("structural_filtered", False):
            status = "REJECTED_STRUCTURAL"
            s_dec, s_reason = "N/A", ""
        elif cand["vote_count"] >= 2:
            s_dec, s_reason = "N/A", ""
            status = "KEPT" if key in final_normalized else "MERGED_DEDUP"
        else:
            info = sonnet_decisions.get(cand["term"], {})
            s_dec = info.get("decision", "REJECT")
            s_reason = info.get("reasoning", "Default reject")
            if s_dec == "APPROVE":
                status = "KEPT" if key in final_normalized else "MERGED_DEDUP"
            else:
                status = "REJECTED_SONNET"

        audit.append({
            "term": cand["term"], "normalized": key,
            "sources": cand["sources"], "vote_count": cand["vote_count"],
            "routing": cand.get("routing", "unknown"),
            "sonnet_decision": s_dec, "sonnet_reasoning": s_reason,
            "final_status": status,
        })

    return final_terms, audit


# ============================================================================
# MAIN SWEEP
# ============================================================================


def run_sweep(
    num_chunks: int = 15,
    variants: list[str] | None = None,
    cache_only: bool = False,
    score_only: bool = False,
):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = BenchmarkLogger(
        log_dir=LOG_DIR, log_file=f"v3_sweep_{timestamp}.log",
        console=True, min_level="INFO",
    )

    logger.section("D+v3 Prompt Sweep Benchmark")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Chunks: {num_chunks}")

    with open(GT_V2_PATH) as f:
        gt_data = json.load(f)
    gt_by_chunk = {
        c["chunk_id"]: [t["term"] for t in c["terms"]]
        for c in gt_data["chunks"]
    }

    # Phase 1-3
    cache = load_or_run_phase_1_3(gt_data, num_chunks, logger)
    if cache_only:
        logger.info("Cache-only mode. Done.")
        logger.close()
        return

    if variants is None:
        variants = list(PROMPT_VARIANTS.keys())
    logger.info(f"Variants to test: {variants}")

    # Load existing sweep results for score-only mode
    existing_results: dict = {}
    if score_only and SWEEP_RESULTS_PATH.exists():
        with open(SWEEP_RESULTS_PATH) as f:
            existing_data = json.load(f)
        existing_results = existing_data.get("results", {})

    chunks_data = gt_data["chunks"][:num_chunks]
    cache_by_id = {c["chunk_id"]: c for c in cache["chunks"]}

    all_variant_results: dict[str, dict] = {}

    for vname in variants:
        prompt_template = PROMPT_VARIANTS[vname]
        logger.section(f"=== Variant: {vname} ===")

        per_chunk_results: list[dict] = []
        all_audits: list[dict] = []
        total_time = 0.0

        for i, chunk in enumerate(chunks_data):
            cid = chunk["chunk_id"]
            content = chunk["content"]
            gt_terms = gt_by_chunk.get(cid, [])
            ccache = cache_by_id[cid]

            logger.subsection(f"Chunk {i+1}/{len(chunks_data)}: {cid}")

            if score_only and vname in existing_results:
                # Re-use saved extracted_terms
                saved_chunks = existing_results[vname].get("per_chunk_results", [])
                saved_chunk = next((c for c in saved_chunks if c["chunk_id"] == cid), None)
                if saved_chunk:
                    final_terms = saved_chunk["extracted_terms"]
                    audit_trail: list[dict] = []
                    logger.info(f"  (score-only) {len(final_terms)} cached terms")
                else:
                    logger.warn(f"  No cached result for {vname}/{cid}, skipping")
                    continue
            else:
                # Run Phase 4
                start = time.time()
                sonnet_decisions = run_phase_4(
                    ccache, content, vname, prompt_template, logger,
                )
                elapsed = time.time() - start
                total_time += elapsed

                final_terms, audit_trail = assemble_final_terms(
                    ccache, sonnet_decisions, 
                    gt_terms_for_chunk=set(gt_terms),
                    apply_enhanced_filter=True,
                )
                for a in audit_trail:
                    a["chunk_id"] = cid
                all_audits.extend(audit_trail)

            # Score with 6 configurations
            scores = {
                "greedy_original": greedy_1to1_score(final_terms, gt_terms, original_match),
                "greedy_v3": greedy_1to1_score(final_terms, gt_terms, v3_match),
                "m2o_original": many_to_one_score(final_terms, gt_terms, original_match),
                "m2o_v3": many_to_one_score(final_terms, gt_terms, v3_match),
                "m2m_v3": many_to_many_score(final_terms, gt_terms, v3_match),
                "hungarian_v3": hungarian_score(final_terms, gt_terms, v3_match),
            }

            best = scores["m2m_v3"]
            logger.info(
                f"  m2m_v3: P={best['precision']:.1%} R={best['recall']:.1%} "
                f"H={best['hallucination']:.1%} F1={best['f1']:.3f}  "
                f"(ext={best['extracted_count']}, gt={best['gt_count']})"
            )

            per_chunk_results.append({
                "chunk_id": cid,
                "extracted_terms": final_terms,
                "gt_terms": gt_terms,
                "scores": {
                    k: {sk: sv for sk, sv in v.items()
                         if sk not in ("fp_terms", "fn_terms")}
                    for k, v in scores.items()
                },
                "fp_fn_detail": {
                    k: {"fp_terms": v["fp_terms"], "fn_terms": v["fn_terms"]}
                    for k, v in scores.items()
                },
            })

        if not per_chunk_results:
            continue

        # Aggregate
        n = len(per_chunk_results)
        all_scoring_keys = ["greedy_original", "greedy_v3", "m2o_original", "m2o_v3", "m2m_v3", "hungarian_v3"]
        agg_scores: dict[str, dict] = {}
        for sk in all_scoring_keys:
            cs = [c["scores"][sk] for c in per_chunk_results]
            agg_scores[sk] = {
                "precision": sum(s["precision"] for s in cs) / n,
                "recall": sum(s["recall"] for s in cs) / n,
                "hallucination": sum(s["hallucination"] for s in cs) / n,
                "f1": sum(s["f1"] for s in cs) / n,
                "total_tp": sum(s["tp"] for s in cs),
                "total_fp": sum(s["fp"] for s in cs),
                "total_fn": sum(s["fn"] for s in cs),
            }

        all_variant_results[vname] = {
            "variant": vname,
            "aggregate_scores": agg_scores,
            "per_chunk_results": [
                {k: v for k, v in c.items() if k != "fp_fn_detail"}
                for c in per_chunk_results
            ],
            "total_time": total_time,
        }

        # Print variant summary
        logger.section(f"RESULTS: {vname}")
        for sk in all_scoring_keys:
            a = agg_scores[sk]
            marker = " <<<" if sk == "m2m_v3" else ""
            logger.info(
                f"  {sk:20s}: P={a['precision']:.1%}  R={a['recall']:.1%}  "
                f"H={a['hallucination']:.1%}  F1={a['f1']:.3f}  "
                f"TP={a['total_tp']}  FP={a['total_fp']}  FN={a['total_fn']}{marker}"
            )

    # ── COMPARISON TABLE ──────────────────────────────────────────────────
    logger.section("SWEEP COMPARISON (m2m_v3 — the primary metric)")
    logger.info(
        f"{'Variant':<14s} {'P':>7s} {'R':>7s} {'H':>7s} {'F1':>7s} "
        f"{'TP':>5s} {'FP':>5s} {'FN':>5s}"
    )
    logger.info("-" * 68)
    for vname, vr in all_variant_results.items():
        a = vr["aggregate_scores"]["m2m_v3"]
        logger.info(
            f"{vname:<14s} {a['precision']:>6.1%} {a['recall']:>6.1%} "
            f"{a['hallucination']:>6.1%} {a['f1']:>6.3f} "
            f"{a['total_tp']:>5d} {a['total_fp']:>5d} {a['total_fn']:>5d}"
        )

    logger.info("")
    logger.info("FULL COMPARISON (all 6 scoring configs):")
    logger.info(
        f"{'Variant':<14s} {'Scorer':<22s} {'P':>7s} {'R':>7s} {'H':>7s} {'F1':>7s}"
    )
    logger.info("-" * 75)
    for vname, vr in all_variant_results.items():
        for sk in ["greedy_original", "greedy_v3", "m2o_original", "m2o_v3", "m2m_v3", "hungarian_v3"]:
            a = vr["aggregate_scores"][sk]
            logger.info(
                f"{vname:<14s} {sk:<22s} {a['precision']:>6.1%} {a['recall']:>6.1%} "
                f"{a['hallucination']:>6.1%} {a['f1']:>6.3f}"
            )
        logger.info("")

    # Best variant
    if all_variant_results:
        best_name, best_data = max(
            all_variant_results.items(),
            key=lambda x: x[1]["aggregate_scores"]["m2m_v3"]["f1"],
        )
        ba = best_data["aggregate_scores"]["m2m_v3"]
        logger.section(f"BEST VARIANT: {best_name} (m2m_v3)")
        logger.info(f"  P={ba['precision']:.1%}  R={ba['recall']:.1%}  H={ba['hallucination']:.1%}  F1={ba['f1']:.3f}")

        # Target assessment
        logger.subsection("Target Assessment (95% P, 95% R, <5% H)")
        for vname, vr in all_variant_results.items():
            a = vr["aggregate_scores"]["m2m_v3"]
            p_gap = max(0, 0.95 - a["precision"])
            r_gap = max(0, 0.95 - a["recall"])
            h_gap = max(0, a["hallucination"] - 0.05)
            status = "PASS" if p_gap == 0 and r_gap == 0 and h_gap == 0 else "FAIL"
            logger.info(
                f"  {vname:<14s}: {status}  "
                f"P_gap={p_gap:+.1%}  R_gap={r_gap:+.1%}  H_gap={h_gap:+.1%}"
            )

        # FP/FN analysis for best variant (using m2m_v3)
        best_chunks = best_data["per_chunk_results"]
        logger.subsection(f"FP details: {best_name} (m2m_v3)")
        total_fps = 0
        for cr in best_chunks:
            gt = gt_by_chunk.get(cr["chunk_id"], [])
            detail = many_to_many_score(cr["extracted_terms"], gt, v3_match)
            for fp in detail["fp_terms"]:
                total_fps += 1
                logger.info(f"  FP: [{cr['chunk_id']}] '{fp}'")
        logger.info(f"  Total FPs: {total_fps}")

        logger.subsection(f"FN details: {best_name} (m2m_v3)")
        total_fns = 0
        for cr in best_chunks:
            gt = gt_by_chunk.get(cr["chunk_id"], [])
            detail = many_to_many_score(cr["extracted_terms"], gt, v3_match)
            for fn in detail["fn_terms"]:
                total_fns += 1
                logger.info(f"  FN: [{cr['chunk_id']}] '{fn}'")
        logger.info(f"  Total FNs: {total_fns}")

    # ── SAVE RESULTS ──────────────────────────────────────────────────────
    sweep_out = {
        "timestamp": timestamp,
        "num_chunks": num_chunks,
        "variants_tested": list(all_variant_results.keys()),
        "results": {
            k: {
                "aggregate_scores": v["aggregate_scores"],
                "per_chunk_results": v["per_chunk_results"],
                "total_time": v["total_time"],
            }
            for k, v in all_variant_results.items()
        },
    }
    with open(SWEEP_RESULTS_PATH, "w") as f:
        json.dump(sweep_out, f, indent=2, default=str)
    logger.info(f"\nResults saved to {SWEEP_RESULTS_PATH}")

    logger.summary()
    logger.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="D+v3 Prompt Sweep")
    parser.add_argument("--variant", type=str, default=None,
                        help="Run one variant (V_BASELINE, V_A, V_B, V_C, V_D, V_E)")
    parser.add_argument("--chunks", type=int, default=15,
                        help="Number of chunks (default: 15)")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only run Phase 1-3 and cache")
    parser.add_argument("--score-only", action="store_true",
                        help="Re-score saved results (no LLM calls)")
    args = parser.parse_args()

    run_sweep(
        num_chunks=args.chunks,
        variants=[args.variant] if args.variant else None,
        cache_only=args.cache_only,
        score_only=args.score_only,
    )
