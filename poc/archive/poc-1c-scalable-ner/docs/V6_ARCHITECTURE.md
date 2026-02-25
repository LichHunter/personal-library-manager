# Strategy v6: Pipeline Architecture

## Overview

Strategy v6 is a 5-stage NER extraction pipeline for StackOverflow technical text. It extracts named software entities (libraries, classes, functions, languages, data types, UI elements, etc.) from short text chunks (200-500 words).

**Key design principle**: Maximize recall in extraction (Stage 1), then progressively filter noise through grounding, heuristic rules, and LLM validation.

**Benchmark**: StackOverflow NER (Tabassum et al., ACL 2020) — 741 train / 249 test / 247 dev documents, 20 entity types, human-annotated BIO tags.

## Pipeline Diagram

```
STAGE 1: EXTRACTION (maximize recall)
├── candidate_verify (Sonnet)     ─┐
│   regex candidates → Sonnet       │
│   classifies ENTITY/REJECT        │
│                                    │
├── Sonnet taxonomy (Sonnet)      ─┤── UNION → ~77 candidates/doc
│   generative 20-type extraction   │   100% recall, ~1:1 signal:noise
│                                    │
├── Heuristic extractor (free)    ─┤
│   CamelCase, dot.path, ALL_CAPS   │
│   function(), .css-class, /paths  │
│                                    │
└── Seed matching (free)          ─┘
    auto_vocab.json (445 terms)
                    │
                    ▼
STAGE 2: GROUNDING + DEDUP
    verify_span() → normalize → track sources
                    │
                    ▼
STAGE 3: NOISE FILTER (rule-based, zero LLM cost)
    stop words, gerunds, URLs, version fragments,
    negatives list, descriptive phrases, generic 3+ word phrases
                    │
                    ▼
STAGE 4: CONFIDENCE ROUTING + VALIDATION
    ├── HIGH confidence → auto-keep
    │   structural + LLM vote, seed + 2 sources,
    │   3+ sources, entity_ratio >= 0.95
    │
    ├── MEDIUM confidence → Sonnet validation
    │   ├── entity_ratio >= 0.90 → skip, auto-keep
    │   └── entity_ratio <  0.90 → Sonnet term-retrieval
    │       (entity_ratio + training snippets → ENTITY/GENERIC)
    │
    └── LOW confidence → dropped (logged)
        heuristic-only ALL_CAPS without corroboration,
        no LLM vote + no seed + low entity_ratio
                    │
                    ▼
STAGE 5: POST-PROCESSING
    span expansion, path-embedded suppression,
    URL filtering, subspan suppression, final dedup
```

## Stage 1: Extraction

Four extraction sources run in parallel, their outputs are unioned for maximum recall.

### 1.1 candidate_verify (1 Sonnet call)

**New in v6** — replaces v5.x's 3x Haiku extraction.

1. **Heuristic candidate generation** (zero cost): regex extracts ALL structurally plausible spans from text:
   - CamelCase identifiers (`ListView`, `NSMutableArray`)
   - lowerCamelCase (`getElementById`, `setChoiceMode`)
   - Dot-separated paths (`System.out`, `R.layout.file`)
   - Function calls (`recv()`, `querySelector()`)
   - ALL_CAPS acronyms (`JSON`, `HTML`, `CPU`)
   - Backtick-wrapped terms
   - CSS class selectors (`.long`, `.container`)
   - Brace-expansion paths (`/usr/bin/{erb,gem,irb}`)
   - Unix file paths (`/usr/bin/ruby`)

2. **Sonnet classification**: Each candidate is sent to Sonnet with the document context. Sonnet classifies each as ENTITY or REJECT.

### 1.2 Sonnet Taxonomy (1 Sonnet call)

Generative extraction using a taxonomy-driven prompt with 20 entity types and examples:

- Library/Framework, Library_Class, Library_Function, Language
- Data_Type, Data_Structure, Application, Operating_System
- Device, File_Type, UI_Element, HTML_Tag
- Error_Name, Version, Website/Org, Keyboard_Key

The prompt explicitly instructs: "Missing an entity is WORSE than including a borderline one."

### 1.3 Heuristic Extractor (zero cost)

Same regex patterns as candidate_verify's Step 1, but used as an independent vote source. Provides +1 source_count for structural patterns.

### 1.4 Seed Matching (zero cost)

Regex word-boundary matching of `auto_vocab.json` terms against document text:
- **bypass** list: high entity_ratio terms (>= 0.55) — auto-kept in Stage 4
- **seeds** list: entity_ratio >= 0.50 — used for seed bypass routing
- **contextual_seeds**: entity_ratio 0.20-0.50 — routed to validation

All 445 terms are data-driven from the 741 training documents.

### Extraction Result

Union of all 4 sources achieves **100% recall** on 10-doc test:
- 771 total candidates (~77/doc average)
- 395 true positives + 376 noise (1:1 signal:noise ratio)
- Only 2/376 noise terms are true hallucinations (parser artifacts)
- All other noise = real text spans that are over-extracted

## Stage 2: Grounding + Dedup

For each candidate from any source:

1. **verify_span()** — confirm the term literally exists in the source text
2. **normalize_term()** — lowercase key for deduplication
3. **Track sources** — which extractors found it (source_count)
4. **Prefer capitalized forms** — keep `React` over `react` for display

Output: `{normalized_key → {term, sources, source_count}}`

## Stage 3: Noise Filter

Rule-based rejection (zero LLM cost):

| Rule | Examples Rejected |
|------|-------------------|
| Empty / single-char | `""`, `"a"` |
| Pure numbers | `"42"`, `"100"` |
| Stop words | `"the"`, `"is"`, `"are"`, `"my"` |
| Action gerunds | `"serializing"`, `"loading"`, `"creating"` |
| URLs | `"https://..."` |
| Smart version filter | Orphan `"1.2"` without `"Python 1.2"` context |
| Negatives list | `tech_domain_negatives.json` (known non-entities) |
| JIRA-style IDs | `"ABC-123"` |
| Descriptive adj + noun | `"hidden fields"`, `"simple example"` |
| Generic 3+ word phrases | Phrases without code markers |

Auto-keep (bypass noise filter):

| Rule | Examples Kept |
|------|--------------|
| Contains code characters | `()._::<>[]` |
| camelCase | `getElementById` |
| ALL_CAPS (>= 2 chars) | `JSON`, `CPU` |
| Keyboard key patterns | `Left`, `PgUp`, `arrow right` |
| Bypass vocabulary | High entity_ratio terms from training |

## Stage 4: Confidence Routing + Validation

### 4a. Routing

Each surviving candidate is routed to one of three tiers:

**HIGH confidence** (auto-keep, no LLM validation):
- Structural pattern (code characters, camelCase, ALL_CAPS) + has at least 1 LLM vote
- Seed term + 2+ sources (or has technical context nearby)
- 3+ sources agree
- entity_ratio >= 0.95 in training data

**MEDIUM confidence** (sent to Sonnet validation):
- Contextual seed + any LLM vote
- 2 sources agree
- entity_ratio >= 0.50
- Single LLM vote with entity_ratio >= 0.0 (v5.2 routing: better to validate than drop)

**LOW confidence** (dropped, logged for analysis):
- Heuristic-only ALL_CAPS without LLM corroboration (and not in seeds/bypass, and entity_ratio < 0.95)
- Everything else that didn't qualify for HIGH or MEDIUM

### 4b. Validation (1 Sonnet call)

MEDIUM confidence terms are validated by Sonnet using **term-retrieval evidence**:

For each term, the prompt includes:
- The current document text
- entity_ratio from training (e.g., "entity in 15 train docs, generic in 3 train docs, ratio=83%")
- Up to 2 positive example snippets (where term WAS annotated as entity)
- Up to 2 negative example snippets (where term appeared but was NOT annotated)

Sonnet classifies each as ENTITY or GENERIC.

**Safety net**: Terms with entity_ratio >= 0.95 are kept even if Sonnet says GENERIC (training data overrides LLM judgment for very high confidence terms).

**Skip validation**: Terms with entity_ratio >= 0.90 skip the Sonnet call entirely and are auto-kept.

## Stage 5: Post-Processing

### 5.1 Span Expansion

Attempts to extend extracted spans to their natural boundary:
- Append parenthesized arguments: `recv` → `recv()`
- Prepend namespace prefix: `send()` → `NetStream.send()`
- Append version info: `Apache` → `Apache Cammel` (when followed by version context)
- Join slash-separated: `latex` → `latex/unicode`
- Extend to compound with `server`/`client` suffix

### 5.2 Path-Embedded Suppression

Remove terms that ONLY appear inside brace-expansion paths (e.g., `erb` from `/usr/bin/{erb,gem,irb}` if `erb` doesn't appear elsewhere in text).

### 5.3 URL Filtering

Remove any remaining terms containing URLs.

### 5.4 Subspan Suppression

Remove shorter terms that are substrings of longer extracted terms:
- `getInputSizes` suppressed by `getInputSizes(ImageFormat.YUV_420_888)`
- BUT: 1-word inside 2-word is preserved (`Right` + `arrow right` both kept)
- Protected seeds are never suppressed

### 5.5 Final Dedup

Deduplicate by normalized key. Output final term list.

## LLM Cost Per Document

| Call | Model | Purpose | ~Time |
|------|-------|---------|-------|
| 1 | Sonnet | candidate_verify — classify heuristic candidates | ~14s |
| 2 | Sonnet | Taxonomy — generative 20-type extraction | ~14s |
| 3 | Sonnet | Term-retrieval validation — MEDIUM confidence | ~14s |

**Total: 3 Sonnet calls per document, ~42s average.**

No Haiku calls (v5.x used 3x Haiku for extraction; v6 replaced with candidate_verify + taxonomy).

## Key Configuration (StrategyConfig)

```python
"strategy_v6": StrategyConfig(
    name="strategy_v6",
    use_candidate_verify=True,          # v6 extraction mode
    use_heuristic_extraction=True,      # regex structural patterns
    use_sonnet_taxonomy=True,           # Sonnet (not Haiku) for taxonomy
    smart_version_filter=True,          # filter orphan version numbers
    ratio_gated_review=True,            # entity_ratio gates LOW confidence
    ratio_auto_approve_threshold=0.70,  # LOW → auto-approve if ratio > 70%
    ratio_auto_reject_threshold=0.30,   # LOW → auto-reject if ratio < 30%
    suppress_path_embedded=True,        # remove brace-expansion-only terms
    high_entity_ratio_threshold=0.95,   # HIGH if ratio >= 95%
    medium_entity_ratio_threshold=0.5,  # MEDIUM if ratio >= 50%
    skip_validation_entity_ratio=0.90,  # skip Sonnet call if ratio >= 90%
    safety_net_ratio=0.95,              # override Sonnet REJECT if ratio >= 95%
    seed_bypass_to_high_confidence=True, # seed terms → HIGH (with context check)
    seed_bypass_require_context=True,
    seed_bypass_min_sources_for_auto=2,
    use_contextual_seeds=True,          # 0.20-0.50 ratio terms as extra seeds
    allcaps_require_corroboration=True, # ALL_CAPS needs LLM vote
    route_single_vote_to_validation=True, # single LLM vote → MEDIUM (not LOW)
    structural_require_llm_vote=True,   # structural patterns need LLM backup
)
```

## Data Dependencies

| File | Purpose | Size |
|------|---------|------|
| `artifacts/train_documents.json` | 741 SO training docs (retrieval corpus) | ~3MB |
| `artifacts/test_documents.json` | 249 SO test docs (evaluation) | ~1MB |
| `artifacts/auto_vocab.json` | Data-driven vocabulary: bypass, seeds, contextual_seeds, negatives | 445 terms |
| `artifacts/tech_domain_negatives.json` | Known non-entity technical words | ~6K entries |

## Evolution from v5.x

| Aspect | v5.x | v6 |
|--------|------|-----|
| Extraction 1 | Haiku few-shot (retrieval) | **candidate_verify** (heuristic → Sonnet) |
| Extraction 2 | Sonnet taxonomy | Sonnet taxonomy (same) |
| Extraction 3 | Haiku simple (specialist prompt) | *(removed — subsumed by candidate_verify)* |
| LLM calls | 3 Haiku + 1 Sonnet taxonomy + 1 Sonnet validation | **2 Sonnet + 1 Sonnet validation** |
| Recall | ~94% | **100%** (at extraction stage) |
| Rationale | Diverse LLM extractors vote | Heuristic catches ALL patterns, Sonnet judges quality |

## Implementation

- **Main pipeline**: `hybrid_ner.py` → `extract_hybrid_v5()` (with `use_candidate_verify=True`)
- **candidate_verify**: `benchmark_prompt_variants.py` → `run_prompt_variant("candidate_verify_v1", ...)`
- **Scoring**: `scoring.py` → `v3_match()` + `many_to_many_score()`
- **Vocab generation**: `generate_vocab.py` → computes entity_ratio from training data
- **Term index**: `hybrid_ner.py` → `build_term_index()` — builds training evidence per term
