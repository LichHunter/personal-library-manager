# POC-1c Research Notes: Scalable NER Extraction Pipeline

## Objective
Extract named software entities from short text chunks (200-500 words) with P >= 90%, R >= 95%.
Benchmark: StackOverflow NER dataset (Tabassum et al., ACL 2020), 20 entity types, BIO-tagged.

## Architecture
Two-stage pipeline: Extract (noisy, max recall) -> Validate (filter noise, max precision).

### Stage 1: Extraction (two diverse extractors, union outputs)
1. **candidate_verify_v1** (Sonnet): Heuristic candidate generation (regex: camelCase, dot-notation, ALL_CAPS, code patterns) -> Sonnet classifies each ENTITY/REJECT
2. **taxonomy_v0** (Sonnet): Generative extraction — Sonnet reads full text, outputs all entities by 20 types

Union of both -> 771 candidates from 10 docs (avg 77/doc).
- **100% recall** (all 208 GT terms captured)
- Signal:noise = 395:376 (~1:1)
- Only 2/376 noise terms are true hallucinations (parser artifacts)
- All other noise = real text spans that are over-extracted

### Stage 2: Validation (the problem area)
Sonnet reads document + candidate terms + training evidence -> ENTITY or REJECT per term.

## Results Summary

| Approach | P | R | F1 | Notes |
|---|---|---|---|---|
| strategy_v6 (full pipeline) | 90.7% | 95.8% | 0.932 | Best. Uses auto_vocab seeds + entity_ratio routing |
| Type-assignment Sonnet | 85.0% | 87.5% | 0.863 | Too conservative — type-assignment kills recall |
| Thinking Sonnet (inclusion-biased) | 71.8% | 95.7% | 0.821 | Too permissive — 78 FPs |

## Data Artifacts
- `artifacts/train_documents.json` — 741 training docs (CLEAN, Answer_to_Question_ID bug fixed)
- `artifacts/test_documents.json` — 249 test docs (CLEAN)
- `artifacts/dev_documents.json` — 247 dev docs (CLEAN)
- `artifacts/results/extraction_raw_10docs.json` — Raw extraction fixture (10 docs, seed=42, 100% recall, 771 terms)
- `artifacts/results/validator_test_results.json` — Thinking Sonnet results (P=71.8%, R=95.7%)

## Bug Fixes
- **Answer_to_Question_ID parser bug**: `parse_so_ner.py` was letting `Answer_to_Question_ID` structural markers fall into document text. Fixed by adding handler (sets metadata_key="answer_id", skips colon and number). All artifacts regenerated.

## Pre-filter Rules (Implemented)
1. **Determiner-strip**: Remove "the X", "a X" etc. ONLY when bare form "X" already exists in union -> 41 terms removed, 0 recall loss
2. **Number-percent**: Remove `^\d+\.?\d*\s+%$` -> 0 recall loss

## Pre-filter Rules (Rejected with evidence)
- Bare numbers -> DANGEROUS, versions like "2.2", "1.1" are GT entities
- Extended adjective+noun -> "max size", "MAX Size" are GT entities
- Trailing suffixes -> "MAX Size" is GT, too risky

## FP Analysis (78 FPs from thinking Sonnet run)

### By entity_ratio from training data:
- **>= 50% (10)**: boost(100%), Weblogic(100%), command line(100%), 64-bit(100%), xml(95%), bin(67%), gem(67%), lib(67%), server(57%), microsoft(54%) — GT annotation misses, Sonnet arguably correct
- **20-50% (8)**: height(44%), config(40%), width(36%), Preview(33%), aspx(33%), endpoint(25%), pen(25%), controls(25%) — ambiguous
- **0-20% (19)**: main(12%), layout(11%), service(9%), resources(8%), handler(8%), Date(8%), model(3%), name(3%), value(2%), error(1%), configuration(0%), line(0%), NULL(0%), std(0%), SOAP(0%), notebook(0%), selector(0%), tall(0%) — likely true FPs
- **UNSEEN (41)**: ZSL, LEVEL-3, fopen, std::fopen, MSVC, ThreadId, irb, rdoc, ri, testrb, Ruby.framework, DSL, classpath, web-service, seed, seeds, latex, cascade, specificity, siblings, h, m, etc. — no training signal

### Key insight
~10-15 FPs are GT annotation misses (Sonnet is right, GT is wrong). ~15-20 are genuinely ambiguous. ~20 are true FPs from unseen terms. Real precision is likely 80-85%.

## Research: Support Systems Investigated

### Ruled Out — Vocabulary-Based
| Tool | Why Rejected |
|---|---|
| StackOverflow tag vocabulary (60K) | Vocabulary-based. Doesn't generalize to new domains. |
| Entity_ratio hard thresholds | Vocabulary-based (training data). No signal for 41/78 unseen FPs. |
| VerifiNER (ACL 2024) | Post-hoc KB lookup verification. Vocabulary with extra steps. No KB for general software terms. |

### Ruled Out — Statistical (Text Too Short)
| Tool | Why Rejected |
|---|---|
| YAKE / KeyBERT | Statistical keyword extraction. Needs frequency/corpus. Useless on 200-500 word chunks. |
| C-value / NC-value / Weirdness (PyATE) | Statistical ATE. Needs corpus frequency. Dead on single short docs. |
| TextRank / PositionRank / graph methods | Keyphrase extraction (top-K keywords). Wrong task. Needs more text. |
| EntropyRank / PromptRank | Information-theoretic keyphrase extraction. Wrong task + needs local LM. |
| Lexical specificity (hypergeometric) | Needs corpus statistics. Dead on single short docs. |
| IBM subword tokenization signal | Catches camelCase/compounds but doesn't help with hard FPs (height, config, service) which tokenize to 1 subword. |

### Ruled Out — Too Expensive
| Tool | Why Rejected |
|---|---|
| Chain-of-Verification (CoVe) | 3-4x LLM cost. Already 15 min for 10 docs. |
| Multi-agent debate | 7-10x cost per term. Impractical for 70+ terms/doc. |

### Ruled Out — Too Weak
| Tool | Why Rejected |
|---|---|
| Haiku as second judge | Too weak to make nuanced entity/non-entity distinctions. |

### Partially Interesting But Insufficient
| Tool | Notes |
|---|---|
| C-ICL (contrastive ICL, EMNLP 2024) | Shows negative examples in prompt. Still prompt engineering; ceiling hit. |
| "Don't Trust LLM Alone" (pipeline structure) | F1=0.51->0.93 via structural validation. Applicable but our pipeline already structured. |

## Prompt Engineering Approaches Tested
- **Inclusion-biased**: Default ENTITY, reject only with evidence. R=95.7%, P=71.8%.
- **Type-assignment**: Classify into 20 types or NONE. P=85%, R=87.5%.
- **Training evidence**: entity_ratio + positive snippets per term. Helps marginally.
- **Calibration rules**: Common-word-entity rule, concept family rule. Marginal.
- **Extended thinking**: thinking_budget=5000. Better reasoning, same fundamental limits.

## Fundamental Constraints Identified

1. **Single-pass LLM judgment at ceiling.** Sonnet maxed out for binary entity/non-entity on short text.
2. **No vocabulary allowed.** Production processes arbitrary user notes — no curated term lists.
3. **Statistical methods useless.** 200-500 word chunks. No frequency signal.
4. **Can't bootstrap from own output.** Even best run has FPs mixed in.
5. **GT is inconsistent.** Same term annotated differently across docs. Theoretical ceiling ~F1=0.95.
6. **No cross-document signal.** User notes are unrelated. No recurring terms.

## The Core Problem (Unsolved)

Given a single short text chunk (~200-500 words) and a candidate term that definitely appears in it, make a binary decision: is this a named software entity or not?

One shot. No corpus. No vocabulary. No recurring signal. Context only.

Sonnet gets 87% of these right. The remaining 13% are genuinely hard cases where context alone is ambiguous (height in CSS = entity?, server in web setup = entity?, cascade in CSS = entity?).

## Status: BLOCKED — Need new approach beyond prompt engineering and vocabulary-based filtering.
