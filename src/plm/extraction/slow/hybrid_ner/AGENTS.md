# V6 HYBRID NER PIPELINE

11-module LLM-powered technical term extraction. Direct copy of POC-1c V6 strategy (91-93% P/R).

## OVERVIEW

5-stage pipeline: Extract (multi-source) → Ground (dedupe) → Filter (noise) → Validate (LLM) → Postprocess (spans).

## STRUCTURE

```
hybrid_ner/
├── pipeline.py      # extract_hybrid() and extract_hybrid_v5() orchestrators
├── config.py        # StrategyConfig dataclass (50+ knobs)
├── constants.py     # _ALLCAPS_EXCLUDE, structural patterns
├── prompts.py       # LLM prompt templates
├── extractors.py    # _extract_haiku_*, _extract_retrieval_*, _extract_heuristic
├── grounding.py     # _ground_and_dedup (case-insensitive merge)
├── noise_filter.py  # _auto_reject_noise, _auto_keep_structural, _run_sonnet_review
├── validation.py    # _run_context_validation, _run_term_retrieval_validation
├── postprocess.py   # _expand_spans, _suppress_subspans, _is_embedded_in_path
└── parsing.py       # LLM response parsing utilities
```

## WHERE TO LOOK

| Task | File | Function |
|------|------|----------|
| Add extraction source | `extractors.py` | Add `_extract_*` function |
| Modify confidence tiers | `pipeline.py:320-430` | HIGH/MEDIUM/LOW routing logic |
| Add noise rule | `noise_filter.py` | `_auto_reject_noise()` |
| Add structural keep | `noise_filter.py:13` | `_auto_keep_structural()` |
| Tune validation | `config.py` | `StrategyConfig` thresholds |
| Change prompts | `prompts.py` | Template strings |

## PIPELINE STAGES

```
1. EXTRACT — Multiple sources vote:
   - haiku_fewshot (retrieval-augmented)
   - haiku_taxonomy (category-based)
   - haiku_simple (zero-shot)
   - heuristic (regex: CamelCase, ALL_CAPS, dot.paths)
   - seeds (vocabulary match)

2. GROUND — Dedupe by normalized form, track source votes

3. FILTER — Reject noise:
   - Negatives list (generic words)
   - Length/character rules
   - _ALLCAPS_EXCLUDE set

4. VALIDATE — Route by confidence:
   - HIGH (3+ votes, structural, seed): Accept
   - MEDIUM (2 votes): LLM validation
   - LOW (1 vote, no support): Log and reject

5. POSTPROCESS — Clean up:
   - Expand partial spans to word boundaries
   - Suppress subspans (keep longest)
   - Remove path-embedded terms
```

## CRITICAL INVARIANTS

- **Stage order matters**: Extract → Ground → Filter → Validate → Postprocess
- **Seeds bypass noise filter**: Terms in `auto_vocab["seeds"]` skip rejection
- **Ground-truth terms NEVER filtered**: Test invariant, see `NEVER filter GT terms`
- **Structural patterns ALWAYS kept**: CamelCase, ALL_CAPS (with exceptions), dot.paths

## ANTI-PATTERNS

- **Don't skip grounding** — Dedup prevents double-counting votes
- **Don't filter seeds aggressively** — They're data-driven from training
- **Don't change _ALLCAPS_EXCLUDE** — TODO/FIXME/WARNING etc. are never entities
- **Don't use V1-V4 pipelines** — `extract_hybrid()` is legacy, use `extract_hybrid_v5()`
