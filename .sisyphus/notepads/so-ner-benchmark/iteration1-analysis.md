# SO NER Benchmark: Iteration 1 Analysis

> Date: 2026-02-09
> Strategy: baseline (K8s prompts as-is)

## Results

| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| Precision | 64.4% | 95% | -30.6% |
| Recall | 91.9% | 95% | -3.1% |
| Hallucination | 35.6% | ≤5% | +30.6% |
| F1 | 0.757 | — | — |

## Root Cause: Task Mismatch

Our pipeline does **keyphrase extraction** (all technical vocabulary).
SO NER does **named entity recognition** (specific named software entities).

These overlap ~60-70% but aren't the same task.

## FP Categorization (88 total)

| Category | Count | % | Examples | Fixable? |
|----------|-------|---|----------|----------|
| Generic programming concepts | ~30 | 34% | function, arguments, element, loop, data | YES - filter/prompt |
| Descriptive multi-word phrases | ~25 | 28% | "download progress events", "loading on demand" | YES - prompt |
| Valid terms GT missed | ~15 | 17% | boost, pom.xml, SOAP, compositewpf | NO - GT quality |
| Partial/variant terms | ~10 | 11% | getInputSizes, error 500 | PARTIAL - matching |
| Truly wrong extractions | ~8 | 9% | Privacy settings, ergonomics, Staff | YES - filter |

**Fixable FPs**: ~63/88 (72%)
**GT quality FPs**: ~15/88 (17%) — not our fault
**Adjusted true noise**: ~73/247 = 29.6% (excluding GT gaps: ~58/247 = 23.5%)

## FN Analysis (15 total)

| Type | Count | Examples | Fix |
|------|-------|----------|-----|
| Common words as entities | 6 | table, page, screen, strings, image, tables | Context-awareness needed |
| Short/unusual terms | 4 | AS3, .long, phone, row | More aggressive extraction |
| Compound special chars | 3 | jQuery-generated, MAX SIZE(4608x3456), MAX Size | Better tokenization |
| Very generic | 2 | long, MAXIMUM | Hard to fix without context |

## Dual Oracle Synthesis

### Oracle 1 (Precision Focus):
- Switch prompt from "keyphrase" to "NER" mode
- Expand noise filter with generic programming stop words
- Add multi-word phrase filter (reject non-CamelCase, non-code phrases)
- Ceiling estimate: ~88-91% precision (Cat 3 limits us)

### Oracle 2 (Strategic):
- The benchmark is measuring the wrong thing for our use case
- Our "FPs" are mostly correct behavior for RAG
- Report adjusted metrics (honest framing)
- Focus on the FN gap (common words as entities) as genuine improvement
- Don't narrow extraction to hurt K8s use case

### Synthesis:
- Create NER-adapted strategy with narrower extraction
- Report BOTH raw and adjusted metrics
- The NER-adapted prompt focuses on proper nouns and specific named entities
- Keep broader extraction as separate strategy for RAG use

## Next Step: Iteration 2

Strategy: `v2_adapted` — software-domain adapted prompts + expanded noise filter
Expected improvement: P 64% → ~78-85%, R 92% → ~88-92%, H 36% → ~15-22%
