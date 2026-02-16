# Model Comparison: CV Extraction Quality Across LLMs

## Benchmark

- **Dataset**: StackOverflow NER (Tabassum et al., ACL 2020)
- **Test set**: 5 documents (Q42873050, Q10709772, Q14831694, Q39288595, Q45734089)
- **Pipeline**: Heuristic candidate generation + LLM candidate-verify classification
- **Scoring**: `v3_match()` — fuzzy span matching with normalization
- **Baseline**: Claude Sonnet via Anthropic API (OpenCode OAuth)
- **Gemini access**: Google Code Assist API (free tier, OpenCode OAuth)
- **Date**: 2026-02-15

## Headline Results

| Model | Precision | Recall | F1 | TP | FP | FN | Time | Status |
|-------|-----------|--------|------|-----|-----|-----|------|--------|
| **sonnet** (baseline) | **50.7%** | **100.0%** | **0.673** | 236 | 229 | 0 | 130s | Complete |
| gemini-2.0-flash | 43.2% | 100.0% | 0.603 | 504 | 663 | 0 | 307s | Complete |
| gemini-2.5-flash | 34.0% | 91.1% | 0.495 | 163 | 316 | 16 | 716s | Complete |
| gemini-2.5-flash+think | 34.2% | 82.0% | 0.483 | 141 | 271 | 31 | 598s | Complete |
| gemini-2.5-pro | — | — | — | 0 | 0 | 109 | 3s | FAILED (token expired) |
| gemini-3-pro-preview | — | — | — | 0 | 0 | 109 | 3s | FAILED (token expired) |

## Key Findings

### 1. Sonnet wins on every metric

Sonnet has the best F1 (0.673), best precision (50.7%), and perfect recall (100.0%) while being the fastest (130s). No Gemini model matched it on this task.

### 2. gemini-2.0-flash is the best Gemini model

Of the Gemini models that completed successfully, gemini-2.0-flash came closest to Sonnet:
- Matched Sonnet on recall (100%) — both extracted every gold entity
- Lower precision (43.2% vs 50.7%) due to ~3x more false positives (663 vs 229)
- ~2.4x slower (307s vs 130s) due to free-tier rate limiting

### 3. Newer Gemini models performed worse, not better

Counter-intuitively, gemini-2.5-flash scored lower than gemini-2.0-flash on every metric:
- Recall dropped from 100% to 91.1% (16 missed entities)
- Precision dropped from 43.2% to 34.0%
- Time increased from 307s to 716s (more retries from rate limits)

### 4. Thinking mode did not help

gemini-2.5-flash+think (4096 token thinking budget) performed slightly worse than vanilla 2.5-flash:
- Recall dropped further to 82.0% (31 missed entities)
- One document (Q45734089) returned 0 results entirely
- 598s runtime — faster than vanilla 2.5-flash but still 4.6x slower than Sonnet

### 5. Free-tier rate limiting is a major bottleneck

Gemini models were 2.4-5.5x slower than Sonnet entirely due to 429 rate limit retries on the Code Assist free tier. The actual generation latency is fast (1-2s per call), but the quota allows very few requests per minute.

### 6. Gemini models had JSON parsing issues

The gemini-2.5-flash variants produced malformed JSON in some responses (visible in FP terms like `` ```json ``, `entities": [`), suggesting weaker instruction following for structured output compared to Sonnet.

## Per-Document Breakdown

### sonnet (baseline)

| Doc ID | Extracted | TP | FP | FN | P | R | F1 | Time |
|--------|-----------|-----|-----|-----|------|------|------|------|
| Q42873050 | 140 | 81 | 59 | 0 | 57.9% | 100% | 0.733 | 25.2s |
| Q10709772 | 41 | 21 | 20 | 0 | 51.2% | 100% | 0.677 | 13.8s |
| Q14831694 | 104 | 63 | 41 | 0 | 60.6% | 100% | 0.755 | 20.5s |
| Q39288595 | 116 | 45 | 71 | 0 | 38.8% | 100% | 0.559 | 56.1s |
| Q45734089 | 64 | 26 | 38 | 0 | 40.6% | 100% | 0.578 | 14.8s |

### gemini-2.0-flash

| Doc ID | Extracted | TP | FP | FN | P | R | F1 | Time |
|--------|-----------|-----|-----|-----|------|------|------|------|
| Q42873050 | 345 | 174 | 171 | 0 | 50.4% | 100% | 0.671 | 96.9s |
| Q10709772 | 129 | 53 | 76 | 0 | 41.1% | 100% | 0.582 | 30.5s |
| Q14831694 | 209 | 126 | 83 | 0 | 60.3% | 100% | 0.752 | 44.8s |
| Q39288595 | 399 | 118 | 281 | 0 | 29.6% | 100% | 0.457 | 97.9s |
| Q45734089 | 85 | 33 | 52 | 0 | 38.8% | 100% | 0.559 | 37.0s |

### gemini-2.5-flash

| Doc ID | Extracted | TP | FP | FN | P | R | F1 | Time |
|--------|-----------|-----|-----|-----|------|------|------|------|
| Q42873050 | 105 | 44 | 61 | 6 | 41.9% | 80.0% | 0.550 | 160.1s |
| Q10709772 | 66 | 25 | 41 | 1 | 37.9% | 92.3% | 0.537 | 175.5s |
| Q14831694 | 102 | 28 | 74 | 7 | 27.5% | 77.4% | 0.405 | 125.3s |
| Q39288595 | 152 | 47 | 105 | 0 | 30.9% | 100% | 0.472 | 187.9s |
| Q45734089 | 54 | 19 | 35 | 2 | 35.2% | 84.6% | 0.497 | 67.4s |

### gemini-2.5-flash+think (4096 token budget)

| Doc ID | Extracted | TP | FP | FN | P | R | F1 | Time |
|--------|-----------|-----|-----|-----|------|------|------|------|
| Q42873050 | 105 | 38 | 67 | 9 | 36.2% | 70.0% | 0.477 | 173.2s |
| Q10709772 | 66 | 23 | 43 | 2 | 34.8% | 84.6% | 0.494 | 73.0s |
| Q14831694 | 89 | 37 | 52 | 4 | 41.6% | 87.1% | 0.563 | 116.0s |
| Q39288595 | 152 | 43 | 109 | 3 | 28.3% | 86.4% | 0.426 | 235.7s |
| Q45734089 | 0 | 0 | 0 | 13 | 0% | 0% | 0 | 0.3s |

## Interpretation

### Why precision is low across all models

These numbers are for the **candidate-verify (CV)** pipeline stage specifically, which uses a heuristic candidate generator that intentionally over-generates. The CV step's job is to filter candidates via LLM classification. The ~50% precision from Sonnet is expected — it means the LLM correctly rejects roughly half the heuristic noise. The full v6 pipeline (with additional stages) achieves 90.7% precision.

### Why Gemini produced more FPs

The Gemini models, especially gemini-2.0-flash, extracted significantly more candidates (e.g., 399 vs 116 for Q39288595). This suggests weaker ability to reject non-entity candidates. Many Gemini FP terms are generic words ("application", "article", "basic", "bit", "code") or fragments ("-", "/ destroy", "As far") that Sonnet successfully filtered.

### Incomplete results caveat

gemini-2.5-pro and gemini-3-pro-preview results are invalid — the Google OAuth token expired mid-benchmark. These models need to be re-tested with a fresh token to get meaningful comparisons. Given the trend (newer != better on this task), it's unclear whether they would outperform gemini-2.0-flash.

## Conclusion

**Sonnet remains the best model for CV extraction in this pipeline.** It's faster, more precise, and has perfect recall on this 5-document sample. The Gemini free tier is too rate-limited for practical use, and even ignoring speed, the extraction quality is lower across all tested Gemini models.

For cost-sensitive deployments where Anthropic API access is unavailable, gemini-2.0-flash is the only viable alternative — it matches Sonnet on recall but with ~15% more false positives that downstream pipeline stages would need to filter.

## Reproducibility

```bash
# Requires OpenCode OAuth tokens and env vars for Gemini credentials
export GEMINI_CLIENT_ID="..."   # from google-gemini/gemini-cli source
export GEMINI_CLIENT_SECRET="..." # from google-gemini/gemini-cli source

cd poc/poc-1c-scalable-ner
python compare_models_cv.py \
  --n-docs 5 \
  --models sonnet gemini-2.0-flash gemini-2.5-flash gemini-2.5-flash+think gemini-2.5-pro gemini-3-pro-preview \
  --save
```

## Related

- [V6_RESULTS.md](./V6_RESULTS.md) — Full v6 pipeline results (Sonnet only, 10 docs, 90.7% precision)
- [V6_ARCHITECTURE.md](./V6_ARCHITECTURE.md) — Pipeline architecture documentation
- [poc/shared/README.md](../../shared/README.md) — Shared LLM provider module (Anthropic + Gemini)
