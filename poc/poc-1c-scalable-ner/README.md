# POC-1c: Scalable NER Extraction Without Vocabulary Lists

## What

Test two scalable NER approaches that eliminate vocabulary dependence:
- **Approach A (Retrieval)**: Embed 741 train.txt documents, retrieve similar ones as few-shot examples
- **Approach B (SLIMER)**: Structured entity definitions + annotation guidelines (zero-shot)

## Why

POC-1b's vocabulary lists (176+ manually curated terms) don't scale beyond 100 docs. Each new document set introduces new FPs/FNs requiring vocabulary expansion.

## Dataset

StackOverflow NER (Tabassum et al., ACL 2020):
- train.txt: 741 docs (retrieval corpus for Approach A)
- test.txt: 249 docs (evaluation)
- dev.txt: 247 docs (unused)

## Setup

```bash
cd poc/poc-1c-scalable-ner
uv sync
source .venv/bin/activate
python parse_so_ner.py  # generates artifacts/{train,test}_documents.json
```

## Usage

```bash
# Quick test (10 docs)
python benchmark_comparison.py --approach slimer --n-docs 10
python benchmark_comparison.py --approach retrieval --n-docs 10

# Full benchmark (100 docs)
nohup python benchmark_comparison.py --approach retrieval --n-docs 100 > /tmp/poc1c_retrieval.log 2>&1 &
nohup python benchmark_comparison.py --approach slimer --n-docs 100 > /tmp/poc1c_slimer.log 2>&1 &

# Run both
python benchmark_comparison.py --approach all --n-docs 100
```

## Baseline (POC-1b iter 29, 100 docs)

| Metric | Value |
|--------|-------|
| Precision | 91.0% |
| Recall | 91.6% |
| Hallucination | 9.0% |
| F1 | 0.913 |

## Results

| Approach | Precision | Recall | Hallucination | F1 | Vocabulary |
|----------|-----------|--------|---------------|------|------------|
| **Baseline (POC-1b)** | 91.0% | 91.6% | 9.0% | 0.913 | 176+ terms |
| **Retrieval few-shot** | 81.6% | 80.6% | 18.4% | 0.811 | 0 terms |
| **SLIMER zero-shot** | 84.9% | 66.0% | 15.1% | 0.743 | 0 terms |

See [RESULTS.md](RESULTS.md) for detailed analysis.

## Conclusion

**Retrieval few-shot is the recommended approach.** It eliminates all vocabulary maintenance with a ~10% F1 drop on the SO NER benchmark. The real-world gap is likely smaller -- many "FPs" are useful tech terms excluded by the GT's conservative annotation scheme, and many "FNs" are common words ("list", "table", "row") that the model conservatively skips.

For our actual use case (knowledge graph for personal library manager), recall matters more than matching SO NER conventions. The FP rate on genuinely useful terms is estimated at 8-12%, not the reported 18%.

**Next step**: Validate on actual documentation corpus to confirm extracted terms are useful for search and knowledge graph construction.

## Files

| File | Purpose |
|------|---------|
| `parse_so_ner.py` | BIO parser -- generates artifacts/{train,test,dev}_documents.json |
| `scoring.py` | v3_match + many_to_many_score (ported from POC-1b) |
| `retrieval_ner.py` | Approach A: FAISS + few-shot prompting |
| `slimer_ner.py` | Approach B: entity definitions + annotation guidelines |
| `benchmark_comparison.py` | CLI runner for benchmarking |
| `utils/llm_provider.py` | Anthropic OAuth provider |
| `utils/logger.py` | Benchmark logger |
| `RESULTS.md` | Detailed analysis with FP/FN breakdowns |
