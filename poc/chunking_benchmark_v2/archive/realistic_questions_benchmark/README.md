# Realistic Questions Benchmark Archive

**Date**: 2026-01-27
**Status**: Complete

## Overview

This benchmark tests retrieval performance on natural user queries that don't use exact documentation terminology. Questions were sourced from the kubefix dataset and transformed into realistic problem descriptions using Claude Haiku.

## Results Summary

| Metric | Value |
|--------|-------|
| Corpus | 1,569 documents → 7,269 chunks |
| Questions | 200 (400 queries with 2 variants) |
| Hit@1 | 20.25% |
| Hit@5 | 40.75% |
| MRR | 0.274 |
| Q1 Hit@5 | 41.5% |
| Q2 Hit@5 | 40.0% |

## Key Finding

**100% of failures are VOCABULARY_MISMATCH**

Realistic user questions use completely different terminology than documentation:
- Users describe symptoms: "my pods keep crashing"
- Docs use technical terms: "container restart policy"

This validates the benchmark's purpose and confirms vocabulary gap is the primary retrieval challenge.

## Files

- `realistic_questions.json` - 200 transformed questions with ground truth
- `realistic_2026-01-27_110955/retrieval_results.json` - Raw benchmark results
- `realistic_2026-01-27_110955/benchmark_report.md` - Failure analysis report

## Methodology

1. **Source**: kubefix HuggingFace dataset (2,563 Q&A pairs with ground truth docs)
2. **Filtering**: 2,502 questions with matching docs in our corpus (97.6%)
3. **Transformation**: Claude Haiku converts bot-like questions to realistic user queries
4. **Quality**: 95% pass rate on transformation quality (19/20 test samples)
5. **Generation**: 200 questions sampled with seed=42, 99% high quality (198/200)
6. **Retrieval**: enriched_hybrid_llm strategy with BGE embedder
7. **Evaluation**: Check if ground truth doc appears in top-5 retrieved chunks

## Comparison with Other Benchmarks

| Benchmark | Hit@5 | Notes |
|-----------|-------|-------|
| Baseline (needle-haystack) | 90% | Human-style documentation queries |
| Adversarial | 65% | Intentionally difficult queries |
| **Realistic questions** | **40.75%** | Natural user problem descriptions |

The realistic benchmark is significantly harder, confirming that bridging the vocabulary gap between user language and documentation is the key challenge.

## Recommendations

1. **Query expansion**: Add synonyms and related terms to queries
2. **Document enrichment**: Extract problem patterns from docs
3. **Fine-tuned embeddings**: Train on user-query to doc-term pairs
4. **Hybrid search tuning**: Increase BM25 weight for exact term matching
