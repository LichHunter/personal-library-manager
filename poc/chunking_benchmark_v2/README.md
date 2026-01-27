# RAG Retrieval Benchmark

Benchmark for evaluating chunking and retrieval strategies for RAG systems.

## Current Status (2026-01-27)

**Recommended Strategy**: `enriched_hybrid_llm`

| Test Type | Pass Rate | Avg Score | Notes |
|-----------|-----------|-----------|-------|
| Baseline (human-style queries) | 90% (18/20) | 8.45/10 | Typical documentation questions |
| Adversarial (hard queries) | 65% (13/20) | 7.55/10 | Version lookups, vocab mismatches |
| **Realistic questions** | **40.75%** (163/400) | N/A | Natural user queries from kubefix |

**Strategy Components**:
- BM25 sparse retrieval
- BGE-base-en-v1.5 semantic embeddings (768-dim)
- YAKE + spaCy keyword enrichment
- Claude Haiku query rewriting (5s timeout)
- RRF fusion (k=60)
- MarkdownSemanticStrategy chunking (target=400, min=50, max=800 tokens)

## Quick Start

```bash
cd /home/fujin/Code/personal-library-manager
source .venv/bin/activate

cd poc/chunking_benchmark_v2

# Run needle-haystack benchmark
python benchmark_needle_haystack.py --questions corpus/needle_questions.json --run-benchmark
```

## Project Structure

```
├── benchmark_needle_haystack.py  # Main benchmark script
├── corpus/
│   ├── kubernetes_sample_200/    # 200 K8s docs test corpus
│   ├── needle_questions.json     # 20 baseline questions
│   ├── needle_questions_adversarial.json  # 20 adversarial questions
│   ├── ALL_TEST_QUESTIONS.md     # All questions documented
│   └── edge_case_queries.json    # 15 hard edge case queries
├── results/
│   ├── needle_haystack_report.md           # Baseline test report (90%)
│   ├── needle_haystack_adversarial_report.md  # Adversarial test report (65%)
│   └── *.json                    # Raw retrieval results
├── retrieval/                    # Retrieval strategy implementations
├── strategies/                   # Chunking strategy implementations
├── enrichment/                   # Enrichment modules (YAKE, spaCy)
└── archive/                      # Old configs, docs, and results
```

## Key Results

### Baseline Needle-in-Haystack Test (90% pass)
- 20 human-style questions about Kubernetes Topology Manager
- 200 K8s documentation files (haystack)
- `enriched_hybrid_llm` finds correct chunks 90% of the time

### Adversarial Needle-in-Haystack Test (65% pass)
- 20 intentionally difficult questions targeting known weaknesses
- Categories: VERSION (40%), COMPARISON (100%), NEGATION (60%), VOCABULARY (60%)
- 65% is within expected range (50-70%) for adversarial questions

### Category Performance

| Category | Pass Rate | Key Finding |
|----------|-----------|-------------|
| VERSION | 40% | Frontmatter metadata is adversarial |
| COMPARISON | 100% | Semantic understanding excels |
| NEGATION | 60% | Mixed results on constraint patterns |
| VOCABULARY | 60% | Synonym matching partially works |

### Realistic Questions Benchmark (2026-01-27)

**Purpose**: Measure retrieval performance on natural user queries that don't use exact documentation terminology.

| Metric | Value |
|--------|-------|
| Corpus | 1,569 docs → 7,269 chunks |
| Questions | 200 (400 queries with 2 variants) |
| Hit@1 | 20.25% |
| Hit@5 | 40.75% |
| MRR | 0.274 |

**Key Finding**: 100% of failures are **VOCABULARY_MISMATCH** - realistic user questions use completely different terminology than documentation. This is the hardest benchmark, confirming vocabulary gap is the primary retrieval challenge.

**Methodology**:
1. Sourced 200 questions from kubefix dataset (K8s Q&A with ground truth docs)
2. Transformed bot-like questions to realistic user queries using Claude Haiku
3. Generated 2 variants per question (q1, q2) for robustness
4. Evaluated against full 1,569-doc Kubernetes corpus

**Results**: `results/realistic_2026-01-27_110955/`

## Known Weaknesses

1. **Vocabulary mismatch** (CRITICAL): Users describe problems with natural language; docs use technical terms. Realistic benchmark shows 59% failure rate due to this.
2. **Frontmatter metadata**: Version numbers in YAML frontmatter not captured by semantic embeddings
3. **Extreme vocabulary mismatches**: Query terms too different from document terms
4. **Chunking boundaries**: Related info sometimes split across chunks

## Recommendations for Improvement

1. **Extract frontmatter metadata** during chunking → VERSION 40% → 80%
2. **Expand query rewriting vocabulary** with domain synonyms → VOCABULARY 60% → 75%
3. **Improve negation pattern handling** → NEGATION 60% → 80%
4. **Optimize chunk overlap** to reduce boundary issues

## Archive

Old configurations, documentation, and results are preserved in `archive/`:
- `FINAL_RESULTS/` - Strategy comparison results
- `old_configs/` - YAML configuration files
- `old_docs/` - Development documentation
- `old_results/` - Intermediate test results

See `archive/README.md` for details.

## Invalid Strategies (Not Recommended)

The following strategies were tested but did not improve over `enriched_hybrid_llm`:
- `adaptive_hybrid` - Adaptive weighting (91% baseline, 59% edge cases)
- `synthetic_variants` - Query variants (93% baseline, 57% edge cases)
- `negation_aware`, `contextual`, `bm25f_hybrid` - No significant improvement

**Stick with `enriched_hybrid_llm` for all use cases.**
