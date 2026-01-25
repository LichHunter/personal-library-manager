# RAG Retrieval Benchmark

Benchmark for evaluating chunking and retrieval strategies for RAG systems.

## Quick Start

```bash
cd /home/fujin/Code/personal-library-manager
source .venv/bin/activate

cd poc/chunking_benchmark_v2
python run_benchmark.py --config config_realistic.yaml
```

## Latest Results (2026-01-25)

| Strategy | Original | Synonym | Problem | Casual | Contextual | Negation | Latency |
|----------|----------|---------|---------|--------|------------|----------|---------|
| semantic | 67.9% | 60.4% | 54.7% | 54.7% | 62.3% | 54.7% | ~10ms |
| hybrid | 75.5% | 69.8% | 54.7% | 64.2% | 60.4% | 52.8% | ~15ms |
| hybrid_rerank | 75.5% | 60.4% | 66.0% | 64.2% | 69.8% | 56.6% | ~50ms |
| enriched_hybrid_fast | 83.0% | 66.0% | 64.2% | 71.7% | 69.8% | 52.8% | ~15ms |
| **enriched_hybrid_llm** | **88.7%** | **73.6%** | **79.2%** | **79.2%** | **75.5%** | **77.4%** | **~960ms** |
| bmx_pure | 56.6% | 41.5% | 56.6% | 50.9% | 45.3% | 45.3% | ~20ms |
| bmx_semantic | 56.6% | 41.5% | 56.6% | 50.9% | 45.3% | 45.3% | ~15ms |
| bmx_wqa | 56.6% | 41.5% | 60.4% | 47.2% | 45.3% | 45.3% | ~1403ms |

Configuration: BGE-base embedder, 512-token chunks, Claude Haiku for query rewriting.

**Best Strategy**: `enriched_hybrid_llm` - 88.7% coverage with LLM query rewriting (suitable for batch/offline retrieval)

**Note**: BMX strategies (entropy-weighted BM25 variant) underperform standard BM25 by 26.4% on this corpus. See BMX Investigation section below.

## Structure

```
├── config_realistic.yaml    # Main config
├── run_benchmark.py         # Benchmark runner
├── corpus/
│   ├── realistic_documents/ # 5 CloudFlow docs (~3400 words each)
│   ├── corpus_metadata_realistic.json
│   └── ground_truth_realistic.json  # 20 queries, 53 facts
├── retrieval/               # Retrieval strategies
│   ├── semantic.py
│   ├── hybrid.py
│   ├── hybrid_rerank.py
│   ├── enriched_hybrid.py      # BM25 + semantic + enrichment
│   ├── enriched_hybrid_llm.py  # + LLM query rewriting (best: 88.7%)
│   ├── enriched_hybrid_bmx.py  # BMX variant (not recommended)
│   ├── query_rewrite.py        # Claude Haiku query rewriting
│   ├── hyde.py
│   ├── multi_query.py
│   └── reverse_hyde.py
├── strategies/              # Chunking strategies
└── results/                 # Benchmark outputs
```

## Key Findings

1. **LLM query rewriting is highly effective**: 88.7% coverage (best result)
2. **Hybrid retrieval (BM25 + semantic)** outperforms pure semantic: 75.5% vs 67.9%
3. **Enrichment improves coverage**: 83.0% vs 75.5% baseline
4. **Reranking helps with difficult queries**: problem queries improved from 54.7% to 66.0%
5. **BGE embedder with prefix** performs well for this corpus
6. **512-token chunks** provide good balance of context and precision
7. **BMX (entropy-weighted BM25) underperforms standard BM25**: 56.6% vs 83.0% - not recommended for this corpus size

## Recommended Strategy

For **best coverage** (88.7%): Use `enriched_hybrid_llm`
```bash
python run_benchmark.py --strategies enriched_hybrid_llm
```

For **low latency** (83.0%): Use `enriched_hybrid_fast`
```bash
python run_benchmark.py --strategies enriched_hybrid_fast
```

## BMX Investigation (2026-01-25)

### Hypothesis
BMX (entropy-weighted BM25 from mixedbread.ai) might outperform standard BM25 for sparse retrieval, especially with Weighted Query Augmentation (WQA).

### Tested Strategies
1. **bmx_pure**: Raw BMX + semantic (no query expansion)
2. **bmx_wqa**: BMX + LLM query rewriting via `search_weighted()` 
3. **bmx_semantic**: Clean BMX + clean semantic (RRF fusion)

### Results
All BMX variants significantly underperformed BM25:
- **enriched_hybrid (BM25)**: 83.0% coverage, 14ms latency
- **bmx_pure**: 56.6% coverage, 20ms latency (-26.4%)
- **bmx_semantic**: 56.6% coverage, 15ms latency (-26.4%)
- **bmx_wqa**: 56.6% coverage, 1403ms latency (-26.4%, 100x slower)

### Key Insights
1. **Query expansion > sparse index choice**: enriched_hybrid's success comes from chunk enrichment (YAKE+spaCy keywords), not the sparse index algorithm
2. **BMX unsuitable for small corpora**: With only 51 chunks, BMX's entropy-weighting doesn't provide advantages over BM25
3. **WQA has severe latency penalty**: 1.4 seconds per query makes it impractical for real-time retrieval
4. **Root cause confirmed**: Previous BMX failure (41.5%) was due to query expansion conflict. Even without expansion, BMX underperforms BM25.

### Recommendation
**Use `enriched_hybrid` (BM25 + semantic + enrichment) for production**. BMX does not provide benefits for this corpus size and use case.

## Configuration

Edit `config_realistic.yaml` to test different combinations:
- Embedding models
- Chunking strategies  
- Retrieval strategies
- Rerankers
