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

Configuration: BGE-base embedder, 512-token chunks, Claude Haiku for query rewriting.

**Best Strategy**: `enriched_hybrid_llm` - 88.7% coverage with LLM query rewriting (suitable for batch/offline retrieval)

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

## Recommended Strategy

For **best coverage** (88.7%): Use `enriched_hybrid_llm`
```bash
python run_benchmark.py --strategies enriched_hybrid_llm
```

For **low latency** (83.0%): Use `enriched_hybrid_fast`
```bash
python run_benchmark.py --strategies enriched_hybrid_fast
```

## Configuration

Edit `config_realistic.yaml` to test different combinations:
- Embedding models
- Chunking strategies  
- Retrieval strategies
- Rerankers
