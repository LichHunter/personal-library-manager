# RAG Retrieval Benchmark

Benchmark for evaluating chunking and retrieval strategies for RAG systems.

## Quick Start

```bash
cd /home/fujin/Code/personal-library-manager
source .venv/bin/activate

cd poc/chunking_benchmark_v2
python run_benchmark.py --config config_realistic.yaml
```

## Latest Results (2026-01-24)

| Strategy | Original | Synonym | Problem | Casual | Contextual | Negation |
|----------|----------|---------|---------|--------|------------|----------|
| semantic | 67.9% | 60.4% | 54.7% | 54.7% | 62.3% | 54.7% |
| hybrid | **75.5%** | **69.8%** | 54.7% | 64.2% | 60.4% | 52.8% |
| hybrid_rerank | **75.5%** | 60.4% | **66.0%** | 64.2% | **69.8%** | **56.6%** |

Configuration: BGE-base embedder, 512-token chunks, MS-MARCO reranker.

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
│   ├── hyde.py
│   ├── multi_query.py
│   └── reverse_hyde.py
├── strategies/              # Chunking strategies
└── results/                 # Benchmark outputs
```

## Key Findings

1. **Hybrid retrieval (BM25 + semantic)** outperforms pure semantic: 75.5% vs 67.9%
2. **Reranking helps with difficult queries**: problem queries improved from 54.7% to 66.0%
3. **BGE embedder with prefix** performs well for this corpus
4. **512-token chunks** provide good balance of context and precision

## Configuration

Edit `config_realistic.yaml` to test different combinations:
- Embedding models
- Chunking strategies  
- Retrieval strategies
- Rerankers
