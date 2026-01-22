# Retrieval Benchmark POC

## What

A comprehensive benchmarking framework to compare different document retrieval strategies for RAG systems. Tests multiple combinations of:

- **Retrieval strategies**: Flat embedding, RAPTOR, LOD with embedding-only search, LOD with LLM-guided navigation
- **Vector backends**: In-memory, ChromaDB, sqlite-vec
- **Embedding models**: Various SentenceTransformer models
- **LLMs**: Various Ollama models (llama, mistral, etc.)

## Why

We need to make informed decisions about our retrieval architecture:

1. **Which retrieval strategy** gives the best accuracy for our use case?
2. **Is the added complexity worth it?** (RAPTOR vs flat search)
3. **What are the performance tradeoffs?** (accuracy vs speed vs cost)
4. **Which vector backend** should we use in production?

This framework provides standardized, comparable benchmarks to answer these questions with data.

## Hypothesis

Based on our RAPTOR research:

1. **Flat embedding** will be fastest but least accurate for broad queries
2. **RAPTOR** will have best accuracy but prohibitive indexing time
3. **LOD with embedding-only** will be a good balance (fast, reasonably accurate)
4. **LOD with LLM-guided** will be most accurate but slow at query time

## Documentation

- [Architecture](./docs/ARCHITECTURE.md) - System design and component interactions
- [Decision Log](./docs/DECISIONS.md) - Record of design decisions and rationale

## Setup

```bash
cd /home/fujin/Code/personal-library-manager
direnv allow  # or: nix develop

cd poc/retrieval_benchmark
uv sync
source .venv/bin/activate

# Ensure Ollama is running (for LLM-based strategies)
ollama serve
ollama pull llama3.2:3b
```

## Usage

### Run a benchmark experiment

```bash
# Quick test (5 docs, 1 run per query)
python -m retrieval_benchmark run --config config/experiments/quick_test.yaml

# Baseline comparison (10 docs, 3 runs)
python -m retrieval_benchmark run --config config/experiments/baseline.yaml

# Full benchmark (all docs, 5 runs)
python -m retrieval_benchmark run --config config/experiments/full.yaml
```

### Run specific strategy only

```bash
python -m retrieval_benchmark run \
  --config config/experiments/baseline.yaml \
  --strategy flat \
  --backend memory
```

### Generate ground truth (if not exists)

```bash
python -m retrieval_benchmark prepare \
  --documents ../test_data/output/documents \
  --output ../test_data/output/ground_truth.json \
  --questions-per-doc 3
```

## Output

Results are saved to `results/{experiment_id}/`:

| File | Description |
|------|-------------|
| `config.yaml` | Copy of configuration used |
| `index_stats.csv` | Indexing metrics per strategy |
| `search_results.csv` | Per-query results (all runs) |
| `summary.csv` | Aggregated metrics for comparison |
| `run.log` | Execution log |

### Key Metrics

| Metric | Description |
|--------|-------------|
| `doc_recall_at_k` | % of queries where correct document in top-k |
| `section_recall_at_k` | % of queries where correct section in top-k |
| `doc_mrr` | Mean Reciprocal Rank for document retrieval |
| `section_mrr` | Mean Reciprocal Rank for section retrieval |
| `avg_search_time_ms` | Average query latency |
| `consistency_score` | % of queries with identical results across runs |
| `index_time_sec` | Time to build index |
| `total_llm_calls` | Number of LLM invocations |

## Strategies

### 1. Flat Embedding (Baseline)

```
Index:  Document → Chunks → Embed all → Store
Search: Query → Embed → Top-K similarity → Return
```

Simple and fast. Baseline for comparison.

### 2. RAPTOR

```
Index:  Chunks → Embed → Cluster → Summarize → Repeat (build tree)
Search: Collapsed tree search OR tree traversal
```

Full hierarchical tree with emergent structure from clustering.

### 3. LOD Embedding-Only

```
Index:  Build 3-level hierarchy from document structure
        Level 2: Document summaries
        Level 1: Section summaries  
        Level 0: Chunks

Search: Query → Search L2 → Filter to top docs
             → Search L1 → Filter to top sections
             → Search L0 → Return chunks
```

Uses explicit document structure. No LLM at search time.

### 4. LOD LLM-Guided

```
Index:  Same as LOD Embedding-Only

Search: Query + L2 summaries → LLM picks docs
             + L1 summaries → LLM picks sections
             → Embedding search in selected sections
```

LLM makes routing decisions. Most accurate but slowest.

## Project Structure

```
poc/retrieval_benchmark/
├── README.md
├── pyproject.toml
├── docs/
│   ├── ARCHITECTURE.md       # System design
│   └── DECISIONS.md          # Decision log
├── config/
│   ├── schema.py             # Pydantic config models
│   └── experiments/          # YAML experiment configs
├── core/
│   ├── types.py              # Data classes
│   ├── protocols.py          # Abstract interfaces
│   └── loader.py             # Data loading utilities
├── backends/                 # Vector store implementations
├── embeddings/               # Embedding model wrappers
├── llms/                     # LLM wrappers
├── strategies/               # Retrieval strategy implementations
├── benchmark/                # Benchmark orchestration
├── cli.py                    # Command-line interface
└── results/                  # Output directory (gitignored)
```

## Requirements

- Python 3.11+
- 8GB+ VRAM (for local LLMs)
- Ollama (for LLM-based strategies)
- Test data from `poc/test_data/output/`
