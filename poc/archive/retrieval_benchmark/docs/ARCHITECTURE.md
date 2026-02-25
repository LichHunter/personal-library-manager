# Retrieval Benchmark Architecture

## Overview

This document describes the architecture of the retrieval benchmark framework. The system is designed to be modular, extensible, and produce comparable results across different retrieval configurations.

## Design Principles

1. **Pluggable Components**: Every major component (embeddings, vector stores, LLMs, strategies) follows a protocol/interface that allows swapping implementations
2. **Configuration-Driven**: Experiments are defined in YAML config files, not code changes
3. **Sequential Execution**: No parallelism to avoid race conditions and LLM rate limits
4. **Reproducible**: Same config + seed = same results (where possible)
5. **Comparable Output**: Standardized CSV format for easy comparison

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                             │
│  - Parse arguments                                               │
│  - Load config                                                   │
│  - Invoke benchmark runner                                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Runner                              │
│  - Load documents and ground truth                               │
│  - Generate config combinations                                  │
│  - Execute each config sequentially                              │
│  - Collect results                                               │
│  - Generate reports                                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Strategy Executor                             │
│  For each (strategy, backend, embedder, llm) combination:        │
│  1. Create components                                            │
│  2. Index documents                                              │
│  3. Run all queries (N runs each)                                │
│  4. Collect metrics                                              │
│  5. Cleanup                                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    Strategy     │ │    Strategy     │ │    Strategy     │
│  (flat, raptor, │ │  (uses)         │ │  (uses)         │
│   lod_embed,    │ │                 │ │                 │
│   lod_llm)      │ │                 │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    Embedder     │ │  Vector Store   │ │      LLM        │
│  (sbert)        │ │  (memory,       │ │  (ollama)       │
│                 │ │   chromadb,     │ │                 │
│                 │ │   sqlite_vec)   │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Component Protocols

### Embedder Protocol

```python
class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...
    
    @property
    def dimension(self) -> int: ...
    
    def embed(self, text: str) -> list[float]: ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

**Implementations**:
- `SBERTEmbedder`: Uses sentence-transformers library

### VectorStore Protocol

```python
class VectorStore(Protocol):
    @property
    def backend_name(self) -> str: ...
    
    def create_collection(self, name: str, dimension: int) -> None: ...
    
    def insert(
        self, 
        collection: str, 
        ids: list[str], 
        embeddings: list[list[float]], 
        metadata: list[dict]
    ) -> None: ...
    
    def search(
        self, 
        collection: str, 
        query_embedding: list[float], 
        top_k: int, 
        filter: dict | None = None
    ) -> list[SearchHit]: ...
    
    def count(self, collection: str) -> int: ...
    
    def delete_collection(self, name: str) -> None: ...
    
    def close(self) -> None: ...
```

**Implementations**:
- `MemoryStore`: NumPy arrays + cosine similarity (baseline)
- `ChromaDBStore`: ChromaDB embedded database
- `SqliteVecStore`: sqlite-vec extension

### LLM Protocol

```python
class LLM(Protocol):
    @property
    def model_name(self) -> str: ...
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 500, 
        temperature: float = 0.1
    ) -> str: ...
```

**Implementations**:
- `OllamaLLM`: Local Ollama models

### RetrievalStrategy Protocol

```python
class RetrievalStrategy(Protocol):
    @property
    def name(self) -> str: ...
    
    @property
    def requires_llm(self) -> bool: ...
    
    def configure(
        self, 
        embedder: Embedder, 
        store: VectorStore, 
        llm: LLM | None
    ) -> None: ...
    
    def index(self, documents: list[Document]) -> IndexStats: ...
    
    def search(self, query: str, top_k: int = 5) -> SearchResponse: ...
    
    def clear(self) -> None: ...
```

**Implementations**:
- `FlatStrategy`: Direct chunking and embedding
- `RaptorStrategy`: Hierarchical tree with clustering
- `LODEmbedStrategy`: Level-of-detail with embedding-only search
- `LODLLMStrategy`: Level-of-detail with LLM-guided navigation

## Data Flow

### Indexing Phase

```
Documents (from test_data)
    │
    ▼
┌─────────────────────────────────────┐
│           Strategy.index()           │
│                                      │
│  1. Parse document structure         │
│  2. Create chunks/levels             │
│  3. Generate embeddings (Embedder)   │
│  4. [Optional] Generate summaries    │
│  5. Store vectors (VectorStore)      │
│                                      │
│  Returns: IndexStats                 │
└─────────────────────────────────────┘
```

### Search Phase

```
Query (from ground truth)
    │
    ▼
┌─────────────────────────────────────┐
│          Strategy.search()           │
│                                      │
│  Flat:                               │
│    1. Embed query                    │
│    2. Search all chunks              │
│    3. Return top-k                   │
│                                      │
│  LOD Embedding:                      │
│    1. Embed query                    │
│    2. Search doc level → top docs    │
│    3. Search section level (filtered)│
│    4. Search chunk level (filtered)  │
│    5. Return top-k                   │
│                                      │
│  LOD LLM:                            │
│    1. Get doc summaries              │
│    2. LLM selects relevant docs      │
│    3. Get section summaries          │
│    4. LLM selects relevant sections  │
│    5. Embed query, search chunks     │
│    6. Return top-k                   │
│                                      │
│  Returns: SearchResponse             │
└─────────────────────────────────────┘
```

### Evaluation Phase

```
SearchResponse + GroundTruth
    │
    ▼
┌─────────────────────────────────────┐
│            Evaluator                 │
│                                      │
│  1. Check if expected doc in results │
│  2. Calculate doc rank               │
│  3. Check if expected section found  │
│  4. Calculate section rank           │
│  5. Record timing and call counts    │
│                                      │
│  Returns: QueryResult                │
└─────────────────────────────────────┘
```

## Configuration System

### Hierarchy

```
ExperimentConfig
├── id: str
├── description: str
├── data: DataConfig
│   ├── documents_dir: str
│   ├── ground_truth_path: str
│   ├── max_documents: int | None
│   └── max_queries: int | None
├── benchmark: BenchmarkConfig
│   ├── runs_per_query: int
│   ├── top_k_values: list[int]
│   └── random_seed: int
├── embeddings: list[EmbeddingConfig]
├── llms: list[LLMConfig]
├── backends: list[BackendConfig]
└── strategies: list[StrategyConfig]
```

### Configuration Matrix

The benchmark runner generates all valid combinations:

```
For each strategy in strategies:
    For each backend in backends:
        For each embedding in embeddings:
            If strategy.requires_llm:
                For each llm in llms:
                    Run benchmark(strategy, backend, embedding, llm)
            Else:
                Run benchmark(strategy, backend, embedding, None)
```

## Output Schema

### index_stats.csv

One row per configuration:

| Column | Type | Description |
|--------|------|-------------|
| strategy | str | Strategy name |
| backend | str | Vector store backend |
| embedding_model | str | Embedding model name |
| llm_model | str? | LLM model (null if not used) |
| duration_sec | float | Indexing time |
| num_documents | int | Documents indexed |
| num_chunks | int | Chunks created |
| num_vectors | int | Vectors stored |
| llm_calls | int | LLM invocations during indexing |
| embed_calls | int | Embedding calls during indexing |

### search_results.csv

One row per (query, run, top_k):

| Column | Type | Description |
|--------|------|-------------|
| query_id | str | Ground truth ID |
| run | int | Run number (1-N) |
| strategy | str | Strategy name |
| backend | str | Vector store backend |
| embedding_model | str | Embedding model |
| llm_model | str? | LLM model |
| question | str | Query text |
| expected_doc | str | Expected document ID |
| expected_sections | str | Expected section IDs (comma-separated) |
| top_k | int | Number of results requested |
| doc_found | bool | Was expected doc in results? |
| doc_rank | int | Position of expected doc (-1 if not found) |
| section_found | bool | Was any expected section in results? |
| section_rank | int | Position of first matching section |
| search_time_ms | float | Query latency |
| embed_calls | int | Embedding calls for this query |
| llm_calls | int | LLM calls for this query |

### summary.csv

One row per configuration with aggregated metrics:

| Column | Type | Description |
|--------|------|-------------|
| strategy | str | Strategy name |
| backend | str | Vector store backend |
| embedding_model | str | Embedding model |
| llm_model | str? | LLM model |
| index_time_sec | float | Indexing time |
| num_vectors | int | Vectors in index |
| doc_recall_at_1 | float | % queries with correct doc at rank 1 |
| doc_recall_at_3 | float | % queries with correct doc in top 3 |
| doc_recall_at_5 | float | % queries with correct doc in top 5 |
| doc_recall_at_10 | float | % queries with correct doc in top 10 |
| section_recall_at_1 | float | % queries with correct section at rank 1 |
| section_recall_at_3 | float | % queries with correct section in top 3 |
| section_recall_at_5 | float | % queries with correct section in top 5 |
| section_recall_at_10 | float | % queries with correct section in top 10 |
| doc_mrr | float | Mean Reciprocal Rank (document) |
| section_mrr | float | Mean Reciprocal Rank (section) |
| avg_search_time_ms | float | Mean query latency |
| p50_search_time_ms | float | Median query latency |
| p95_search_time_ms | float | 95th percentile latency |
| consistency_score | float | % queries with same results across runs |
| total_llm_calls | int | Total LLM invocations |
| total_embed_calls | int | Total embedding calls |
| queries_evaluated | int | Number of unique queries |
| runs_per_query | int | Runs per query |

## Directory Structure

```
poc/retrieval_benchmark/
├── README.md                     # Overview and usage
├── pyproject.toml                # Dependencies
│
├── docs/
│   ├── ARCHITECTURE.md           # This document
│   └── DECISIONS.md              # Decision log
│
├── config/
│   ├── schema.py                 # Pydantic models for config validation
│   └── experiments/
│       ├── quick_test.yaml       # 5 docs, 1 run (development)
│       ├── baseline.yaml         # 10 docs, 3 runs (quick comparison)
│       └── full.yaml             # All docs, 5 runs (comprehensive)
│
├── core/
│   ├── __init__.py
│   ├── types.py                  # Data classes (Document, Chunk, etc.)
│   ├── protocols.py              # Abstract interfaces
│   └── loader.py                 # Load data from test_data output
│
├── backends/
│   ├── __init__.py               # Factory function: create_backend()
│   ├── base.py                   # VectorStore ABC
│   ├── memory.py                 # In-memory NumPy implementation
│   ├── chromadb_store.py         # ChromaDB implementation
│   └── sqlite_vec_store.py       # sqlite-vec implementation
│
├── embeddings/
│   ├── __init__.py               # Factory function: create_embedder()
│   ├── base.py                   # Embedder ABC
│   └── sbert.py                  # SentenceTransformers implementation
│
├── llms/
│   ├── __init__.py               # Factory function: create_llm()
│   ├── base.py                   # LLM ABC
│   └── ollama_llm.py             # Ollama implementation
│
├── strategies/
│   ├── __init__.py               # Factory function: create_strategy()
│   ├── base.py                   # RetrievalStrategy ABC
│   ├── flat.py                   # Flat embedding strategy
│   ├── raptor.py                 # RAPTOR strategy
│   ├── lod_embed.py              # LOD with embedding-only search
│   └── lod_llm.py                # LOD with LLM-guided navigation
│
├── benchmark/
│   ├── __init__.py
│   ├── runner.py                 # Main benchmark executor
│   ├── evaluator.py              # Metric computation
│   └── reporter.py               # CSV report generation
│
├── cli.py                        # Click-based CLI
│
└── results/                      # Output directory (gitignored)
    └── {experiment_id}/
        ├── config.yaml
        ├── index_stats.csv
        ├── search_results.csv
        ├── summary.csv
        └── run.log
```

## Error Handling

### Strategy-Level Errors

If a strategy fails during indexing or search:
1. Log the error with full traceback
2. Mark all queries for that config as failed
3. Continue with next configuration
4. Include error summary in final report

### Query-Level Errors

If a single query fails:
1. Log the error
2. Record the query as failed (special marker in CSV)
3. Continue with next query
4. Don't count failed queries in aggregates

### Resource Cleanup

After each configuration:
1. Clear strategy index
2. Close vector store connection
3. Force garbage collection (release GPU memory)

## Performance Considerations

### Memory Management

- Batch embedding calls (reduce GPU kernel launches)
- Clear embeddings after storing (don't keep in memory)
- Use generators for large document sets

### Timing Accuracy

- Use `time.perf_counter()` for high-resolution timing
- Exclude setup time from search timing
- Warm up embedder before timing (first call is slower)

### Reproducibility

- Set random seeds where applicable
- Document model versions in output
- Save exact config used (not just reference to file)
