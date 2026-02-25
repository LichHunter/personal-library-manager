# SEARCH MODULE

Hybrid retrieval system with switchable sparse retriever (BM25 or SPLADE) + semantic embedding search + RRF fusion.

## OVERVIEW

`HybridRetriever` orchestrates: ingest documents → enrich content → embed → sparse index (BM25/SPLADE) → query with fusion.

## STRUCTURE

```
search/
├── retriever.py          # HybridRetriever - main orchestrator
├── config.py             # RetrievalConfig, SparseRetrieverType, factory
├── pipeline.py           # High-level pipeline (unused?)
├── types.py              # Query, RewrittenQuery, ExpandedQuery
├── components/
│   ├── sparse/           # Switchable sparse retrievers
│   │   ├── base.py       # SparseRetriever ABC
│   │   ├── bm25_retriever.py   # BM25 implementation
│   │   └── splade_retriever.py # SPLADE implementation
│   ├── bm25.py           # BM25Index wrapper (bm25s library)
│   ├── semantic.py       # SemanticSearch (FAISS)
│   ├── embedder.py       # EmbeddingEncoder (sentence-transformers)
│   ├── enricher.py       # ContentEnricher (keywords → enriched text)
│   ├── expander.py       # QueryExpander (domain term expansion)
│   ├── rrf.py            # RRF fusion (standalone, not used by retriever)
│   └── query_rewriter.py # LLM-based query rewriting
├── storage/
│   └── sqlite.py         # SQLiteStorage - chunks, embeddings, documents
├── service/
│   ├── app.py            # FastAPI endpoints (/query, /health, /status)
│   ├── cli.py            # plm-query CLI (POSTs to FastAPI)
│   └── watcher.py        # DirectoryWatcher for auto-ingestion
└── adapters/
    └── gliner_adapter.py # Dead code (GLiNER rejected)
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Switch BM25 ↔ SPLADE | Set `PLM_SPARSE_RETRIEVER=splade` | Environment variable |
| Enable SPLADE-only mode | Set `PLM_SEMANTIC_ENABLED=false` | Disables semantic, no RRF |
| Change RRF weights | `retriever.py:73-84` | `DEFAULT_*` / `EXPANDED_*` constants |
| Add new sparse retriever | `components/sparse/` | Implement `SparseRetriever` ABC |
| Change embedding model | `components/embedder.py` | `MODEL_NAME` constant |
| Add query preprocessing | `components/query_rewriter.py` | Uses Claude for rewriting |
| Add storage backend | `storage/` | Subclass pattern from `sqlite.py` |

## RETRIEVAL MODES

| Mode | Sparse | Semantic | Fusion | Config |
|------|--------|----------|--------|--------|
| **BM25 + Semantic** | BM25 | BGE | RRF | Default |
| **SPLADE + Semantic** | SPLADE | BGE | RRF | `PLM_SPARSE_RETRIEVER=splade` |
| **SPLADE-only** | SPLADE | Disabled | None | `PLM_SPARSE_RETRIEVER=splade PLM_SEMANTIC_ENABLED=false` |

**SPLADE-only achieves +26.7% MRR on informed queries vs BM25+Semantic hybrid.**

## CRITICAL INVARIANTS

```python
# RRF MUST process semantic FIRST, BM25 SECOND
for rank, idx in enumerate(sem_ranks[:n_candidates]):
    rrf_scores[idx] = rrf_scores.get(idx, 0) + sem_weight / (rrf_k + rank)
for rank, result in enumerate(bm25_results):
    rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_weight / (rrf_k + rank)
```

- Changing order breaks POC parity
- `dict.get(idx, 0)` accumulation pattern required
- Embeddings are L2-normalized by `EmbeddingEncoder` — cosine = dot product

## ADAPTIVE WEIGHTS

| Condition | RRF k | BM25 Weight | Sem Weight | Candidates |
|-----------|-------|-------------|------------|------------|
| Default | 60 | 1.0 | 1.0 | 10x k |
| Query expanded | 10 | 3.0 | 0.3 | 20x k |

## ANTI-PATTERNS

- **Don't use `adapters/gliner_adapter.py`** — GLiNER rejected for software NER
- **Don't use `components/rrf.py` directly** — HybridRetriever has its own inline RRF
- **Don't change RRF order** — Semantic first, sparse second is load-bearing
- **Don't use SPLADE+Semantic hybrid** — POC showed it performs worse than SPLADE-only

## ENVIRONMENT VARIABLES

| Variable | Values | Default |
|----------|--------|---------|
| `PLM_SPARSE_RETRIEVER` | `bm25`, `splade` | `bm25` |
| `PLM_SPLADE_MODEL` | HuggingFace model name | `naver/splade-cocondenser-ensembledistil` |
| `PLM_SPLADE_DEVICE` | `cpu`, `cuda` | auto-detect |
| `PLM_SEMANTIC_ENABLED` | `true`, `false` | `true` |
| `PLM_LOG_LEVEL` | `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR` | `INFO` |
| `PLM_LOG_DIR` | Path | `/data/logs` |
| `PLM_LOG_TO_FILE` | `true`, `false` | `true` |

## REQUEST TRACING

Every request is traced with a `request_id` returned in the response. Set `PLM_LOG_LEVEL=TRACE` to see full pipeline traces:

```
[request_id] [receive] query=... k=... rewrite=...
[request_id] [rewrite] input=... output=...
[request_id] [expand] added=[...]
[request_id] [sparse] found=... top5=[...]
[request_id] [semantic] computed=... top5=[...]
[request_id] [rrf] fused=... top_k=[...]
[request_id] [complete] results=...
```

Trace a specific request:
```bash
grep "request_id" /data/logs/search_trace.log
```
