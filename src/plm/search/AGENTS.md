# SEARCH MODULE

Hybrid retrieval system combining BM25 lexical + semantic embedding search with RRF fusion.

## OVERVIEW

`HybridRetriever` orchestrates: ingest documents → enrich content → embed → BM25 index → query with fusion.

## STRUCTURE

```
search/
├── retriever.py          # HybridRetriever - main orchestrator
├── pipeline.py           # High-level pipeline (unused?)
├── types.py              # Query, RewrittenQuery, ExpandedQuery
├── components/           # Search primitives
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
| Change RRF weights | `retriever.py:73-84` | `DEFAULT_*` / `EXPANDED_*` constants |
| Add retrieval signal | `retriever.py:retrieve()` | Insert between semantic/BM25 and RRF fusion |
| Change embedding model | `components/embedder.py` | `MODEL_NAME` constant |
| Add query preprocessing | `components/query_rewriter.py` | Uses Claude for rewriting |
| Add storage backend | `storage/` | Subclass pattern from `sqlite.py` |

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
- **Don't change RRF order** — Semantic first, BM25 second is load-bearing
