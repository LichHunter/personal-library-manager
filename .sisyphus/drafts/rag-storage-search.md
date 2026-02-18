# Draft: RAG Storage & Search System

## Requirements (confirmed)

- **One-to-one port of POC flow**: Direct translation of `poc/modular_retrieval_pipeline/` to production code
- **Only ingestion changes**: Instead of YAKE/spaCy inline extraction, consume GLiNER fast system output
- **Retrieval flow unchanged**: Same RRF fusion, same weights, same components

## POC Flow to Port

### Indexing (POC)
```
Content → KeywordExtractor (YAKE) → EntityExtractor (spaCy) → ContentEnricher 
        → Enriched text = "keywords | entities\n\noriginal_content"
        → EmbeddingEncoder (BGE) → Vector Index (numpy array)
        → BM25Okapi → Lexical Index
```

### Indexing (New - GLiNER ingestion)
```
GLiNER DocumentResult (JSON) → Adapter (map ChunkResult.terms → keywords, entities)
        → ContentEnricher → Enriched text
        → EmbeddingEncoder (BGE) → Vector Index
        → BM25Okapi → Lexical Index
```

### Retrieval (Unchanged - direct port)
```
Query → QueryExpander (domain terms) → expanded query
      → Encode → SimilarityScorer (cosine) → Semantic results
      → BM25Scorer → BM25 results
      → RRFFuser (k=60 default, adaptive weights) → Final ranked results
```

## Technical Decisions

- **Embedding model**: BAAI/bge-base-en-v1.5 (same as POC)
- **BM25**: rank_bm25.BM25Okapi (same as POC)
- **RRF parameters**: k=60 default, adaptive (k=10, bm25=3.0, sem=0.3) when expanded
- **Content enrichment format**: "keywords | entities\n\noriginal_content"

## Files to Port (one-to-one)

| POC File | Production Location | Change |
|----------|--------------------|---------| 
| `types.py` | `src/plm/search/types.py` | Direct port |
| `base.py` | `src/plm/search/pipeline.py` | Direct port |
| `components/content_enricher.py` | `src/plm/search/components/enricher.py` | Adapt for GLiNER format |
| `components/embedding_encoder.py` | `src/plm/search/components/embedder.py` | Direct port |
| `components/bm25_scorer.py` | `src/plm/search/components/bm25.py` | Direct port |
| `components/similarity_scorer.py` | `src/plm/search/components/semantic.py` | Direct port |
| `components/query_expander.py` | `src/plm/search/components/expander.py` | Direct port |
| `components/rrf_fuser.py` | `src/plm/search/components/rrf.py` | Direct port |
| `modular_enriched_hybrid.py` | `src/plm/search/retriever.py` | Adapt for GLiNER ingestion |

## GLiNER Output Format (source)

```python
@dataclass
class ChunkResult:
    text: str
    terms: list[str]          # → maps to keywords
    entities: list[ExtractedEntity]  # → maps to entities
    start_char: int
    end_char: int

@dataclass  
class HeadingSection:
    heading: str
    level: int
    chunks: list[ChunkResult]

@dataclass
class DocumentResult:
    source_file: str
    headings: list[HeadingSection]
    avg_confidence: float
    total_entities: int
    is_low_confidence: bool
```

## Storage Decision (confirmed)

- **Database**: SQLite (not in-memory like POC)
- **Deployment**: SQLite file in Docker volume
- **Architecture**: Python app connects to SQLite file mounted in volume
- **Indices**: Store embeddings in SQLite (BLOB), BM25 in separate directory (bm25s)

## Oracle Review Updates (applied to plan)

1. **YAKE Integration**: Add YAKE to fast extraction system, output `keywords` field alongside GLiNER `terms`
2. **BM25S Library**: Replace rank-bm25 with bm25s (built-in persistence, 10-180x faster)
3. **Error Handling**: Add try/except for embedding model loading
4. **Accuracy Test**: Add regression test comparing to POC 90% baseline

## Database Schema (proposed)

```sql
-- Documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table (from GLiNER output)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    heading TEXT,
    heading_level INTEGER,
    content TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    terms TEXT,  -- JSON array of extracted terms
    entities TEXT,  -- JSON array of entities
    enriched_content TEXT,  -- "keywords | entities\n\ncontent"
    embedding BLOB  -- numpy array serialized
);

-- BM25 tokenized content (for rebuilding index)
CREATE TABLE bm25_tokens (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id),
    tokens TEXT  -- JSON array of lowercase tokens
);
```

## Scope Boundaries

- **IN**: 
  - All retrieval components from POC
  - GLiNER output ingestion adapter
  - Hybrid retrieval (BM25 + semantic + RRF)
  
- **OUT**:
  - Redis caching (defer)
  - Cross-encoder reranking (defer)
  - LLM query rewriting (POC showed it's unnecessary)
