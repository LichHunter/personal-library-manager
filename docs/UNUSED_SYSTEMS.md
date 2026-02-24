# PLM Search Service: Tools & Data Analysis

**Generated:** 2026-02-21  
**Purpose:** Analysis of data and components that are stored/available but unused in retrieval

---

## Executive Summary

| Category | Status | Issue |
|----------|--------|-------|
| **BM25 Index** | Used | Built on enriched content (keywords + entities prepended) |
| **Semantic Embeddings** | Used | Generated from enriched content |
| **Keywords** | Partially Used | Baked into `enriched_content`, but `keywords_json` column never queried |
| **Entities** | Partially Used | Baked into `enriched_content`, but `entities_json` column never queried |
| **Entity Labels/Scores** | Unused | Stored but discarded during enrichment |
| **Heading Embeddings** | Underused | Only in `retrieve_headings()`, not in main `retrieve()` |
| **Document Embeddings** | Underused | Only in `retrieve_documents()`, not in main `retrieve()` |
| **Character Offsets** | Unused | Stored but never used for context expansion |
| **Standalone Components** | Dead Code | `semantic.py`, `rrf.py` exist but HybridRetriever has inline implementations |

---

## 1. Data Storage vs Usage

### SQLite Schema (3 Tables, 3 Levels)

```
+-----------------------------------------------------------------------------+
| documents                                                                    |
| +- id, source_file, created_at                                              |
| +- embedding (BLOB) <- aggregated from headings      [!] Only in retrieve_documents()
| +- keywords_json (TEXT) <- deduplicated from chunks  [X] NEVER READ BACK    |
| +- entities_json (TEXT) <- deduplicated from chunks  [X] NEVER READ BACK    |
+-----------------------------------------------------------------------------+
| headings                                                                     |
| +- id, doc_id, heading_text, heading_level                                  |
| +- embedding (BLOB) <- aggregated from chunks        [!] Only in retrieve_headings()
| +- keywords_json (TEXT)                              [X] NEVER READ BACK    |
| +- entities_json (TEXT)                              [X] NEVER READ BACK    |
| +- start_char, end_char                              [X] UNUSED             |
+-----------------------------------------------------------------------------+
| chunks (PRIMARY RETRIEVAL UNIT)                                              |
| +- id, doc_id, heading_id, chunk_index                                      |
| +- content (TEXT)            [OK] Returned in search results                |
| +- enriched_content (TEXT)   [OK] BM25 index + embedding generation         |
| +- embedding (BLOB)          [OK] Semantic search                           |
| +- heading (TEXT)            [OK] Result metadata                           |
| +- keywords_json (TEXT)      [X] NEVER READ BACK (only written)             |
| +- entities_json (TEXT)      [X] NEVER READ BACK (only written)             |
| +- start_char, end_char      [X] UNUSED                                     |
+-----------------------------------------------------------------------------+
```

### The Enrichment Pattern

```
Keywords: ["kubernetes", "autoscaler"]                 --+
Entities: [{"text": "HPA", "label": "technology"}]     --+--> ContentEnricher --> "kubernetes, autoscaler | HPA\n\nOriginal content..."
Content: "Original content..."                         --+                                         |
                                                                                                   v
                                                                                       +------------------------+
                                                                                       |   enriched_content     |
                                                                                       |   (stored in SQLite)   |
                                                                                       +------------------------+
                                                                                                   |
                                                    +----------------------------------------------+----------------------------------------------+
                                                    v                                                                                              v
                                           EmbeddingEncoder                                                                                 BM25Index
                                           (BGE-base-en-v1.5)                                                                            (bm25s library)
                                                    |                                                                                              |
                                                    v                                                                                              v
                                              embedding BLOB                                                                              lexical index
                                              (768-dim, normalized)                                                                      (enriched content)
```

**Key insight**: Keywords and entities ARE used, but only indirectly via `enriched_content`. The structured JSON columns are write-only metadata.

---

## 2. Component Inventory

### Actively Used Components

| Component | File | Purpose | Data Used |
|-----------|------|---------|-----------|
| `HybridRetriever` | `retriever.py` | Main orchestrator | All chunk data |
| `ContentEnricher` | `enricher.py` | Formats keywords/entities into text | keywords, entities dict |
| `EmbeddingEncoder` | `embedder.py` | Text -> 768-dim vector | enriched_content |
| `BM25Index` | `bm25.py` | Lexical search | enriched_content |
| `QueryExpander` | `expander.py` | Domain term expansion | Query string |
| `QueryRewriter` | `query_rewriter.py` | LLM query rewrite (optional) | Query string |

### Standalone Components (NOT Used by HybridRetriever)

| Component | File | Reason Unused |
|-----------|------|---------------|
| `SimilarityScorer` | `semantic.py` | HybridRetriever has inline dot-product implementation |
| `RRFFuser` | `rrf.py` | HybridRetriever has inline RRF implementation |
| `GLiNERAdapter` | `adapters/gliner_adapter.py` | Dead code (GLiNER rejected for software NER) |

### Type System Gap

| Type | Defined In | Used? |
|------|------------|-------|
| `Query` | `types.py` | Yes |
| `RewrittenQuery` | `types.py` | Yes |
| `ExpandedQuery` | `types.py` | Yes |
| `EmbeddedQuery` | `types.py` | No (HybridRetriever returns dicts) |
| `ScoredChunk` | `types.py` | No (HybridRetriever returns dicts) |
| `FusionConfig` | `types.py` | No (HybridRetriever uses class constants) |
| `PipelineResult` | `types.py` | No |

---

## 3. Configuration Parameters

### RRF Weights (Primary Tuning Points)

| Mode | RRF k | BM25 Weight | Semantic Weight | Candidates |
|------|-------|-------------|-----------------|------------|
| **Default** | 60 | 1.0 | 1.0 | 10x k |
| **Expanded** (query expansion triggered) | 10 | 3.0 | 0.3 | 20x k |

### Hardcoded Values (Not Configurable)

| Parameter | Value | Location |
|-----------|-------|----------|
| Embedding model | `BAAI/bge-base-en-v1.5` | `embedder.py` |
| Embedding dim | 768 | `embedder.py` |
| BM25 tokenization | lowercase whitespace split | `bm25.py` |
| Max keywords in enrichment | 7 | `enricher.py` |
| Max entities in enrichment | 5 (2 per type) | `enricher.py` |
| Query rewrite timeout | 5.0s | `retriever.py` |

---

## 4. Unused Data Opportunities

### 4.1 `keywords_json` / `entities_json` Columns

**Current state**: Written during ingestion, never read back during retrieval.

**Potential uses**:
1. **Query-time expansion**: Parse stored keywords to expand user queries
2. **Faceted search**: Filter by entity type ("show only results mentioning 'database' entities")
3. **Keyword highlighting**: Highlight extracted keywords in result snippets
4. **Analytics**: Corpus-wide keyword/entity statistics

### 4.2 `start_char` / `end_char` Offsets

**Current state**: Stored in chunks table, returned in results, but never used for logic.

**Potential uses**:
1. **Context expansion**: Fetch N chars before/after chunk for surrounding context
2. **Precise citations**: Link to exact source location
3. **Deduplication**: Detect overlapping chunks

### 4.3 Heading/Document Embeddings

**Current state**: Aggregated (mean of child embeddings) and stored, but only used in separate `retrieve_headings()` / `retrieve_documents()` methods.

**Potential uses**:
1. **Hierarchical retrieval**: Score at document -> heading -> chunk levels
2. **Re-ranking**: Use document relevance to boost/demote chunks
3. **Clustering**: Group related documents by embedding similarity

### 4.4 Entity Labels and Confidence Scores

**Current state**: Stored in `entities_json` but discarded during enrichment (only text used).

**Potential uses**:
1. **Type-aware enrichment**: `"Technology: Kubernetes | Database: Redis"` instead of `"Kubernetes, Redis"`
2. **Confidence filtering**: Drop low-score entities before enrichment
3. **Type-based boosting**: Weight entities differently by type

---

## 5. Dead Code / Redundancy

| Item | Location | Issue | Recommendation |
|------|----------|-------|----------------|
| `SimilarityScorer` | `components/semantic.py` | Duplicated by inline code in retriever | Remove or refactor retriever to use it |
| `RRFFuser` | `components/rrf.py` | Duplicated by inline code in retriever | Remove or refactor retriever to use it |
| `GLiNERAdapter` | `adapters/gliner_adapter.py` | GLiNER rejected for software NER | Remove |
| Type classes | `types.py` | 4 of 7 types unused | Clean up or adopt in retriever |
| `FusionConfig` defaults | `types.py:92-101` | Different from retriever constants | Consolidate |

---

## 6. Benchmark-Informed Findings

From PLM vs RAG benchmark (`poc/plm_vs_rag_benchmark/results/DETAILED_RESULTS.md`):

| Finding | Implication |
|---------|-------------|
| **Enrichment helps needle/informed (+5-8%)** | Keep enrichment for technical queries |
| **Enrichment hurts realistic (-5.6%)** | Consider query-type detection |
| **BM25 + RRF is the value driver** | Keep hybrid approach |
| **Embedding enrichment consistently -2 to -5%** | May want raw embeddings + enriched BM25 |

---

## 7. Recommendations

### Quick Wins (Low Effort)

1. **Remove dead code**: Delete `semantic.py`, `rrf.py`, `gliner_adapter.py` (or document as "reference implementations")
2. **Document JSON columns**: Mark `keywords_json`/`entities_json` as "metadata-only, not used in retrieval"
3. **Add entity labels to enrichment**: Change format from `"K8s, Redis"` to `"technology: K8s | database: Redis"`

### Medium Effort

4. **Query-time keyword expansion**: Parse `keywords_json` from top results to expand queries
5. **Confidence filtering**: Filter entities with score < 0.5 before enrichment
6. **Context expansion**: Use `start_char`/`end_char` to fetch surrounding paragraphs

### Larger Refactors

7. **Query-type detection**: Use enriched BM25 for technical queries, raw BM25 for natural language
8. **Hierarchical retrieval**: Incorporate heading/document embeddings in chunk scoring
9. **Raw embeddings experiment**: Re-embed with raw content (not enriched) to test if retrieval improves

---

## 8. File Reference

### Search Service
- `src/plm/search/retriever.py` - HybridRetriever (main orchestrator)
- `src/plm/search/storage/sqlite.py` - SQLite schema definition
- `src/plm/search/components/enricher.py` - ContentEnricher
- `src/plm/search/components/embedder.py` - EmbeddingEncoder
- `src/plm/search/components/bm25.py` - BM25Index
- `src/plm/search/components/expander.py` - QueryExpander
- `src/plm/search/components/semantic.py` - SimilarityScorer (UNUSED)
- `src/plm/search/components/rrf.py` - RRFFuser (UNUSED)
- `src/plm/search/adapters/gliner_adapter.py` - GLiNERAdapter (DEAD CODE)
- `src/plm/search/types.py` - Type definitions (4/7 unused)

### Extraction Service
- `src/plm/extraction/fast/` - GLiNER + YAKE extraction
- `src/plm/extraction/slow/` - V6 LLM pipeline

### Benchmark Results
- `poc/plm_vs_rag_benchmark/results/DETAILED_RESULTS.md` - Full benchmark analysis

---

*Last updated: 2026-02-21*
