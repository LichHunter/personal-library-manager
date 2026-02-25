# SPLADE Integration Design: Switchable Retrieval Architecture

**Created**: 2026-02-22  
**Status**: DRAFT  
**Goal**: Enable switching between current BM25+Semantic hybrid and SPLADE-based retrieval without code changes

---

## 1. Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HybridRetriever                            │
│                                                                 │
│  Query ──► Expander ──► ┬──► Semantic (BGE) ──┐                │
│                         │                      ├──► RRF ──► Results
│                         └──► BM25 (bm25s) ────┘                │
│                                                                 │
│  Optional: QueryRewriter (before) + CrossEncoder (after)       │
└─────────────────────────────────────────────────────────────────┘
```

### Current Components

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| `BM25Index` | `bm25s` library | Lexical/sparse retrieval |
| `EmbeddingEncoder` | `sentence-transformers` (BGE) | Dense semantic retrieval |
| `ContentEnricher` | Keywords/entities prefix | Content preprocessing |
| `QueryExpander` | Rule-based expansion | Query preprocessing |
| `CrossEncoderReranker` | `cross-encoder/ms-marco` | Optional reranking |

### Current Flow (retrieve method)

1. Optionally rewrite query (LLM-based)
2. Expand query with domain terms
3. Select adaptive RRF parameters based on expansion
4. Get semantic scores (embedding similarity)
5. Get BM25 scores (lexical match)
6. Fuse with RRF (semantic FIRST, BM25 SECOND)
7. Optionally rerank with cross-encoder
8. Return top-k results

---

## 2. Proposed Architecture

### 2.1 Strategy Pattern for Sparse Retrieval

Replace direct BM25 usage with an abstract `SparseRetriever` interface:

```
┌─────────────────────────────────────────────────────────────────┐
│                      HybridRetriever                            │
│                                                                 │
│  Query ──► Expander ──► ┬──► Semantic (BGE) ──┐                │
│                         │                      ├──► RRF ──► Results
│                         └──► SparseRetriever ─┘                │
│                              ▲                                  │
│                              │                                  │
│              ┌───────────────┼───────────────┐                 │
│              │               │               │                 │
│         BM25Retriever   SPLADERetriever   (future)            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Configuration-Based Switching

Configuration file (`config.yaml` or environment variables):

```yaml
retrieval:
  # Sparse retriever: "bm25" | "splade"
  sparse_retriever: "bm25"
  
  # SPLADE-specific settings (only used if sparse_retriever: "splade")
  splade:
    model: "naver/splade-cocondenser-ensembledistil"  # or "splade-tiny"
    device: "cpu"  # or "cuda"
    max_length: 512
    batch_size: 16
  
  # Dense retriever settings
  semantic:
    enabled: true  # Can disable for SPLADE-only mode
    model: "BAAI/bge-base-en-v1.5"
  
  # Fusion settings
  fusion:
    method: "rrf"  # or "convex" (future)
    rrf_k: 60
    bm25_weight: 1.0
    semantic_weight: 1.0
```

### 2.3 Retrieval Modes

| Mode | Sparse | Dense | Fusion | Use Case |
|------|--------|-------|--------|----------|
| **hybrid-bm25** | BM25 | BGE | RRF | Current production |
| **hybrid-splade** | SPLADE | BGE | RRF | Test SPLADE with semantic backup |
| **splade-only** | SPLADE | Disabled | None | Best quality (per POC results) |
| **bm25-only** | BM25 | Disabled | None | Baseline comparison |

---

## 3. Component Design

### 3.1 SparseRetriever Interface

Abstract base class that both BM25 and SPLADE implement:

**Interface Contract**:

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `index(documents)` | List of document texts | None | Build index from documents |
| `search(query, k)` | Query string, top-k | List of (index, score) | Retrieve top-k matches |
| `save(path)` | Directory path | None | Persist index to disk |
| `load(path)` | Directory path | Instance | Load index from disk |
| `encode_query(query)` | Query string | Sparse vector | Encode query (for SPLADE) |

**Key Difference**:
- BM25: No query encoding needed (term matching)
- SPLADE: Query encoding required (BERT inference)

### 3.2 BM25Retriever (Existing)

Wrapper around current `BM25Index`:

| Aspect | Value |
|--------|-------|
| Index format | bm25s native |
| Query encoding | None (term matching) |
| Storage | Single directory |
| Dependencies | `bm25s` |

### 3.3 SPLADERetriever (New)

New component wrapping SPLADE model:

| Aspect | Value |
|--------|-------|
| Index format | Sparse CSR matrix + inverted index |
| Query encoding | BERT MLM head (~20-800ms depending on model) |
| Storage | Two files: `doc_vectors.npz` + `metadata.json` |
| Dependencies | `transformers`, `torch`, `scipy` |

**SPLADE Index Structure**:

```
splade_index/
├── doc_vectors.npz      # Sparse document vectors (CSR format)
├── term_to_docs.pkl     # Inverted index: term_id → [(doc_id, weight), ...]
├── metadata.json        # Model name, vocab size, num_docs
└── config.json          # Encoding parameters
```

### 3.4 RetrieverFactory

Factory that creates appropriate retriever based on configuration:

**Factory Logic**:

```
Input: config dict
Output: SparseRetriever instance

If config.sparse_retriever == "bm25":
    Return BM25Retriever(config.bm25)
Elif config.sparse_retriever == "splade":
    Return SPLADERetriever(config.splade)
Else:
    Raise ConfigurationError
```

---

## 4. HybridRetriever Modifications

### 4.1 Initialization Changes

**Current**:
```
__init__(db_path, bm25_index_path)
    → self.bm25_index = BM25Index.load(bm25_index_path)
```

**Proposed**:
```
__init__(db_path, index_path, config=None)
    → self.config = load_config(config)
    → self.sparse_retriever = RetrieverFactory.create(self.config, index_path)
    → self.semantic_enabled = self.config.semantic.enabled
```

### 4.2 Ingestion Changes

**Current**:
- Enrich content
- Generate embedding
- Store chunk
- Rebuild BM25 index

**Proposed**:
- Enrich content
- Generate embedding (if semantic enabled)
- Store chunk
- Rebuild sparse index (BM25 OR SPLADE)

**SPLADE Ingestion Consideration**:
- SPLADE encoding is slow (~200ms/doc on CPU)
- Batch encoding essential for ingestion
- May want async/background indexing option

### 4.3 Retrieval Changes

**Current _retrieve_rrf**:
```
1. Expand query
2. Get semantic scores (embedding similarity)
3. Get BM25 results (bm25_index.search)
4. RRF fusion
```

**Proposed _retrieve_rrf**:
```
1. Expand query
2. If semantic_enabled:
     Get semantic scores (embedding similarity)
3. Get sparse results (sparse_retriever.search)  # BM25 or SPLADE
4. If semantic_enabled:
     RRF fusion (semantic + sparse)
   Else:
     Return sparse results directly
```

### 4.4 SPLADE-Only Mode

When `semantic.enabled: false` and `sparse_retriever: "splade"`:

```
Query ──► Expander ──► SPLADE ──► (Optional Reranker) ──► Results
```

No RRF fusion needed. Simpler pipeline, best quality per POC results.

---

## 5. Index Migration Strategy

### 5.1 Dual Index Support

During transition, support both indexes simultaneously:

```
index/
├── bm25/              # Current BM25 index
│   └── ...
└── splade/            # New SPLADE index
    ├── doc_vectors.npz
    └── ...
```

### 5.2 Index Building

| Scenario | Action |
|----------|--------|
| Fresh install | Build only configured index type |
| Upgrade (BM25 → SPLADE) | Build SPLADE index, keep BM25 as fallback |
| Rollback (SPLADE → BM25) | Switch config, BM25 index still available |

### 5.3 Rebuild Command

CLI command to rebuild index for new retriever type:

```bash
plm-index rebuild --type splade --model splade-tiny
plm-index rebuild --type bm25
```

---

## 6. Configuration Examples

### 6.1 Current Production (BM25 + Semantic)

```yaml
retrieval:
  sparse_retriever: "bm25"
  semantic:
    enabled: true
    model: "BAAI/bge-base-en-v1.5"
  fusion:
    method: "rrf"
    rrf_k: 60
```

### 6.2 SPLADE-Only (Recommended)

```yaml
retrieval:
  sparse_retriever: "splade"
  splade:
    model: "naver/splade-cocondenser-ensembledistil"
    device: "cpu"
  semantic:
    enabled: false  # Disabled - SPLADE handles semantics
  fusion:
    method: "none"  # No fusion needed
```

### 6.3 SPLADE + Semantic Hybrid

```yaml
retrieval:
  sparse_retriever: "splade"
  splade:
    model: "naver/splade-cocondenser-ensembledistil"
    device: "cpu"
  semantic:
    enabled: true
    model: "BAAI/bge-base-en-v1.5"
  fusion:
    method: "rrf"
    rrf_k: 60
    sparse_weight: 1.0  # Note: "sparse" not "bm25"
    semantic_weight: 1.0
```

### 6.4 Low-Latency SPLADE (splade-tiny)

```yaml
retrieval:
  sparse_retriever: "splade"
  splade:
    model: "prithivida/Splade_PP_en_v1"  # splade-tiny equivalent
    device: "cpu"
    max_length: 256  # Shorter for speed
  semantic:
    enabled: false
```

---

## 7. Environment Variable Overrides

For quick switching without config file changes:

| Variable | Values | Default |
|----------|--------|---------|
| `PLM_SPARSE_RETRIEVER` | `bm25`, `splade` | `bm25` |
| `PLM_SPLADE_MODEL` | Model name/path | `naver/splade-cocondenser-ensembledistil` |
| `PLM_SPLADE_DEVICE` | `cpu`, `cuda` | `cpu` |
| `PLM_SEMANTIC_ENABLED` | `true`, `false` | `true` |

**Precedence**: Environment variables > Config file > Defaults

---

## 8. Backward Compatibility

### 8.1 API Compatibility

| Method | Change |
|--------|--------|
| `HybridRetriever.__init__` | New optional `config` parameter |
| `HybridRetriever.retrieve` | No change |
| `HybridRetriever.ingest_document` | No change (internal index type changes) |
| `BM25Index` | Kept as-is, wrapped by `BM25Retriever` |

### 8.2 Index Compatibility

- Existing BM25 indexes continue to work
- SPLADE requires new index build
- Both can coexist in same index directory

### 8.3 Default Behavior

With no config changes, system behaves exactly as current production:
- BM25 + Semantic + RRF
- All existing tests pass

---

## 9. Implementation Phases

### Phase 1: Interface & BM25 Wrapper (1-2 days)

1. Define `SparseRetriever` abstract base class
2. Create `BM25Retriever` wrapping existing `BM25Index`
3. Create `RetrieverFactory`
4. Modify `HybridRetriever.__init__` to use factory
5. Ensure all existing tests pass

**Deliverable**: Refactored code with same behavior

### Phase 2: SPLADE Retriever (2-3 days)

1. Create `SPLADERetriever` class
2. Implement index building (document encoding)
3. Implement search (query encoding + scoring)
4. Implement save/load
5. Add SPLADE-specific tests

**Deliverable**: SPLADE retriever working standalone

### Phase 3: Integration (1-2 days)

1. Integrate `SPLADERetriever` into `HybridRetriever`
2. Add configuration parsing
3. Add environment variable support
4. Handle semantic disable mode
5. Integration tests for all modes

**Deliverable**: Full switchable system

### Phase 4: CLI & Documentation (1 day)

1. Add `plm-index rebuild` command
2. Update configuration documentation
3. Add migration guide
4. Performance benchmarks in docs

**Deliverable**: User-ready feature

---

## 10. File Structure Changes

### New Files

```
src/plm/search/
├── components/
│   ├── sparse/                    # New directory
│   │   ├── __init__.py
│   │   ├── base.py                # SparseRetriever ABC
│   │   ├── bm25_retriever.py      # BM25Retriever (wraps BM25Index)
│   │   └── splade_retriever.py    # SPLADERetriever
│   └── ...
├── config.py                      # Configuration loading
└── ...
```

### Modified Files

| File | Changes |
|------|---------|
| `retriever.py` | Use `SparseRetriever` instead of `BM25Index` |
| `service/cli.py` | Add `rebuild` command |
| `service/app.py` | Load config on startup |

### Unchanged Files

| File | Reason |
|------|--------|
| `components/bm25.py` | Kept as-is, wrapped by `BM25Retriever` |
| `components/embedder.py` | No changes needed |
| `components/enricher.py` | No changes needed |
| `storage/sqlite.py` | No changes needed |

---

## 11. Testing Strategy

### Unit Tests

| Test | Purpose |
|------|---------|
| `test_bm25_retriever.py` | BM25Retriever matches BM25Index behavior |
| `test_splade_retriever.py` | SPLADE encoding/search works |
| `test_retriever_factory.py` | Factory creates correct retriever |
| `test_config.py` | Config parsing and validation |

### Integration Tests

| Test | Purpose |
|------|---------|
| `test_hybrid_bm25.py` | Current behavior preserved |
| `test_hybrid_splade.py` | SPLADE + Semantic works |
| `test_splade_only.py` | SPLADE-only mode works |
| `test_mode_switching.py` | Can switch modes via config |

### Regression Tests

| Test | Purpose |
|------|---------|
| `test_benchmark_parity.py` | MRR matches POC results |
| `test_latency.py` | Latency within bounds |

---

## 12. Rollback Plan

If SPLADE causes issues in production:

1. Set `PLM_SPARSE_RETRIEVER=bm25` (instant rollback)
2. Or update config file and restart
3. BM25 index still available (not deleted)
4. No data loss, no reindexing needed

---

## 13. Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| All existing tests pass | 100% |
| Mode switching works | Config change only, no code changes |
| BM25 mode MRR | Same as current |
| SPLADE mode MRR | Match POC results (±2%) |
| Rollback time | < 1 minute |

---

## 14. Open Questions

1. **SPLADE model caching**: Should we keep model in memory between queries, or load on demand?
   - Recommendation: Keep in memory (load once at startup)

2. **Async indexing**: Should SPLADE indexing be async/background?
   - Recommendation: Start with sync, add async later if needed

3. **GPU support**: How to handle GPU availability detection?
   - Recommendation: Default to CPU, allow `device: "cuda"` in config

4. **Query expansion with SPLADE**: Should we skip query expansion when using SPLADE?
   - Recommendation: Test both, SPLADE may already handle expansion

---

## 15. References

- Current retriever: `src/plm/search/retriever.py`
- SPLADE POC: `poc/splade_benchmark/`
- SPLADE results: `poc/splade_benchmark/RESULTS.md`
- Informed improvements plan: `.sisyphus/plans/informed-query-improvements.md`

---

*Document version: 1.0*
*Created: 2026-02-22*
