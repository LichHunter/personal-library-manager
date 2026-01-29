# Draft: Modular Pipeline Fixes

## Requirements (confirmed)

### Goal
Make the modular pipeline benchmark produce **identical results** to enriched_hybrid_llm by:
1. Fixing component configurations to match original
2. Implementing missing retrieval pipeline components
3. Creating end-to-end retrieval that replicates the exact algorithm

### User Decision
- Full replication of enriched_hybrid_llm behavior
- Must pass accuracy comparison benchmark

## Research Findings

### From Explore Agent 1 (Architecture):
- Component protocol: `process(InputT) -> OutputT` - stateless, immutable
- Pipeline builder: Fluent API with type validation
- Type system: Query chain (Query → RewrittenQuery → ExpandedQuery → ScoredChunk)
- Caching: CacheableComponent wrapper with hash-based keys

### From Explore Agent 2 (enriched_hybrid_llm):
- **INDEX PHASE**: Enrich chunks → Embed → BM25 index
- **RETRIEVE PHASE**: Rewrite → Expand → Score (BM25 + semantic) → Adaptive RRF fusion
- **ADAPTIVE WEIGHTING**: When expansion triggered: bm25=3.0, sem=0.3, rrf_k=10
- **CANDIDATE MULTIPLIER**: 10x normal, 20x when expanded

### From Explore Agent 3 (Existing Components):
- BM25Scorer: Works but expects raw chunks (not enriched)
- EmbeddingEncoder: Works correctly (BGE-base-en-v1.5)
- SimilarityScorer: Works but returns empty content
- RRFFuser: **CRITICAL GAP** - No adaptive weighting
- Reranker: Not used in enriched_hybrid_llm

## Technical Decisions

### Architecture Approach
**Decision**: Create new types and components rather than modifying existing ones
- Keeps existing components working for other use cases
- New `IndexedChunkStore` type for stateful index
- New `AdaptiveRRFFuser` component for adaptive weighting
- New `ModularRetriever` orchestrator component

### Component Design
1. Fix existing components where configuration is wrong
2. Create new types for index storage
3. Create new orchestrator component for retrieval

## Open Questions
- None - all requirements are clear from analysis

## Scope Boundaries

### INCLUDE:
- Fix KeywordExtractor YAKE config
- Fix EntityExtractor entity types
- Fix ContentEnricher format
- Create IndexedChunkStore type
- Create ChunkIndexer component
- Create AdaptiveRRFFuser component
- Create ModularRetriever orchestrator
- Update benchmark to use full pipeline
- Verify accuracy matches baseline

### EXCLUDE:
- Modifying enriched_hybrid_llm.py
- Adding new retrieval features not in original
- Cross-encoder reranking (not in original)

---

## Detailed Gap Analysis

### 1. KeywordExtractor (YAKE Config)

| Parameter | Original (FastEnricher) | Current Modular | Fix |
|-----------|------------------------|-----------------|-----|
| `n` | 2 (bigrams max) | 3 (trigrams max) | Change to 2 |
| `dedupFunc` | "seqm" | None (default) | Add "seqm" |
| `windowsSize` | 1 | None (default) | Add 1 |
| `top` | 10 | 10 | OK |
| `dedupLim` | 0.9 | 0.9 | OK |

### 2. EntityExtractor (Entity Types)

| Original (FastEnricher) | Current Modular |
|------------------------|-----------------|
| ORG | ORG |
| PRODUCT | PRODUCT |
| GPE | MISSING |
| PERSON | PERSON |
| WORK_OF_ART | MISSING |
| LAW | MISSING |
| EVENT | MISSING |
| FAC | MISSING |
| NORP | MISSING |
| - | TECH (WRONG) |

### 3. ContentEnricher (Format)

| Aspect | Original (FastEnricher) | Current Modular |
|--------|------------------------|-----------------|
| Keywords | First 7 only | All keywords |
| Entities | First 2 per type, 5 total | All entities |
| Entity format | No type labels | With type labels |
| Separator | ` \| ` | ` \| ` |
| Example | `"k1, k2 \| e1, e2\n\ncontent"` | `"k1, k2 \| ORG: e1\n\ncontent"` |

### 4. RRFFuser (Adaptive Weighting)

| Parameter | Normal | Expansion Triggered |
|-----------|--------|---------------------|
| bm25_weight | 1.0 | 3.0 |
| sem_weight | 1.0 | 0.3 |
| rrf_k | 60 | 10 |
| multiplier | 10x | 20x |

**Current modular**: Fixed weights from config, no adaptive logic

### 5. Missing: Index Storage

enriched_hybrid_llm stores:
- `self.chunks: list[Chunk]` - Original chunks
- `self.embeddings: np.ndarray` - Embeddings of enriched content
- `self.bm25: BM25Okapi` - BM25 index of enriched content
- `self._enriched_contents: list[str]` - Enriched content strings

**Modular pipeline**: Stateless components, no index storage

### 6. Missing: Retrieval Orchestration

enriched_hybrid_llm retrieve() does:
1. Query rewriting (LLM)
2. Query expansion (domain terms)
3. Adaptive weight selection
4. Semantic scoring (dot product)
5. BM25 scoring
6. RRF fusion
7. Return top-k

**Modular pipeline**: Components exist but not wired together

---

## Implementation Plan

### Task List

1. **Fix KeywordExtractor YAKE config** - Quick fix
2. **Fix EntityExtractor entity types** - Quick fix
3. **Fix ContentEnricher format** - Medium (logic change)
4. **Create IndexedChunkStore type** - New type for index storage
5. **Create ChunkIndexer component** - New component for indexing
6. **Create AdaptiveRRFFuser component** - New component with adaptive weighting
7. **Create ModularRetriever orchestrator** - New component for end-to-end retrieval
8. **Update benchmark** - Wire everything together
9. **Verify accuracy** - Run benchmark and compare

### Estimated Effort
- Config fixes (1-3): ~30 min each
- New types/components (4-7): ~1-2 hours each
- Integration (8-9): ~1-2 hours
- **Total**: ~8-12 hours
