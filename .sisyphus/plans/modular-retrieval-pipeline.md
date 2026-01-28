# Modular Retrieval Pipeline Architecture

## TL;DR

> **Quick Summary**: Build a fine-grained, composable retrieval pipeline with maximum granularity where each operation (query rewriting, keyword extraction, BM25 scoring, etc.) is a separate, swappable component that can be assembled in any order via a fluent API.
> 
> **Deliverables**: 
> - Base pipeline abstractions and type system
> - 10+ fine-grained components (QueryRewriter, BM25Scorer, RRFFuser, etc.)
> - Pipeline builder with fluent API
> - Complete end-to-end benchmark comparison
> 
> **Estimated Effort**: Large (15-20 focused work sessions)
> **Parallel Execution**: NO - sequential implementation
> **Critical Path**: Types → Base → Components → Pipeline → Benchmark

---

## Context

### Original Request
User wants to split retrieval/enrichment/query technologies into puzzle pieces that can be plugged in any order and in any place - a modular, composable architecture.

### Interview Summary
**Key Discussions**:
- **Granularity**: Maximum granularity - each operation as separate component
- **Architecture**: Linear pipeline like Unix pipes (sequential only)
- **Configuration**: Code-based fluent API, not config files
- **State**: Pure functions with immutable objects
- **Error handling**: Fail fast on any component error
- **Caching**: External cache for expensive operations
- **Migration**: Keep both systems (old for production, new for experiments)
- **Priority**: Build full pipeline first for end-to-end functionality

**Research Findings**:
- **Haystack**: Most mature component architecture with explicit contracts
- **RRF standard**: k=60 for fusion, adaptive weights for query expansion
- **Industry patterns**: Protocol contracts, component registry, hash-based caching
- **Current system**: Monolithic strategies with internal coupling

### Metis Review
**Identified Gaps** (addressed):
- **Type design**: Need immutable types with provenance tracking → Added comprehensive type system
- **Embedding cache**: Currently uncached (28% of time) → Added caching component
- **Pure functions**: Mixed I/O and logic → Strict separation enforced
- **Component testing**: No isolation strategy → Each component independently testable
- **Performance baseline**: No metrics → Added benchmark requirements

---

## Work Objectives

### Core Objective
Create a modular retrieval pipeline where each operation is a separate, stateless component that can be composed in any order through a fluent API. Components follow Unix pipe philosophy: output of one component becomes input to the next, with immutable data flowing through the pipeline.

### Concrete Deliverables
- `poc/modular_retrieval_pipeline/` - New POC directory for experimental pipeline
- Immutable type system with provenance tracking
- 10+ fine-grained components (one per operation)
- Linear pipeline builder with fluent API
- External cache integration
- End-to-end benchmark against existing system

### Definition of Done
- [ ] Pipeline achieves comparable accuracy to enriched_hybrid_llm (90% baseline) - PARTIAL: Components work, full integration needed
- [x] All components are pure functions (no side effects)
- [x] Each component is independently testable
- [x] Fluent API allows arbitrary component ordering
- [x] Performance metrics documented

### Must Have
- Maximum granularity (separate component for each operation)
- Immutable data objects with frozen dataclasses
- Sequential execution only (no parallelism)
- Fail-fast error handling
- External caching for expensive operations

### Must NOT Have (Guardrails)
- No mutable state in components
- No implicit dependencies between components
- No configuration files (code-only)
- No modifications to existing enriched_hybrid_llm
- No runtime component swapping (static pipeline)
- No parallel execution paths

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest in POCs)
- **User wants tests**: Manual-only (focus on architecture first)
- **Framework**: pytest

### Automated Verification Only

Each component and pipeline must be verifiable through automated commands:

**Component Testing**:
```bash
# Test individual components in isolation
python -m pytest poc/modular_retrieval_pipeline/tests/components/test_query_rewriter.py
# Assert: Component returns expected output for known input
```

**Pipeline Testing**:
```bash
# Test full pipeline execution
python poc/modular_retrieval_pipeline/test_pipeline.py
# Assert: Pipeline processes test queries successfully
```

**Benchmark Verification**:
```bash
# Compare against existing system
python poc/modular_retrieval_pipeline/benchmark.py --baseline enriched_hybrid_llm
# Assert: Accuracy within 5% of baseline (85% minimum)
```

---

## Execution Strategy

### Sequential Implementation
Since parallelism is explicitly excluded, all tasks execute sequentially.

### Dependency Chain
```
Types (1) → Base Abstractions (2) → Cache (3) → Components (4-13) → Pipeline (14) → Benchmark (15)
```

### Critical Path
Task 1 → Task 2 → Tasks 4-13 → Task 14 → Task 15

---

## TODOs

- [x] 1. Define Immutable Type System

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/types.py`
  - Define frozen dataclasses for Query, RewrittenQuery, ExpandedQuery, EmbeddedQuery
  - Define ScoredChunk with provenance tracking
  - Define FusionConfig for RRF parameters
  - Define PipelineResult with full transformation history

  **Must NOT do**:
  - Use mutable types (lists, dicts as attributes)
  - Allow attribute modification after creation
  - Create circular dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure type definitions, no complex logic
  - **Skills**: []
    - No special skills needed for type definitions

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: All other tasks
  - **Blocked By**: None (can start immediately)

  **References**:
  - `poc/chunking_benchmark_v2/strategies/base.py:Chunk` - Existing Chunk type to understand fields
  - `poc/chunking_benchmark_v2/strategies/base.py:Document` - Document type structure
  - `poc/chunking_benchmark_v2/enrichment/__init__.py:EnrichmentResult` - Pattern for result types
  - Haystack docs: `https://docs.haystack.deepset.ai/docs/data-classes` - Industry standard for data classes

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "from poc.modular_retrieval_pipeline.types import Query; q = Query('test'); q.text = 'modified'"
  # Assert: AttributeError (frozen dataclass)
  
  python -c "from poc.modular_retrieval_pipeline.types import ScoredChunk; import dataclasses; print(dataclasses.is_dataclass(ScoredChunk) and ScoredChunk.__dataclass_fields__['chunk_id'].frozen)"
  # Assert: Output is "True"
  ```

  **Commit**: YES
  - Message: `feat(modular): define immutable type system for pipeline`
  - Files: `poc/modular_retrieval_pipeline/types.py`

---

- [x] 2. Create Base Pipeline Abstractions

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/base.py`
  - Define Component protocol with process() method
  - Define Pipeline class with add() method for chaining
  - **Unix pipe data flow**: output of component N automatically becomes input to component N+1
  - Example flow: `data = c1.process(data); data = c2.process(data); data = c3.process(data)`
  - Components return new immutable objects (never modify input)
  - Implement error propagation (fail-fast)
  - Add type validation between components

  **Must NOT do**:
  - Allow runtime component modification
  - Create stateful components
  - Implement parallel execution

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Core architecture design but straightforward patterns
  - **Skills**: []
    - Base Python patterns only

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 3-15
  - **Blocked By**: Task 1

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/base.py:RetrievalStrategy` - Abstract base pattern
  - Haystack component pattern: `https://docs.haystack.deepset.ai/docs/custom-components` - Component contract design
  - `poc/modular_retrieval_pipeline/types.py` - Type definitions from Task 1

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.base import Pipeline, Component
class TestComponent(Component):
    def process(self, data): return data.upper()
p = Pipeline().add(TestComponent())
result = p.run('hello')
print(result)
"
  # Assert: Output is "HELLO"
  ```

  **Commit**: YES
  - Message: `feat(modular): implement base pipeline abstractions`
  - Files: `poc/modular_retrieval_pipeline/base.py`

---

- [x] 3. Implement Cache Integration

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/cache.py`
  - Implement CacheableComponent wrapper
  - Add hash-based cache key generation
  - Integrate with existing EnrichmentCache
  - Support both disk and Redis backends

  **Must NOT do**:
  - Implement caching inside components
  - Use mutable objects as cache keys
  - Create new cache implementation (reuse existing)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wrapper pattern around existing cache
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Components that need caching (Tasks 6, 7)
  - **Blocked By**: Task 2

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/cache.py:EnrichmentCache` - Existing cache to wrap
  - `poc/modular_retrieval_pipeline/base.py` - Component interface from Task 2

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.cache import CacheableComponent
class Slow:
    def process(self, data):
        import time
        time.sleep(0.1)
        return data.upper()
cached = CacheableComponent(Slow(), 'test_cache')
import time
t0 = time.time()
cached.process('hello')
t1 = time.time()
cached.process('hello')  # Should be cached
t2 = time.time()
print(f'First: {t1-t0:.3f}s, Second: {t2-t1:.3f}s')
"
  # Assert: Second call is <0.01s (cached)
  ```

  **Commit**: YES
  - Message: `feat(modular): add cache integration for expensive operations`
  - Files: `poc/modular_retrieval_pipeline/cache.py`

---

- [x] 4. Implement QueryRewriter Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/query_rewriter.py`
  - Wrap existing rewrite_query function from enriched_hybrid_llm
  - Accept Query, return RewrittenQuery
  - Handle LLM timeout (5s default)
  - Pure function interface

  **Must NOT do**:
  - Store LLM client as instance variable
  - Modify input Query object
  - Implement new rewriting logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple wrapper around existing function
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py:rewrite_query` - Function to wrap
  - `poc/modular_retrieval_pipeline/types.py:Query,RewrittenQuery` - Input/output types
  - `poc/modular_retrieval_pipeline/base.py:Component` - Interface to implement

  **Acceptance Criteria**:
  ```bash
  # Agent runs (mock LLM call):
  python -c "
from poc.modular_retrieval_pipeline.components.query_rewriter import QueryRewriter
from poc.modular_retrieval_pipeline.types import Query
from unittest.mock import patch
with patch('poc.modular_retrieval_pipeline.components.query_rewriter.rewrite_query', return_value='token expiration TTL'):
    rewriter = QueryRewriter(timeout=5.0)
    query = Query('why does my token expire')
    result = rewriter.process(query)
    print(f'Original: {result.original.text}, Rewritten: {result.rewritten}')
"
  # Assert: Shows original and rewritten queries
  ```

  **Commit**: YES
  - Message: `feat(modular): implement QueryRewriter component`
  - Files: `poc/modular_retrieval_pipeline/components/query_rewriter.py`

---

- [x] 5. Implement QueryExpander Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/query_expander.py`
  - Port expand_query logic with DOMAIN_EXPANSIONS dict
  - Accept RewrittenQuery, return ExpandedQuery
  - Track which expansions were applied
  - Pure function, no state

  **Must NOT do**:
  - Modify DOMAIN_EXPANSIONS at runtime
  - Store expansion history as mutable state

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Port existing logic with minor adaptations
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:20-78` - DOMAIN_EXPANSIONS and expand_query
  - `poc/modular_retrieval_pipeline/types.py:RewrittenQuery,ExpandedQuery` - Input/output types

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.components.query_expander import QueryExpander
from poc.modular_retrieval_pipeline.types import Query, RewrittenQuery
expander = QueryExpander()
rq = RewrittenQuery(Query('token authentication'), 'token auth', 'mock')
result = expander.process(rq)
print(f'Expanded: {result.expanded}')
print(f'Applied: {result.expansions_applied}')
"
  # Assert: Shows JWT-related expansions applied
  ```

  **Commit**: YES
  - Message: `feat(modular): implement QueryExpander with domain terms`
  - Files: `poc/modular_retrieval_pipeline/components/query_expander.py`

---

- [x] 6. Implement KeywordExtractor Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/keyword_extractor.py`
  - Wrap YAKE keyword extraction from FastEnricher
  - **Input**: data dict with 'content' field
  - **Output**: new dict with added 'keywords' field (preserves all input fields)
  - Support caching via CacheableComponent
  - Extract top 10 keywords by default

  **Must NOT do**:
  - Initialize YAKE in __init__ (keep stateless)
  - Modify chunk objects directly

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wrapper around YAKE functionality
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Tasks 2, 3

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:FastEnricher` - YAKE implementation
  - `poc/modular_retrieval_pipeline/cache.py:CacheableComponent` - For caching

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.components.keyword_extractor import KeywordExtractor
extractor = KeywordExtractor(max_keywords=5)
content = 'Kubernetes horizontal pod autoscaler scales replicas based on CPU utilization metrics'
result = extractor.process(content)
print(f'Keywords found: {len(result.keywords)}')
print(f'Sample keywords: {result.keywords[:3]}')
"
  # Assert: Returns 5 keywords including technical terms
  ```

  **Commit**: YES
  - Message: `feat(modular): add YAKE keyword extraction component`
  - Files: `poc/modular_retrieval_pipeline/components/keyword_extractor.py`

---

- [x] 7. Implement EntityExtractor Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/entity_extractor.py`
  - Wrap spaCy NER from FastEnricher
  - **Input**: data dict (may already have 'keywords' from previous component)
  - **Output**: new dict with added 'entities' field (preserves all input fields including keywords)
  - Support caching via CacheableComponent
  - Extract ORG, PRODUCT, PERSON, TECH entities

  **Must NOT do**:
  - Load spaCy model in __init__ (keep stateless)
  - Combine with keyword extraction (separate concerns)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wrapper around spaCy NER
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Tasks 2, 3

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:FastEnricher._extract_entities` - spaCy NER logic
  - `poc/modular_retrieval_pipeline/cache.py:CacheableComponent` - For caching

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.components.entity_extractor import EntityExtractor
extractor = EntityExtractor()
content = 'Google Cloud Platform offers Kubernetes Engine for container orchestration'
result = extractor.process(content)
print(f'Entities: {result.entities}')
"
  # Assert: Finds "Google Cloud Platform" as ORG, "Kubernetes Engine" as PRODUCT
  ```

  **Commit**: YES
  - Message: `feat(modular): add spaCy entity extraction component`
  - Files: `poc/modular_retrieval_pipeline/components/entity_extractor.py`

---

- [x] 8. Implement ContentEnricher Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/content_enricher.py`
  - **Input**: data dict with 'content', 'keywords', 'entities' (from previous components)
  - **Output**: enriched string: "keywords | entities\n\noriginal_content"
  - Pure string formatting using accumulated data from pipeline
  - Example: YAKE → spaCy → ContentEnricher (receives both keywords AND entities)

  **Must NOT do**:
  - Call keyword/entity extraction (separate components)
  - Store enrichment format as mutable state

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple string formatting
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:FastEnricher.enrich` - Formatting pattern

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.components.content_enricher import ContentEnricher
enricher = ContentEnricher()
data = {
    'content': 'Original text here',
    'keywords': ['key1', 'key2'],
    'entities': {'ORG': ['Company']}
}
result = enricher.process(data)
print(result)
"
  # Assert: Output shows "key1, key2 | Company\n\nOriginal text here"
  ```

  **Commit**: YES
  - Message: `feat(modular): add content enrichment formatter`
  - Files: `poc/modular_retrieval_pipeline/components/content_enricher.py`

---

- [x] 9. Implement BM25Scorer Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/bm25_scorer.py`
  - Wrap rank_bm25.BM25Okapi scoring
  - Accept query and chunks, return scored chunks
  - Build index in process() (stateless)
  - Return ScoredChunk objects with BM25 scores

  **Must NOT do**:
  - Store BM25 index as instance state
  - Modify chunk objects

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wrapper around BM25 library
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:BM25Okapi` - BM25 usage pattern
  - `poc/modular_retrieval_pipeline/types.py:ScoredChunk` - Output type

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.components.bm25_scorer import BM25Scorer
scorer = BM25Scorer()
chunks = ['kubernetes pod', 'docker container', 'kubernetes deployment']
results = scorer.process({'query': 'kubernetes', 'chunks': chunks})
print(f'Top result: {results[0].chunk_id} with score {results[0].score:.2f}')
"
  # Assert: Returns scored chunks with "kubernetes" chunks scoring higher
  ```

  **Commit**: YES
  - Message: `feat(modular): implement BM25 scoring component`
  - Files: `poc/modular_retrieval_pipeline/components/bm25_scorer.py`

---

- [x] 10. Implement EmbeddingEncoder Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/embedding_encoder.py`
  - Wrap sentence-transformers encoding
  - Accept text, return embedding vector
  - Support batch encoding
  - Cache embeddings by content hash

  **Must NOT do**:
  - Load model in __init__ (use lazy loading)
  - Store embeddings as instance state

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wrapper around embedder
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Tasks 2, 3

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/base.py:EmbedderMixin` - Embedding pattern
  - BGE model docs: `https://huggingface.co/BAAI/bge-base-en-v1.5`

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.components.embedding_encoder import EmbeddingEncoder
encoder = EmbeddingEncoder(model='BAAI/bge-base-en-v1.5')
result = encoder.process('test text')
print(f'Embedding shape: {len(result.embedding)}')
print(f'First 3 values: {result.embedding[:3]}')
"
  # Assert: Returns 768-dimensional embedding
  ```

  **Commit**: YES
  - Message: `feat(modular): add embedding encoder component`
  - Files: `poc/modular_retrieval_pipeline/components/embedding_encoder.py`

---

- [x] 11. Implement SimilarityScorer Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/similarity_scorer.py`
  - Compute cosine similarity between query and chunk embeddings
  - Accept embeddings, return ScoredChunk objects
  - Use numpy for efficient computation
  - Pure mathematical function

  **Must NOT do**:
  - Compute embeddings (separate component)
  - Use approximate methods (exact similarity only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple cosine similarity calculation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/semantic.py` - Cosine similarity pattern
  - NumPy docs for dot product: `https://numpy.org/doc/stable/reference/generated/numpy.dot.html`

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.components.similarity_scorer import SimilarityScorer
import numpy as np
scorer = SimilarityScorer()
query_emb = np.array([1.0, 0.0, 0.0])
chunk_embs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
results = scorer.process({'query': query_emb, 'chunks': chunk_embs})
print(f'Similarity scores: {[r.score for r in results]}')
"
  # Assert: First chunk has similarity 1.0, second has 0.0
  ```

  **Commit**: YES
  - Message: `feat(modular): implement cosine similarity scorer`
  - Files: `poc/modular_retrieval_pipeline/components/similarity_scorer.py`

---

- [x] 12. Implement RRFFuser Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/rrf_fuser.py`
  - Port reciprocal_rank_fusion from gem_utils
  - Accept multiple scored chunk lists, return fused list
  - Support configurable k parameter (default 60)
  - Support weighted fusion

  **Must NOT do**:
  - Modify input score lists
  - Store fusion history as state

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Port existing RRF algorithm
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Task 2

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/gem_utils.py:reciprocal_rank_fusion` - RRF implementation
  - `poc/modular_retrieval_pipeline/types.py:FusionConfig,ScoredChunk` - Config and types

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline.components.rrf_fuser import RRFFuser
from poc.modular_retrieval_pipeline.types import ScoredChunk, FusionConfig
fuser = RRFFuser(FusionConfig(rrf_k=60))
list1 = [ScoredChunk('a', 10.0, 'bm25', 1), ScoredChunk('b', 8.0, 'bm25', 2)]
list2 = [ScoredChunk('b', 0.9, 'semantic', 1), ScoredChunk('a', 0.7, 'semantic', 2)]
result = fuser.process([list1, list2])
print(f'Fused top: {result[0].chunk_id} with score {result[0].score:.3f}')
"
  # Assert: Returns fused ranking with RRF scores
  ```

  **Commit**: YES
  - Message: `feat(modular): add RRF fusion component`
  - Files: `poc/modular_retrieval_pipeline/components/rrf_fuser.py`

---

- [x] 13. Implement Reranker Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/reranker.py`
  - Wrap cross-encoder reranking model
  - Accept query and chunks, return reranked chunks
  - Support different reranker models
  - Cache reranking scores

  **Must NOT do**:
  - Load model in __init__
  - Modify original ranking

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wrapper around reranker model
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Tasks 2, 3

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/base.py:RerankerMixin` - Reranking pattern
  - Cross-encoder docs: `https://www.sbert.net/docs/cross_encoder/usage.html`

  **Acceptance Criteria**:
  ```bash
  # Agent runs (with mock):
  python -c "
from poc.modular_retrieval_pipeline.components.reranker import Reranker
from unittest.mock import MagicMock
reranker = Reranker(model='cross-encoder/ms-marco-MiniLM-L-6-v2')
# Mock the model to avoid download
reranker._get_model = MagicMock(return_value=MagicMock(predict=lambda x: [0.9, 0.3]))
chunks = ['relevant text', 'irrelevant text']
result = reranker.process({'query': 'test query', 'chunks': chunks})
print(f'Reranked: {result}')
"
  # Assert: Returns reranked chunks based on scores
  ```

  **Commit**: YES
  - Message: `feat(modular): implement cross-encoder reranker`
  - Files: `poc/modular_retrieval_pipeline/components/reranker.py`

---

- [x] 14. Build Complete Pipeline with Fluent API

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/pipeline_builder.py`
  - Implement fluent API for pipeline construction
  - Add type checking between components
  - Create preset pipelines (semantic_only, hybrid, enriched_hybrid)
  - Add execution with fail-fast error handling

  **Must NOT do**:
  - Allow runtime component modification
  - Implement parallel branches
  - Use configuration files

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core integration of all components with type safety
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 15
  - **Blocked By**: Tasks 1-13

  **References**:
  - `poc/modular_retrieval_pipeline/base.py` - Pipeline base class
  - `poc/modular_retrieval_pipeline/components/` - All component implementations
  - Haystack pipeline builder: `https://docs.haystack.deepset.ai/docs/pipelines` - Fluent API patterns

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "
from poc.modular_retrieval_pipeline import PipelineBuilder

# Example 1: Query processing pipeline
query_pipeline = (PipelineBuilder()
    .add_query_rewriter()    # Output: RewrittenQuery
    .add_query_expander()     # Input: RewrittenQuery, Output: ExpandedQuery
    .build())

# Example 2: Multiple enrichers (Unix pipe style accumulation)
enrichment_pipeline = (PipelineBuilder()
    .add_keyword_extractor()  # Output: {content: '...', keywords: [...]}
    .add_entity_extractor()   # Output: {content: '...', keywords: [...], entities: {...}}
    .add_content_enricher()   # Output: 'keywords | entities\n\ncontent'
    .build())

# Example 3: Full retrieval pipeline
full_pipeline = (PipelineBuilder()
    .add_query_rewriter()
    .add_query_expander()
    .add_bm25_scorer()
    .add_similarity_scorer()
    .add_rrf_fuser()
    .build())

result = full_pipeline.run('test query', chunks=['chunk1', 'chunk2'])
print(f'Pipeline result: {result}')
"
  # Assert: Data flows through pipeline with each component's output becoming next component's input
  ```

  **Commit**: YES
  - Message: `feat(modular): implement fluent pipeline builder API`
  - Files: `poc/modular_retrieval_pipeline/pipeline_builder.py`

---

- [x] 15. Benchmark Against Existing System

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/benchmark.py`
  - Load needle-in-haystack test questions
  - Run both enriched_hybrid_llm and new pipeline
  - Compare accuracy, latency, and memory usage
  - Generate comparison report

  **Must NOT do**:
  - Modify existing benchmark code
  - Change test questions
  - Skip performance metrics

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Straightforward benchmark comparison
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None (final task)
  - **Blocked By**: Task 14

  **References**:
  - `poc/chunking_benchmark_v2/benchmark_needle_haystack.py` - Benchmark framework
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Test questions
  - `poc/modular_retrieval_pipeline/pipeline_builder.py` - New pipeline

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python poc/modular_retrieval_pipeline/benchmark.py --questions poc/chunking_benchmark_v2/corpus/needle_questions.json
  # Assert: Generates report showing:
  # - Baseline (enriched_hybrid_llm): 90% accuracy
  # - Modular pipeline: ≥85% accuracy
  # - Latency comparison
  # - Memory usage comparison
  ```

  **Commit**: YES
  - Message: `feat(modular): add benchmark comparison with existing system`
  - Files: `poc/modular_retrieval_pipeline/benchmark.py`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(modular): define immutable type system for pipeline` | types.py |
| 2 | `feat(modular): implement base pipeline abstractions` | base.py |
| 3 | `feat(modular): add cache integration for expensive operations` | cache.py |
| 4-13 | `feat(modular): implement [Component] component` | components/*.py |
| 14 | `feat(modular): implement fluent pipeline builder API` | pipeline_builder.py |
| 15 | `feat(modular): add benchmark comparison with existing system` | benchmark.py |

---

## Success Criteria

### Verification Commands
```bash
# Run all component tests
python -m pytest poc/modular_retrieval_pipeline/tests/

# Run integration test
python poc/modular_retrieval_pipeline/test_pipeline.py

# Run benchmark comparison
python poc/modular_retrieval_pipeline/benchmark.py --baseline enriched_hybrid_llm
```

### Final Checklist
- [x] All components are pure functions (no side effects)
- [ ] Pipeline achieves ≥85% accuracy on baseline test - PARTIAL: Baseline verified, modular integration needed
- [x] Each component independently testable
- [x] Fluent API allows arbitrary ordering
- [x] External cache reduces redundant computation
- [x] No modifications to existing system