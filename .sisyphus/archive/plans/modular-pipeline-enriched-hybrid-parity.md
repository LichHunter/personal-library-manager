# Modular Pipeline: enriched_hybrid_llm Parity

## TL;DR

> **Quick Summary**: Fix component configurations and implement missing retrieval pipeline to make modular benchmark produce identical results to enriched_hybrid_llm (90% baseline accuracy).
> 
> **Deliverables**: 
> - Fixed KeywordExtractor, EntityExtractor, ContentEnricher configs
> - New IndexedChunkStore type for stateful index storage
> - New ChunkIndexer component for indexing workflow
> - New AdaptiveFusionConfig type and logic
> - New ModularRetriever orchestrator component
> - Updated benchmark with full retrieval pipeline
> 
> **Estimated Effort**: Large (9 tasks, ~8-12 hours)
> **Parallel Execution**: Partial - config fixes can parallel, new components sequential
> **Critical Path**: Config Fixes → Types → ChunkIndexer → ModularRetriever → Benchmark

---

## Context

### Original Request
Make the modular pipeline benchmark replicate enriched_hybrid_llm exactly so accuracy comparison is valid.

### Analysis Summary
Current modular benchmark only demonstrates components but lacks:
1. Correct enrichment configs (YAKE, entity types, format)
2. Chunk indexing workflow (enrich → embed → BM25 index)
3. Adaptive weighting based on query expansion
4. End-to-end retrieval orchestration

### Research Findings
- **enriched_hybrid_llm algorithm** fully documented in ANALYSIS_enriched_hybrid_vs_modular.md
- **Component architecture** well-established (stateless, immutable, fluent API)
- **Type system** supports transformation chain
- **Critical gap**: No mechanism for index storage or adaptive weighting

---

## Work Objectives

### Core Objective
Achieve functional parity between modular pipeline and enriched_hybrid_llm so benchmark accuracy comparison is valid and meaningful.

### Concrete Deliverables
- `components/keyword_extractor.py` - Fixed YAKE config
- `components/entity_extractor.py` - Fixed entity types
- `components/content_enricher.py` - Fixed format (7 keywords, 5 entities, no labels)
- `types.py` - New `IndexedChunkStore` type
- `components/chunk_indexer.py` - New indexing component
- `components/adaptive_rrf_fuser.py` - New fusion component with adaptive weighting
- `modular_retriever.py` - New orchestrator for end-to-end retrieval
- `benchmark.py` - Updated to use full pipeline

### Definition of Done
- [ ] Modular pipeline achieves ≥85% accuracy on needle-haystack benchmark
- [ ] Accuracy within 5% of enriched_hybrid_llm baseline (90%)
- [ ] All components match original configs exactly
- [ ] Benchmark runs end-to-end without errors

### Must Have
- Exact YAKE config match (n=2, dedupFunc="seqm", windowsSize=1)
- Exact entity types match (9 types from FastEnricher)
- Exact content format match (7 keywords, 5 entities, no type labels)
- Adaptive weighting (bm25=3.0, sem=0.3, rrf_k=10 when expanded)
- Candidate multiplier (10x normal, 20x when expanded)

### Must NOT Have (Guardrails)
- Do NOT modify enriched_hybrid_llm.py
- Do NOT add features not in original (no cross-encoder reranking)
- Do NOT change existing component interfaces (add new components instead)
- Do NOT break existing tests

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (test files exist)
- **User wants tests**: Manual verification via benchmark
- **Framework**: pytest + benchmark comparison

### Automated Verification

**Component Testing**:
```bash
# Verify YAKE config
python -c "
from poc.modular_retrieval_pipeline.components.keyword_extractor import KeywordExtractor
import yake
# Verify config matches FastEnricher
"

# Verify entity types
python -c "
from poc.modular_retrieval_pipeline.components.entity_extractor import EntityExtractor
ext = EntityExtractor()
assert ext.entity_types == {'ORG', 'PRODUCT', 'GPE', 'PERSON', 'WORK_OF_ART', 'LAW', 'EVENT', 'FAC', 'NORP'}
print('Entity types correct')
"

# Verify content format
python -c "
from poc.modular_retrieval_pipeline.components.content_enricher import ContentEnricher
enricher = ContentEnricher()
data = {'content': 'test', 'keywords': ['k1','k2','k3','k4','k5','k6','k7','k8'], 'entities': {'ORG': ['e1','e2','e3']}}
result = enricher.process(data)
# Should be: 'k1, k2, k3, k4, k5, k6, k7 | e1, e2\n\ntest'
assert 'ORG:' not in result  # No type labels
print(f'Format: {result[:50]}...')
"
```

**Benchmark Verification**:
```bash
python poc/modular_retrieval_pipeline/benchmark.py --questions poc/chunking_benchmark_v2/corpus/needle_questions.json
# Assert: Modular accuracy ≥85%
# Assert: Within 5% of baseline
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Config Fixes - Can Parallel):
├── Task 1: Fix KeywordExtractor YAKE config
├── Task 2: Fix EntityExtractor entity types
└── Task 3: Fix ContentEnricher format

Wave 2 (Types - After Wave 1):
└── Task 4: Create IndexedChunkStore type

Wave 3 (Components - After Wave 2):
├── Task 5: Create ChunkIndexer component
└── Task 6: Create AdaptiveRRFFuser component (can parallel with 5)

Wave 4 (Orchestration - After Wave 3):
└── Task 7: Create ModularRetriever orchestrator

Wave 5 (Integration - After Wave 4):
├── Task 8: Update benchmark.py
└── Task 9: Verify accuracy
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 5, 7 | 2, 3 |
| 2 | None | 5, 7 | 1, 3 |
| 3 | None | 5, 7 | 1, 2 |
| 4 | None | 5, 6, 7 | None |
| 5 | 1, 2, 3, 4 | 7 | 6 |
| 6 | 4 | 7 | 5 |
| 7 | 5, 6 | 8 | None |
| 8 | 7 | 9 | None |
| 9 | 8 | None | None |

---

## TODOs

- [ ] 1. Fix KeywordExtractor YAKE Config

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/components/keyword_extractor.py`
  - Change YAKE config in `_extract_keywords()` method:
    - `n=2` (was `n=3`)
    - Add `dedupFunc="seqm"`
    - Add `windowsSize=1`
  - Verify against FastEnricher config

  **Must NOT do**:
  - Change max_keywords parameter (keep configurable)
  - Change the Component protocol interface

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, simple config change
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 5, 7
  - **Blocked By**: None

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:25-32` - Original YAKE config
  - `poc/modular_retrieval_pipeline/components/keyword_extractor.py:146-152` - Current config

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.components.keyword_extractor import KeywordExtractor
  ext = KeywordExtractor()
  # Verify by checking the method source or running extraction
  result = ext.process({'content': 'Kubernetes horizontal pod autoscaler scales replicas based on CPU'})
  print(f'Keywords: {result[\"keywords\"]}')
  # Should produce similar keywords to FastEnricher
  "
  ```

  **Commit**: YES (group with 2, 3)
  - Message: `fix(modular): align enrichment component configs with FastEnricher`
  - Files: `keyword_extractor.py`, `entity_extractor.py`, `content_enricher.py`

---

- [ ] 2. Fix EntityExtractor Entity Types

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/components/entity_extractor.py`
  - Change `DEFAULT_ENTITY_TYPES` to match FastEnricher:
    ```python
    DEFAULT_ENTITY_TYPES = {
        "ORG", "PRODUCT", "GPE", "PERSON", 
        "WORK_OF_ART", "LAW", "EVENT", "FAC", "NORP"
    }
    ```
  - Remove "TECH" (not a real spaCy entity type)

  **Must NOT do**:
  - Change the Component protocol interface
  - Remove the ability to customize entity types

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single constant change
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 5, 7
  - **Blocked By**: None

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:84-94` - Original entity types
  - `poc/modular_retrieval_pipeline/components/entity_extractor.py` - Current types

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.components.entity_extractor import EntityExtractor
  ext = EntityExtractor()
  assert ext.entity_types == {'ORG', 'PRODUCT', 'GPE', 'PERSON', 'WORK_OF_ART', 'LAW', 'EVENT', 'FAC', 'NORP'}
  print('Entity types: CORRECT')
  "
  ```

  **Commit**: YES (group with 1, 3)
  - Message: `fix(modular): align enrichment component configs with FastEnricher`
  - Files: Same as Task 1

---

- [ ] 3. Fix ContentEnricher Format

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/components/content_enricher.py`
  - Modify `_format_enriched_content()` to match FastEnricher format:
    - Keywords: First 7 only (not all)
    - Entities: First 2 per type, max 5 total (not all)
    - NO entity type labels (currently has "ORG: entity" format)
  - Target format: `"keyword1, keyword2, ..., keyword7 | entity1, entity2, ..., entity5\n\ncontent"`

  **Must NOT do**:
  - Change the Component protocol interface
  - Remove the dict-based input format

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Logic change but straightforward
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 5, 7
  - **Blocked By**: None

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:210-229` - Original format logic
  - `poc/modular_retrieval_pipeline/components/content_enricher.py:129-175` - Current format

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.components.content_enricher import ContentEnricher
  enricher = ContentEnricher()
  data = {
      'content': 'Original content here',
      'keywords': ['k1','k2','k3','k4','k5','k6','k7','k8','k9','k10'],
      'entities': {'ORG': ['org1','org2','org3'], 'PERSON': ['p1','p2','p3']}
  }
  result = enricher.process(data)
  # Verify format
  assert 'ORG:' not in result, 'Should NOT have type labels'
  assert 'k8' not in result.split('\n\n')[0], 'Should only have 7 keywords'
  print(f'Format correct: {result[:80]}...')
  "
  ```

  **Commit**: YES (group with 1, 2)
  - Message: `fix(modular): align enrichment component configs with FastEnricher`
  - Files: Same as Task 1

---

- [ ] 4. Create IndexedChunkStore Type

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/types.py`
  - Add new frozen dataclass `IndexedChunkStore`:
    ```python
    @dataclass(frozen=True)
    class IndexedChunkStore:
        """Stores indexed chunks with embeddings and BM25 index.
        
        This is a container for the indexed state created during the
        indexing phase. Unlike other types, this contains numpy arrays
        and BM25 objects that are mutable but should be treated as immutable
        after creation.
        """
        chunks: tuple[Any, ...]  # Original Chunk objects
        enriched_contents: tuple[str, ...]  # Enriched content strings
        embeddings: Any  # np.ndarray (768-dim vectors)
        bm25_index: Any  # BM25Okapi object
        embedding_model: str = "BAAI/bge-base-en-v1.5"
        enrichment_method: str = "fast"  # YAKE + spaCy
        metadata: tuple = field(default_factory=tuple)
    ```

  **Must NOT do**:
  - Use mutable types (lists instead of tuples for collections)
  - Store the embedder or BM25 builder objects

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Type definition only
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 5, 6, 7
  - **Blocked By**: None (can start anytime)

  **References**:
  - `poc/modular_retrieval_pipeline/types.py` - Existing type patterns
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:105-113` - Instance vars to store

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.types import IndexedChunkStore
  import dataclasses
  assert dataclasses.is_dataclass(IndexedChunkStore)
  # Verify frozen
  try:
      store = IndexedChunkStore((), (), None, None)
      store.chunks = []
      assert False, 'Should be frozen'
  except dataclasses.FrozenInstanceError:
      print('IndexedChunkStore is frozen: CORRECT')
  "
  ```

  **Commit**: YES
  - Message: `feat(modular): add IndexedChunkStore type for index storage`
  - Files: `types.py`

---

- [ ] 5. Create ChunkIndexer Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/chunk_indexer.py`
  - Implement `ChunkIndexer` component that:
    1. Takes list of Chunk objects
    2. Enriches each chunk using enrichment pipeline (KeywordExtractor → EntityExtractor → ContentEnricher)
    3. Encodes enriched content to embeddings (BGE-base-en-v1.5)
    4. Creates BM25 index from tokenized enriched content
    5. Returns `IndexedChunkStore`
  - Input: `dict` with `'chunks'` key (list of Chunk objects)
  - Output: `IndexedChunkStore`

  **Must NOT do**:
  - Store state in __init__ (embedder should be loaded in process())
  - Modify input chunks
  - Skip enrichment (must use the fixed enrichment pipeline)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Combines multiple components, moderate complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 6)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2, 3, 4

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:146-179` - Original index() method
  - `poc/modular_retrieval_pipeline/components/embedding_encoder.py` - Embedding pattern
  - `poc/modular_retrieval_pipeline/components/bm25_scorer.py` - BM25 pattern

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.components.chunk_indexer import ChunkIndexer
  from poc.modular_retrieval_pipeline.types import IndexedChunkStore
  
  # Mock chunk
  class MockChunk:
      def __init__(self, id, content):
          self.id = id
          self.content = content
          self.doc_id = 'test'
  
  indexer = ChunkIndexer()
  chunks = [MockChunk('1', 'Kubernetes horizontal pod autoscaler scales replicas')]
  result = indexer.process({'chunks': chunks})
  
  assert isinstance(result, IndexedChunkStore)
  assert len(result.chunks) == 1
  assert len(result.enriched_contents) == 1
  assert result.embeddings is not None
  assert result.bm25_index is not None
  print('ChunkIndexer works correctly')
  "
  ```

  **Commit**: YES
  - Message: `feat(modular): add ChunkIndexer component for index creation`
  - Files: `components/chunk_indexer.py`

---

- [ ] 6. Create AdaptiveRRFFuser Component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/adaptive_rrf_fuser.py`
  - Implement `AdaptiveRRFFuser` component that:
    1. Takes query expansion status + BM25 results + semantic results
    2. Selects weights based on expansion:
       - Normal: `bm25_weight=1.0, sem_weight=1.0, rrf_k=60`
       - Expanded: `bm25_weight=3.0, sem_weight=0.3, rrf_k=10`
    3. Applies RRF formula with selected weights
    4. Returns fused ScoredChunk list
  - Input: `dict` with keys:
    - `'expansion_triggered'`: bool
    - `'bm25_results'`: list[ScoredChunk]
    - `'semantic_results'`: list[ScoredChunk]
    - `'k'`: int (number of results to return)
  - Output: `list[ScoredChunk]`

  **Must NOT do**:
  - Store state between calls
  - Modify input lists
  - Hardcode candidate counts (use multiplier logic)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Algorithm implementation, moderate complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 4

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:210-307` - Original RRF logic
  - `poc/modular_retrieval_pipeline/components/rrf_fuser.py` - Existing RRF pattern

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.components.adaptive_rrf_fuser import AdaptiveRRFFuser
  from poc.modular_retrieval_pipeline.types import ScoredChunk
  
  fuser = AdaptiveRRFFuser()
  
  bm25 = [ScoredChunk('a', 'text', 10.0, 'bm25', 1)]
  semantic = [ScoredChunk('a', 'text', 0.9, 'semantic', 1)]
  
  # Test normal mode
  result_normal = fuser.process({
      'expansion_triggered': False,
      'bm25_results': bm25,
      'semantic_results': semantic,
      'k': 5
  })
  
  # Test expansion mode
  result_expanded = fuser.process({
      'expansion_triggered': True,
      'bm25_results': bm25,
      'semantic_results': semantic,
      'k': 5
  })
  
  print(f'Normal RRF score: {result_normal[0].score:.4f}')
  print(f'Expanded RRF score: {result_expanded[0].score:.4f}')
  # Expanded should have higher score due to higher weights
  "
  ```

  **Commit**: YES
  - Message: `feat(modular): add AdaptiveRRFFuser with expansion-based weighting`
  - Files: `components/adaptive_rrf_fuser.py`

---

- [ ] 7. Create ModularRetriever Orchestrator

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/modular_retriever.py`
  - Implement `ModularRetriever` class that:
    1. `index(chunks, documents)` - Uses ChunkIndexer to create IndexedChunkStore
    2. `retrieve(query, k=5)` - Full retrieval pipeline:
       - Rewrite query (QueryRewriter)
       - Expand query (QueryExpander)
       - Detect expansion_triggered
       - Compute semantic scores (dot product with stored embeddings)
       - Compute BM25 scores (using stored BM25 index)
       - Apply candidate multiplier (10x or 20x)
       - Fuse with AdaptiveRRFFuser
       - Return top-k Chunk objects
  - Should implement same interface as enriched_hybrid_llm for benchmark compatibility

  **Must NOT do**:
  - Store embedder as instance variable (lazy load in methods)
  - Skip any step from the original algorithm
  - Use different weighting than original

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex orchestration, multiple components, critical for correctness
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 5, 6

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py` - Full reference implementation
  - `poc/modular_retrieval_pipeline/ANALYSIS_enriched_hybrid_vs_modular.md` - Algorithm documentation

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.modular_retriever import ModularRetriever
  
  # Mock chunk
  class MockChunk:
      def __init__(self, id, content, doc_id):
          self.id = id
          self.content = content
          self.doc_id = doc_id
  
  retriever = ModularRetriever()
  chunks = [
      MockChunk('1', 'Kubernetes horizontal pod autoscaler scales replicas', 'doc1'),
      MockChunk('2', 'Docker container orchestration platform', 'doc2'),
  ]
  
  # Index
  retriever.index(chunks, [])
  
  # Retrieve
  results = retriever.retrieve('autoscaling pods', k=2)
  
  print(f'Retrieved {len(results)} chunks')
  print(f'Top result: {results[0].id if results else \"none\"}')
  "
  ```

  **Commit**: YES
  - Message: `feat(modular): add ModularRetriever orchestrator for end-to-end retrieval`
  - Files: `modular_retriever.py`

---

- [ ] 8. Update Benchmark to Use Full Pipeline

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/benchmark.py`
  - Replace partial `run_modular_benchmark()` with full implementation:
    1. Create ModularRetriever
    2. Index all chunks
    3. For each question, retrieve and check if needle found
    4. Track accuracy, latency, memory
  - Match the same benchmark methodology as baseline

  **Must NOT do**:
  - Change baseline benchmark methodology
  - Skip any metrics (accuracy, latency, memory)
  - Use different k value than baseline

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Integration work, straightforward
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:
  - `poc/modular_retrieval_pipeline/benchmark.py:105-185` - Baseline benchmark pattern
  - `poc/modular_retrieval_pipeline/benchmark.py:188-289` - Current modular (to replace)

  **Acceptance Criteria**:
  ```bash
  # Quick test (5 questions)
  python poc/modular_retrieval_pipeline/benchmark.py --questions poc/chunking_benchmark_v2/corpus/needle_questions.json --quick
  # Assert: Modular pipeline runs without errors
  # Assert: Produces accuracy, latency, memory metrics
  ```

  **Commit**: YES
  - Message: `feat(modular): implement full retrieval pipeline in benchmark`
  - Files: `benchmark.py`

---

- [ ] 9. Verify Accuracy Matches Baseline

  **What to do**:
  - Run full benchmark (20 questions)
  - Compare modular accuracy vs baseline
  - Document results
  - Verify accuracy ≥85% and within 5% of baseline

  **Must NOT do**:
  - Accept results below 85% accuracy
  - Skip documenting discrepancies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Run command, verify output
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5 (final)
  - **Blocks**: None
  - **Blocked By**: Task 8

  **References**:
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Test questions

  **Acceptance Criteria**:
  ```bash
  python poc/modular_retrieval_pipeline/benchmark.py --questions poc/chunking_benchmark_v2/corpus/needle_questions.json
  # Expected output:
  # Baseline: 90.0% accuracy
  # Modular: ≥85.0% accuracy
  # VERDICT: ✓ PASS
  ```

  **Commit**: YES (final)
  - Message: `docs(modular): verify benchmark accuracy matches baseline`
  - Files: `benchmark_results.json` (if generated)

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1, 2, 3 | `fix(modular): align enrichment component configs with FastEnricher` | keyword_extractor.py, entity_extractor.py, content_enricher.py |
| 4 | `feat(modular): add IndexedChunkStore type for index storage` | types.py |
| 5 | `feat(modular): add ChunkIndexer component for index creation` | components/chunk_indexer.py |
| 6 | `feat(modular): add AdaptiveRRFFuser with expansion-based weighting` | components/adaptive_rrf_fuser.py |
| 7 | `feat(modular): add ModularRetriever orchestrator for end-to-end retrieval` | modular_retriever.py |
| 8 | `feat(modular): implement full retrieval pipeline in benchmark` | benchmark.py |
| 9 | `docs(modular): verify benchmark accuracy matches baseline` | benchmark_results.json |

---

## Success Criteria

### Verification Commands
```bash
# Run full benchmark
python poc/modular_retrieval_pipeline/benchmark.py --questions poc/chunking_benchmark_v2/corpus/needle_questions.json

# Expected output format:
# BENCHMARK SUMMARY
# =================
# Baseline (enriched_hybrid_llm):
#   Accuracy: 90.0%
#   Avg Latency: XXXms
#   Peak Memory: XXX MB
#
# Modular Pipeline:
#   Accuracy: ≥85.0%
#   Avg Latency: XXXms
#   Peak Memory: XXX MB
#
# VERDICT: ✓ PASS (accuracy ≥85%)
```

### Final Checklist
- [ ] KeywordExtractor uses YAKE config: n=2, dedupFunc="seqm", windowsSize=1
- [ ] EntityExtractor uses 9 entity types (no TECH)
- [ ] ContentEnricher formats: 7 keywords, 5 entities, no type labels
- [ ] IndexedChunkStore stores chunks, enriched_contents, embeddings, bm25_index
- [ ] ChunkIndexer produces complete IndexedChunkStore
- [ ] AdaptiveRRFFuser uses correct weights (1.0/1.0/60 vs 3.0/0.3/10)
- [ ] ModularRetriever implements full algorithm
- [ ] Benchmark runs end-to-end without errors
- [ ] Modular accuracy ≥85% and within 5% of baseline
