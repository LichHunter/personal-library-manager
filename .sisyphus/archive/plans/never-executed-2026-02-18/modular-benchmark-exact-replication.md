# Modular Benchmark: Exact Replication of enriched_hybrid_llm

## TL;DR

> **Quick Summary**: Create `ModularEnrichedHybridLLM` - a stateful orchestrator that uses modular components to produce **100% IDENTICAL** results to the original `enriched_hybrid_llm` strategy.
> 
> **Architecture**: Stateful orchestrator owns index state, delegates to stateless components
> 
> **Deliverables**: 
> - `ModularEnrichedHybridLLM` class with `index()` and `retrieve()` methods
> - Fixed enrichment components to match FastEnricher exactly
> - Golden test comparing original vs modular output
> 
> **Estimated Effort**: Medium (7 tasks, ~6-8 hours)
> **Parallel Execution**: Partial - enrichment fixes can parallel
> **Critical Path**: Enrichment Fixes → Orchestrator → Verification

---

## Context

### Oracle Consultation Summary

**Recommended Architecture**: Stateful orchestrator + stateless components

```
┌─────────────────────────────────────────────────────────────┐
│          ModularEnrichedHybridLLM (Stateful)                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ State (stored after index()):                          │ │
│  │   - chunks: list[Chunk]                                │ │
│  │   - embeddings: np.ndarray                             │ │
│  │   - bm25: BM25Okapi                                    │ │
│  │   - _enriched_contents: list[str]                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  index(chunks) ────────────────────────────────────────►    │
│     │                                                       │
│     ├─► FastEnricher.enrich() for each chunk               │
│     ├─► embedder.encode(enriched_contents)                 │
│     └─► BM25Okapi(tokenized_enriched)                      │
│                                                             │
│  retrieve(query, k=5) ─────────────────────────────────►    │
│     │                                                       │
│     ├─► QueryRewriter.process(query)                       │
│     ├─► QueryExpander.process(rewritten)                   │
│     ├─► Detect expansion_triggered                         │
│     ├─► Select adaptive weights                            │
│     ├─► Semantic scoring (dot product)                     │
│     ├─► BM25 scoring                                       │
│     ├─► RRF fusion (semantic FIRST, then BM25)            │
│     └─► Return top-k Chunks                                │
└─────────────────────────────────────────────────────────────┘
```

### Critical Details for 100% Identical Results

1. **RRF Insertion Order**: MUST process semantic results FIRST, then BM25
   - This affects tie-breaking when RRF scores are equal
   - Python's `sorted()` is stable, so insertion order matters

2. **Use `dict.get(idx, 0)`** not `setdefault()` for RRF accumulation

3. **Exact String Formatting** for enrichment:
   - Format: `" | ".join(prefix_parts)` + `"\n\n"` + content
   - Keywords: First 7, comma-joined: `", ".join(keywords[:7])`
   - Entities: First 2 per type, max 5 total: `", ".join(entity_values[:5])`

4. **No Random Seeds**: YAKE, spaCy, BM25, BGE are all deterministic

---

## Work Objectives

### Core Objective
Create a modular implementation that produces **byte-for-byte identical** retrieval results as `enriched_hybrid_llm` on any input.

### Concrete Deliverables
- `poc/modular_retrieval_pipeline/modular_enriched_hybrid_llm.py` - Main orchestrator
- Fixed `components/keyword_extractor.py` - YAKE config match
- Fixed `components/entity_extractor.py` - Entity types match
- Fixed `components/content_enricher.py` - Format match
- `poc/modular_retrieval_pipeline/test_exact_replication.py` - Golden test

### Definition of Done
- [x] Golden test passes: `original.retrieve(q) == modular.retrieve(q)` for all test queries
- [x] Intermediate outputs match at each step (enrichment, embeddings, BM25 scores, RRF scores)
- [x] Benchmark shows identical accuracy (not just "comparable")

### Must Have
- Exact YAKE config: `n=2, top=10, dedupLim=0.9, dedupFunc="seqm", windowsSize=1`
- Exact entity types: `{ORG, PRODUCT, GPE, PERSON, WORK_OF_ART, LAW, EVENT, FAC, NORP}`
- Exact format: `"kw1, kw2, ..., kw7 | e1, e2, ..., e5\n\n{content}"`
- Exact RRF order: semantic first (lines 270-278), BM25 second (lines 280-288)
- Exact weights: `bm25=3.0, sem=0.3, rrf_k=10` when expanded; `bm25=1.0, sem=1.0, rrf_k=60` when not
- Exact candidate selection: `n_candidates = min(k * multiplier, len(chunks))`

### Must NOT Have (Guardrails)
- Do NOT modify original `enriched_hybrid_llm.py`
- Do NOT change any algorithm parameters
- Do NOT introduce new dependencies
- Do NOT "optimize" the algorithm (must be identical)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (PARALLEL - Enrichment Fixes):
├── Task 1: Fix KeywordExtractor to use FastEnricher's YAKE config
├── Task 2: Fix EntityExtractor to use FastEnricher's entity types
└── Task 3: Fix ContentEnricher to use FastEnricher's exact format

Wave 2 (Sequential - Orchestrator):
└── Task 4: Create ModularEnrichedHybridLLM orchestrator

Wave 3 (Sequential - Integration):
├── Task 5: Create golden test for exact comparison
├── Task 6: Update benchmark.py to use ModularEnrichedHybridLLM
└── Task 7: Verify 100% identical results
```

---

## TODOs

- [x] 1. Fix KeywordExtractor YAKE Config (EXACT MATCH)

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/components/keyword_extractor.py`
  - Change `_extract_keywords()` method to use EXACT FastEnricher config:
    ```python
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=2,           # WAS: n=3
        top=10,        # OK
        dedupLim=0.9,  # OK
        dedupFunc="seqm",   # WAS: missing
        windowsSize=1,      # WAS: missing
    )
    ```
  - Also add code block removal logic (FastEnricher lines 157-158):
    ```python
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
    code_ratio = _calculate_code_ratio(content)
    text_for_nlp = CODE_BLOCK_PATTERN.sub(" ", content) if code_ratio > 0.3 else content
    ```

  **Must NOT do**:
  - Change the Component interface
  - Remove configurable max_keywords

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 2, 3)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:19-33` - YAKE extractor setup
  - `poc/chunking_benchmark_v2/enrichment/fast.py:52-62` - Code block removal
  - `poc/chunking_benchmark_v2/enrichment/fast.py:157-158` - Code ratio check

  **Acceptance Criteria**:
  ```bash
  python -c "
  import yake
  # Verify YAKE config matches FastEnricher
  from poc.modular_retrieval_pipeline.components.keyword_extractor import KeywordExtractor
  ext = KeywordExtractor()
  # Test with same content
  content = 'Kubernetes horizontal pod autoscaler scales replicas based on CPU utilization metrics'
  result = ext.process({'content': content})
  print(f'Keywords: {result[\"keywords\"]}')
  # Compare with FastEnricher
  from poc.chunking_benchmark_v2.enrichment.fast import FastEnricher
  fe = FastEnricher()
  fe_result = fe.enrich(content)
  print(f'FastEnricher keywords: {fe_result.keywords}')
  "
  ```

  **Commit**: YES (group with Tasks 2, 3)
  - Message: `fix(modular): exact match enrichment configs with FastEnricher`

---

- [x] 2. Fix EntityExtractor Entity Types (EXACT MATCH)

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/components/entity_extractor.py`
  - Change `DEFAULT_ENTITY_TYPES` to EXACT FastEnricher set:
    ```python
    DEFAULT_ENTITY_TYPES = {
        "ORG",
        "PRODUCT",
        "GPE",
        "PERSON",
        "WORK_OF_ART",
        "LAW",
        "EVENT",
        "FAC",
        "NORP",
    }
    ```
  - Remove "TECH" (not a real spaCy entity type)
  - Add spaCy text limit (FastEnricher line 188): `doc = nlp(text_for_nlp[:5000])`
  - Add per-type limit (FastEnricher line 197-198): `entities[label] = entities[label][:5]`
  - Add code block removal (same as Task 1)

  **Must NOT do**:
  - Change the Component interface
  - Remove configurable entity_types

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 1, 3)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:84-94` - DEFAULT_ENTITY_TYPES
  - `poc/chunking_benchmark_v2/enrichment/fast.py:188` - 5000 char limit
  - `poc/chunking_benchmark_v2/enrichment/fast.py:197-198` - 5 per type limit

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.components.entity_extractor import EntityExtractor
  ext = EntityExtractor()
  expected = {'ORG', 'PRODUCT', 'GPE', 'PERSON', 'WORK_OF_ART', 'LAW', 'EVENT', 'FAC', 'NORP'}
  assert ext.entity_types == expected, f'Got {ext.entity_types}'
  print('Entity types: EXACT MATCH')
  "
  ```

  **Commit**: YES (group with Tasks 1, 3)

---

- [x] 3. Fix ContentEnricher Format (EXACT MATCH)

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/components/content_enricher.py`
  - Change `_format_enriched_content()` to use EXACT FastEnricher format:
    ```python
    def _format_enriched_content(self, content, keywords, entities):
        prefix_parts = []
        
        # Keywords: first 7, comma-joined (line 213)
        if keywords:
            prefix_parts.append(", ".join(keywords[:7]))
        
        # Entities: first 2 per type, max 5 total, NO type labels (lines 216-220)
        if entities:
            entity_values = []
            for label, values in entities.items():
                entity_values.extend(values[:2])
            if entity_values:
                prefix_parts.append(", ".join(entity_values[:5]))
        
        # Combine with " | " separator (line 224)
        if prefix_parts:
            prefix = " | ".join(prefix_parts)
            return f"{prefix}\n\n{content}"
        else:
            return content
    ```

  **Critical differences from current**:
  - Keywords: First 7 ONLY (not all)
  - Entities: First 2 per type, max 5 total (not all)
  - NO entity type labels (current has "ORG: entity")
  - Iteration order: `for label, values in entities.items()` (dict order)

  **Must NOT do**:
  - Change the Component interface
  - Add type labels to entities

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 1, 2)
  - **Blocks**: Task 4
  - **Blocked By**: None

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/fast.py:210-229` - Format logic

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.components.content_enricher import ContentEnricher
  enricher = ContentEnricher()
  data = {
      'content': 'Original content',
      'keywords': ['k1','k2','k3','k4','k5','k6','k7','k8','k9','k10'],
      'entities': {'ORG': ['org1','org2','org3'], 'PERSON': ['p1','p2','p3']}
  }
  result = enricher.process(data)
  # Expected: 'k1, k2, k3, k4, k5, k6, k7 | org1, org2, p1, p2\n\nOriginal content'
  assert 'k8' not in result.split('\n\n')[0], 'Should only have 7 keywords'
  assert 'ORG:' not in result, 'Should NOT have type labels'
  print(f'Format: {result}')
  "
  ```

  **Commit**: YES (group with Tasks 1, 2)

---

- [x] 4. Create ModularEnrichedHybridLLM Orchestrator

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/modular_enriched_hybrid_llm.py`
  - Implement EXACT same algorithm as `enriched_hybrid_llm.py`:

  ```python
  """Modular implementation of EnrichedHybridLLMRetrieval.
  
  This class produces 100% IDENTICAL results to the original
  enriched_hybrid_llm by using the exact same algorithm with
  modular components.
  """
  
  import numpy as np
  from rank_bm25 import BM25Okapi
  from typing import Optional
  
  from strategies import Chunk, Document
  
  # Import modular components
  from .components.query_rewriter import QueryRewriter
  from .components.query_expander import QueryExpander, DOMAIN_EXPANSIONS
  from .components.keyword_extractor import KeywordExtractor
  from .components.entity_extractor import EntityExtractor
  from .components.content_enricher import ContentEnricher
  from .base import Pipeline
  from .types import Query
  
  
  class ModularEnrichedHybridLLM:
      """Stateful orchestrator using modular components.
      
      Produces identical results to EnrichedHybridLLMRetrieval.
      """
      
      def __init__(
          self,
          rrf_k: int = 60,
          candidate_multiplier: int = 10,
          rewrite_timeout: float = 5.0,
          debug: bool = False,
      ):
          self.rrf_k = rrf_k
          self.candidate_multiplier = candidate_multiplier
          self.rewrite_timeout = rewrite_timeout
          self.debug = debug
          
          # Build enrichment pipeline (matches FastEnricher)
          self.enrichment_pipeline = (
              Pipeline()
              .add(KeywordExtractor(max_keywords=10))
              .add(EntityExtractor())
              .add(ContentEnricher())
              .build()
          )
          
          # Query processing components
          self.query_rewriter = QueryRewriter(timeout=rewrite_timeout)
          self.query_expander = QueryExpander()
          
          # State (populated by index())
          self.chunks: Optional[list[Chunk]] = None
          self.embeddings: Optional[np.ndarray] = None
          self.bm25: Optional[BM25Okapi] = None
          self._enriched_contents: list[str] = []
          self.embedder = None  # Set via set_embedder()
      
      def set_embedder(self, embedder):
          """Set the embedder (sentence-transformers model)."""
          self.embedder = embedder
      
      def index(self, chunks: list[Chunk], documents: Optional[list[Document]] = None):
          """Index chunks using modular enrichment pipeline.
          
          EXACTLY matches enriched_hybrid_llm.index() lines 146-179.
          """
          if self.embedder is None:
              raise ValueError("Embedder not set. Call set_embedder() first.")
          
          self.chunks = chunks
          enriched_contents = []
          
          # Enrich each chunk (line 160-162)
          for chunk in chunks:
              enriched = self.enrichment_pipeline.run({"content": chunk.content})
              enriched_contents.append(enriched)
          
          # Encode embeddings (line 175)
          self.embeddings = self.embedder.encode(
              enriched_contents,
              normalize_embeddings=True,
              show_progress_bar=False
          )
          
          # Build BM25 index (line 177-178)
          tokenized = [content.lower().split() for content in enriched_contents]
          self.bm25 = BM25Okapi(tokenized)
          self._enriched_contents = enriched_contents
      
      def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
          """Retrieve top-k chunks for query.
          
          EXACTLY matches enriched_hybrid_llm.retrieve() lines 181-339.
          """
          if self.chunks is None or self.embeddings is None or self.bm25 is None:
              return []
          
          # Step 1: Rewrite query (line 190-192)
          query_obj = Query(query)
          rewritten = self.query_rewriter.process(query_obj)
          rewritten_query = rewritten.rewritten
          
          # Step 2: Expand query (line 205)
          expanded = self.query_expander.process(rewritten)
          expanded_query = expanded.expanded
          
          # Step 3: Detect expansion triggered (line 206)
          expansion_triggered = expanded_query != rewritten_query
          
          # Step 4: Select adaptive weights (lines 210-217)
          if expansion_triggered:
              bm25_weight = 3.0
              sem_weight = 0.3
              multiplier = self.candidate_multiplier * 2
              rrf_k = 10
          else:
              bm25_weight = 1.0
              sem_weight = 1.0
              multiplier = self.candidate_multiplier
              rrf_k = self.rrf_k
          
          # Step 5: Calculate n_candidates (line 233)
          n_candidates = min(k * multiplier, len(self.chunks))
          
          # Step 6: Semantic scoring (lines 238-240)
          q_emb = self.embedder.encode(
              expanded_query,
              normalize_embeddings=True,
              show_progress_bar=False
          )
          sem_scores = np.dot(self.embeddings, q_emb)
          sem_ranks = np.argsort(sem_scores)[::-1]
          
          # Step 7: BM25 scoring (lines 253-254)
          bm25_scores = self.bm25.get_scores(expanded_query.lower().split())
          bm25_ranks = np.argsort(bm25_scores)[::-1]
          
          # Step 8: RRF fusion (lines 267-288)
          # CRITICAL: Process semantic FIRST, then BM25 (insertion order matters for tie-breaking)
          rrf_scores: dict[int, float] = {}
          
          # Semantic contribution FIRST (lines 270-278)
          for rank, idx in enumerate(sem_ranks[:n_candidates]):
              sem_component = sem_weight / (rrf_k + rank)
              rrf_scores[idx] = rrf_scores.get(idx, 0) + sem_component
          
          # BM25 contribution SECOND (lines 280-288)
          for rank, idx in enumerate(bm25_ranks[:n_candidates]):
              bm25_component = bm25_weight / (rrf_k + rank)
              rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_component
          
          # Step 9: Sort and return top-k (lines 305-309)
          top_idx = sorted(
              rrf_scores.keys(),
              key=lambda x: rrf_scores[x],
              reverse=True
          )[:k]
          
          return [self.chunks[i] for i in top_idx]
  ```

  **Critical implementation details**:
  - Line 267: Use `dict[int, float]` type hint
  - Line 272, 282: Use `dict.get(idx, 0)` not `setdefault()`
  - Lines 270-278: Process semantic FIRST
  - Lines 280-288: Process BM25 SECOND
  - Line 305: Sort by `rrf_scores[x]` descending

  **Must NOT do**:
  - Change algorithm order
  - Use different data structures
  - "Optimize" the RRF loop

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Critical orchestrator, must be exact
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Tasks 5, 6, 7
  - **Blocked By**: Tasks 1, 2, 3

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py` - ENTIRE FILE

  **Acceptance Criteria**:
  ```bash
  python -c "
  from poc.modular_retrieval_pipeline.modular_enriched_hybrid_llm import ModularEnrichedHybridLLM
  from sentence_transformers import SentenceTransformer
  
  # Create instance
  modular = ModularEnrichedHybridLLM()
  embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')
  modular.set_embedder(embedder)
  
  # Mock chunk
  class MockChunk:
      def __init__(self, id, content, doc_id):
          self.id = id
          self.content = content
          self.doc_id = doc_id
  
  chunks = [
      MockChunk('1', 'Kubernetes horizontal pod autoscaler', 'doc1'),
      MockChunk('2', 'Docker container platform', 'doc2'),
  ]
  
  modular.index(chunks)
  results = modular.retrieve('autoscaling pods', k=2)
  print(f'Retrieved: {[c.id for c in results]}')
  "
  ```

  **Commit**: YES
  - Message: `feat(modular): add ModularEnrichedHybridLLM exact replication orchestrator`

---

- [x] 5. Create Golden Test for Exact Comparison

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/test_exact_replication.py`
  - Test that original and modular produce IDENTICAL results:
    - Same chunk IDs in same order
    - Same RRF scores (within floating point tolerance)
    - Same intermediate outputs (enriched content, embeddings, BM25 scores)

  ```python
  """Golden test: verify modular produces identical results to original."""
  
  import numpy as np
  from sentence_transformers import SentenceTransformer
  
  # Import original
  import sys
  sys.path.insert(0, "poc/chunking_benchmark_v2")
  from retrieval import create_retrieval_strategy
  
  # Import modular
  from modular_enriched_hybrid_llm import ModularEnrichedHybridLLM
  
  
  def test_identical_results():
      """Test that original and modular produce identical results."""
      # Setup
      embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')
      
      original = create_retrieval_strategy("enriched_hybrid_llm", debug=False)
      original.set_embedder(embedder)
      
      modular = ModularEnrichedHybridLLM(debug=False)
      modular.set_embedder(embedder)
      
      # Load test chunks
      chunks = load_test_chunks()  # Same chunks for both
      
      # Index
      original.index(chunks)
      modular.index(chunks)
      
      # Test queries
      test_queries = [
          "What is the Topology Manager?",
          "How does token authentication work?",
          "What is the RPO for disaster recovery?",
          "How do I configure autoscaling?",
      ]
      
      for query in test_queries:
          orig_results = original.retrieve(query, k=5)
          mod_results = modular.retrieve(query, k=5)
          
          # Verify identical chunk IDs
          orig_ids = [c.id for c in orig_results]
          mod_ids = [c.id for c in mod_results]
          assert orig_ids == mod_ids, f"Query '{query}': {orig_ids} != {mod_ids}"
          
          print(f"✓ Query '{query[:30]}...' - identical results")
      
      print("\n✓ All queries produce identical results!")
  
  
  def test_identical_enrichment():
      """Test that enrichment produces identical strings."""
      from poc.chunking_benchmark_v2.enrichment.fast import FastEnricher
      from components.keyword_extractor import KeywordExtractor
      from components.entity_extractor import EntityExtractor
      from components.content_enricher import ContentEnricher
      from base import Pipeline
      
      original = FastEnricher()
      modular = (Pipeline()
          .add(KeywordExtractor())
          .add(EntityExtractor())
          .add(ContentEnricher())
          .build())
      
      test_contents = [
          "Kubernetes horizontal pod autoscaler scales replicas based on CPU utilization",
          "Google Cloud Platform offers Kubernetes Engine for container orchestration",
      ]
      
      for content in test_contents:
          orig_result = original.enrich(content).enhanced_content
          mod_result = modular.run({"content": content})
          
          assert orig_result == mod_result, f"Enrichment mismatch:\nOrig: {orig_result}\nMod: {mod_result}"
          print(f"✓ Enrichment identical for '{content[:40]}...'")
  
  
  if __name__ == "__main__":
      test_identical_enrichment()
      test_identical_results()
  ```

  **Must NOT do**:
  - Accept "close enough" results
  - Skip intermediate comparisons

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 7
  - **Blocked By**: Task 4

  **References**:
  - Oracle's verification strategy recommendations

  **Acceptance Criteria**:
  ```bash
  python poc/modular_retrieval_pipeline/test_exact_replication.py
  # Expected: All tests pass with "✓" output
  ```

  **Commit**: YES
  - Message: `test(modular): add golden test for exact replication verification`

---

- [x] 6. Update Benchmark to Use ModularEnrichedHybridLLM

  **What to do**:
  - Open `poc/modular_retrieval_pipeline/benchmark.py`
  - Replace `run_modular_benchmark()` with implementation using `ModularEnrichedHybridLLM`
  - Should produce IDENTICAL results to baseline

  **Must NOT do**:
  - Change baseline methodology
  - Use different k value

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 7
  - **Blocked By**: Task 4

  **Acceptance Criteria**:
  ```bash
  python poc/modular_retrieval_pipeline/benchmark.py --quick
  # Expected: Modular results IDENTICAL to baseline (same accuracy)
  ```

  **Commit**: YES
  - Message: `feat(modular): use ModularEnrichedHybridLLM in benchmark`

---

- [x] 7. Verify 100% Identical Results

  **What to do**:
  - Run full benchmark
  - Verify modular accuracy EQUALS baseline accuracy
  - Document results

  **Must NOT do**:
  - Accept "close" results (must be identical)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: None
  - **Blocked By**: Tasks 5, 6

  **Acceptance Criteria**:
  ```bash
  python poc/modular_retrieval_pipeline/benchmark.py
  # Expected output:
  # Baseline: 90.0% accuracy (18/20)
  # Modular:  90.0% accuracy (18/20)  <-- MUST BE IDENTICAL
  # VERDICT: ✓ IDENTICAL
  ```

  **Commit**: YES
  - Message: `docs(modular): verify 100% identical benchmark results`

---

## Commit Strategy

| After Task(s) | Message |
|---------------|---------|
| 1, 2, 3 | `fix(modular): exact match enrichment configs with FastEnricher` |
| 4 | `feat(modular): add ModularEnrichedHybridLLM exact replication orchestrator` |
| 5 | `test(modular): add golden test for exact replication verification` |
| 6 | `feat(modular): use ModularEnrichedHybridLLM in benchmark` |
| 7 | `docs(modular): verify 100% identical benchmark results` |

---

## Success Criteria

### Verification Commands
```bash
# Run golden test
python poc/modular_retrieval_pipeline/test_exact_replication.py

# Run full benchmark
python poc/modular_retrieval_pipeline/benchmark.py
```

### Final Checklist
- [x] YAKE config: `n=2, dedupLim=0.9, dedupFunc="seqm", windowsSize=1`
- [x] Entity types: 9 types (no TECH)
- [x] Content format: `"kw1, ..., kw7 | e1, ..., e5\n\n{content}"`
- [x] Keywords: First 7 only
- [x] Entities: First 2 per type, max 5 total, NO type labels
- [x] Code block removal: Same regex pattern
- [x] RRF order: Semantic FIRST, BM25 SECOND
- [x] Weights: `3.0/0.3/10` when expanded, `1.0/1.0/60` when not
- [x] Multiplier: `10x` normal, `20x` when expanded
- [x] Dict accumulation: Use `dict.get(idx, 0)`
- [x] Golden test passes
- [x] Benchmark accuracy IDENTICAL (not just "comparable")
