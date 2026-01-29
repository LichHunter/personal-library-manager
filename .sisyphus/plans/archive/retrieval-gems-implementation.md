# Retrieval Gems Implementation Plan

## Overview

**Goal**: Implement, test, and verify 5 hidden gem strategies to improve RAG retrieval from 94% to 98-99%

**Methodology**: 
1. Implement each strategy SEPARATELY as a new retrieval class
2. Establish individual baselines through manual testing
3. Cross-breed best-performing strategies
4. All tests executed against manual_test.py with human verification

**Based on**: `.sisyphus/notepads/hidden-gems-recommendations.md` and `.sisyphus/notepads/experiment-plan.md`

---

## Phase 0: Test Infrastructure Setup

### Task 0.1: Create Edge Case Test Dataset

**What**: Create a JSON file with the 16 failure queries from research for consistent testing

**File**: `poc/chunking_benchmark_v2/corpus/edge_case_queries.json`

**Content**:
```json
{
  "queries": [
    {"id": "mh_002", "type": "multi-hop", "query": "If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?", "baseline_score": 5, "target_score": 8, "root_causes": ["VOCABULARY_MISMATCH", "EMBEDDING_BLIND"]},
    {"id": "mh_004", "type": "multi-hop", "query": "How do the HPA scaling parameters relate to the API Gateway resource requirements?", "baseline_score": 6, "target_score": 8, "root_causes": ["EMBEDDING_BLIND"]},
    {"id": "tmp_004", "type": "temporal", "query": "How long does it take for workflow definition cache changes to propagate?", "baseline_score": 4, "target_score": 7, "root_causes": ["VOCABULARY_MISMATCH", "EMBEDDING_BLIND"]},
    {"id": "cmp_001", "type": "comparative", "query": "What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?", "baseline_score": 2, "target_score": 8, "root_causes": ["BM25_MISS", "EMBEDDING_BLIND"]},
    {"id": "cmp_003", "type": "comparative", "query": "What's the difference between /health and /ready endpoints?", "baseline_score": 5, "target_score": 7, "root_causes": ["EMBEDDING_BLIND", "CORPUS_GAP"]},
    {"id": "neg_001", "type": "negation", "query": "What should I NOT do when I'm rate limited?", "baseline_score": 6, "target_score": 8, "root_causes": ["NEGATION_BLIND"]},
    {"id": "neg_002", "type": "negation", "query": "Why doesn't HS256 work for JWT token validation?", "baseline_score": 7, "target_score": 8, "root_causes": ["NEGATION_BLIND", "CORPUS_GAP"]},
    {"id": "neg_003", "type": "negation", "query": "Why can't I schedule workflows more frequently than every minute?", "baseline_score": 7, "target_score": 8, "root_causes": ["NEGATION_BLIND"]},
    {"id": "neg_004", "type": "negation", "query": "What happens if I don't implement token refresh logic?", "baseline_score": 6, "target_score": 8, "root_causes": ["NEGATION_BLIND"]},
    {"id": "neg_005", "type": "negation", "query": "Why shouldn't I hardcode API keys in workflow definitions?", "baseline_score": 5, "target_score": 8, "root_causes": ["NEGATION_BLIND"]},
    {"id": "imp_001", "type": "implicit", "query": "Best practice for handling long-running data processing that might exceed time limits", "baseline_score": 6, "target_score": 8, "root_causes": ["VOCABULARY_MISMATCH"]},
    {"id": "tmp_003", "type": "temporal", "query": "What's the sequence of events when a workflow execution times out?", "baseline_score": 7, "target_score": 8, "root_causes": ["VOCABULARY_MISMATCH"]},
    {"id": "tmp_005", "type": "temporal", "query": "What's the timeline for automatic failover when the database primary fails?", "baseline_score": 7, "target_score": 8, "root_causes": ["RANKING_ERROR"]},
    {"id": "cmp_002", "type": "comparative", "query": "How do fixed, linear, and exponential backoff strategies differ for retries?", "baseline_score": 6, "target_score": 8, "root_causes": ["RANKING_ERROR"]},
    {"id": "imp_003", "type": "implicit", "query": "How to debug why my API calls are slow", "baseline_score": 7, "target_score": 8, "root_causes": ["VOCABULARY_MISMATCH"]},
    {"id": "mh_001", "type": "multi-hop", "query": "Compare JWT expiration in Auth Service vs the API documentation - are they consistent?", "baseline_score": 8, "target_score": 9, "root_causes": []}
  ],
  "metadata": {
    "total_failures": 16,
    "baseline_avg": 5.9,
    "target_avg": 7.8
  }
}
```

### Task 0.2: Create Manual Test Runner Script

**What**: Script to run edge case tests and output results for manual grading

**File**: `poc/chunking_benchmark_v2/test_edge_cases.py`

**Functionality**:
- Load edge case queries
- Run specified retrieval strategy
- Output retrieved chunks with formatting for manual review
- Save results to markdown file for grading
- Calculate aggregate scores after grading

---

## Phase 1: Individual Strategy Implementation

### Strategy 1: Adaptive Hybrid Weights

**File**: `poc/chunking_benchmark_v2/retrieval/adaptive_hybrid.py`

**Class**: `AdaptiveHybridRetrieval`

**Key Components**:

```python
def detect_technical_query(query: str) -> float:
    """
    Patterns to detect:
    - camelCase: PgBouncer, minReplicas, maxConnections
    - snake_case: max_connections, pool_mode
    - ALL_CAPS: HS256, JWT, TTL, YAML, HPA
    - URL paths: /health, /ready
    - Numbers with units: 3600s, 70%
    - Technical keywords: configuration, parameter, endpoint
    
    Returns: 0.0-1.0 (0=natural language, 1=highly technical)
    """

def calculate_adaptive_weights(query: str) -> tuple[float, float]:
    """
    Returns: (bm25_weight, semantic_weight)
    
    If technical_score > 0.3:
        return (0.7, 0.3)  # Favor BM25 for exact matches
    else:
        return (0.4, 0.6)  # Favor semantic for natural language
    """
```

**Target Failures**: mh_002, mh_004, tmp_004, cmp_001, cmp_003, imp_003, mh_001 (7 queries)

**Expected Improvement**: 7 failures → 2 failures (71%)

**Verification**:
1. Run `test_edge_cases.py --strategy adaptive_hybrid`
2. Manually grade each of 7 target queries
3. Verify technical queries get BM25 weight 0.7
4. Verify PgBouncer/HPA YAML chunks are retrieved
5. Record scores in grading spreadsheet

---

### Strategy 2: Negation-Aware Filtering

**File**: `poc/chunking_benchmark_v2/retrieval/negation_aware.py`

**Class**: `NegationAwareRetrieval`

**Key Components**:

```python
def detect_negation(query: str) -> dict:
    """
    Negation types:
    - prohibition: "shouldn't", "can't", "don't", "avoid", "never"
    - failure: "doesn't work", "why can't", "not supported"
    - limitation: "can't schedule", "minimum", "maximum"
    - consequence: "what happens if I don't", "without"
    
    Returns: {has_negation, negation_type, matched_keywords}
    """

def expand_negation_query(query: str, negation_info: dict) -> list[str]:
    """
    Query variants by type:
    - prohibition: + "anti-patterns mistakes to avoid warnings"
    - failure: + "not supported limitations alternatives"
    - limitation: + "minimum maximum limits constraints"
    - consequence: + "consequences failure modes errors"
    """

def filter_negation_results(results: list, negation_info: dict) -> list:
    """
    Post-retrieval filtering:
    - Boost: warning, caution, avoid, never, don't, limitation
    - Penalize: how to, recommended, implement, setup
    """
```

**Target Failures**: neg_001, neg_002, neg_003, neg_004, neg_005 (5 queries)

**Expected Improvement**: 5 failures → 1 failure (80%)

**Verification**:
1. Run `test_edge_cases.py --strategy negation_aware`
2. Manually grade each of 5 negation queries
3. Verify negation detection catches all 5 queries
4. Verify warning/caution chunks are boosted
5. Verify positive-only advice is penalized

---

### Strategy 3: Synthetic Query Variants

**File**: `poc/chunking_benchmark_v2/retrieval/synthetic_variants.py`

**Class**: `SyntheticVariantsRetrieval`

**Key Components**:

```python
VARIANT_PROMPT = """Generate 3 diverse search queries for this question.
Make them DIFFERENT - vary terminology, specificity, and framing.

Question: {query}

Output exactly 3 queries, one per line:"""

def generate_variants(query: str) -> list[str]:
    """
    Single LLM call to generate 3 diverse variants.
    Returns: [original, variant1, variant2, variant3]
    """

def parallel_search(variants: list[str]) -> list[list[Chunk]]:
    """
    Search with all variants in parallel (async).
    Returns: list of result sets
    """

def reciprocal_rank_fusion(result_sets: list[list[Chunk]]) -> list[Chunk]:
    """
    Fuse results using RRF with equal weights.
    """
```

**Target Failures**: tmp_004, imp_001, tmp_003, mh_002, neg_001-neg_005 (9 queries)

**Expected Improvement**: 9 failures → 2 failures (78%)

**Verification**:
1. Run `test_edge_cases.py --strategy synthetic_variants`
2. Manually grade all 9 target queries
3. Verify variant generation produces diverse queries
4. Verify vocabulary mismatch queries improve
5. Log LLM calls and latency

---

### Strategy 4: BM25F Field Weighting

**File**: `poc/chunking_benchmark_v2/retrieval/bm25f_hybrid.py`

**Class**: `BM25FHybridRetrieval`

**Key Components**:

```python
def parse_chunk_fields(chunk: Chunk) -> dict:
    """
    Extract fields from chunk:
    - heading: From chunk.metadata['heading'] or first line
    - first_paragraph: First 200 chars
    - body: Remaining content
    - code: Content within ``` blocks
    """

def bm25f_score(query: str, chunk_fields: dict) -> float:
    """
    Field-weighted BM25 scoring:
    - heading_weight: 3.0
    - first_paragraph_weight: 2.0
    - body_weight: 1.0
    - code_weight: 0.5 (code doesn't match well)
    
    score = sum(bm25(query, field) * weight for field, weight)
    """
```

**Target Failures**: mh_001, mh_002, cmp_001, tmp_005, cmp_002, neg_005 (6 queries)

**Expected Improvement**: 6 failures → 2 failures (67%)

**Verification**:
1. Run `test_edge_cases.py --strategy bm25f_hybrid`
2. Manually grade all 6 target queries
3. Verify heading matches are boosted
4. Verify section-relevant chunks rank higher
5. Compare field parsing accuracy

---

### Strategy 5: Contextual Retrieval (Anthropic-style)

**File**: `poc/chunking_benchmark_v2/retrieval/contextual.py`

**Class**: `ContextualRetrieval`

**Key Components**:

```python
CONTEXT_PROMPT = """Provide a short context (1-2 sentences) for this chunk within its document.
Include: what section it's from, what topic it covers, any key facts it contains.

Document: {doc_title}
Chunk:
{chunk_content}

Context:"""

def enrich_chunk_with_context(chunk: Chunk, doc: Document) -> str:
    """
    Prepend LLM-generated context to chunk content.
    Called at INDEXING time, not query time.
    
    Returns: f"{context}\n\n{chunk.content}"
    """
```

**Target Failures**: tmp_004, imp_001, mh_002, mh_004, cmp_003, tmp_003 (10 queries - highest coverage)

**Expected Improvement**: 10 failures → 3 failures (70%)

**Note**: Requires re-indexing. Higher latency at indexing, but no query-time cost.

**Verification**:
1. Re-index corpus with contextual enrichment
2. Run `test_edge_cases.py --strategy contextual`
3. Manually grade all 10 target queries
4. Verify context improves disambiguation
5. Log indexing cost (LLM tokens, time)

---

## Phase 2: Baseline Testing

### Task 2.1: Establish Baseline Scores

**For each strategy**:
1. Run against ALL 16 failure queries
2. Generate markdown report with retrieved chunks
3. Manually grade each query (1-10 scale)
4. Record in `.sisyphus/notepads/strategy-baselines.md`

**Output Format**:
```markdown
# Strategy Baseline Results

## Strategy: adaptive_hybrid

### Query: mh_002
**Query**: "If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?"
**Previous Score**: 5/10
**New Score**: ___/10
**Retrieved Chunks**:
1. [chunk preview...]
2. [chunk preview...]
**Notes**: ___

### Summary
| Query | Prev | New | Delta |
|-------|------|-----|-------|
| mh_002 | 5 | ___ | ___ |
...

**Overall**: X/16 improved, Y/16 regressed, Z/16 unchanged
**Average Score**: ___/10 (prev: 5.9)
```

### Task 2.2: Regression Testing

**For each strategy**:
1. Run against 8 PASSING queries (score >7)
2. Verify no regressions
3. Record any regressions for investigation

---

## Phase 3: Cross-Breeding Best Strategies

### Task 3.1: Analyze Phase 2 Results

After individual testing, identify:
1. **Best performers by root cause**:
   - NEGATION_BLIND → Strategy 2 or 3?
   - VOCABULARY_MISMATCH → Strategy 1 or 3?
   - EMBEDDING_BLIND → Strategy 1 or 4?
   
2. **Complementary strategies** (non-overlapping improvements)
3. **Conflicting strategies** (same query, different results)

### Task 3.2: Design Hybrid Strategies

Based on Phase 2 results, create hybrid strategies:

**Option A: Sequential Pipeline**
```
Query → Negation Detection → Adaptive Weights → BM25F Search → Semantic Search → RRF Fusion → Results
```

**Option B: Parallel + Fusion**
```
Query → [Adaptive Hybrid, Negation-Aware, Synthetic Variants] → RRF Fusion → Results
```

**Option C: Query-Type Routing**
```
Query → Classify Type → Route to Best Strategy → Results
- Negation queries → Negation-Aware
- Technical queries → Adaptive Hybrid
- Vocabulary mismatch → Synthetic Variants
```

### Task 3.3: Implement Top Hybrid

**File**: `poc/chunking_benchmark_v2/retrieval/hybrid_gems.py`

**Class**: `HybridGemsRetrieval`

Implementation depends on Phase 3.1 analysis.

---

## Phase 4: Final Verification

### Task 4.1: Full Test Suite

1. Run hybrid strategy against ALL 24 queries (16 failed + 8 passing)
2. Generate full grading report
3. Manual verification of EVERY query
4. Calculate final metrics

### Task 4.2: A/B Comparison

**Compare**:
- `enriched_hybrid_llm` (current best: 94%)
- `hybrid_gems` (new hybrid)

**Metrics**:
- Overall accuracy (target: 98-99%)
- Per-query-type accuracy
- Latency (query-time)
- Cost (LLM calls)

### Task 4.3: Document Results

**Create**: `.sisyphus/notepads/gems-implementation-results.md`

**Content**:
- Strategy-by-strategy results
- Best hybrid configuration
- Final accuracy achieved
- Lessons learned
- Recommendations for production

---

## Task Checklist

### Phase 0: Infrastructure
- [ ] 0.1: Create edge case test dataset (edge_case_queries.json)
- [ ] 0.2: Create manual test runner (test_edge_cases.py)

### Phase 1: Implementation
- [ ] 1.1: Implement AdaptiveHybridRetrieval
- [ ] 1.2: Implement NegationAwareRetrieval
- [ ] 1.3: Implement SyntheticVariantsRetrieval
- [ ] 1.4: Implement BM25FHybridRetrieval
- [ ] 1.5: Implement ContextualRetrieval

### Phase 2: Baseline Testing
- [ ] 2.1.1: Test adaptive_hybrid (7 queries)
- [ ] 2.1.2: Test negation_aware (5 queries)
- [ ] 2.1.3: Test synthetic_variants (9 queries)
- [ ] 2.1.4: Test bm25f_hybrid (6 queries)
- [ ] 2.1.5: Test contextual (10 queries)
- [ ] 2.2: Run regression tests on all strategies

### Phase 3: Cross-Breeding
- [ ] 3.1: Analyze Phase 2 results
- [ ] 3.2: Design hybrid strategy
- [ ] 3.3: Implement HybridGemsRetrieval

### Phase 4: Verification
- [ ] 4.1: Full test suite (24 queries)
- [ ] 4.2: A/B comparison with enriched_hybrid_llm
- [ ] 4.3: Document final results

---

## Success Criteria

### Per-Strategy Targets

| Strategy | Target Queries | Expected Fix Rate | Min Acceptable |
|----------|---------------|-------------------|----------------|
| Adaptive Hybrid | 7 | 71% (5/7) | 57% (4/7) |
| Negation-Aware | 5 | 80% (4/5) | 60% (3/5) |
| Synthetic Variants | 9 | 78% (7/9) | 56% (5/9) |
| BM25F Hybrid | 6 | 67% (4/6) | 50% (3/6) |
| Contextual | 10 | 70% (7/10) | 50% (5/10) |

### Final Hybrid Target

- **Minimum**: 94% → 96% accuracy (fix 8/16 failures)
- **Target**: 94% → 98% accuracy (fix 12/16 failures)
- **Stretch**: 94% → 99% accuracy (fix 15/16 failures)

### Non-Regression Requirement

- 0 regressions on passing queries (8 queries must stay >7)
- Max 1 regression allowed if offset by 2+ improvements

---

## Manual Verification Protocol

### Grading Rubric (1-10)

**10**: Perfect - All info retrieved, directly answers question
**9**: Excellent - Complete answer, minor noise
**8**: Very Good - Answer present but requires parsing
**7**: Good - Core answer present, missing details
**6**: Adequate - Partial answer, useful
**5**: Borderline - Some relevant info, misses key point
**4**: Poor - Tangentially related
**3**: Very Poor - Mostly irrelevant
**2**: Bad - Almost entirely irrelevant
**1**: Failed - No relevant content

### Verification Steps

For EACH query:
1. Read the query and expected answer
2. Read ALL retrieved chunks (top 5)
3. Assign score using rubric
4. Write 1-2 sentence justification
5. Note if correct chunk is present but ranked low

### Aggregation

- Calculate mean score per strategy
- Calculate improvement delta from baseline
- Identify patterns in failures
- Document any surprises

---

## Risk Mitigation

### Risk 1: Strategy Interactions

**Problem**: Strategies may interfere when combined
**Mitigation**: Test combinations incrementally
**Fallback**: Use query-type routing instead of fusion

### Risk 2: Latency Increase

**Problem**: Multiple strategies increase query latency
**Mitigation**: Profile each strategy, set latency budget (500ms)
**Fallback**: Pre-compute what's possible (contextual enrichment)

### Risk 3: Corpus Gaps

**Problem**: Some failures (CORPUS_GAP) can't be fixed by retrieval
**Mitigation**: Document gaps separately, don't count against strategy
**Fallback**: Recommend corpus additions

### Risk 4: Overfitting to Test Set

**Problem**: Strategies may overfit to 16 failure queries
**Mitigation**: Test on additional queries from ground_truth_realistic.json
**Fallback**: Use 80/20 train/test split on failures

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0 | 0.5 days | None |
| Phase 1 | 2-3 days | Phase 0 |
| Phase 2 | 1-2 days | Phase 1 |
| Phase 3 | 1 day | Phase 2 |
| Phase 4 | 1 day | Phase 3 |
| **Total** | **5-7 days** | |

---

## Files to Create

```
poc/chunking_benchmark_v2/
├── corpus/
│   └── edge_case_queries.json          # Phase 0.1
├── retrieval/
│   ├── adaptive_hybrid.py              # Phase 1.1
│   ├── negation_aware.py               # Phase 1.2
│   ├── synthetic_variants.py           # Phase 1.3
│   ├── bm25f_hybrid.py                 # Phase 1.4
│   ├── contextual.py                   # Phase 1.5
│   └── hybrid_gems.py                  # Phase 3.3
├── test_edge_cases.py                  # Phase 0.2
└── results/
    └── gems_baseline_[date].md         # Phase 2.1

.sisyphus/notepads/
├── strategy-baselines.md               # Phase 2.1
├── crossbreed-analysis.md              # Phase 3.1
└── gems-implementation-results.md      # Phase 4.3
```

---

## Existing Infrastructure to Leverage

### Current Best Strategy
- `enriched_hybrid_llm.py`: BM25 + semantic + LLM query rewriting (88.7% automated, 94% manual)

### Reusable Components
- `EmbedderMixin`: Embedding encoding
- `RetrievalStrategy`: Base class
- `EnrichmentCache`: Caching for LLM enrichments
- `call_llm`: LLM provider abstraction
- `reciprocal_rank_fusion`: RRF implementation in enriched_hybrid_llm.py
- `manual_test.py`: Manual testing framework

### Test Framework
- `run_benchmark.py`: Automated benchmark runner
- `manual_test.py`: Manual testing with Claude grading
- `ground_truth_realistic.json`: 53 facts across 20 queries

---

## Definition of Done

### Strategy Implementation
- [ ] Code compiles without errors
- [ ] Passes type checking
- [ ] Integrates with test_edge_cases.py
- [ ] Produces comparable output to enriched_hybrid_llm

### Strategy Verification
- [ ] All target queries tested
- [ ] All queries manually graded
- [ ] Results documented in markdown
- [ ] No regressions on passing queries

### Final Hybrid
- [ ] Achieves ≥96% accuracy (target: 98%)
- [ ] Latency <500ms per query
- [ ] A/B comparison documented
- [ ] Production recommendations written
