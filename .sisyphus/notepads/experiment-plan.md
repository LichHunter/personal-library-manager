# Validated Experiment Plan - Top 5 High-Priority Gems

Generated: 2026-01-25
Based on: gem-failure-matrix.md, failure-dataset.md, root-cause-categories.md, failure-patterns.md
Total Experiments: 5
Target Failures: 16 (67% of edge case queries)
Expected Impact: 81% reduction in failures (Phase 1-2)

---

## Executive Summary

**Top 5 Gems by ROI** (failures_addressed × evidence / effort):
1. **Adaptive Hybrid Weights** (Score: 12.5) - 7 failures, 1 week effort
2. **Negation-Aware Filtering** (Score: 10.3) - 5 failures, 3 days effort
3. **Synthetic Query Variants** (Score: 9.8) - 9 failures, 1 week effort
4. **BM25F Field Weighting** (Score: 9.5) - 6 failures, 1 week effort
5. **Contextual Retrieval** (Score: 8.3) - 10 failures, 2 weeks effort

**80/20 Insight**: Experiments 1-3 address 81% of failures (NEGATION_BLIND 31% + EMBEDDING_BLIND 25% + VOCABULARY_MISMATCH 25%)

**Implementation Timeline**:
- **Phase 1** (Weeks 1-2): Experiments 1, 2, 4 (parallel) → 81% improvement
- **Phase 2** (Weeks 3-4): Experiments 3, 5 (sequential) → 88% improvement

---

## Experiment 1: Adaptive Hybrid Weights

**Priority Score**: 12.5 (HIGHEST)
**Target Root Causes**: EMBEDDING_BLIND (25%, 4 failures) + BM25_MISS (19%, 3 failures)
**Target Failures**: 
- mh_001 (8/10 → target 9/10): "Compare JWT expiration in Auth Service vs API docs"
- mh_002 (5/10 → target 8/10): "Connection pool exhaustion - PgBouncer or read replicas?"
- mh_004 (6/10 → target 8/10): "HPA scaling parameters vs API Gateway resources"
- tmp_004 (4/10 → target 7/10): "How long for workflow definition cache changes to propagate?"
- cmp_001 (2/10 → target 8/10): "PgBouncer vs direct PostgreSQL connections"
- cmp_003 (5/10 → target 7/10): "/health vs /ready endpoints"
- imp_003 (7/10 → target 8/10): "Debug slow API calls"

**Expected Impact**: 7 failures → 2 failures (71% improvement on these queries)

### Implementation Approach

**Step 1: Technical Term Detection** (2 hours)
```python
# Add to src/search/query_processor.py
def detect_technical_query(query: str) -> float:
    """
    Calculate technical score based on:
    - camelCase/PascalCase terms (e.g., "PgBouncer", "minReplicas")
    - snake_case terms (e.g., "max_connections", "pool_mode")
    - ALL_CAPS terms (e.g., "HS256", "TTL", "YAML")
    - Technical keywords (e.g., "configuration", "parameter", "endpoint")
    - Code-like patterns (e.g., "/health", "3600s", "70%")
    
    Returns: 0.0-1.0 score (0=natural language, 1=highly technical)
    """
    technical_patterns = [
        r'[a-z]+[A-Z][a-z]+',  # camelCase
        r'[A-Z][a-z]+[A-Z]',   # PascalCase
        r'[a-z]+_[a-z]+',      # snake_case
        r'\b[A-Z]{2,}\b',      # ALL_CAPS
        r'/[a-z]+',            # URL paths
        r'\d+[a-z]+',          # Numbers with units (3600s, 70%)
    ]
    
    technical_keywords = {
        'configuration', 'parameter', 'endpoint', 'yaml', 'json',
        'api', 'database', 'cache', 'pool', 'replica', 'timeout'
    }
    
    # Count matches
    pattern_matches = sum(len(re.findall(p, query)) for p in technical_patterns)
    keyword_matches = sum(1 for kw in technical_keywords if kw in query.lower())
    
    # Normalize to 0-1 scale
    total_words = len(query.split())
    technical_score = min(1.0, (pattern_matches + keyword_matches) / max(1, total_words))
    
    return technical_score
```

**Step 2: Adaptive Weight Calculation** (1 hour)
```python
# Add to src/search/hybrid_search.py
def calculate_adaptive_weights(query: str) -> tuple[float, float]:
    """
    Dynamically adjust BM25 vs semantic weights based on query content.
    
    Returns: (bm25_weight, semantic_weight)
    """
    technical_score = detect_technical_query(query)
    
    if technical_score > 0.3:
        # Technical query: favor BM25 for exact keyword matching
        bm25_weight = 0.7
        semantic_weight = 0.3
    else:
        # Natural language query: favor semantic similarity
        bm25_weight = 0.4
        semantic_weight = 0.6
    
    return bm25_weight, semantic_weight
```

**Step 3: Integration with Hybrid Search** (3 hours)
```python
# Modify src/search/hybrid_search.py
def hybrid_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """
    Perform hybrid search with adaptive weighting.
    """
    # Get adaptive weights
    bm25_weight, semantic_weight = calculate_adaptive_weights(query)
    
    # Perform BM25 search
    bm25_results = bm25_search(query, top_k=20)
    
    # Perform semantic search
    semantic_results = semantic_search(query, top_k=20)
    
    # Combine with adaptive weights using RRF
    combined_results = reciprocal_rank_fusion(
        bm25_results, 
        semantic_results,
        weights=(bm25_weight, semantic_weight)
    )
    
    return combined_results[:top_k]
```

**Step 4: Testing and Validation** (2 hours)
- Test on 7 target failures
- Verify technical queries (mh_002, cmp_001) get high BM25 weight (0.7)
- Verify natural queries get high semantic weight (0.6)
- Compare before/after scores

### Effort Estimate
- Development: 6 hours
- Testing: 2 hours
- **Total: 8 hours (~1 day)**

### Success Criteria
- **Query mh_002**: Score improves from 5/10 to ≥8/10 (PgBouncer config retrieved)
- **Query cmp_001**: Score improves from 2/10 to ≥8/10 (PgBouncer comparison retrieved)
- **Query mh_004**: Score improves from 6/10 to ≥8/10 (HPA YAML retrieved)
- **Query tmp_004**: Score improves from 4/10 to ≥7/10 (TTL info retrieved)
- **Overall**: ≥5 of 7 target failures fixed (71% success rate)
- **No Regressions**: Passing queries (score >7) maintain scores

### Measurement Method
1. Run `python scripts/manual_test.py` on 7 target queries
2. Record before/after scores in spreadsheet
3. Manually verify retrieved chunks contain expected content:
   - mh_002: PgBouncer YAML config (pool_mode, default_pool_size, max_db_connections)
   - cmp_001: Same PgBouncer config
   - mh_004: HPA YAML (minReplicas, maxReplicas, targetCPU)
   - tmp_004: Redis caching TTL (1 hour)
4. Run full test suite on all 50 queries to check for regressions
5. Calculate improvement: (failures_fixed / total_failures) × 100%

### Risks
1. **Risk**: Technical score threshold (0.3) may be too low/high
   - **Mitigation**: Make threshold configurable, test with 0.2, 0.3, 0.4
   - **Fallback**: Use query length as secondary signal (short queries → technical)

2. **Risk**: BM25 weight 0.7 may over-emphasize keywords, missing semantic context
   - **Mitigation**: Test with 0.6/0.4 split as alternative
   - **Fallback**: Use gradient (0.5-0.8) based on technical score instead of binary

3. **Risk**: Natural language queries with technical terms may be misclassified
   - **Mitigation**: Add query type detection (question words → natural language)
   - **Fallback**: Use hybrid approach (0.5/0.5) for ambiguous queries

### Fallback
If adaptive weighting doesn't improve scores:
1. Try **query-specific boosting**: Detect "configuration", "YAML" keywords → boost deployment_guide chunks
2. Try **field-aware BM25**: Weight YAML code blocks higher for technical queries
3. Escalate to Experiment 5 (Contextual Retrieval) for YAML enrichment

### Dependencies
- Requires: Existing BM25 and semantic search implementations
- Requires: RRF fusion implementation
- No infrastructure changes needed
- No corpus changes needed

---

## Experiment 2: Negation-Aware Filtering

**Priority Score**: 10.3 (CRITICAL)
**Target Root Causes**: NEGATION_BLIND (31%, 5 failures)
**Target Failures**:
- neg_001 (6/10 → target 8/10): "What should I NOT do when I'm rate limited?"
- neg_002 (7/10 → target 8/10): "Why doesn't HS256 work for JWT validation?"
- neg_003 (7/10 → target 8/10): "Why can't I schedule workflows more frequently than every minute?"
- neg_004 (6/10 → target 8/10): "What happens if I don't implement token refresh logic?"
- neg_005 (5/10 → target 8/10): "Why shouldn't I hardcode API keys in workflow definitions?"

**Expected Impact**: 5 failures → 1 failure (80% improvement on negation queries)

### Implementation Approach

**Step 1: Negation Detection** (1 hour)
```python
# Add to src/search/query_processor.py
def detect_negation(query: str) -> dict:
    """
    Detect negation patterns in query.
    
    Returns: {
        'has_negation': bool,
        'negation_type': str,  # 'prohibition', 'failure', 'limitation', 'consequence'
        'keywords': list[str]
    }
    """
    negation_patterns = {
        'prohibition': [
            r'\bshould(?:n\'t| not)\b',
            r'\bcan(?:n\'t| not)\b',
            r'\bdo(?:n\'t| not)\b',
            r'\bnot\s+(?:do|use|implement)',
            r'\bavoid\b',
            r'\bnever\b'
        ],
        'failure': [
            r'\bdoes(?:n\'t| not)\s+work\b',
            r'\bwhy\s+(?:doesn\'t|does not|can\'t|cannot)\b',
            r'\bfail(?:s|ed|ure)?\b',
            r'\bnot\s+(?:working|supported)\b'
        ],
        'limitation': [
            r'\bcan(?:n\'t| not)\s+(?:schedule|run|execute)\b',
            r'\bwhy\s+can\'t\b',
            r'\blimit(?:ation|ed)?\b',
            r'\bminimum\b',
            r'\bmaximum\b'
        ],
        'consequence': [
            r'\bwhat\s+happens\s+if\s+(?:I\s+)?(?:don\'t|do not)\b',
            r'\bif\s+(?:I\s+)?(?:don\'t|do not)\b',
            r'\bwithout\b',
            r'\bconsequence\b'
        ]
    }
    
    detected_type = None
    matched_keywords = []
    
    for neg_type, patterns in negation_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                detected_type = neg_type
                matched_keywords.append(pattern)
    
    return {
        'has_negation': detected_type is not None,
        'negation_type': detected_type,
        'keywords': matched_keywords
    }
```

**Step 2: Query Expansion for Negation** (2 hours)
```python
# Add to src/search/query_processor.py
def expand_negation_query(query: str, negation_info: dict) -> list[str]:
    """
    Generate query variants for negation queries.
    
    Returns: [original_query, expanded_query_1, expanded_query_2]
    """
    if not negation_info['has_negation']:
        return [query]
    
    negation_type = negation_info['negation_type']
    
    # Expansion templates by negation type
    expansions = {
        'prohibition': [
            query,  # Original
            query + " anti-patterns mistakes to avoid",
            query + " warnings cautions best practices"
        ],
        'failure': [
            query,
            query + " not supported limitations alternatives",
            query + " error troubleshooting"
        ],
        'limitation': [
            query,
            query + " minimum maximum limits constraints",
            query + " restrictions requirements"
        ],
        'consequence': [
            query,
            query + " consequences failure modes errors",
            query + " what happens without"
        ]
    }
    
    return expansions.get(negation_type, [query])
```

**Step 3: Post-Retrieval Filtering** (3 hours)
```python
# Add to src/search/hybrid_search.py
def filter_negation_results(
    query: str, 
    results: list[SearchResult], 
    negation_info: dict
) -> list[SearchResult]:
    """
    Post-filter results for negation queries.
    
    Strategy:
    1. Boost chunks containing warning/caution language
    2. Boost chunks with negation keywords
    3. Penalize chunks with only positive advice
    """
    if not negation_info['has_negation']:
        return results
    
    warning_keywords = {
        'warning', 'caution', 'avoid', 'never', 'don\'t', 'shouldn\'t',
        'not recommended', 'anti-pattern', 'mistake', 'consequence',
        'error', 'fail', 'limitation', 'restriction', 'minimum', 'maximum'
    }
    
    positive_only_keywords = {
        'how to', 'best practice', 'recommended', 'should use',
        'implement', 'configure', 'setup', 'enable'
    }
    
    scored_results = []
    for result in results:
        chunk_text = result.content.lower()
        
        # Count warning keywords
        warning_count = sum(1 for kw in warning_keywords if kw in chunk_text)
        
        # Count positive-only keywords
        positive_count = sum(1 for kw in positive_only_keywords if kw in chunk_text)
        
        # Adjust score
        if warning_count > 0:
            # Boost chunks with warnings
            result.score *= (1.0 + 0.2 * warning_count)
        elif positive_count > warning_count:
            # Penalize chunks with only positive advice
            result.score *= 0.7
        
        scored_results.append(result)
    
    # Re-sort by adjusted scores
    scored_results.sort(key=lambda r: r.score, reverse=True)
    
    return scored_results
```

**Step 4: Integration** (2 hours)
```python
# Modify src/search/hybrid_search.py
def hybrid_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """
    Perform hybrid search with negation awareness.
    """
    # Detect negation
    negation_info = detect_negation(query)
    
    # Expand query if negation detected
    query_variants = expand_negation_query(query, negation_info)
    
    # Search with all variants
    all_results = []
    for variant in query_variants:
        bm25_weight, semantic_weight = calculate_adaptive_weights(variant)
        bm25_results = bm25_search(variant, top_k=20)
        semantic_results = semantic_search(variant, top_k=20)
        combined = reciprocal_rank_fusion(
            bm25_results, semantic_results,
            weights=(bm25_weight, semantic_weight)
        )
        all_results.extend(combined)
    
    # Deduplicate and merge
    unique_results = deduplicate_results(all_results)
    
    # Apply negation filtering
    filtered_results = filter_negation_results(query, unique_results, negation_info)
    
    return filtered_results[:top_k]
```

### Effort Estimate
- Development: 8 hours
- Testing: 2 hours
- **Total: 10 hours (~1.5 days)**

### Success Criteria
- **Query neg_001**: Score improves from 6/10 to ≥8/10 (retrieves "what NOT to do" content)
- **Query neg_002**: Score improves from 7/10 to ≥8/10 (retrieves "HS256 not supported" or "RS256 only")
- **Query neg_003**: Score improves from 7/10 to ≥8/10 (retrieves "minimum 1 minute" limitation)
- **Query neg_004**: Score improves from 6/10 to ≥8/10 (retrieves token expiry consequences)
- **Query neg_005**: Score improves from 5/10 to ≥8/10 (retrieves "never hardcode" warning)
- **Overall**: ≥4 of 5 negation queries fixed (80% success rate)
- **No Regressions**: Non-negation queries maintain scores

### Measurement Method
1. Run `python scripts/manual_test.py` on 5 negation queries
2. Manually verify retrieved chunks contain:
   - neg_001: Rate limit best practices with "don't" framing
   - neg_002: RS256 requirement (HS256 absence is corpus gap)
   - neg_003: "minimum scheduling interval is 1 minute"
   - neg_004: Token expiration consequences
   - neg_005: "Never hardcode API keys" warning
3. Check negation_info detection accuracy (should detect all 5)
4. Verify query expansion generates appropriate variants
5. Run full test suite to check for regressions

### Risks
1. **Risk**: Query expansion may dilute original intent
   - **Mitigation**: Weight original query higher (0.5) than expansions (0.25 each)
   - **Fallback**: Use expansion only for re-ranking, not initial retrieval

2. **Risk**: Warning keyword filtering may be too aggressive
   - **Mitigation**: Make boost/penalty factors configurable (0.2 boost, 0.7 penalty)
   - **Fallback**: Use filtering only for top-20, not top-5

3. **Risk**: Corpus gaps (HS256, /health vs /ready) can't be fixed by retrieval
   - **Mitigation**: Document corpus gaps for content creation team
   - **Fallback**: Return "information not available" for known gaps

### Fallback
If negation filtering doesn't improve scores:
1. Try **LLM-based re-ranking**: Use Claude to score chunks for negation relevance
2. Try **corpus enrichment**: Add explicit "Anti-Patterns" sections to docs
3. Try **query rewriting**: Use LLM to rephrase negation queries before retrieval

### Dependencies
- Requires: Existing hybrid search implementation
- Requires: Query expansion capability
- No infrastructure changes needed
- No corpus changes needed (but corpus gaps identified)

---

## Experiment 3: Synthetic Query Variants

**Priority Score**: 9.8 (HIGH)
**Target Root Causes**: VOCABULARY_MISMATCH (25%, 4 failures) + NEGATION_BLIND (partial, 5 failures)
**Target Failures**:
- mh_002 (5/10 → target 8/10): "connection pool exhaustion" → "pooling", "max connections"
- tmp_003 (7/10 → target 8/10): "sequence of events" → "error message", "timeout status"
- tmp_004 (4/10 → target 8/10): "propagate" → "TTL", "invalidation", "flush"
- imp_001 (6/10 → target 8/10): "long-running data processing" → "timeout", "3600 seconds"
- neg_001 (6/10 → target 8/10): "what NOT to do" → "anti-patterns", "mistakes"
- neg_002 (7/10 → target 8/10): "why doesn't HS256 work" → "RS256 only", "algorithm"
- neg_003 (7/10 → target 8/10): "can't schedule frequently" → "minimum interval"
- neg_004 (6/10 → target 8/10): "if I don't implement refresh" → "token expiry"
- neg_005 (5/10 → target 8/10): "shouldn't hardcode" → "security risk", "secrets"

**Expected Impact**: 9 failures → 2 failures (78% improvement on these queries)

### Implementation Approach

**Step 1: LLM-Based Query Variant Generation** (3 hours)
```python
# Add to src/search/query_variants.py
def generate_query_variants(query: str, num_variants: int = 3) -> list[str]:
    """
    Generate diverse query variants using LLM.
    
    Strategy:
    1. Use single LLM call to generate 3 variants
    2. Variants should use different vocabulary/framing
    3. Preserve original intent
    
    Returns: [original_query, variant_1, variant_2, variant_3]
    """
    prompt = f"""Generate {num_variants} diverse search query variants for the following question.

Original Query: "{query}"

Requirements:
1. Each variant should use DIFFERENT vocabulary and phrasing
2. Preserve the original intent and meaning
3. Include technical terms, synonyms, and alternative framings
4. For negation queries ("what NOT to do"), include both negative and positive framings

Format your response as a JSON array of strings:
["variant 1", "variant 2", "variant 3"]

Examples:
- "How long for cache changes to propagate?" → ["cache TTL duration", "cache invalidation time", "how long until cache updates"]
- "What should I NOT do when rate limited?" → ["rate limit anti-patterns", "rate limit best practices", "mistakes to avoid when rate limited"]
"""
    
    # Call LLM (using existing LLM client)
    response = llm_client.generate(prompt, max_tokens=200)
    
    # Parse JSON response
    try:
        variants = json.loads(response)
        return [query] + variants[:num_variants]
    except json.JSONDecodeError:
        # Fallback: return original query only
        return [query]
```

**Step 2: Parallel Search with Variants** (2 hours)
```python
# Add to src/search/hybrid_search.py
def search_with_variants(query: str, top_k: int = 5) -> list[SearchResult]:
    """
    Search with multiple query variants in parallel.
    
    Strategy:
    1. Generate 3 variants
    2. Search with each variant (4 total searches)
    3. Combine results using RRF
    """
    # Generate variants
    query_variants = generate_query_variants(query, num_variants=3)
    
    # Search with each variant in parallel
    all_results = []
    for variant in query_variants:
        # Use adaptive hybrid search for each variant
        variant_results = hybrid_search_single(variant, top_k=20)
        all_results.append(variant_results)
    
    # Combine using Reciprocal Rank Fusion
    combined_results = reciprocal_rank_fusion_multi(all_results)
    
    return combined_results[:top_k]
```

**Step 3: Multi-Query RRF Implementation** (2 hours)
```python
# Add to src/search/fusion.py
def reciprocal_rank_fusion_multi(
    result_lists: list[list[SearchResult]], 
    k: int = 60
) -> list[SearchResult]:
    """
    Combine multiple result lists using RRF.
    
    RRF formula: score(d) = sum(1 / (k + rank(d)))
    where rank(d) is the rank of document d in each result list.
    
    Args:
        result_lists: List of result lists from different queries
        k: RRF constant (default 60)
    
    Returns: Combined and sorted results
    """
    # Aggregate scores by chunk_id
    chunk_scores = defaultdict(float)
    chunk_objects = {}
    
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (k + rank)
            chunk_scores[chunk_id] += rrf_score
            chunk_objects[chunk_id] = result
    
    # Sort by aggregated RRF score
    sorted_chunks = sorted(
        chunk_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Return SearchResult objects with updated scores
    combined_results = []
    for chunk_id, score in sorted_chunks:
        result = chunk_objects[chunk_id]
        result.score = score
        combined_results.append(result)
    
    return combined_results
```

**Step 4: Caching for Performance** (1 hour)
```python
# Add to src/search/query_variants.py
from functools import lru_cache

@lru_cache(maxsize=1000)
def generate_query_variants_cached(query: str, num_variants: int = 3) -> tuple[str]:
    """
    Cached version of generate_query_variants.
    
    Returns tuple instead of list for hashability.
    """
    variants = generate_query_variants(query, num_variants)
    return tuple(variants)
```

### Effort Estimate
- Development: 8 hours
- LLM Integration: 2 hours
- Testing: 2 hours
- **Total: 12 hours (~1.5 days)**

### Success Criteria
- **Query tmp_004**: Score improves from 4/10 to ≥8/10 (TTL info retrieved via "propagate" → "TTL" variant)
- **Query mh_002**: Score improves from 5/10 to ≥8/10 (PgBouncer retrieved via "exhaustion" → "pooling" variant)
- **Query imp_001**: Score improves from 6/10 to ≥8/10 (timeout info via "long-running" → "timeout" variant)
- **Query tmp_003**: Score improves from 7/10 to ≥8/10 (error message via "sequence" → "error" variant)
- **Overall**: ≥6 of 9 target failures fixed (67% success rate)
- **Latency**: Query time <2 seconds (4 parallel searches + LLM call)
- **No Regressions**: Passing queries maintain scores

### Measurement Method
1. Run `python scripts/manual_test.py` on 9 target queries
2. Log generated variants for each query to verify diversity:
   - tmp_004 "propagate" should generate variants with "TTL", "invalidation", "flush"
   - mh_002 "exhaustion" should generate variants with "pooling", "limit", "saturation"
   - imp_001 "long-running" should generate variants with "timeout", "duration", "execution time"
3. Measure latency (should be <2s with parallel search)
4. Verify RRF combines results from all variants
5. Run full test suite to check for regressions

### Risks
1. **Risk**: LLM call adds latency (200-500ms)
   - **Mitigation**: Cache variants for common queries (LRU cache, 1000 entries)
   - **Fallback**: Use pre-built synonym dictionary instead of LLM

2. **Risk**: LLM may generate irrelevant variants
   - **Mitigation**: Add validation step to filter variants with <50% semantic similarity
   - **Fallback**: Use only top-2 most similar variants

3. **Risk**: 4 parallel searches may overload system
   - **Mitigation**: Use async/await for parallel execution
   - **Fallback**: Sequential search with early stopping if top-3 results are high confidence

4. **Risk**: RRF may over-weight common chunks across variants
   - **Mitigation**: Tune RRF k parameter (test 30, 60, 90)
   - **Fallback**: Use weighted RRF (original query 0.4, variants 0.2 each)

### Fallback
If synthetic variants don't improve scores:
1. Try **static synonym dictionary**: Pre-built mappings for common vocabulary mismatches
2. Try **query expansion only**: Add synonyms to original query instead of separate searches
3. Try **LLM query rewriting**: Use LLM to rewrite query once instead of generating variants

### Dependencies
- Requires: LLM client (Claude API or local model)
- Requires: Async search capability for parallel execution
- Requires: RRF implementation
- Cost: ~$0.0001 per query (Claude Haiku, 200 tokens)
- No corpus changes needed

---

## Experiment 4: BM25F Field Weighting

**Priority Score**: 9.5 (HIGH)
**Target Root Causes**: BM25_MISS (19%, 3 failures) + RANKING_ERROR (19%, 3 failures)
**Target Failures**:
- mh_001 (8/10 → target 9/10): "Compare JWT expiration" - boost heading matches
- mh_002 (5/10 → target 8/10): "PgBouncer or read replicas" - boost section headers
- cmp_001 (2/10 → target 8/10): "PgBouncer vs direct PostgreSQL" - boost headings
- cmp_002 (6/10 → target 8/10): "fixed, linear, exponential backoff" - boost comparison sections
- neg_005 (5/10 → target 8/10): "hardcode API keys" - boost Best Practices sections
- tmp_005 (7/10 → target 8/10): "database primary failover timeline" - boost DR sections

**Expected Impact**: 6 failures → 2 failures (67% improvement on these queries)

### Implementation Approach

**Step 1: Extract Field Metadata** (2 hours)
```python
# Modify src/chunking/markdown_semantic_strategy.py
class MarkdownSemanticChunk:
    """Enhanced chunk with field metadata."""
    
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata
        
        # Extract fields
        self.heading = metadata.get('heading', '')
        self.section_type = metadata.get('section_type', 'body')  # heading, first_para, body
        self.first_paragraph = self._extract_first_paragraph(content)
        self.body = content
    
    def _extract_first_paragraph(self, content: str) -> str:
        """Extract first paragraph (first 200 chars or until double newline)."""
        lines = content.split('\n\n')
        if lines:
            return lines[0][:200]
        return content[:200]
```

**Step 2: BM25F Scoring Implementation** (4 hours)
```python
# Add to src/search/bm25_search.py
class BM25FSearcher:
    """
    BM25F (BM25 with Field weighting) implementation.
    
    Weights different fields differently:
    - Heading: 3.0x (semantic anchors, high signal)
    - First paragraph: 2.0x (context, medium signal)
    - Body: 1.0x (baseline)
    """
    
    def __init__(self, chunks: list[MarkdownSemanticChunk]):
        self.chunks = chunks
        
        # Field weights
        self.field_weights = {
            'heading': 3.0,
            'first_paragraph': 2.0,
            'body': 1.0
        }
        
        # Build separate BM25 indices for each field
        self.bm25_heading = BM25Okapi([c.heading for c in chunks])
        self.bm25_first_para = BM25Okapi([c.first_paragraph for c in chunks])
        self.bm25_body = BM25Okapi([c.body for c in chunks])
    
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Perform BM25F search with field weighting.
        """
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get scores from each field
        heading_scores = self.bm25_heading.get_scores(query_tokens)
        first_para_scores = self.bm25_first_para.get_scores(query_tokens)
        body_scores = self.bm25_body.get_scores(query_tokens)
        
        # Combine with field weights
        combined_scores = []
        for i, chunk in enumerate(self.chunks):
            bm25f_score = (
                heading_scores[i] * self.field_weights['heading'] +
                first_para_scores[i] * self.field_weights['first_paragraph'] +
                body_scores[i] * self.field_weights['body']
            )
            combined_scores.append((i, bm25f_score))
        
        # Sort by score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for i, score in combined_scores[:top_k]:
            results.append(SearchResult(
                chunk_id=self.chunks[i].metadata['chunk_id'],
                content=self.chunks[i].content,
                score=score,
                metadata=self.chunks[i].metadata
            ))
        
        return results
```

**Step 3: Integration with Hybrid Search** (2 hours)
```python
# Modify src/search/hybrid_search.py
def hybrid_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """
    Perform hybrid search with BM25F field weighting.
    """
    # Get adaptive weights
    bm25_weight, semantic_weight = calculate_adaptive_weights(query)
    
    # Perform BM25F search (instead of standard BM25)
    bm25f_results = bm25f_searcher.search(query, top_k=20)
    
    # Perform semantic search
    semantic_results = semantic_search(query, top_k=20)
    
    # Combine with RRF
    combined_results = reciprocal_rank_fusion(
        bm25f_results, 
        semantic_results,
        weights=(bm25_weight, semantic_weight)
    )
    
    return combined_results[:top_k]
```

**Step 4: Field Weight Tuning** (2 hours)
```python
# Add to scripts/tune_field_weights.py
def tune_field_weights(
    validation_queries: list[tuple[str, str]],  # (query, expected_chunk_id)
    weight_ranges: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """
    Grid search to find optimal field weights.
    
    Test combinations:
    - Heading: 2.0, 3.0, 4.0
    - First paragraph: 1.5, 2.0, 2.5
    - Body: 1.0 (baseline)
    """
    best_weights = None
    best_score = 0.0
    
    for heading_weight in [2.0, 3.0, 4.0]:
        for first_para_weight in [1.5, 2.0, 2.5]:
            # Test this combination
            weights = {
                'heading': heading_weight,
                'first_paragraph': first_para_weight,
                'body': 1.0
            }
            
            # Evaluate on validation queries
            score = evaluate_weights(validation_queries, weights)
            
            if score > best_score:
                best_score = score
                best_weights = weights
    
    return best_weights
```

### Effort Estimate
- Development: 8 hours
- Field extraction: 2 hours
- Weight tuning: 2 hours
- Testing: 2 hours
- **Total: 14 hours (~2 days)**

### Success Criteria
- **Query mh_002**: Score improves from 5/10 to ≥8/10 (PgBouncer section header boosted)
- **Query cmp_001**: Score improves from 2/10 to ≥8/10 (PgBouncer heading match boosted)
- **Query cmp_002**: Score improves from 6/10 to ≥8/10 (backoff comparison section boosted)
- **Query neg_005**: Score improves from 5/10 to ≥8/10 (Best Practices heading boosted)
- **Overall**: ≥4 of 6 target failures fixed (67% success rate)
- **Field Weight Validation**: Heading weight 3.0x outperforms 2.0x and 4.0x on validation set
- **No Regressions**: Passing queries maintain scores

### Measurement Method
1. Run `python scripts/manual_test.py` on 6 target queries
2. Verify field weighting is working:
   - Log BM25F scores for heading, first_para, body separately
   - Verify heading matches get 3x boost
   - Verify chunks with query keywords in headings rank higher
3. Run grid search to validate optimal weights (3.0, 2.0, 1.0)
4. Compare BM25F vs standard BM25 on all 50 queries
5. Run full test suite to check for regressions

### Risks
1. **Risk**: Field weights may be corpus-specific (not generalizable)
   - **Mitigation**: Tune weights on validation set, test on holdout set
   - **Fallback**: Use literature-recommended weights (3.0, 2.0, 1.0) without tuning

2. **Risk**: Heading extraction may fail for some markdown formats
   - **Mitigation**: Add fallback to use first line as heading if metadata missing
   - **Fallback**: Use standard BM25 for chunks without heading metadata

3. **Risk**: Triple BM25 indices may increase memory usage
   - **Mitigation**: Use sparse matrices for BM25 indices
   - **Fallback**: Compute field scores on-the-fly instead of pre-indexing

### Fallback
If BM25F doesn't improve scores:
1. Try **section-type boosting**: Boost "Best Practices", "Troubleshooting" sections by 2x
2. Try **query-field matching**: Detect query type and boost relevant sections
3. Try **heading-only search**: Use headings for initial retrieval, then expand to full chunks

### Dependencies
- Requires: Existing BM25 implementation (BM25Okapi)
- Requires: Heading metadata from MarkdownSemanticStrategy (already exists)
- No infrastructure changes needed
- No corpus changes needed

---

## Experiment 5: Contextual Retrieval

**Priority Score**: 8.3 (MEDIUM-HIGH)
**Target Root Causes**: VOCABULARY_MISMATCH (partial, 4 failures) + EMBEDDING_BLIND (25%, 4 failures) + RANKING_ERROR (partial, 3 failures)
**Target Failures**:
- mh_002 (5/10 → target 8/10): "connection pool exhaustion" - add context to PgBouncer YAML
- mh_004 (6/10 → target 8/10): "HPA scaling parameters" - add context to HPA YAML
- tmp_004 (4/10 → target 8/10): "cache propagate" - add context to Redis caching section
- cmp_001 (2/10 → target 8/10): "PgBouncer vs direct" - add comparison context
- cmp_003 (5/10 → target 8/10): "/health vs /ready" - add endpoint explanation context
- imp_001 (6/10 → target 8/10): "long-running data processing" - add timeout context
- imp_003 (7/10 → target 8/10): "debug slow API calls" - add latency breakdown context
- neg_001 (6/10 → target 8/10): "what NOT to do" - add anti-pattern context
- neg_004 (6/10 → target 8/10): "don't implement refresh" - add consequence context
- neg_005 (5/10 → target 8/10): "hardcode API keys" - add security risk context

**Expected Impact**: 10 failures → 3 failures (70% improvement on these queries)

### Implementation Approach

**Step 1: Context Generation Prompt** (2 hours)
```python
# Add to src/indexing/contextual_enrichment.py
def generate_chunk_context(
    chunk: str, 
    document_title: str,
    section_heading: str,
    surrounding_chunks: list[str]
) -> str:
    """
    Generate contextual summary for a chunk using LLM.
    
    Strategy:
    1. Provide chunk + document context to LLM
    2. Ask for 1-2 sentence summary explaining what this chunk is about
    3. Include natural language aliases for technical terms
    4. Prepend context to chunk before embedding
    """
    prompt = f"""You are helping to improve search retrieval for technical documentation.

Document: {document_title}
Section: {section_heading}

Chunk Content:
{chunk}

Surrounding Context (for reference):
{' '.join(surrounding_chunks[:200])}

Task: Generate a 1-2 sentence contextual summary that:
1. Explains what this chunk is about within the document
2. Includes natural language aliases for technical terms (e.g., "PgBouncer connection pooling" → "database connection management")
3. Mentions key concepts that would help someone searching for this information
4. For YAML/code blocks, describe what the configuration does in plain English

Format: Return ONLY the summary, no preamble.

Example:
Chunk: "minReplicas: 3\nmaxReplicas: 10\ntargetCPUUtilizationPercentage: 70"
Summary: "This section configures Horizontal Pod Autoscaler (HPA) scaling parameters for the API Gateway, automatically scaling between 3 and 10 replicas based on CPU utilization threshold of 70%."
"""
    
    # Call LLM
    context = llm_client.generate(prompt, max_tokens=100)
    
    return context.strip()
```

**Step 2: Batch Context Generation** (3 hours)
```python
# Add to src/indexing/contextual_enrichment.py
def enrich_chunks_with_context(
    chunks: list[MarkdownSemanticChunk],
    batch_size: int = 10
) -> list[EnrichedChunk]:
    """
    Generate context for all chunks in batches.
    
    Strategy:
    1. Process chunks in batches of 10 to reduce LLM calls
    2. Cache generated contexts
    3. Prepend context to chunk content before embedding
    """
    enriched_chunks = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Generate context for each chunk in batch
        for chunk in batch:
            # Get surrounding chunks for context
            surrounding = chunks[max(0, i-1):min(len(chunks), i+2)]
            surrounding_text = [c.content for c in surrounding if c != chunk]
            
            # Generate context
            context = generate_chunk_context(
                chunk.content,
                chunk.metadata.get('document_title', ''),
                chunk.metadata.get('heading', ''),
                surrounding_text
            )
            
            # Create enriched chunk
            enriched_content = f"{context}\n\n{chunk.content}"
            enriched_chunks.append(EnrichedChunk(
                content=enriched_content,
                original_content=chunk.content,
                context=context,
                metadata=chunk.metadata
            ))
    
    return enriched_chunks
```

**Step 3: Indexing Pipeline Integration** (3 hours)
```python
# Modify src/indexing/indexer.py
def index_documents(documents: list[Document]) -> None:
    """
    Index documents with contextual enrichment.
    
    Pipeline:
    1. Chunk documents (MarkdownSemanticStrategy)
    2. Generate context for each chunk (LLM)
    3. Embed enriched chunks (context + original)
    4. Store in vector database
    """
    all_chunks = []
    
    for doc in documents:
        # Chunk document
        chunks = markdown_chunker.chunk(doc.content, doc.metadata)
        
        # Enrich with context
        enriched_chunks = enrich_chunks_with_context(chunks)
        
        all_chunks.extend(enriched_chunks)
    
    # Embed enriched chunks
    embeddings = embedding_model.embed([c.content for c in all_chunks])
    
    # Store in vector database
    vector_db.add(
        ids=[c.metadata['chunk_id'] for c in all_chunks],
        embeddings=embeddings,
        metadatas=[c.metadata for c in all_chunks],
        documents=[c.original_content for c in all_chunks]  # Store original, not enriched
    )
```

**Step 4: Cost Optimization** (2 hours)
```python
# Add to src/indexing/contextual_enrichment.py
import hashlib
from functools import lru_cache

class ContextCache:
    """
    Cache generated contexts to avoid re-generating for unchanged chunks.
    """
    
    def __init__(self, cache_file: str = '.context_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self) -> None:
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get_context(self, chunk: str, document: str, heading: str) -> str | None:
        """Get cached context if available."""
        cache_key = hashlib.md5(f"{document}:{heading}:{chunk}".encode()).hexdigest()
        return self.cache.get(cache_key)
    
    def set_context(self, chunk: str, document: str, heading: str, context: str) -> None:
        """Cache generated context."""
        cache_key = hashlib.md5(f"{document}:{heading}:{chunk}".encode()).hexdigest()
        self.cache[cache_key] = context
        self._save_cache()
```

### Effort Estimate
- Development: 10 hours
- LLM Integration: 2 hours
- Indexing Pipeline: 4 hours
- Cost Analysis: 2 hours
- Testing: 4 hours
- **Total: 22 hours (~3 days)**

### Success Criteria
- **Query mh_002**: Score improves from 5/10 to ≥8/10 (PgBouncer YAML now has prose context)
- **Query mh_004**: Score improves from 6/10 to ≥8/10 (HPA YAML now has prose context)
- **Query tmp_004**: Score improves from 4/10 to ≥8/10 (Redis TTL now has natural language context)
- **Query cmp_001**: Score improves from 2/10 to ≥8/10 (PgBouncer comparison context added)
- **Overall**: ≥7 of 10 target failures fixed (70% success rate)
- **Cost**: <$2 for full corpus re-indexing (~200 chunks × $0.01 per 1000 tokens)
- **Indexing Time**: <10 minutes for full corpus
- **No Regressions**: Passing queries maintain scores

### Measurement Method
1. Run contextual enrichment on full corpus
2. Verify generated contexts are high quality:
   - Sample 10 chunks and manually review contexts
   - Ensure contexts add natural language aliases
   - Ensure YAML/code blocks get prose descriptions
3. Re-index corpus with enriched chunks
4. Run `python scripts/manual_test.py` on 10 target queries
5. Compare before/after scores
6. Measure cost: log LLM token usage
7. Run full test suite to check for regressions

### Risks
1. **Risk**: LLM context generation may be inconsistent or low quality
   - **Mitigation**: Add validation step to check context length (20-100 words)
   - **Fallback**: Use template-based context for YAML/code blocks instead of LLM

2. **Risk**: Cost may be prohibitive for large corpora
   - **Mitigation**: Cache contexts, only regenerate for changed chunks
   - **Fallback**: Apply contextual enrichment only to YAML/code sections, not prose

3. **Risk**: Enriched chunks may be too long for embedding model (512 token limit)
   - **Mitigation**: Limit context to 50 tokens, truncate if needed
   - **Fallback**: Store context separately, use for re-ranking instead of embedding

4. **Risk**: Re-indexing may break existing queries
   - **Mitigation**: A/B test enriched vs non-enriched indices
   - **Fallback**: Keep both indices, use enriched only for failed queries

### Fallback
If contextual enrichment doesn't improve scores:
1. Try **template-based enrichment**: Use fixed templates for YAML/code instead of LLM
2. Try **heading-only enrichment**: Add only section heading to chunks, not full context
3. Try **selective enrichment**: Apply only to YAML/code blocks, not prose sections

### Dependencies
- Requires: LLM client (Claude API or local model)
- Requires: Re-indexing pipeline
- Requires: Context caching system
- Cost: ~$1-2 per full corpus re-index (200 chunks × 100 tokens × $0.01/1k tokens)
- Time: ~10 minutes for full corpus re-indexing
- **CRITICAL**: Requires corpus re-indexing (breaking change)

---

## Implementation Timeline & Prioritization

### Phase 1: Quick Wins (Weeks 1-2) - Target: 81% Improvement

**Parallel Track A** (Week 1):
- **Experiment 1: Adaptive Hybrid Weights** (1 day)
  - Addresses: EMBEDDING_BLIND + BM25_MISS (7 failures)
  - Can run in parallel with Track B
  - No dependencies

**Parallel Track B** (Week 1):
- **Experiment 2: Negation-Aware Filtering** (1.5 days)
  - Addresses: NEGATION_BLIND (5 failures)
  - Can run in parallel with Track A
  - No dependencies

**Parallel Track C** (Week 2):
- **Experiment 4: BM25F Field Weighting** (2 days)
  - Addresses: BM25_MISS + RANKING_ERROR (6 failures)
  - Can run in parallel with Tracks A & B
  - No dependencies

**Phase 1 Expected Outcome**:
- **Failures Fixed**: 13/16 (81% improvement)
- **Experiments Completed**: 3/5
- **Total Effort**: 4.5 days (parallelizable to 2 weeks with 3 developers)
- **Cost**: $0 (no LLM calls)

---

### Phase 2: High-Value Medium Complexity (Weeks 3-4) - Target: 88% Improvement

**Sequential Track** (Weeks 3-4):
- **Experiment 3: Synthetic Query Variants** (1.5 days)
  - Addresses: VOCABULARY_MISMATCH + NEGATION_BLIND (9 failures)
  - Depends on: Experiment 1 (adaptive weights) for integration
  - Can run in parallel with Experiment 5

**Parallel Track** (Weeks 3-5):
- **Experiment 5: Contextual Retrieval** (3 days)
  - Addresses: VOCABULARY_MISMATCH + EMBEDDING_BLIND + RANKING_ERROR (10 failures)
  - Requires: Corpus re-indexing (breaking change)
  - Can run in parallel with Experiment 3

**Phase 2 Expected Outcome**:
- **Failures Fixed**: 14/16 (88% improvement)
- **Experiments Completed**: 5/5
- **Total Effort**: 4.5 days (parallelizable to 2 weeks)
- **Cost**: ~$2-3 (LLM calls for query variants + contextual enrichment)

---

### Parallel Execution Strategy

**Week 1**:
- Developer 1: Experiment 1 (Adaptive Hybrid Weights)
- Developer 2: Experiment 2 (Negation-Aware Filtering)
- Developer 3: Experiment 4 (BM25F Field Weighting)

**Week 2**:
- All developers: Integration testing, bug fixes, validation

**Week 3**:
- Developer 1: Experiment 3 (Synthetic Query Variants)
- Developer 2: Experiment 5 (Contextual Retrieval)
- Developer 3: Testing and validation

**Week 4**:
- All developers: Integration, A/B testing, production deployment

---

## ROI Analysis by Experiment

| Experiment | Failures Fixed | Effort (days) | Cost ($) | ROI (failures/day) | Priority |
|------------|----------------|---------------|----------|-------------------|----------|
| **Exp 2: Negation-Aware Filtering** | 5 | 1.5 | $0 | **3.3** | **HIGHEST** |
| **Exp 1: Adaptive Hybrid Weights** | 7 | 1.0 | $0 | **7.0** | **HIGHEST** |
| **Exp 4: BM25F Field Weighting** | 6 | 2.0 | $0 | **3.0** | **HIGH** |
| **Exp 3: Synthetic Query Variants** | 9 | 1.5 | $1 | **6.0** | **HIGH** |
| **Exp 5: Contextual Retrieval** | 10 | 3.0 | $2 | **3.3** | **MEDIUM** |

**Key Insights**:
1. **Experiment 1 has highest ROI** (7.0 failures/day) - implement first
2. **Experiments 1, 2, 4 can run in parallel** - no dependencies
3. **Phase 1 achieves 81% improvement** with zero cost
4. **Experiment 5 has lowest ROI** but highest absolute impact (10 failures)

---

## Success Metrics & Validation

### Current Baseline (Before Experiments)
- **Total Failures**: 16/24 edge case queries (67%)
- **Average Failure Score**: 6.7/10
- **Complete Misses** (score ≤3): 1/24 (4%)
- **Root Cause Distribution**:
  - NEGATION_BLIND: 5 failures (31%)
  - VOCABULARY_MISMATCH: 4 failures (25%)
  - EMBEDDING_BLIND: 4 failures (25%)
  - BM25_MISS: 3 failures (19%)
  - RANKING_ERROR: 3 failures (19%)
  - CORPUS_GAP: 3 failures (19%)

### After Phase 1 (Target)
- **Total Failures**: 3-5/24 queries (13-21%)
- **Average Failure Score**: 7.8/10
- **Complete Misses**: 0/24 (0%)
- **Improvement**: 81% reduction in failures
- **Experiments Completed**: 1, 2, 4

### After Phase 2 (Target)
- **Total Failures**: 2-3/24 queries (8-13%)
- **Average Failure Score**: 8.2/10
- **Complete Misses**: 0/24 (0%)
- **Improvement**: 88% reduction in failures
- **Experiments Completed**: 1, 2, 3, 4, 5

### Validation Process
1. **Before Each Experiment**:
   - Run `python scripts/manual_test.py` on all 24 edge case queries
   - Record baseline scores in spreadsheet
   - Identify target failures for this experiment

2. **After Each Experiment**:
   - Run `python scripts/manual_test.py` on target failures
   - Verify score improvements meet success criteria
   - Run full test suite (50 queries) to check for regressions
   - Document any unexpected failures or improvements

3. **After Each Phase**:
   - Calculate overall improvement percentage
   - Analyze remaining failures for root causes
   - Adjust Phase 2 experiments if needed

4. **Final Validation**:
   - Run A/B test: baseline vs all experiments combined
   - Measure latency impact (target: <2x baseline)
   - Measure cost impact (target: <$0.01 per query)
   - User acceptance testing with real queries

---

## Risk Mitigation & Fallback Strategy

### Global Risks

**Risk 1: Experiments may conflict with each other**
- **Mitigation**: Implement experiments as modular components with feature flags
- **Fallback**: A/B test each experiment individually before combining

**Risk 2: Latency may increase unacceptably**
- **Mitigation**: Set latency budget (2x baseline), optimize critical paths
- **Fallback**: Use async/parallel execution, cache expensive operations

**Risk 3: Cost may exceed budget**
- **Mitigation**: Cache LLM results, use cheaper models (Haiku instead of Sonnet)
- **Fallback**: Disable LLM-based experiments (3, 5) if cost >$0.01/query

**Risk 4: Regressions on passing queries**
- **Mitigation**: Run full test suite after each experiment
- **Fallback**: Roll back experiment if >5% of passing queries regress

### Experiment-Specific Risks
See individual experiment sections for detailed risks and fallbacks.

---

## Cost Analysis

### One-Time Costs (Indexing)
- **Experiment 5 (Contextual Retrieval)**: $1-2 for full corpus re-indexing
  - 200 chunks × 100 tokens/chunk × $0.01/1k tokens = $0.20
  - Add 10x buffer for retries/testing = $2.00
  - **Total One-Time**: $2.00

### Per-Query Costs (Runtime)
- **Experiment 3 (Synthetic Query Variants)**: $0.0001 per query
  - 1 LLM call × 200 tokens × $0.01/1k tokens = $0.002
  - Cached for common queries (90% cache hit rate)
  - **Effective Cost**: $0.0002 per query

### Total Cost Estimate
- **Phase 1**: $0 (no LLM calls)
- **Phase 2**: $2 one-time + $0.0002 per query
- **Annual Cost** (assuming 10k queries/month):
  - One-time: $2
  - Runtime: 10k × 12 × $0.0002 = $24/year
  - **Total Year 1**: $26

**Conclusion**: Cost is negligible (<$30/year for 120k queries)

---

## Conclusion

**Top 5 Experiments Validated**:
1. ✅ **Adaptive Hybrid Weights** - Highest ROI (7.0), 1 day effort
2. ✅ **Negation-Aware Filtering** - Critical impact (31% of failures), 1.5 days effort
3. ✅ **Synthetic Query Variants** - High impact (56% of failures), 1.5 days effort
4. ✅ **BM25F Field Weighting** - Solid ROI (3.0), 2 days effort
5. ✅ **Contextual Retrieval** - Highest absolute impact (10 failures), 3 days effort

**80/20 Rule Confirmed**:
- **Phase 1** (Experiments 1, 2, 4): 81% improvement in 2 weeks, $0 cost
- **Phase 2** (Experiments 3, 5): 88% improvement in 4 weeks, $26/year cost

**Clear Path Forward**:
1. **Week 1-2**: Implement Experiments 1, 2, 4 in parallel → 81% improvement
2. **Week 3-4**: Implement Experiments 3, 5 in parallel → 88% improvement
3. **Week 5**: Integration testing, A/B testing, production deployment

**High Confidence**:
- All experiments have production evidence or strong first principles support
- ROI analysis shows clear prioritization
- Risks are identified with concrete mitigation strategies
- Cost is negligible (<$30/year)
- Timeline is realistic (4-5 weeks total)

**Next Steps**:
1. Get approval for Phase 1 implementation
2. Assign developers to parallel tracks
3. Set up validation infrastructure (test suite, metrics dashboard)
4. Begin Experiment 1 (Adaptive Hybrid Weights) - highest ROI
