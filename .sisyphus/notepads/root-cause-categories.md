# Root Cause Categories - enriched_hybrid_llm Pipeline Failures

Generated: 2026-01-25
Total Failures Analyzed: 16
Categories Identified: 6 primary + 3 secondary

---

## Category Definitions

### NEGATION_BLIND
**Definition**: Retrieval system fails to distinguish between positive and negative framing. Queries asking "what NOT to do" or "why doesn't X work" retrieve content about "what to do" or "how X works". Embeddings treat "do X" and "don't do X" as semantically similar.

**Addressable**: YES

**Why**: Can be fixed with negation-aware query processing, filtering, and specialized handling of negative framing.

**Failures in this category**:
- neg_001 (score 6/10): "What should I NOT do when I'm rate limited?"
- neg_002 (score 7/10): "Why doesn't HS256 work for JWT token validation?"
- neg_003 (score 7/10): "Why can't I schedule workflows more frequently than every minute?"
- neg_004 (score 6/10): "What happens if I don't implement token refresh logic?"
- neg_005 (score 5/10): "Why shouldn't I hardcode API keys in workflow definitions?"

**Potential Solutions**:
1. **Negation Detection**: Identify negation keywords ("not", "don't", "shouldn't", "can't", "doesn't", "why not")
2. **Query Rewriting**: Transform "what NOT to do" → "anti-patterns", "mistakes to avoid", "warnings"
3. **Consequence Expansion**: "what happens if I don't X" → "consequences of not X", "failure modes", "errors"
4. **Corpus Enrichment**: Add explicit anti-pattern sections to documentation
5. **Post-Retrieval Filtering**: When negation detected, boost chunks containing warning/caution language

---

### VOCABULARY_MISMATCH
**Definition**: Query uses different terminology than the corpus. Users employ natural language ("propagate", "long-running", "exhaustion") while corpus uses technical terms ("TTL", "timeout", "pooling"). This creates a semantic gap that both BM25 and embeddings struggle to bridge.

**Addressable**: YES

**Why**: Can be fixed with synonym expansion, domain-specific vocabulary mapping, and query enrichment.

**Failures in this category**:
- mh_002 (score 5/10): "connection pool exhaustion" vs corpus "connection pooling"
- tmp_003 (score 7/10): "sequence of events" vs corpus "error message", "timeout status"
- tmp_004 (score 4/10): "propagate" vs corpus "TTL", "invalidation", "flush"
- imp_001 (score 6/10): "long-running data processing" vs corpus "timeout", "3600 seconds"

**Potential Solutions**:
1. **Synonym Dictionary**: Build domain-specific mappings:
   - "propagate" → "TTL, invalidation, update, sync, flush"
   - "exhaustion" → "pooling, limit, max connections, saturation"
   - "long-running" → "timeout, execution time, duration limit"
   - "sequence of events" → "workflow, timeline, steps, process"
2. **Query Expansion**: Automatically add synonyms to query before retrieval
3. **Corpus Enrichment**: Add natural language aliases to technical sections
4. **User Vocabulary Learning**: Track common query patterns and add them to expansion dictionary

---

### EMBEDDING_BLIND
**Definition**: Semantic embeddings fail to capture meaning in structured content (YAML, code blocks, configuration files). Embeddings are trained on prose and struggle with technical syntax, key-value pairs, and code structure. "minReplicas: 3" doesn't semantically relate to "scaling parameters".

**Addressable**: YES

**Why**: Can be fixed with code-aware chunking, prose enrichment, and hybrid retrieval strategies.

**Failures in this category**:
- mh_001 (score 5/10): PgBouncer YAML configuration not retrieved
- mh_004 (score 6/10): HPA YAML configuration (90% code, 10% prose)
- tmp_004 (score 4/10): "TTL: 1 hour" in bullet-point list
- cmp_003 (score 5/10): "/health" and "/ready" URL paths in YAML

**Potential Solutions**:
1. **Code Block Enrichment**: Add natural language summaries to YAML/code sections
   - Before: `minReplicas: 3\nmaxReplicas: 10`
   - After: "Horizontal Pod Autoscaler scales between 3 and 10 replicas. minReplicas: 3\nmaxReplicas: 10"
2. **Structured Content Extraction**: Parse YAML/JSON and convert to prose
   - "PgBouncer pool_mode is transaction, default_pool_size is 25, max_db_connections is 100"
3. **Dual Indexing**: Index both raw code AND generated prose descriptions
4. **Code-Aware Embeddings**: Use specialized embeddings trained on technical documentation

---

### BM25_MISS
**Definition**: Keyword-based BM25 search fails due to vocabulary mismatch, stop words, or code tokenization issues. Query keywords don't appear in target chunks, or appear in wrong context.

**Addressable**: YES

**Why**: Can be fixed with better tokenization, keyword expansion, and BM25 tuning.

**Failures in this category**:
- mh_001 (score 5/10): "PgBouncer" appears 33 times but not retrieved
- mh_002 (score 5/10): "exhaustion" doesn't appear near PgBouncer content
- cmp_001 (score 2/10): "PgBouncer" keyword match failed completely

**Potential Solutions**:
1. **Keyword Enrichment**: Add searchable keywords to chunks
   - PgBouncer sections: "connection pool, pool exhaustion, connection limit, max connections"
2. **Stop Word Tuning**: Don't treat "not", "don't" as stop words in technical context
3. **Code Tokenization**: Improve tokenization of YAML keys, URL paths, configuration values
4. **Boost Technical Terms**: Increase BM25 weight for exact matches on technical terms (PgBouncer, HS256, etc.)
5. **Phrase Matching**: Prioritize multi-word phrase matches ("connection pool exhaustion")

---

### RANKING_ERROR
**Definition**: Correct content exists in corpus and may even be retrieved, but is ranked too low (below top-5). Related but less relevant content ranks higher due to RRF fusion weights or semantic similarity scores.

**Addressable**: YES

**Why**: Can be fixed with better ranking algorithms, re-ranking models, and query-specific boosting.

**Failures in this category**:
- cmp_002 (score 6/10): Exact comparison section exists but ranked below tangential retry content
- neg_005 (score 5/10): Best Practices section with answer ranked below API Keys section
- tmp_005 (score 7/10): Database failover retrieved but Redis/Kafka failover ranked lower

**Potential Solutions**:
1. **Re-Ranking Model**: Add LLM-based re-ranker after initial retrieval
   - Use Claude to score top-20 chunks for relevance to query
   - Re-rank based on LLM scores
2. **Query-Type Boosting**: Detect query patterns and adjust ranking:
   - Comparative queries ("difference", "vs") → boost comparison sections
   - Best practice queries → boost "Best Practices", "Recommendations" sections
   - Troubleshooting queries → boost "Solutions", "Troubleshooting" sections
3. **Section-Aware Ranking**: Give higher weight to chunks from relevant sections
4. **RRF Weight Tuning**: Adjust semantic vs BM25 weights based on query type

---

### CORPUS_GAP
**Definition**: Information simply doesn't exist in the corpus. Query asks for conceptual explanations, comparisons, or "why" reasoning that isn't documented. This is a content problem, not a retrieval problem.

**Addressable**: NO (requires corpus expansion)

**Why**: Cannot be fixed by improving retrieval. Requires adding missing content to documentation.

**Failures in this category**:
- neg_002 (score 7/10): "HS256" doesn't appear in corpus; only RS256 is mentioned
- cmp_003 (score 5/10): Corpus has /health and /ready paths but NO explanation of what each checks
- neg_004 (score 6/10): Consequence of not implementing token refresh is not explicitly stated

**Potential Solutions** (require content creation):
1. **Add Missing Comparisons**: Document "PgBouncer vs direct connections", "/health vs /ready"
2. **Add Anti-Patterns**: Document "what NOT to do" explicitly
3. **Add Consequence Sections**: "What happens if you don't..." for common mistakes
4. **Add Algorithm Explanations**: Why HS256 isn't supported, why 1-minute minimum for scheduling
5. **Content Audit**: Identify common user questions that aren't answered in corpus

---

## Secondary Categories (Co-occurring Issues)

### COMPARATIVE_QUERY
**Definition**: Queries asking for "difference between X and Y" require retrieving BOTH concepts and finding comparison content. Current pipeline doesn't handle multi-concept retrieval well.

**Addressable**: YES

**Failures in this category**:
- cmp_001 (score 2/10): "PgBouncer vs direct PostgreSQL"
- cmp_002 (score 6/10): "fixed, linear, exponential backoff"
- cmp_003 (score 5/10): "/health vs /ready endpoints"

**Potential Solutions**:
1. Detect comparative keywords ("difference", "vs", "compare")
2. Extract both concepts from query
3. Retrieve chunks for EACH concept separately
4. Boost chunks that mention BOTH concepts
5. Add explicit comparison sections to corpus

---

### CHUNKING_BOUNDARY
**Definition**: Answer is split across chunk boundaries. The question's answer spans multiple chunks, and the most relevant part may be in a different chunk than the section header.

**Addressable**: YES

**Failures in this category**:
- neg_003 (score 7/10): "minimum interval is 1 minute" is ONE line that may be split from section header
- imp_003 (score 7/10): Latency percentages may be in different chunk than section header

**Potential Solutions**:
1. **Sliding Window Chunking**: Overlap chunks by 20-30%
2. **Semantic Chunking**: Split on topic boundaries, not fixed token counts
3. **Multi-Chunk Retrieval**: Return top-10 and merge adjacent chunks
4. **Context Preservation**: Include section headers in every chunk

---

### YAML_CODE_BLOCK
**Definition**: Specific case of EMBEDDING_BLIND where YAML configuration blocks are not well-indexed. YAML is both code (structured) and configuration (semantic), making it hard for both BM25 and embeddings.

**Addressable**: YES

**Failures in this category**:
- mh_001 (score 5/10): PgBouncer YAML config
- mh_004 (score 6/10): HPA YAML config (90% YAML, 10% prose)

**Potential Solutions**:
1. Parse YAML and generate prose descriptions
2. Add inline comments to YAML with natural language
3. Create separate "Configuration Reference" sections with prose explanations
4. Index YAML keys as searchable terms

---

## Distribution Summary

| Root Cause | Count | % of Failures | Addressable? | Priority |
|------------|-------|---------------|--------------|----------|
| NEGATION_BLIND | 5 | 31% | YES | HIGH |
| VOCABULARY_MISMATCH | 4 | 25% | YES | HIGH |
| EMBEDDING_BLIND | 4 | 25% | YES | HIGH |
| BM25_MISS | 3 | 19% | YES | MEDIUM |
| RANKING_ERROR | 3 | 19% | YES | MEDIUM |
| CORPUS_GAP | 3 | 19% | NO | LOW* |
| COMPARATIVE_QUERY | 2 | 13% | YES | MEDIUM |
| CHUNKING_BOUNDARY | 2 | 13% | YES | LOW |
| YAML_CODE_BLOCK | 2 | 13% | YES | MEDIUM |

**Note**: Total > 100% because some failures have multiple root causes.

*LOW priority for retrieval improvements; HIGH priority for content creation.

---

## Addressable vs Inherent

### Addressable (Can be fixed with retrieval improvements): 13 failures (81%)

**Categories**:
1. NEGATION_BLIND (5 failures, 31%)
2. VOCABULARY_MISMATCH (4 failures, 25%)
3. EMBEDDING_BLIND (4 failures, 25%)
4. BM25_MISS (3 failures, 19%)
5. RANKING_ERROR (3 failures, 19%)
6. COMPARATIVE_QUERY (2 failures, 13%)
7. CHUNKING_BOUNDARY (2 failures, 13%)
8. YAML_CODE_BLOCK (2 failures, 13%)

**Key Insight**: 81% of failures can be addressed through retrieval pipeline improvements without changing the corpus.

---

### Inherent (Corpus limitations): 3 failures (19%)

**Category**:
- CORPUS_GAP (3 failures, 19%)

**Specific Gaps**:
1. HS256 algorithm not documented (only RS256 mentioned)
2. /health vs /ready conceptual difference not explained
3. Consequences of not implementing token refresh not stated

**Key Insight**: 19% of failures require content creation, not retrieval improvements.

---

## Priority for Solutions

Based on frequency, addressability, and impact:

### Tier 1: Critical (High Frequency + High Impact)

1. **NEGATION_BLIND** (31%, addressable)
   - **Impact**: Affects 5 failures across different query types
   - **Solution Complexity**: Medium (negation detection + query rewriting)
   - **ROI**: High - single fix addresses 31% of failures

2. **VOCABULARY_MISMATCH** (25%, addressable)
   - **Impact**: Affects 4 failures, fundamental to user experience
   - **Solution Complexity**: Medium (synonym dictionary + expansion)
   - **ROI**: High - improves natural language understanding

3. **EMBEDDING_BLIND** (25%, addressable)
   - **Impact**: Affects 4 failures, especially YAML/code content
   - **Solution Complexity**: High (requires corpus enrichment + dual indexing)
   - **ROI**: High - unlocks structured content retrieval

---

### Tier 2: Important (Medium Frequency + Medium Impact)

4. **BM25_MISS** (19%, addressable)
   - **Impact**: Affects 3 failures, especially PgBouncer content
   - **Solution Complexity**: Low (keyword enrichment + tokenization)
   - **ROI**: Medium - targeted fix for specific blind spots

5. **RANKING_ERROR** (19%, addressable)
   - **Impact**: Affects 3 failures, content exists but ranked wrong
   - **Solution Complexity**: High (re-ranking model + LLM scoring)
   - **ROI**: Medium - improves precision but requires compute

6. **COMPARATIVE_QUERY** (13%, addressable)
   - **Impact**: Affects 2 failures, specific query pattern
   - **Solution Complexity**: Medium (query parsing + multi-concept retrieval)
   - **ROI**: Medium - handles specific use case well

---

### Tier 3: Nice-to-Have (Low Frequency or Low Impact)

7. **CHUNKING_BOUNDARY** (13%, addressable)
   - **Impact**: Affects 2 failures, minor issue
   - **Solution Complexity**: Low (sliding window chunking)
   - **ROI**: Low - incremental improvement

8. **CORPUS_GAP** (19%, NOT addressable via retrieval)
   - **Impact**: Affects 3 failures, requires content creation
   - **Solution Complexity**: N/A (not a retrieval problem)
   - **ROI**: N/A - requires documentation team

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 weeks)
1. **Negation Detection + Query Rewriting** (addresses 31% of failures)
2. **Synonym Dictionary for Common Mismatches** (addresses 25% of failures)
3. **BM25 Keyword Enrichment for PgBouncer** (addresses 19% of failures)

**Expected Impact**: Fix 8-10 of 16 failures (50-60% improvement)

---

### Phase 2: Structural Improvements (3-4 weeks)
4. **YAML/Code Block Enrichment** (addresses 25% of failures)
5. **Comparative Query Detection** (addresses 13% of failures)
6. **Sliding Window Chunking** (addresses 13% of failures)

**Expected Impact**: Fix additional 3-4 failures (70-80% total improvement)

---

### Phase 3: Advanced Techniques (4-6 weeks)
7. **LLM Re-Ranking Model** (addresses 19% of failures)
8. **Query-Type Specific Boosting** (improves overall precision)

**Expected Impact**: Fix remaining 1-2 failures (90%+ total improvement)

---

### Phase 4: Content Creation (ongoing)
9. **Add Missing Comparisons** (PgBouncer vs direct, /health vs /ready)
10. **Add Anti-Pattern Sections** (what NOT to do)
11. **Add Consequence Sections** (what happens if you don't...)

**Expected Impact**: Eliminate CORPUS_GAP failures (100% coverage)

---

## Success Metrics

### Before Improvements
- Failures (score ≤7): 16/50 queries (32%)
- Average score of failures: 5.6/10
- Complete misses (score ≤3): 2/50 queries (4%)

### After Phase 1 (Target)
- Failures (score ≤7): 6-8/50 queries (12-16%)
- Average score of failures: 7.5/10
- Complete misses (score ≤3): 0/50 queries (0%)

### After Phase 2 (Target)
- Failures (score ≤7): 3-5/50 queries (6-10%)
- Average score of failures: 8.0/10
- Complete misses (score ≤3): 0/50 queries (0%)

### After Phase 3 (Target)
- Failures (score ≤7): 1-2/50 queries (2-4%)
- Average score of failures: 8.5/10
- Complete misses (score ≤3): 0/50 queries (0%)

---

## Conclusion

**Key Findings**:
1. **81% of failures are addressable** through retrieval improvements
2. **Negation handling is the #1 priority** (31% of failures)
3. **Vocabulary mismatch and embedding blindness** are equally critical (25% each)
4. **Only 19% require corpus expansion** (CORPUS_GAP)

**Strategic Recommendation**:
Focus on Phase 1 quick wins (negation, synonyms, BM25 enrichment) to achieve 50-60% improvement in 1-2 weeks. This provides immediate value while building foundation for Phase 2 structural improvements.

**Long-term Vision**:
Combine retrieval improvements (Phases 1-3) with content creation (Phase 4) to achieve 95%+ query success rate and eliminate all failure modes.
