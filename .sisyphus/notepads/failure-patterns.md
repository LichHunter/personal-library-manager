# Cross-Cutting Failure Patterns - 80/20 Analysis

Generated: 2026-01-25
Dataset: 16 failed queries (score ≤7) from enriched_hybrid_llm strategy
Analysis: Cross-cutting patterns across query types, content types, document sections, keywords, and position

---

## Executive Summary: The 80/20 Rule

**Two patterns explain 81% of all failures:**

1. **Negation/Implicit Query Framing** (56% of failures, 9/16)
2. **YAML/Code Content Blindness** (25% of failures, 4/16)

These are the critical leverage points for improvement. Fixing these two patterns would resolve 13 of 16 failures.

---

## Pattern 1: Negation/Implicit Query Framing

**Coverage**: 9/16 failures (56%)

**Description**: Queries framed with negation ("what NOT to do", "why doesn't X work") or implicit vocabulary ("long-running", "propagate") fail because embeddings don't distinguish positive from negative framing, and user vocabulary doesn't match corpus technical terms.

### Sub-Pattern 1A: Negation Queries (5 failures, 31%)

**Failures**:
- **neg_001** (6/10): "What should I NOT do when I'm rate limited?"
- **neg_002** (7/10): "Why doesn't HS256 work for JWT token validation?"
- **neg_003** (7/10): "Why can't I schedule workflows more frequently than every minute?"
- **neg_004** (6/10): "What happens if I don't implement token refresh logic?"
- **neg_005** (5/10): "Why shouldn't I hardcode API keys in workflow definitions?"

**Failure Rate**: 5/5 negation queries failed (100%)

**Why This Fails**:
- Embeddings treat "do X" and "don't do X" as semantically similar
- BM25 treats "not", "don't", "shouldn't" as stop words
- Corpus describes what TO do, not what NOT to do
- Query rewriting may convert negation to positive form, losing intent

**Evidence**:
- neg_001: Query asks "what NOT to do", corpus has "Best Practices" (positive framing)
- neg_002: Query asks "why doesn't HS256 work", corpus only mentions RS256 is used
- neg_005: Query asks "why shouldn't hardcode", corpus says "Never hardcode" but ranked low

**Statistical Support**:
- 100% of negation queries failed (5/5)
- Average score: 6.2/10 (borderline failure)
- All 5 have NEGATION_BLIND as root cause

### Sub-Pattern 1B: Implicit Vocabulary Queries (4 failures, 25%)

**Failures**:
- **imp_001** (6/10): "Best practice for handling long-running data processing that might exceed time limits"
- **tmp_003** (7/10): "What's the sequence of events when a workflow execution times out?"
- **tmp_004** (4/10): "How long does it take for workflow definition cache changes to propagate?"
- **mh_002** (5/10): "If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?"

**Failure Rate**: 4/4 implicit queries failed (100%)

**Why This Fails**:
- Users use natural language: "propagate", "long-running", "exhaustion", "sequence of events"
- Corpus uses technical terms: "TTL", "timeout", "pooling", "error message"
- Semantic gap that both BM25 and embeddings struggle to bridge

**Evidence**:
- tmp_004: "propagate" vs corpus "TTL", "invalidation", "flush" - complete vocabulary mismatch
- imp_001: "long-running data processing" vs corpus "timeout", "3600 seconds"
- mh_002: "connection pool exhaustion" vs corpus "connection pooling" (noun vs problem state)
- tmp_003: "sequence of events" vs corpus "error message", "TIMEOUT status"

**Statistical Support**:
- 100% of implicit queries failed (4/4)
- Average score: 5.5/10 (worst performing query type)
- All 4 have VOCABULARY_MISMATCH as root cause

### Pattern 1 Impact

**Total Coverage**: 9/16 failures (56%)

**Addressable**: YES

**Solutions**:
1. **Negation Detection**: Identify negation keywords ("not", "don't", "shouldn't", "can't", "doesn't", "why not")
2. **Negation-Aware Query Expansion**: 
   - "what NOT to do" → "anti-patterns", "mistakes to avoid", "warnings", "cautions"
   - "why doesn't X work" → "X not supported", "X limitations", "alternatives to X"
3. **Synonym Dictionary for Vocabulary Mismatch**:
   - "propagate" → "TTL, invalidation, update, sync, flush"
   - "exhaustion" → "pooling, limit, max connections, saturation"
   - "long-running" → "timeout, execution time, duration limit"
   - "sequence of events" → "workflow, timeline, steps, process, error handling"
4. **Corpus Enrichment**: Add natural language aliases to technical sections

**Expected Impact**: Fixing this pattern would resolve 9/16 failures (56% improvement)

---

## Pattern 2: YAML/Code Content Blindness

**Coverage**: 4/16 failures (25%)

**Description**: Queries about configuration values, YAML settings, or code blocks fail because semantic embeddings are trained on prose and struggle with structured content. "minReplicas: 3" doesn't semantically relate to "scaling parameters".

**Failures**:
- **mh_002** (5/10): "If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?"
  - Missing: PgBouncer YAML config (pool_mode=transaction, default_pool_size=25, max_db_connections=100)
  - Location: deployment_guide.md lines 473-561 (90% YAML code)
- **mh_004** (6/10): "How do the HPA scaling parameters relate to the API Gateway resource requirements?"
  - Missing: HPA YAML config (minReplicas: 3, maxReplicas: 10, targetCPUUtilizationPercentage: 70)
  - Location: deployment_guide.md lines 804-857 (90% YAML, 10% prose)
- **cmp_001** (2/10): "What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?"
  - Missing: Same PgBouncer YAML config as mh_002
  - Location: deployment_guide.md lines 473-561
- **cmp_003** (5/10): "What's the difference between /health and /ready endpoints?"
  - Missing: Health check paths in YAML (livenessProbe: path: /health, readinessProbe: path: /ready)
  - Location: deployment_guide.md lines 277-290

**Failure Rate**: 4/4 YAML-heavy queries failed (100%)

**Why This Fails**:
- Semantic embeddings trained on prose, not Kubernetes YAML or configuration files
- YAML key-value pairs don't have strong semantic meaning in embedding space
- BM25 tokenization struggles with YAML structure (colons, indentation, keys)
- Section headers don't contain enough keywords ("### PgBouncer Configuration" doesn't mention "connection pool exhaustion")

**Evidence**:
- mh_002: "PgBouncer" appears 33 times in corpus but retrieval returned 0 PgBouncer chunks
- mh_004: HPA section is 90% YAML code (lines 806-850), only 1 line of prose before 44 lines of YAML
- cmp_001: Worst failure (2/10) - complete retrieval miss despite "PgBouncer" keyword
- tmp_004: "TTL: 1 hour" is in a bullet-point list (line 491), not prose

**Statistical Support**:
- 100% of YAML-heavy queries failed (4/4)
- Average score: 4.5/10 (severe failures)
- All 4 have EMBEDDING_BLIND as root cause
- 3/4 also have BM25_MISS as secondary root cause

**Addressable**: YES

**Solutions**:
1. **Code Block Enrichment**: Add natural language summaries to YAML/code sections
   - Before: `minReplicas: 3\nmaxReplicas: 10`
   - After: "Horizontal Pod Autoscaler scales between 3 and 10 replicas. minReplicas: 3\nmaxReplicas: 10"
2. **Structured Content Extraction**: Parse YAML/JSON and convert to prose
   - "PgBouncer pool_mode is transaction, default_pool_size is 25, max_db_connections is 100"
3. **Dual Indexing**: Index both raw code AND generated prose descriptions
4. **Keyword Enrichment**: Add searchable keywords to YAML sections
   - PgBouncer sections: "connection pool, pool exhaustion, connection limit, max connections"
5. **Code-Aware Embeddings**: Use specialized embeddings trained on technical documentation

**Expected Impact**: Fixing this pattern would resolve 4/16 failures (25% improvement)

---

## Pattern 3: deployment_guide Section Retrieval Failures

**Coverage**: 6/16 failures (38%)

**Description**: Queries requiring information from deployment_guide.md sections consistently fail. This overlaps heavily with Pattern 2 (YAML/Code) because deployment_guide contains mostly YAML configurations.

**Failures**:
- **mh_002** (5/10): PgBouncer Configuration (lines 473-561)
- **mh_004** (6/10): Horizontal Pod Autoscaling (lines 804-857)
- **cmp_001** (2/10): PgBouncer Configuration (lines 473-561)
- **cmp_003** (5/10): Health Check Configuration (lines 277-290)
- **neg_005** (5/10): Best Practices section (lines 723-737)
- **imp_001** (6/10): Timeout solutions (referenced from troubleshooting_guide)

**Why This Fails**:
- deployment_guide.md is heavily YAML-focused (60-70% code blocks)
- Sections are late in the document (PgBouncer at line 473, HPA at line 804)
- Section headers are technical ("### PgBouncer Configuration") without natural language keywords
- Embeddings struggle with configuration-heavy content

**Overlap with Pattern 2**: 4/6 failures are YAML-related (67% overlap)

**Addressable**: YES (same solutions as Pattern 2)

**Expected Impact**: This is a symptom of Pattern 2, not a separate root cause. Fixing Pattern 2 will resolve most deployment_guide failures.

---

## Pattern 4: Comparative Query Failures

**Coverage**: 3/5 comparative queries failed (60%)

**Description**: Queries asking for "difference between X and Y" require retrieving BOTH concepts and finding comparison content. Current pipeline doesn't handle multi-concept retrieval well.

**Failures**:
- **cmp_001** (2/10): "What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?"
- **cmp_002** (6/10): "How do fixed, linear, and exponential backoff strategies differ for retries?"
- **cmp_003** (5/10): "What's the difference between /health and /ready endpoints?"

**Failure Rate**: 3/5 comparative queries failed (60%)

**Why This Fails**:
- Comparative queries need BOTH concepts retrieved
- Corpus often doesn't have explicit comparison sections
- Embeddings don't prioritize chunks mentioning both concepts
- BM25 may match one concept but not the other

**Evidence**:
- cmp_001: Needs both "PgBouncer" AND "direct PostgreSQL" - only got PostgreSQL content
- cmp_002: Exact comparison exists (user_guide.md lines 523-539) but ranked below tangential retry content
- cmp_003: Corpus has /health and /ready paths but NO explanation of conceptual difference

**Statistical Support**:
- 60% of comparative queries failed (3/5)
- Average score: 4.3/10 (severe failures)
- 2/3 have COMPARATIVE_QUERY as root cause
- 1/3 has CORPUS_GAP (cmp_003 - comparison doesn't exist in corpus)

**Addressable**: PARTIAL (2/3 addressable, 1/3 requires corpus expansion)

**Solutions**:
1. **Comparative Query Detection**: Identify keywords ("difference", "vs", "compare", "differ")
2. **Multi-Concept Extraction**: Parse query to extract both concepts
3. **Dual Retrieval**: Retrieve chunks for EACH concept separately
4. **Boost Multi-Mention Chunks**: Prioritize chunks mentioning BOTH concepts
5. **Corpus Enrichment**: Add explicit comparison sections (e.g., "PgBouncer vs Direct Connections")

**Expected Impact**: Fixing this pattern would resolve 2/16 failures (13% improvement)

---

## Pattern 5: Vocabulary Mismatch Keywords

**Coverage**: 4/16 failures (25%)

**Description**: Specific keyword mismatches between user queries and corpus terminology cause retrieval failures.

**Common Mismatches**:

| User Query Term | Corpus Term | Failures |
|-----------------|-------------|----------|
| "propagate" | "TTL", "invalidation", "flush" | tmp_004 (4/10) |
| "exhaustion" | "pooling", "limit" | mh_002 (5/10) |
| "long-running" | "timeout", "3600 seconds" | imp_001 (6/10) |
| "sequence of events" | "error message", "status" | tmp_003 (7/10) |

**Why This Fails**:
- Users employ natural, descriptive language
- Corpus uses precise technical terminology
- No synonym expansion in current pipeline
- Embeddings don't bridge this semantic gap well

**Addressable**: YES (same as Pattern 1B solutions)

**Expected Impact**: This is a sub-pattern of Pattern 1, already counted in that 56% coverage.

---

## Pattern 6: troubleshooting_guide Section Misses

**Coverage**: 4/16 failures (25%)

**Description**: Queries requiring troubleshooting solutions or error handling information fail to retrieve the correct troubleshooting_guide.md sections.

**Failures**:
- **tmp_003** (7/10): Timeout Errors section (lines 518-587) - retrieved timeout info but not error message/status
- **imp_001** (6/10): Timeout Solutions (lines 549-587) - retrieved limits but not solutions
- **mh_002** (5/10): Connection Pool Exhaustion (line 380) - didn't retrieve PgBouncer as solution
- **neg_001** (6/10): Rate limit handling - retrieved code but not "what NOT to do" framing

**Why This Fails**:
- troubleshooting_guide sections ranked lower than user_guide or architecture_overview
- Solutions sections buried within long troubleshooting entries
- Query framing doesn't match troubleshooting vocabulary

**Addressable**: YES

**Solutions**:
1. **Query-Type Boosting**: Detect troubleshooting queries and boost troubleshooting_guide sections
2. **Section-Aware Ranking**: Give higher weight to "Solutions", "Troubleshooting" sections
3. **Keyword Enrichment**: Add problem keywords to solution sections

**Expected Impact**: Fixing this pattern would resolve 4/16 failures (25% improvement), but overlaps with Patterns 1 and 2.

---

## The 80/20 Rule: Critical Leverage Points

### Top 2 Patterns Explain 81% of Failures

**Pattern 1: Negation/Implicit Query Framing**
- Coverage: 9/16 failures (56%)
- Addressable: YES
- Complexity: Medium (negation detection + synonym dictionary)
- ROI: **HIGHEST** - single fix addresses majority of failures

**Pattern 2: YAML/Code Content Blindness**
- Coverage: 4/16 failures (25%)
- Addressable: YES
- Complexity: High (corpus enrichment + dual indexing)
- ROI: **HIGH** - unlocks structured content retrieval

**Combined Coverage**: 13/16 failures (81%)

### Why These Two Patterns?

1. **Minimal Overlap**: Only 1 failure (mh_002) has both patterns
2. **Distinct Root Causes**: Pattern 1 is query-side, Pattern 2 is corpus-side
3. **High Addressability**: Both can be fixed without corpus expansion
4. **Complementary Solutions**: Can be implemented in parallel

---

## Secondary Patterns (Remaining 19%)

**Pattern 3: Comparative Queries** (3 failures, 19%)
- Overlaps with Pattern 2 (cmp_001 is YAML-related)
- 1 failure (cmp_003) has CORPUS_GAP - requires content creation

**Pattern 4: Ranking Errors** (3 failures, 19%)
- Correct content exists but ranked too low
- Can be addressed with re-ranking model

**Pattern 5: Corpus Gaps** (3 failures, 19%)
- Information doesn't exist in corpus
- Requires content creation, not retrieval improvements

---

## Statistical Summary

### Pattern Coverage Distribution

| Pattern | Failures | % Coverage | Addressable | Priority |
|---------|----------|------------|-------------|----------|
| Negation/Implicit Framing | 9 | 56% | YES | **CRITICAL** |
| YAML/Code Blindness | 4 | 25% | YES | **HIGH** |
| deployment_guide Misses | 6 | 38% | YES | Medium* |
| Comparative Queries | 3 | 19% | PARTIAL | Medium |
| troubleshooting_guide Misses | 4 | 25% | YES | Medium* |
| Vocabulary Mismatch | 4 | 25% | YES | Medium* |

*These are symptoms of Patterns 1 and 2, not separate root causes.

### Query Type Failure Rates

| Query Type | Total | Failed | Failure Rate | Avg Score |
|------------|-------|--------|--------------|-----------|
| Negation | 5 | 5 | **100%** | 6.2/10 |
| Implicit | 4 | 4 | **100%** | 5.5/10 |
| Comparative | 5 | 3 | 60% | 6.6/10 |
| Multi-hop | 5 | 3 | 60% | 6.4/10 |
| Temporal | 5 | 2 | 40% | 7.2/10 |

### Content Type Failure Rates

| Content Type | Failures | % of Total | Avg Score |
|--------------|----------|------------|-----------|
| YAML/Code | 4 | 25% | 4.5/10 |
| Configuration Values | 3 | 19% | 5.3/10 |
| Prose | 9 | 56% | 6.4/10 |

---

## Recommended Action Plan

### Phase 1: Address Pattern 1 (Negation/Implicit Framing)

**Target**: 9/16 failures (56% improvement)

**Actions**:
1. Implement negation detection in query preprocessing
2. Build synonym dictionary for common vocabulary mismatches
3. Add negation-aware query expansion
4. Test on all 9 negation/implicit queries

**Timeline**: 1-2 weeks

**Expected Outcome**: 
- Negation queries: 5/5 → 4/5 passing (80% improvement)
- Implicit queries: 4/4 → 3/4 passing (75% improvement)
- Total: 9 failures → 2-3 failures

### Phase 2: Address Pattern 2 (YAML/Code Blindness)

**Target**: 4/16 failures (25% improvement)

**Actions**:
1. Add prose summaries to YAML/code blocks in deployment_guide
2. Implement structured content extraction (YAML → prose)
3. Add keyword enrichment to PgBouncer, HPA sections
4. Test dual indexing (code + prose)

**Timeline**: 3-4 weeks

**Expected Outcome**:
- YAML queries: 4/4 → 1/4 failing (75% improvement)
- Total: 4 failures → 1 failure

### Phase 3: Address Remaining Patterns

**Target**: 3/16 failures (19% improvement)

**Actions**:
1. Implement comparative query detection
2. Add LLM re-ranking for top-20 chunks
3. Identify and fill corpus gaps (HS256, /health vs /ready)

**Timeline**: 4-6 weeks

**Expected Outcome**:
- Comparative queries: 3/5 → 1/5 failing (60% improvement)
- Total: 3 failures → 1 failure

---

## Success Metrics

### Current State (Baseline)
- Total failures: 16/24 queries (67%)
- Average failure score: 6.7/10
- Complete misses (≤3): 1/24 (4%)

### After Phase 1 (Target)
- Total failures: 7-8/24 queries (29-33%)
- Average failure score: 7.5/10
- Complete misses (≤3): 0/24 (0%)
- **Improvement**: 50-56% reduction in failures

### After Phase 2 (Target)
- Total failures: 3-4/24 queries (13-17%)
- Average failure score: 8.0/10
- Complete misses (≤3): 0/24 (0%)
- **Improvement**: 75-81% reduction in failures

### After Phase 3 (Target)
- Total failures: 1-2/24 queries (4-8%)
- Average failure score: 8.5/10
- Complete misses (≤3): 0/24 (0%)
- **Improvement**: 88-94% reduction in failures

---

## Conclusion

**The 80/20 Rule Identified:**

Two patterns explain 81% of all retrieval failures:
1. **Negation/Implicit Query Framing** (56%)
2. **YAML/Code Content Blindness** (25%)

**Key Insights:**

1. **Query framing is the #1 failure mode**: Negation and implicit vocabulary account for 56% of failures. Users ask "what NOT to do" and use natural language, while the system retrieves "what to do" in technical terms.

2. **Structured content is invisible**: YAML configurations, code blocks, and key-value pairs don't embed well. 100% of YAML-heavy queries failed.

3. **High addressability**: 81% of failures can be fixed through retrieval improvements without corpus expansion.

4. **Clear priority**: Fix Pattern 1 first (56% impact, medium complexity), then Pattern 2 (25% impact, high complexity).

**Strategic Recommendation:**

Focus on Phase 1 (negation detection + synonym expansion) to achieve 50-56% improvement in 1-2 weeks. This provides immediate value and validates the approach before investing in the more complex Phase 2 (YAML enrichment).

**Long-term Vision:**

Combining Phases 1-3 can reduce failures from 67% to 4-8%, achieving a 90%+ query success rate on edge cases.
