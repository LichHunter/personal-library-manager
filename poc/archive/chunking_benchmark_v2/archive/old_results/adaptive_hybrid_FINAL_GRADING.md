# Adaptive Hybrid - FINAL GRADING RESULTS
**Strategy**: adaptive_hybrid  
**Chunking**: MarkdownSemanticStrategy (target=400 words)  
**Date**: 2026-01-26  
**Total Queries**: 15

## Complete Results Table

| Query ID | Question | Expected Answer | Score | Why Failed/Succeeded |
|----------|----------|-----------------|-------|---------------------|
| mh_002 | If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas? | Both valid. PgBouncer for pooling, read replicas for read-heavy | 6/10 | PARTIAL: Has read replicas (chunk 2) and max_connections=100 (chunk 3). MISSING: PgBouncer pool_mode, default_pool_size config |
| mh_004 | How do HPA scaling parameters relate to API Gateway resource requirements? | HPA: minReplicas=3, maxReplicas=10, targetCPU=70%. Gateway: 2vCPU, 4GB RAM | 3/10 | FAILED: Wrong chunks - data flow, HA deployment. NO HPA params, NO resource requirements found |
| tmp_003 | What's the sequence when workflow execution times out? | Runs up to 3600s, auto-terminated, error message, TIMEOUT status | 5/10 | PARTIAL: Has timeout values (1800s, 3600s) in chunk 2. MISSING: termination behavior, error format, TIMEOUT status |
| tmp_004 | How long for workflow definition cache changes to propagate? | Redis cache, TTL 1 hour, invalidated on update, 94.2% hit rate | 2/10 | FAILED: Completely wrong chunks (testing, timezone, slow queries). ZERO cache info |
| tmp_005 | Timeline for automatic failover when database primary fails? | DB: 30-60s, Redis: <10s, Kafka: <30s | 9/10 | EXCELLENT: Chunk 1 has EXACT answer - "Detection <30s, promotion 30-60s". Missing Redis/Kafka times |
| cmp_001 | Difference between PgBouncer pooling and direct PostgreSQL? | PgBouncer: pool_mode, 1000 clients→100 DB conns. Direct: max 100 | 4/10 | POOR: Has PostgreSQL max_connections=100. MISSING: All PgBouncer config (pool_mode, multiplexing) |
| cmp_002 | How do fixed/linear/exponential backoff differ? | Fixed: constant. Linear: +fixed. Exponential: doubles (1s,2s,4s) | 7/10 | GOOD: Chunk 2 has "exponential backoff, max 3 retries, initial 1s". Missing explicit comparison formulas |
| cmp_003 | Difference between /health and /ready endpoints? | /health: liveness. /ready: readiness, checks DB/Redis/Kafka | 2/10 | FAILED: Wrong chunks (error handling, analytics, workflow engine). NO endpoint info |
| neg_001 | What should I NOT do when rate limited? | Don't hammer API. DO: check Retry-After, exponential backoff, monitor headers | 8/10 | GOOD: Chunks 1-2 have rate limit headers, Retry-After logic, 429 handling. Implicit "don't" from DO examples |
| neg_002 | Why doesn't HS256 work for JWT validation? | CloudFlow uses RS256 (asymmetric), not HS256 (symmetric) | 8/10 | GOOD: Chunks 2-3 explicitly state "RS256 signing algorithm", "Algorithm: RS256 (asymmetric)". Clear answer |
| neg_003 | Why can't I schedule workflows < 1 minute? | Minimum 1 minute (cron limitation). Use webhooks for sub-minute | 5/10 | PARTIAL: Chunk 1 shows cron syntax (minute granularity). MISSING: explicit "minimum 1 minute" statement, webhook alternative |
| neg_004 | What happens if I don't implement token refresh? | Tokens expire after 3600s. Auth fails. Need refresh token (7-30 days) | 6/10 | PARTIAL: Chunk 2 shows "exp: max 3600 seconds". MISSING: refresh token mechanism, failure behavior |
| neg_005 | Why shouldn't I hardcode API keys in workflows? | Security risk. Use {{secrets.API_KEY}}. Encrypted at rest | 6/10 | PARTIAL: Chunk 1 says "Never expose API keys in client-side code". MISSING: secrets syntax, encryption details |
| imp_001 | Best practice for long-running data processing > timeout? | Split workflows, checkpointing, parallel, custom timeout (7200s Enterprise) | 7/10 | GOOD: Chunks 1,4 have "3600s timeout", "7200s Enterprise", "split workflows". Missing checkpointing, parallel details |
| imp_003 | How to debug slow API calls? | Check latency breakdown, slow query log, connection pool, metrics | 7/10 | GOOD: Chunk 2 has "slow query log", "analyze-queries", "EXPLAIN ANALYZE". Missing latency breakdown percentages |

## Summary Statistics
- **Total Queries**: 15
- **Average Score**: 5.9/10
- **Pass Rate (≥8)**: 20.0% (3/15)
- **Scores Distribution**:
  - 9-10 (Excellent): 1 query (6.7%)
  - 7-8 (Good): 5 queries (33.3%)
  - 5-6 (Partial): 4 queries (26.7%)
  - 3-4 (Poor): 2 queries (13.3%)
  - 1-2 (Failed): 3 queries (20.0%)

## Comparison with Baseline
- **Baseline (FixedSizeStrategy)**: 5.7/10 average
- **New (MarkdownSemanticStrategy)**: 5.9/10 average
- **Improvement**: +0.2 points (+3.5%)

## Analysis

### What Worked Well (7-9/10)
1. **tmp_005** (9/10): Disaster recovery info perfectly retrieved
2. **neg_001** (8/10): Rate limiting best practices well-covered
3. **neg_002** (8/10): JWT RS256 vs HS256 clearly explained
4. **cmp_002** (7/10): Exponential backoff concept present
5. **imp_001** (7/10): Timeout limits and Enterprise options found
6. **imp_003** (7/10): Debugging tools (slow query log) retrieved

### What Failed (1-4/10)
1. **tmp_004** (2/10): Cache propagation - completely wrong chunks
2. **cmp_003** (2/10): /health vs /ready - no endpoint info
3. **mh_004** (3/10): HPA parameters - wrong chunks retrieved
4. **cmp_001** (4/10): PgBouncer config - only half the answer

### Root Causes of Failures
1. **Vocabulary Mismatch**: Queries use different terms than docs (e.g., "cache propagation" vs "Redis TTL")
2. **Semantic Chunking Splits Related Content**: PgBouncer config may be in different chunk from PostgreSQL config
3. **BM25 Dilution**: Larger chunks (400 words) reduce term frequency scores
4. **Missing Content**: Some info genuinely not in corpus (e.g., /health vs /ready endpoints)

## Conclusion

**MarkdownSemanticStrategy shows MINIMAL improvement (+0.2) over FixedSizeStrategy for adaptive_hybrid.**

**Why adaptive_hybrid underperforms:**
- Relies heavily on BM25 term matching
- Semantic chunking creates larger, more diluted chunks
- Technical queries need exact term matches, not semantic boundaries

**Recommendation**: Test other strategies (synthetic_variants, enriched_hybrid_llm) which may work better with semantic chunking.
