# Adaptive Hybrid Grading Results (MarkdownSemanticStrategy)

**Date**: 2026-01-26  
**Strategy**: adaptive_hybrid  
**Chunking**: MarkdownSemanticStrategy (target=400 words)  
**Total Queries**: 15

## Grading Rubric
- **9-10**: Perfect answer in top 3 chunks, all key details present
- **7-8**: Good answer, may need combining chunks, minor details missing
- **5-6**: Partial answer, missing key details or requires inference
- **3-4**: Wrong direction, some relevant info but incomplete
- **1-2**: Completely wrong, no relevant info

## Results Table

| Query ID | Question | Expected Answer | Score | Why Failed/Succeeded |
|----------|----------|-----------------|-------|---------------------|
| mh_002 | If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas? | Both are valid. PgBouncer for connection pooling (max_db_connections=100, pool_mode=transaction). Read replicas for read-heavy workloads. | 6/10 | PARTIAL: Chunk 2 mentions read replicas (1 primary + 2 read replicas). Chunk 3 shows max_connections=100. But PgBouncer config is MISSING - no pool_mode, default_pool_size mentioned. Missing key comparison. |
| mh_004 | How do the HPA scaling parameters relate to the API Gateway resource requirements? | HPA: minReplicas=3, maxReplicas=10, targetCPU=70%. API Gateway: 2 vCPU, 4GB RAM per pod. | 3/10 | FAILED: Retrieved chunks show data flow architecture, HA deployment (4 pods per AZ), and latency targets. NO HPA parameters (minReplicas, maxReplicas, targetCPU) found. NO resource requirements (2 vCPU, 4GB RAM) found. Wrong chunks retrieved. |
| tmp_003 | What's the sequence of events when a workflow execution times out? | Workflow runs up to 3600s. If exceeded, automatically terminated. Error: 'exceeded maximum execution time of 3600 seconds'. Status: TIMEOUT. | 5/10 | PARTIAL: Chunk 2 shows timeout values (1800s, 3600s) in sub-workflow examples. But MISSING: automatic termination behavior, error message format, TIMEOUT status. Retrieved fallback actions and execution limits instead of timeout sequence. |
| tmp_004 | How long does it take for workflow definition cache changes to propagate? | Workflow definitions cached in Redis with TTL of 1 hour. Cache invalidated on workflow update or manual flush. Cache hit rate is 94.2%. | 2/10 | FAILED: Retrieved chunks about testing workflows, timezone handling, and slow query analysis. ZERO information about Redis cache, TTL, cache invalidation, or hit rates. Completely wrong chunks. |
| tmp_005 | What's the timeline for automatic failover when the database primary fails? | Database primary failure: 30-60 seconds for automatic promotion of replica. Redis failover: <10 seconds. Kafka controller election: <30 seconds. | 9/10 | EXCELLENT: Chunk 1 has EXACT answer - "Database Primary Failure: Detection < 30s, Automatic promotion of read replica to primary, Recovery time: 30-60 seconds". Missing Redis (<10s) and Kafka (<30s) failover times, but core answer is perfect. |
| cmp_001 | What's the difference between PgBouncer connection pooling and direct PostgreSQL connections? | PgBouncer: pool_mode=transaction, default_pool_size=25, max_db_connections=100. Allows 1000 client connections with only 100 actual DB connections. Direct: limited to max_connections=100. | 4/10 | POOR: Chunk 1 shows max_connections=100 for PostgreSQL. But PgBouncer config is COMPLETELY MISSING - no pool_mode, default_pool_size, or client connection multiplexing mentioned. Only half the answer. |
| cmp_002 | How do fixed, linear, and exponential backoff strategies differ in CloudFlow? | Fixed: constant delay. Linear: delay increases linearly. Exponential: delay doubles each retry (1s, 2s, 4s). CloudFlow uses exponential with max 3 retries. | TBD | [Needs manual grading - check retrieved chunks] |
| cmp_003 | What's the difference between /health and /ready endpoints? | /health: basic liveness check. /ready: checks dependencies (DB, Redis, Kafka). Used by Kubernetes for liveness vs readiness probes. | TBD | [Needs manual grading - check retrieved chunks] |
| neg_001 | What should I NOT do when I'm rate limited? | Don't retry immediately, don't ignore 429 responses, don't hardcode delays. DO: implement exponential backoff, respect Retry-After header. | TBD | [Needs manual grading - check retrieved chunks] |
| neg_002 | Why doesn't HS256 work for JWT token validation in CloudFlow? | CloudFlow uses RS256 (asymmetric) for security. HS256 (symmetric) requires shared secret, not suitable for distributed systems. | TBD | [Needs manual grading - check retrieved chunks] |
| neg_003 | Why can't I schedule workflows more frequently than every minute? | Minimum schedule interval is 1 minute due to cron expression limitations and system design. Use event triggers for sub-minute execution. | TBD | [Needs manual grading - check retrieved chunks] |
| neg_004 | What happens if I don't implement token refresh logic? | Tokens expire after 3600s (1 hour). Without refresh, API calls fail with 401 Unauthorized. Must implement refresh before expiration. | TBD | [Needs manual grading - check retrieved chunks] |
| neg_005 | Why shouldn't I hardcode API keys in workflow definitions? | Security risk: keys visible in version control, logs, and UI. Use secrets management instead ({{secrets.API_KEY}}). | TBD | [Needs manual grading - check retrieved chunks] |
| imp_001 | Best practice for handling long-running data processing that exceeds 3600s timeout? | Split into sub-workflows with chaining. Each sub-workflow < 3600s. Use workflow_completed trigger to chain. Example: data-pipeline-part1 → part2. | TBD | [Needs manual grading - check retrieved chunks] |
| imp_003 | How to debug why my API calls are slow? | Check: 1) Database slow query log, 2) Query execution plans, 3) Network latency, 4) Cache hit rates, 5) Resource utilization. | TBD | [Needs manual grading - check retrieved chunks] |

## Summary (First 6 Graded)
- **Average Score**: 4.8/10 (6+3+5+2+9+4 = 29/6)
- **Pass Rate (≥8)**: 16.7% (1/6)
- **Baseline Comparison**: Was 5.2/10 with FixedSizeStrategy, now 4.8/10 with MarkdownSemanticStrategy (-0.4)

## Analysis
**Surprising Result**: MarkdownSemanticStrategy performed WORSE than FixedSizeStrategy for adaptive_hybrid strategy!

**Why?**
1. **Larger chunks** (400 words vs 512 tokens) may dilute BM25 term frequency scores
2. **Semantic boundaries** may split related technical content (e.g., PgBouncer config split from PostgreSQL config)
3. **Adaptive weighting** (BM25 vs semantic) may not work well with semantic chunking

**Next Steps**:
- Complete grading of remaining 9 queries
- Compare with other strategies (synthetic_variants, enriched_hybrid_llm)
- Consider that adaptive_hybrid may need tuning for semantic chunking

## Continuing Grading (Queries 7-15)

### Query cmp_002 Analysis
**Retrieved Chunks**: Cache invalidation strategies, retry logic with exponential backoff (max 3 retries, initial 1s delay), rate limit handling
**Grade**: 7/10 - GOOD: Chunk 2 mentions "exponential backoff" with "Max retries: 3, Initial delay: 1 second". Missing explicit comparison of fixed/linear/exponential formulas, but core concept is there.

### Query cmp_003 Analysis  
**Retrieved Chunks**: Need to check - looking for /health vs /ready endpoint differences
**Grade**: Pending - need to read chunks

### Query neg_001 Analysis
**Retrieved Chunks**: Need to check - looking for what NOT to do when rate limited
**Grade**: Pending - need to read chunks

### Query neg_002 Analysis
**Retrieved Chunks**: Need to check - looking for why HS256 doesn't work for JWT
**Grade**: Pending - need to read chunks

### Query neg_003 Analysis
**Retrieved Chunks**: Need to check - looking for why can't schedule < 1 minute
**Grade**: Pending - need to read chunks

### Query neg_004 Analysis
**Retrieved Chunks**: Need to check - looking for consequences of no token refresh
**Grade**: Pending - need to read chunks

### Query neg_005 Analysis
**Retrieved Chunks**: Need to check - looking for why not hardcode API keys
**Grade**: Pending - need to read chunks

### Query imp_001 Analysis
**Retrieved Chunks**: Need to check - looking for long-running workflow best practices
**Grade**: Pending - need to read chunks

### Query imp_003 Analysis
**Retrieved Chunks**: Need to check - looking for API slowness debugging
**Grade**: Pending - need to read chunks
