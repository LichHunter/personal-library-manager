# Gem Strategy Test Results

**Strategy**: adaptive_hybrid
**Date**: 2026-01-26T08:03:58.018484
**Queries Tested**: 15

## Query: mh_002
**Type**: multi-hop
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND

**Query**: If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?

**Expected Answer**: Both are valid. PgBouncer for connection pooling (max_db_connections=100, pool_mode=transaction). Read replicas for read-heavy workloads. Troubleshooting guide recommends PgBouncer first.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 5/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: mh_004
**Type**: multi-hop
**Root Causes**: YAML_BLIND, EMBEDDING_BLIND

**Query**: How do the HPA scaling parameters relate to the API Gateway resource requirements?

**Expected Answer**: HPA: minReplicas=3, maxReplicas=10, targetCPU=70%. API Gateway: 2 vCPU, 4GB RAM per pod. Scales when CPU exceeds 70% of 2 vCPU.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: tmp_003
**Type**: temporal
**Root Causes**: EMBEDDING_BLIND

**Query**: What's the sequence of events when a workflow execution times out?

**Expected Answer**: Workflow runs up to 3600s. If exceeded, automatically terminated. Error: 'exceeded maximum execution time of 3600 seconds'. Status: TIMEOUT. Can request custom timeout up to 7200s on Enterprise.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: tmp_004
**Type**: temporal
**Root Causes**: EMBEDDING_BLIND

**Query**: How long does it take for workflow definition cache changes to propagate?

**Expected Answer**: Workflow definitions cached in Redis with TTL of 1 hour. Cache invalidated on workflow update or manual flush. Cache hit rate is 94.2%.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 4/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: tmp_005
**Type**: temporal
**Root Causes**: EMBEDDING_BLIND

**Query**: What's the timeline for automatic failover when the database primary fails?

**Expected Answer**: Database primary failure: 30-60 seconds for automatic promotion of replica. Redis failover: <10 seconds. Kafka controller election: <30 seconds.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: cmp_001
**Type**: comparative
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND

**Query**: What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?

**Expected Answer**: PgBouncer: pool_mode=transaction, default_pool_size=25, max_db_connections=100. Allows 1000 client connections with only 100 actual DB connections. Direct: limited to max_connections=100.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 2/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: cmp_002
**Type**: comparative
**Root Causes**: EMBEDDING_BLIND

**Query**: How do fixed, linear, and exponential backoff strategies differ for retries?

**Expected Answer**: Fixed: same wait time (1s, 1s, 1s). Linear: increase by fixed amount (1s, 2s, 3s). Exponential: double each time (1s, 2s, 4s). Exponential is recommended.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: cmp_003
**Type**: comparative
**Root Causes**: EMBEDDING_BLIND

**Query**: What's the difference between /health and /ready endpoints?

**Expected Answer**: /health: liveness check, returns basic status. /ready: readiness check, checks dependencies like database and redis connectivity.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 5/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_001
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: What should I NOT do when I'm rate limited?

**Expected Answer**: Don't keep hammering the API. Instead: check Retry-After header, implement exponential backoff, monitor X-RateLimit-Remaining, cache responses, consider upgrading tier.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_002
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: Why doesn't HS256 work for JWT token validation in CloudFlow?

**Expected Answer**: CloudFlow uses RS256 (asymmetric) not HS256 (symmetric). RS256 requires private key for signing, public key for validation. HS256 would fail with algorithm mismatch error.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_003
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: Why can't I schedule workflows more frequently than every minute?

**Expected Answer**: Minimum scheduling interval is 1 minute. Expressions evaluating to more frequent executions will be rejected. For near real-time, use webhook or event-based triggers instead.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_004
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: What happens if I don't implement token refresh logic?

**Expected Answer**: Tokens expire after 3600 seconds (1 hour). Without refresh logic, authentication will fail after expiry. Need to implement refresh using refresh token (valid 7-30 days).

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_005
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: Why shouldn't I hardcode API keys in workflow definitions?

**Expected Answer**: Security risk - keys could be exposed. Use secrets instead: {{secrets.API_TOKEN}}. Secrets are encrypted at rest. Store in Settings > Secrets.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 5/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: imp_001
**Type**: implicit
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND

**Query**: Best practice for handling long-running data processing that might exceed time limits

**Expected Answer**: Workflow timeout is 3600s. Solutions: split into smaller workflows, enable checkpointing (every 300s), use parallel workers, request custom timeout (up to 7200s on Enterprise).

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: imp_003
**Type**: implicit
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND

**Query**: How to debug why my API calls are slow

**Expected Answer**: Check latency breakdown: Auth (18%), DB Query (64%), Business Logic (13%), Serialization (5%). Use cloudflow metrics latency-report. Check slow query log. Review connection pool status.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---
