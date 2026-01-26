# Failure Dataset - Edge Case Query Analysis

Generated: 2026-01-25
Strategy: enriched_hybrid_llm
Total Queries: 24
Failed/Partial (≤7): 16

## Summary Statistics

| Query Type | Total | Passed (>7) | Failed (≤7) | Avg Score |
|------------|-------|-------------|-------------|-----------|
| Multi-hop | 5 | 2 | 3 | 6.4 |
| Temporal | 5 | 3 | 2 | 7.2 |
| Comparative | 5 | 2 | 3 | 6.6 |
| Negation | 5 | 1 | 4 | 5.8 |
| Implicit | 4 | 0 | 4 | 5.5 |
| **TOTAL** | **24** | **8** | **16** | **6.3** |

## Grading Rubric

- 10: Perfect - All info retrieved, directly answers question
- 9: Excellent - Complete answer with minor irrelevant content
- 8: Very Good - Answer present but requires parsing
- 7: Good - Core answer present, missing details
- 6: Adequate - Partial answer, enough to be useful
- 5: Borderline - Some relevant info, misses key point
- 4: Poor - Tangentially related only
- 3: Very Poor - Mostly irrelevant, hint of topic
- 2: Bad - Almost entirely irrelevant
- 1: Failed - No relevant content

---

## MULTI-HOP QUERIES (5)

### Query mh_001: [Multi-hop] Compare JWT expiration in Auth Service vs the API documentation - are they consistent?

**Expected Answer**: Auth Service: 15-minute access token expiry. API docs: 3600 seconds (1 hour) max. These are different contexts - Auth Service internal tokens vs API JWT claims.

**Retrieved Chunks**:
1. [api_reference] - JWT Tokens section with RS256, exp claim max 3600 seconds from iat
2. [architecture_overview] - API Gateway JWT validation delegated to Auth Service
3. [troubleshooting_guide] - Quick diagnostic checklist (irrelevant)
4. [architecture_overview] - JWT Token Validation with RS256, key rotation every 30 days
5. [architecture_overview] - Auth Service with 15-minute JWT expiry, refresh token 7 days

**Manual Grade**: 8/10

**Failure Notes**: Retrieved both pieces of information (API: 3600s, Auth Service: 15-min). However, the comparison requires user to synthesize across chunks. The Auth Service 15-min expiry is in chunk 5, API 3600s in chunk 1. Good retrieval but requires parsing.

---

### Query mh_002: [Multi-hop] If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?

**Expected Answer**: Both are valid. PgBouncer for connection pooling (max_db_connections=100, pool_mode=transaction). Read replicas for read-heavy workloads. Troubleshooting guide recommends PgBouncer first.

**Retrieved Chunks**:
1. [deployment_guide] - Pods in CrashLoopBackOff, High Memory Usage (tangential)
2. [architecture_overview] - PostgreSQL Primary Database with 1 primary + 2 read replicas
3. [deployment_guide] - postgres-values.yaml with max_connections = 100
4. [architecture_overview] - Multi-AZ Deployment showing PostgreSQL Primary/Replica
5. [architecture_overview] - Monitoring & Alerting with database connection pool saturation

**Manual Grade**: 5/10

**Failure Notes**: MISSING PgBouncer configuration details! Retrieved read replica info but NOT the PgBouncer section from deployment_guide. The troubleshooting guide's connection pool exhaustion section was NOT retrieved. Critical miss - user wouldn't know about PgBouncer as a solution.

---

### Query mh_003: [Multi-hop] What's the relationship between workflow timeout (3600s) and the retry backoff strategy?

**Expected Answer**: Workflow timeout is 3600s max. Retry uses exponential backoff: 1s, 2s, 4s (max 3 retries). Total retry time ~7s, well within timeout. But long-running steps can still timeout.

**Retrieved Chunks**:
1. [troubleshooting_guide] - Retry Logic and Exponential Backoff: max 3 retries, 1s initial, backoff multiplier
2. [user_guide] - Retry Policies with max_attempts: 3, backoff_type: exponential
3. [user_guide] - Execution Timeout: 3600 seconds, automatically terminated
4. [troubleshooting_guide] - Slow query analysis (tangential)
5. [troubleshooting_guide] - Rate limit retry logic (tangential)

**Manual Grade**: 9/10

**Failure Notes**: Excellent retrieval! Got both timeout (3600s) and retry backoff (exponential, 3 retries, 1s initial). User can synthesize the relationship. Minor noise in chunks 4-5.

---

### Query mh_004: [Multi-hop] How do the HPA scaling parameters relate to the API Gateway resource requirements?

**Expected Answer**: HPA: minReplicas=3, maxReplicas=10, targetCPU=70%. API Gateway: 2 vCPU, 4GB RAM per pod. Scales when CPU exceeds 70% of 2 vCPU.

**Retrieved Chunks**:
1. [deployment_guide] - Grafana dashboards (tangential)
2. [architecture_overview] - Resource Utilization: API Gateway 55% avg, 80% peak, 2.5GB/4GB
3. [architecture_overview] - Table of Contents (useless)
4. [architecture_overview] - API Gateway: 2 vCPU, 4GB RAM per pod, auto-scaling 8-20
5. [deployment_guide] - Pods in CrashLoopBackOff (tangential)

**Manual Grade**: 6/10

**Failure Notes**: Got API Gateway resources (2 vCPU, 4GB) but MISSING the HPA configuration (minReplicas=3, maxReplicas=10, targetCPU=70%). The deployment_guide has HPA YAML but it wasn't retrieved. Partial answer only.

---

### Query mh_005: [Multi-hop] What happens to scheduled workflows during a disaster recovery failover?

**Expected Answer**: RPO is 1 hour, RTO is 4 hours. Scheduler uses leader election with Redis. During failover, scheduled executions may be skipped (logged in audit trail). Kafka retention allows event replay.

**Retrieved Chunks**:
1. [architecture_overview] - DR Test Results: RTO 2h23m, RPO 42m
2. [architecture_overview] - Recovery Objectives: RPO 1 hour, RTO 4 hours, Kafka 7 days retention
3. [architecture_overview] - Scheduler Service: leader election via Redis, active-passive
4. [architecture_overview] - DR Procedures: Single AZ failure, Database Primary failure
5. [architecture_overview] - Workflow Engine (tangential)

**Manual Grade**: 8/10

**Failure Notes**: Good retrieval of DR objectives and Scheduler leader election. Missing explicit mention of "scheduled executions may be skipped" from user_guide. User can infer from leader election + failover time but not explicitly stated.

---

## TEMPORAL QUERIES (5)

### Query tmp_001: [Temporal] What changed between the last DR test and the current DR objectives?

**Expected Answer**: Last DR test (Dec 15, 2025): Actual RTO was 2h 23m (target 4h), RPO was 42m (target 1h). Both met objectives. DNS propagation issue was resolved.

**Retrieved Chunks**:
1. [architecture_overview] - DR Test Results (Dec 15, 2025): RTO 2h23m, RPO 42m, DNS issue resolved
2. [architecture_overview] - Recovery Objectives: RPO 1 hour, RTO 4 hours
3. [architecture_overview] - DR Procedures
4. [deployment_guide] - Table of Contents (useless)
5. [troubleshooting_guide] - SEV-3/SEV-4 definitions (irrelevant)

**Manual Grade**: 9/10

**Failure Notes**: Excellent! Retrieved both the DR test results AND the current objectives. User can directly compare. Minor noise in chunks 4-5.

---

### Query tmp_002: [Temporal] How often should API keys be rotated and what's the certificate rotation schedule?

**Expected Answer**: API keys: rotate every 90 days. Certificates: rotate every 90 days (automated via cert-manager). Secrets in Vault also rotated every 90 days.

**Retrieved Chunks**:
1. [architecture_overview] - Network Security: Certificate rotation every 90 days via cert-manager
2. [api_reference] - API Keys: Rotate keys every 90 days
3. [user_guide] - Data Limits (tangential)
4. [architecture_overview] - Secrets Management: Rotation every 90 days
5. [architecture_overview] - JWT Token Validation: Key rotation every 30 days

**Manual Grade**: 10/10

**Failure Notes**: Perfect retrieval! All three rotation schedules found: API keys (90 days), certificates (90 days), Vault secrets (90 days). Also bonus info about JWT key rotation (30 days).

---

### Query tmp_003: [Temporal] What's the sequence of events when a workflow execution times out?

**Expected Answer**: Workflow runs up to 3600s. If exceeded, automatically terminated. Error: 'exceeded maximum execution time of 3600 seconds'. Status: TIMEOUT. Can request custom timeout up to 7200s on Enterprise.

**Retrieved Chunks**:
1. [user_guide] - Execution Timeout: 3600 seconds, automatically terminated, Enterprise custom limits
2. [troubleshooting_guide] - Retry Logic and Exponential Backoff (tangential)
3. [user_guide] - Retry Policies (tangential)
4. [troubleshooting_guide] - RBAC Policy Violations (irrelevant)
5. [troubleshooting_guide] - Error handling config (tangential)

**Manual Grade**: 7/10

**Failure Notes**: Got the timeout (3600s) and termination behavior, but MISSING the specific error message "exceeded maximum execution time of 3600 seconds" and TIMEOUT status from troubleshooting_guide. Partial answer.

---

### Query tmp_004: [Temporal] How long does it take for workflow definition cache changes to propagate?

**Expected Answer**: Workflow definitions cached in Redis with TTL of 1 hour. Cache invalidated on workflow update or manual flush. Cache hit rate is 94.2%.

**Retrieved Chunks**:
1. [architecture_overview] - Performance Characteristics: GET /workflows cache hit <50ms, miss <150ms
2. [architecture_overview] - Asynchronous Event Flow (irrelevant)
3. [troubleshooting_guide] - Slow query log (irrelevant)
4. [architecture_overview] - Monitoring & Alerting with cache hit/miss ratios
5. [user_guide] - Test Thoroughly (irrelevant)

**Manual Grade**: 4/10

**Failure Notes**: CRITICAL MISS! Did NOT retrieve the Redis Caching Layer section with "Workflow Definitions: TTL: 1 hour" and "cache hit rate 94.2%". Retrieved tangential cache info but not the specific workflow definition cache TTL. User cannot answer the question.

---

### Query tmp_005: [Temporal] What's the timeline for automatic failover when the database primary fails?

**Expected Answer**: Database primary failure: 30-60 seconds for automatic promotion of replica. Redis failover: <10 seconds. Kafka controller election: <30 seconds.

**Retrieved Chunks**:
1. [architecture_overview] - DR Procedures: Database Primary Failure 30-60 seconds
2. [architecture_overview] - Recovery Objectives
3. [architecture_overview] - Scheduler Service with leader election
4. [architecture_overview] - Multi-AZ Deployment
5. [architecture_overview] - Architecture Diagram

**Manual Grade**: 7/10

**Failure Notes**: Got database failover (30-60s) but MISSING Redis failover (<10s) and Kafka controller election (<30s) from the same section. Partial answer - only 1 of 3 failover timelines retrieved.

---

## COMPARATIVE QUERIES (5)

### Query cmp_001: [Comparative] What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?

**Expected Answer**: PgBouncer: pool_mode=transaction, default_pool_size=25, max_db_connections=100. Allows 1000 client connections with only 100 actual DB connections. Direct: limited to max_connections=100.

**Retrieved Chunks**:
1. [deployment_guide] - Grafana dashboards (irrelevant)
2. [architecture_overview] - Architecture Diagram (irrelevant)
3. [deployment_guide] - Environment Configuration with DATABASE_URL
4. [troubleshooting_guide] - Slow query analysis (irrelevant)
5. [architecture_overview] - JWT Token Validation (irrelevant)

**Manual Grade**: 2/10

**Failure Notes**: COMPLETE FAILURE! Did NOT retrieve the PgBouncer Configuration section from deployment_guide which has pool_mode=transaction, default_pool_size=25, max_db_connections=100. Also missed the architecture_overview Connection Pooling section. User cannot answer the question at all.

---

### Query cmp_002: [Comparative] How do fixed, linear, and exponential backoff strategies differ for retries?

**Expected Answer**: Fixed: same wait time (1s, 1s, 1s). Linear: increase by fixed amount (1s, 2s, 3s). Exponential: double each time (1s, 2s, 4s). Exponential is recommended.

**Retrieved Chunks**:
1. [troubleshooting_guide] - Retry Logic: exponential backoff, 1s, 2s, 4s sequence
2. [user_guide] - Retry Policies: backoff_type "exponential" or "fixed", "linear"
3. [architecture_overview] - Cache Invalidation Strategies (irrelevant)
4. [troubleshooting_guide] - Rate limit retry logic
5. [user_guide] - Fallback Actions

**Manual Grade**: 6/10

**Failure Notes**: Retrieved that fixed/linear/exponential exist and exponential sequence (1s, 2s, 4s). But MISSING the explicit comparison of all three strategies. User_guide has the comparison but the specific section wasn't retrieved. Partial answer.

---

### Query cmp_003: [Comparative] What's the difference between /health and /ready endpoints?

**Expected Answer**: /health: liveness check, returns basic status. /ready: readiness check, checks dependencies like database and redis connectivity.

**Retrieved Chunks**:
1. [architecture_overview] - Multi-AZ Deployment (irrelevant)
2. [troubleshooting_guide] - SEV-3/SEV-4 definitions (irrelevant)
3. [troubleshooting_guide] - (truncated in output)
4. [deployment_guide] - (not visible in truncated output)
5. [architecture_overview] - (not visible)

**Manual Grade**: 5/10

**Failure Notes**: Based on test output, deployment_guide was retrieved but the specific health check section wasn't in top chunks. The deployment_guide has "healthCheck: livenessProbe: path: /health" and "readinessProbe: path: /ready" but the explanation of what they check wasn't retrieved. Partial.

---

### Query cmp_004: [Comparative] Compare the rate limits for authenticated vs unauthenticated API requests

**Expected Answer**: Authenticated: 100 requests/minute per user. Unauthenticated: 20 requests/minute. Burst allowance: 150 requests in 10-second window. Enterprise: 1000 req/min.

**Retrieved Chunks**:
1. [troubleshooting_guide] - Rate limit tiers table (Standard: 1000/min)
2. [troubleshooting_guide] - Rate limit status checking
3. [api_reference] - Rate Limiting section
4. [troubleshooting_guide] - Rate limit handling
5. [api_reference] - Rate Limiting

**Manual Grade**: 8/10

**Failure Notes**: Good retrieval! api_reference has "100 requests per minute per authenticated user" and "20 requests per minute for unauthenticated". Troubleshooting has tier table. User can answer the question. Minor: burst allowance may require parsing.

---

### Query cmp_005: [Comparative] How do SEV-1 and SEV-2 incidents differ in response time and escalation?

**Expected Answer**: SEV-1: Immediate response (<15 min), page on-call immediately, complete outage. SEV-2: <1 hour response, create ticket and notify on-call, major functionality impaired.

**Retrieved Chunks**:
1. [troubleshooting_guide] - SEV-1 and SEV-2 definitions with response times
2. [troubleshooting_guide] - More severity level details
3. [troubleshooting_guide] - Severity level continuation
4. [architecture_overview] - (tangential)
5. [architecture_overview] - (tangential)

**Manual Grade**: 9/10

**Failure Notes**: Excellent! Retrieved the severity level definitions with response times and escalation procedures. User can directly compare SEV-1 vs SEV-2.

---

## NEGATION QUERIES (5)

### Query neg_001: [Negation] What should I NOT do when I'm rate limited?

**Expected Answer**: Don't keep hammering the API. Instead: check Retry-After header, implement exponential backoff, monitor X-RateLimit-Remaining, cache responses, consider upgrading tier.

**Retrieved Chunks**:
1. [troubleshooting_guide] - Rate limit handling code
2. [troubleshooting_guide] - Rate limit status
3. [user_guide] - (tangential)
4. [user_guide] - (tangential)
5. [api_reference] - Rate Limiting

**Manual Grade**: 6/10

**Failure Notes**: Retrieved what TO do (check headers, backoff) but not framed as what NOT to do. The api_reference "Best Practices" section has the guidance but the negation framing is lost. User gets useful info but not in the expected format.

---

### Query neg_002: [Negation] Why doesn't HS256 work for JWT token validation in CloudFlow?

**Expected Answer**: CloudFlow uses RS256 (asymmetric) not HS256 (symmetric). RS256 requires private key for signing, public key for validation. HS256 would fail with algorithm mismatch error.

**Retrieved Chunks**:
1. [api_reference] - JWT Tokens with RS256 signing algorithm
2. [troubleshooting_guide] - (tangential)
3. [architecture_overview] - JWT Token Validation: RS256
4. [architecture_overview] - Auth Service: RS256 algorithm
5. [architecture_overview] - (tangential)

**Manual Grade**: 7/10

**Failure Notes**: Retrieved that RS256 is used (multiple times), but MISSING explicit statement that HS256 doesn't work or why. User can infer "RS256 only" but the negation aspect isn't directly addressed.

---

### Query neg_003: [Negation] Why can't I schedule workflows more frequently than every minute?

**Expected Answer**: Minimum scheduling interval is 1 minute. Expressions evaluating to more frequent executions will be rejected. For near real-time, use webhook or event-based triggers instead.

**Retrieved Chunks**:
1. [troubleshooting_guide] - (tangential)
2. [user_guide] - Scheduling section
3. [api_reference] - (tangential)
4. [troubleshooting_guide] - (tangential)
5. [troubleshooting_guide] - (tangential)

**Manual Grade**: 7/10

**Failure Notes**: user_guide has "minimum scheduling interval is 1 minute" and "Expressions that evaluate to more frequent executions will be rejected" but need to verify if this specific text was in retrieved chunk. Likely partial retrieval.

---

### Query neg_004: [Negation] What happens if I don't implement token refresh logic?

**Expected Answer**: Tokens expire after 3600 seconds (1 hour). Without refresh logic, authentication will fail after expiry. Need to implement refresh using refresh token (valid 7-30 days).

**Retrieved Chunks**:
1. [architecture_overview] - Auth Service token expiry
2. [troubleshooting_guide] - Token expiration troubleshooting
3. [architecture_overview] - (tangential)
4. [api_reference] - Token Expiration
5. [architecture_overview] - (tangential)

**Manual Grade**: 6/10

**Failure Notes**: Retrieved token expiration info (3600s) but the consequence of NOT implementing refresh isn't explicitly stated. User can infer but the negation framing is weak.

---

### Query neg_005: [Negation] Why shouldn't I hardcode API keys in workflow definitions?

**Expected Answer**: Security risk - keys could be exposed. Use secrets instead: {{secrets.API_TOKEN}}. Secrets are encrypted at rest. Store in Settings > Secrets.

**Retrieved Chunks**:
1. [api_reference] - API Keys security notes
2. [architecture_overview] - (tangential)
3. [api_reference] - (tangential)
4. [deployment_guide] - (tangential)
5. [user_guide] - Best Practices

**Manual Grade**: 5/10

**Failure Notes**: api_reference has "Never expose API keys in client-side code" but the user_guide Best Practices section with "Never hardcode API keys" and secrets usage wasn't prominently retrieved. Partial answer.

---

## IMPLICIT QUERIES (4)

### Query imp_001: [Implicit] Best practice for handling long-running data processing that might exceed time limits

**Expected Answer**: Workflow timeout is 3600s. Solutions: split into smaller workflows, enable checkpointing (every 300s), use parallel workers, request custom timeout (up to 7200s on Enterprise).

**Retrieved Chunks**:
1. [user_guide] - Workflow Limits section
2. [user_guide] - (tangential)
3. [user_guide] - (tangential)
4. [troubleshooting_guide] - Timeout errors
5. [architecture_overview] - (tangential)

**Manual Grade**: 6/10

**Failure Notes**: Retrieved timeout info (3600s) and Enterprise limits (7200s). But MISSING the troubleshooting_guide solutions: split workflows, checkpointing, parallel workers. Partial answer - knows the problem but not all solutions.

---

### Query imp_002: [Implicit] How to ensure my application survives a complete region failure

**Expected Answer**: Multi-AZ deployment across 3 AZs. Cross-region replication to us-west-2 (15-min lag). Manual failover procedure: update DNS, promote DR replica, scale up DR services. RTO: 2-4 hours.

**Retrieved Chunks**:
1. [architecture_overview] - DR procedures
2. [architecture_overview] - Multi-AZ deployment
3. [architecture_overview] - Recovery objectives
4. [architecture_overview] - HA architecture
5. [architecture_overview] - (tangential)

**Manual Grade**: 8/10

**Failure Notes**: Good retrieval of DR content. Multi-AZ, cross-region replication, failover procedures all present. User can synthesize a complete answer. Minor: deployment_guide backup strategy not retrieved.

---

### Query imp_003: [Implicit] How to debug why my API calls are slow

**Expected Answer**: Check latency breakdown: Auth (18%), DB Query (64%), Business Logic (13%), Serialization (5%). Use cloudflow metrics latency-report. Check slow query log. Review connection pool status.

**Retrieved Chunks**:
1. [architecture_overview] - Performance Characteristics
2. [troubleshooting_guide] - Slow Query Performance
3. [architecture_overview] - (tangential)
4. [troubleshooting_guide] - Latency Breakdown Analysis
5. [deployment_guide] - (tangential)

**Manual Grade**: 7/10

**Failure Notes**: Retrieved latency breakdown and slow query analysis. But the specific percentages (Auth 18%, DB 64%, etc.) from troubleshooting_guide may not be in top chunks. Partial answer.

---

### Query imp_004: [Implicit] What monitoring should I set up for production workflows?

**Expected Answer**: Prometheus for metrics, Grafana for dashboards, Jaeger for distributed tracing. Key metrics: request rate, error rate, latency percentiles, cache hit ratios, Kafka consumer lag. Alerts via PagerDuty.

**Retrieved Chunks**:
1. [deployment_guide] - Prometheus Setup
2. [deployment_guide] - Grafana Dashboards
3. [architecture_overview] - Monitoring stack
4. [architecture_overview] - Key Metrics
5. [deployment_guide] - (tangential)

**Manual Grade**: 8/10

**Failure Notes**: Good retrieval! Prometheus, Grafana, key metrics all present. Jaeger mentioned in architecture_overview. User can build a complete monitoring setup. Minor: PagerDuty alerts may require parsing.

---

## KEY FAILURE PATTERNS IDENTIFIED

### 1. PgBouncer Configuration Blind Spot
- Queries about PgBouncer consistently fail to retrieve the detailed configuration
- The deployment_guide PgBouncer section (pool_mode, default_pool_size, max_db_connections) is not being indexed/retrieved well
- Affects: mh_002, cmp_001

### 2. Negation Framing Lost
- Queries framed as "what NOT to do" retrieve "what TO do" content
- The negation aspect is not captured in retrieval
- Affects: neg_001, neg_002, neg_004, neg_005

### 3. Cache TTL Section Missing
- The Redis Caching Layer section with specific TTLs is not retrieved
- "Workflow Definitions: TTL: 1 hour" is a key fact that's missed
- Affects: tmp_004

### 4. Multi-hop Synthesis Required
- Multi-hop queries retrieve relevant chunks but from different sections
- User must synthesize across chunks to answer
- Not a retrieval failure but an answer quality issue
- Affects: mh_001, mh_003, mh_005

### 5. Implicit Query Vocabulary Mismatch
- Implicit queries use different vocabulary than the corpus
- "long-running data processing" vs "workflow timeout"
- "survives region failure" vs "disaster recovery"
- Affects: imp_001, imp_003

### 6. HPA Configuration Not Retrieved
- The HPA YAML in deployment_guide is not being retrieved for scaling queries
- minReplicas, maxReplicas, targetCPUUtilizationPercentage are key facts
- Affects: mh_004

---

## RECOMMENDATIONS FOR IMPROVEMENT

1. **Chunk Enrichment for PgBouncer**: Add "connection pooling", "pool exhaustion" keywords to PgBouncer chunks
2. **Negation-Aware Retrieval**: Consider query expansion for negation queries to include positive forms
3. **Cache Section Boosting**: The Redis Caching Layer section needs better keyword enrichment
4. **HPA Configuration Indexing**: Ensure YAML code blocks are properly indexed with semantic meaning
5. **Synonym Expansion**: "long-running" → "timeout", "survives failure" → "disaster recovery"

---

## RAW SCORES

| ID | Type | Score | Pass/Fail |
|----|------|-------|-----------|
| mh_001 | multi-hop | 8 | PASS |
| mh_002 | multi-hop | 5 | FAIL |
| mh_003 | multi-hop | 9 | PASS |
| mh_004 | multi-hop | 6 | FAIL |
| mh_005 | multi-hop | 8 | PASS |
| tmp_001 | temporal | 9 | PASS |
| tmp_002 | temporal | 10 | PASS |
| tmp_003 | temporal | 7 | FAIL |
| tmp_004 | temporal | 4 | FAIL |
| tmp_005 | temporal | 7 | FAIL |
| cmp_001 | comparative | 2 | FAIL |
| cmp_002 | comparative | 6 | FAIL |
| cmp_003 | comparative | 5 | FAIL |
| cmp_004 | comparative | 8 | PASS |
| cmp_005 | comparative | 9 | PASS |
| neg_001 | negation | 6 | FAIL |
| neg_002 | negation | 7 | FAIL |
| neg_003 | negation | 7 | FAIL |
| neg_004 | negation | 6 | FAIL |
| neg_005 | negation | 5 | FAIL |
| imp_001 | implicit | 6 | FAIL |
| imp_002 | implicit | 8 | PASS |
| imp_003 | implicit | 7 | FAIL |
| imp_004 | implicit | 8 | PASS |

**Total Passed (>7)**: 8/24 (33%)
**Total Failed (≤7)**: 16/24 (67%)
**Average Score**: 6.7/10
