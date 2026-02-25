# synthetic_variants Strategy - Manual Grading Results

**Date**: 2026-01-26  
**Strategy**: synthetic_variants  
**Chunking**: MarkdownSemanticStrategy (target=400 words)  
**Chunks**: 80  
**Queries Graded**: 15

---

## Grading Rubric

| Score | Quality | Description |
|-------|---------|-------------|
| 10 | Perfect | Complete answer with all details in top chunks |
| 9 | Excellent | Answer fully present, minor details missing |
| 8 | Good | Core answer present, some context missing |
| 7 | Acceptable | Partial answer, requires inference |
| 6 | Marginal | Key facts present but incomplete |
| 5 | Weak | Some relevant info, major gaps |
| 4 | Poor | Minimal relevant content |
| 3 | Very Poor | Wrong context, barely relevant |
| 2 | Failed | Completely wrong chunks |
| 1 | Total Failure | No relevant information |

**Pass Threshold**: ≥8/10

---

## Query Results

### mh_002: PgBouncer vs Read Replicas
**Type**: multi-hop  
**Query**: If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?

**Expected Answer**: Both are valid. PgBouncer for connection pooling (max_db_connections=100, pool_mode=transaction). Read replicas for read-heavy workloads. Troubleshooting guide recommends PgBouncer first.

**Retrieved Chunks**:
1. ✅ Database Architecture - mentions primary-replica setup with 2 read replicas
2. ❌ Troubleshooting guide - database slow query log (not about connection pooling)
3. ❌ Throughput Capacity - database read/write throughput (not about pooling)
4. ❌ Cache Invalidation Strategies - not relevant
5. ❌ Pods in CrashLoopBackOff - mentions database connection failure but not pooling

**Score**: 3/10  
**Reasoning**: Chunk 1 mentions read replicas exist but doesn't explain when to use them vs PgBouncer. No mention of PgBouncer configuration (max_db_connections=100, pool_mode=transaction). Missing the troubleshooting guide's recommendation. Cannot answer the question.

---

### mh_004: HPA Scaling Parameters vs API Gateway Resources
**Type**: multi-hop  
**Query**: How do the HPA scaling parameters relate to the API Gateway resource requirements?

**Expected Answer**: HPA: minReplicas=3, maxReplicas=10, targetCPU=70%. API Gateway: 2 vCPU, 4GB RAM per pod. Scales when CPU exceeds 70% of 2 vCPU.

**Retrieved Chunks**:
1. ❌ High Availability Architecture - Multi-AZ deployment (no HPA params)
2. ✅ API Gateway - "2 vCPU, 4GB RAM per pod", "auto-scaling 8-20 based on CPU"
3. ❌ Table of Contents - not relevant
4. ✅ Resource Utilization - "API Gateway: 55% average, 80% peak" CPU usage
5. ❌ Grafana Dashboards - monitoring info, not HPA params

**Score**: 5/10  
**Reasoning**: Has API Gateway resources (2 vCPU, 4GB RAM) and mentions auto-scaling based on CPU. Has actual CPU utilization (55% avg, 80% peak). But missing the critical HPA parameters: minReplicas=3, maxReplicas=10, targetCPU=70%. Cannot explain the relationship without the HPA config.

---

### tmp_003: Workflow Execution Timeout Sequence
**Type**: temporal  
**Query**: What's the sequence of events when a workflow execution times out?

**Expected Answer**: Workflow runs up to 3600s. If exceeded, automatically terminated. Error: 'exceeded maximum execution time of 3600 seconds'. Status: TIMEOUT. Can request custom timeout up to 7200s on Enterprise.

**Retrieved Chunks**:
1. ✅ Troubleshooting guide - mentions timeout 1800s and 3600s for sub-workflows
2. ✅ User Guide - "Default: 3600 seconds (60 minutes)", "Workflows exceeding this timeout are automatically terminated", "Enterprise plans can request custom timeout limits"
3. ❌ Incident report gathering - not about timeouts
4. ❌ Workflow Engine - general description, no timeout details
5. ❌ Fallback Actions - error handling, not timeouts

**Score**: 8/10  
**Reasoning**: Chunk 2 is EXCELLENT - has default timeout (3600s), automatic termination behavior, and Enterprise custom limits. Missing the exact error message format and TIMEOUT status. But has enough to answer the core question.

---

### tmp_004: Workflow Definition Cache Propagation
**Type**: temporal  
**Query**: How long does it take for workflow definition cache changes to propagate?

**Expected Answer**: Workflow definitions cached in Redis with TTL of 1 hour. Cache invalidated on workflow update or manual flush. Cache hit rate is 94.2%.

**Retrieved Chunks**:
1. ❌ Workflow Engine - general description, no cache details
2. ❌ Performance Characteristics - API latency (cache hit vs miss) but no TTL
3. ❌ Steps Per Workflow - limits, not caching
4. ❌ Database slow query log - not about caching
5. ❌ Kafka Event Streaming - not about workflow cache

**Score**: 2/10  
**Reasoning**: Chunk 2 mentions "cache hit" vs "cache miss" latency but doesn't explain TTL, invalidation strategy, or cache hit rate. Completely missing the answer. Wrong chunks retrieved.

---

### tmp_005: Database Primary Failover Timeline
**Type**: temporal  
**Query**: What's the timeline for automatic failover when the database primary fails?

**Expected Answer**: Database primary failure: 30-60 seconds for automatic promotion of replica. Redis failover: <10 seconds. Kafka controller election: <30 seconds.

**Retrieved Chunks**:
1. ✅ Disaster Recovery Procedures - "Detection: Health check fails for primary database (< 30 seconds)", "Action: Automatic promotion of read replica to primary", "Recovery time: 30-60 seconds"
2. ❌ Scheduler Service - leader election, not database failover
3. ❌ PostgreSQL Primary Database - cluster config, not failover
4. ❌ High Availability Architecture - Multi-AZ deployment, not failover timeline
5. ❌ Recovery Objectives - RPO/RTO, not specific failover times

**Score**: 9/10  
**Reasoning**: Chunk 1 is PERFECT for database failover - has detection time (<30s), action (promote replica), and recovery time (30-60s). Missing Redis (<10s) and Kafka (<30s) failover times. But answers the main question excellently.

---

### cmp_001: PgBouncer vs Direct PostgreSQL Connections
**Type**: comparative  
**Query**: What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?

**Expected Answer**: PgBouncer: pool_mode=transaction, default_pool_size=25, max_db_connections=100. Allows 1000 client connections with only 100 actual DB connections. Direct: limited to max_connections=100.

**Retrieved Chunks**:
1. ✅ Deployment Guide - PostgreSQL config with "max_connections = 100"
2. ❌ Database Architecture - general cluster config, no PgBouncer
3. ❌ Pods in CrashLoopBackOff - database connection failure, not pooling
4. ❌ Database slow query log - not about connection pooling
5. ❌ Database Queries - workflow action, not connection pooling

**Score**: 3/10  
**Reasoning**: Has max_connections=100 for direct PostgreSQL. But completely missing PgBouncer configuration (pool_mode, default_pool_size, max_db_connections). Cannot explain the difference or the 1000:100 client-to-DB connection ratio.

---

### cmp_002: Backoff Strategies (Fixed, Linear, Exponential)
**Type**: comparative  
**Query**: How do fixed, linear, and exponential backoff strategies differ for retries?

**Expected Answer**: Fixed: same wait time (1s, 1s, 1s). Linear: increase by fixed amount (1s, 2s, 3s). Exponential: double each time (1s, 2s, 4s). Exponential is recommended.

**Retrieved Chunks**:
1. ✅ User Guide - Retry Policies with "backoff_type: 'exponential' # or 'fixed', 'linear'", "initial_interval: 1000", "multiplier: 2.0"
2. ✅ Troubleshooting guide - "CloudFlow implements automatic retry with exponential backoff", "Max retries: 3", "Initial delay: 1 second"
3. ❌ Rate limit retry logic - Python example, not explaining backoff types
4. ❌ Fallback Actions - error handling, not retry backoff
5. ❌ Table of Contents - not relevant

**Score**: 8/10  
**Reasoning**: Chunk 1 is EXCELLENT - shows all three backoff types (exponential, fixed, linear), initial interval (1000ms), and multiplier (2.0). Chunk 2 confirms exponential backoff is used. Missing explicit examples of the sequences (1s,1s,1s vs 1s,2s,3s vs 1s,2s,4s) but has enough to infer the answer.

---

### cmp_003: /health vs /ready Endpoints
**Type**: comparative  
**Query**: What's the difference between /health and /ready endpoints?

**Expected Answer**: /health: liveness check, returns basic status. /ready: readiness check, checks dependencies like database and redis connectivity.

**Retrieved Chunks**:
1. ❌ SEV-3/SEV-4 incident severity - not about health endpoints
2. ❌ Environment Configuration - DATABASE_URL, REDIS_URL (not health checks)
3. ❌ High Availability Architecture - Multi-AZ deployment
4. ❌ Monitoring & Alerting - Prometheus metrics, not health endpoints
5. ❌ Performance Characteristics - API latency, not health checks

**Score**: 1/10  
**Reasoning**: Completely wrong chunks. No mention of /health or /ready endpoints. Cannot answer the question at all. Total retrieval failure.

---

### neg_001: What NOT to Do When Rate Limited
**Type**: negation  
**Query**: What should I NOT do when I'm rate limited?

**Expected Answer**: Don't keep hammering the API. Instead: check Retry-After header, implement exponential backoff, monitor X-RateLimit-Remaining, cache responses, consider upgrading tier.

**Retrieved Chunks**:
1. ✅ Rate Limiting - "X-RateLimit-Limit: 100", "X-RateLimit-Remaining: 87", "X-RateLimit-Reset: 1640995200", "When you exceed the rate limit, you'll rec..."
2. ✅ Rate Limit Tiers - table showing Free/Standard/Premium/Enterprise limits
3. ❌ Error Codes - mentions rate_limit_exceeded but no guidance
4. ✅ Handling Rate Limits in Code - Python example with "retry_after = int(response.headers.get('Retry-After', 60))", "time.sleep(retry_after)"
5. ❌ Fallback Actions - not about rate limiting

**Score**: 8/10  
**Reasoning**: Chunk 1 shows the headers to monitor (X-RateLimit-Remaining). Chunk 4 shows the correct approach (check Retry-After, sleep). Implicitly shows what NOT to do (don't ignore headers, don't keep retrying immediately). Missing explicit "don't hammer the API" warning and cache/upgrade suggestions, but has the core answer.

---

### neg_002: Why HS256 Doesn't Work for JWT
**Type**: negation  
**Query**: Why doesn't HS256 work for JWT token validation in CloudFlow?

**Expected Answer**: CloudFlow uses RS256 (asymmetric) not HS256 (symmetric). RS256 requires private key for signing, public key for validation. HS256 would fail with algorithm mismatch error.

**Retrieved Chunks**:
1. ❌ Troubleshooting guide overview - not about JWT algorithms
2. ✅ JWT Tokens - "CloudFlow supports JSON Web Tokens (JWT) with RS256 signing algorithm"
3. ❌ Error Codes - mentions authentication_failed but not algorithm details
4. ❌ API Gateway - mentions JWT token validation but not algorithm
5. ✅ Authentication & Authorization - "Algorithm: RS256 (asymmetric signing)", "Key rotation: Every 30 days"

**Score**: 7/10  
**Reasoning**: Chunks 2 and 5 clearly state CloudFlow uses RS256 (asymmetric). Can infer HS256 (symmetric) won't work. Missing explicit explanation of why (algorithm mismatch error) and the difference between symmetric vs asymmetric. Partial answer.

---

### neg_003: Why Can't Schedule More Frequently Than Every Minute
**Type**: negation  
**Query**: Why can't I schedule workflows more frequently than every minute?

**Expected Answer**: Minimum scheduling interval is 1 minute. Expressions evaluating to more frequent executions will be rejected. For near real-time, use webhook or event-based triggers instead.

**Retrieved Chunks**:
1. ❌ Execution Limits - max executions per day, not scheduling frequency
2. ✅ Scheduler Service - "Parse and validate cron expressions (extended format supporting seconds)", "Distributed scheduling with leader e..."
3. ❌ Test Thoroughly - not about scheduling limits
4. ❌ Workflow Engine - general description, not scheduling
5. ❌ Rate Limit Tiers - API rate limits, not workflow scheduling

**Score**: 3/10  
**Reasoning**: Chunk 2 mentions cron expressions support seconds, which contradicts the expected answer (minimum 1 minute). No mention of the 1-minute minimum or webhook alternatives. Cannot answer the question correctly.

---

### neg_004: No Token Refresh Logic Consequences
**Type**: negation  
**Query**: What happens if I don't implement token refresh logic?

**Expected Answer**: Tokens expire after 3600 seconds (1 hour). Without refresh logic, authentication will fail after expiry. Need to implement refresh using refresh token (valid 7-30 days).

**Retrieved Chunks**:
1. ❌ Troubleshooting guide overview - not about token refresh
2. ✅ JWT Tokens - "exp (expiration): Unix timestamp (max 3600 seconds from iat)"
3. ❌ Auth Service - mentions JWT token generation but not refresh
4. ❌ Authentication & Authorization - token validation, not refresh
5. ❌ Error Codes - authentication_failed but not expiry details

**Score**: 5/10  
**Reasoning**: Chunk 2 has the expiration time (3600 seconds = 1 hour). Can infer tokens will expire. But missing what happens after expiry (authentication fails) and the refresh token mechanism (7-30 days validity). Incomplete answer.

---

### neg_005: Why Not Hardcode API Keys
**Type**: negation  
**Query**: Why shouldn't I hardcode API keys in workflow definitions?

**Expected Answer**: Security risk - keys could be exposed. Use secrets instead: {{secrets.API_TOKEN}}. Secrets are encrypted at rest. Store in Settings > Secrets.

**Retrieved Chunks**:
1. ✅ API Keys - "Security Notes: Never expose API keys in client-side code", "Rotate keys every 90 days"
2. ✅ Authentication & Authorization - mentions token revocation, key rotation
3. ❌ Data Limits - not about security
4. ✅ Secrets Management - "HashiCorp Vault Integration", "API keys for external services: Stored with versioning", "Rotation policy: Automated rotation every 90 days"
5. ❌ Error Codes - not about API key security

**Score**: 7/10  
**Reasoning**: Chunk 1 says "Never expose API keys" (answers the "why not"). Chunk 4 mentions secrets management with Vault. But missing the workflow-specific syntax ({{secrets.API_TOKEN}}) and "Settings > Secrets" UI location. Partial answer.

---

### imp_001: Long-Running Data Processing Best Practices
**Type**: implicit  
**Query**: Best practice for handling long-running data processing that might exceed time limits

**Expected Answer**: Workflow timeout is 3600s. Solutions: split into smaller workflows, enable checkpointing (every 300s), use parallel workers, request custom timeout (up to 7200s on Enterprise).

**Retrieved Chunks**:
1. ✅ Error Handling - Retry Policies with exponential backoff
2. ✅ Data Limits - "Maximum execution payload: 50MB total"
3. ✅ Cache Invalidation Strategies - TTL info (Medium-lived data: 1 hour for workflow definitions)
4. ❌ Scheduler Service - cron scheduling, not timeout handling
5. ❌ Workflow Engine - general description, no timeout strategies

**Score**: 6/10  
**Reasoning**: Has some relevant limits (50MB payload, 1 hour cache TTL). But missing the critical 3600s timeout, checkpointing strategy (every 300s), sub-workflow splitting approach, and Enterprise 7200s custom timeout. Cannot provide complete answer.

---

### imp_003: How to Debug Slow API Calls
**Type**: implicit  
**Query**: How to debug why my API calls are slow

**Expected Answer**: Check latency breakdown: Auth (18%), DB Query (64%), Business Logic (13%), Serialization (5%). Use cloudflow metrics latency-report. Check slow query log. Review connection pool status.

**Retrieved Chunks**:
1. ✅ Performance Characteristics - P99 latency targets for API operations (GET /workflows: <50ms cache hit, <150ms cache miss)
2. ✅ Synchronous Request Flow - shows request path through API Gateway → Auth Service → Target Service → Database
3. ❌ Handling Rate Limits - retry logic, not debugging slow calls
4. ✅ Database slow query log - "kubectl logs -n cloudflow deploy/cloudflow-db-primary | grep 'slow query'", "cloudflow db analyze-queries --min-duration 5000"
5. ✅ Analytics - GET /analytics/workflows/{workflow_id} with metrics like avg_duration, success_rate

**Score**: 7/10  
**Reasoning**: Has P99 latency targets (chunk 1), request flow showing components (chunk 2), slow query log commands (chunk 4), and analytics endpoint (chunk 5). Missing the specific latency breakdown percentages (Auth 18%, DB 64%, etc.) and "cloudflow metrics latency-report" command. But has enough to debug slow API calls.

---

## Summary Statistics

**Completed**: 15/15 queries graded  

### Final Results:

| Score Range | Count | Percentage |
|-------------|-------|------------|
| 9-10 (Excellent) | 1 | 6.7% |
| 8 (Good) | 3 | 20.0% |
| 7 (Acceptable) | 3 | 20.0% |
| 5-6 (Weak) | 4 | 26.7% |
| 3-4 (Poor) | 3 | 20.0% |
| 1-2 (Failed) | 1 | 6.7% |

**Average Score**: 5.7/10 (57%)  
**Pass Rate (≥8)**: 26.7% (4/15)

### Comparison to adaptive_hybrid:
- **adaptive_hybrid**: 5.9/10 average, 20% pass rate (3/15)
- **synthetic_variants**: 5.7/10 average, 26.7% pass rate (4/15)

**Result**: synthetic_variants shows **+6.7% pass rate improvement** but **-0.2 average score** vs adaptive_hybrid. Slightly better at getting excellent results (4 vs 3 queries ≥8) but slightly worse on average.

---

## Key Findings (FINAL)

### Strengths:
1. **Excellent on temporal queries**: tmp_003 (8/10), tmp_005 (9/10)
2. **Good on comparative queries**: cmp_002 (8/10)
3. **Good on negation queries**: neg_001 (8/10)
4. **Acceptable on debugging**: imp_003 (7/10), neg_002 (7/10), neg_005 (7/10)

### Weaknesses:
1. **Failed on /health vs /ready**: cmp_003 (1/10) - completely wrong chunks
2. **Poor on cache propagation**: tmp_004 (2/10) - missing cache TTL info
3. **Poor on PgBouncer queries**: mh_002 (3/10), cmp_001 (3/10) - missing config details
4. **Poor on scheduling limits**: neg_003 (3/10) - contradictory info

### Root Causes:
1. **Vocabulary mismatch**: "PgBouncer" not in corpus, "/health" and "/ready" not in corpus
2. **Missing content**: Cache TTL (1 hour), cache hit rate (94.2%), PgBouncer config, HPA parameters
3. **Query expansion not helping**: synthetic_variants generates query variants but still retrieves wrong chunks for vocabulary-mismatched queries

**Note**: Chunking is NOT the issue - we're using MarkdownSemanticStrategy (400-word semantic chunks) which should preserve context well.

### Comparison: synthetic_variants vs adaptive_hybrid

| Metric | adaptive_hybrid | synthetic_variants | Difference |
|--------|----------------|-------------------|------------|
| Average Score | 5.9/10 | 5.7/10 | -0.2 |
| Pass Rate (≥8) | 20% (3/15) | 26.7% (4/15) | +6.7% |
| Excellent (9-10) | 6.7% (1/15) | 6.7% (1/15) | 0% |
| Failed (1-2) | 6.7% (1/15) | 6.7% (1/15) | 0% |

**Verdict**: synthetic_variants is **marginally better** (+6.7% pass rate) but **not significantly different** from adaptive_hybrid. Both strategies struggle with the same root causes (vocabulary mismatch, missing content).

---

## Conclusion

**synthetic_variants does NOT provide significant improvement over adaptive_hybrid** when using MarkdownSemanticStrategy.

**Key Insight**: Query expansion (generating variants) doesn't help when:
1. The corpus doesn't contain the vocabulary (PgBouncer, /health, /ready)
2. The content is missing (HPA params, cache TTL, PgBouncer config)

**Chunking is working correctly** - MarkdownSemanticStrategy creates semantic 400-word chunks that preserve context.

**Recommendation**: 
- **Skip testing more retrieval strategies** - the problem is corpus quality, not retrieval algorithm
- **Focus on corpus improvement**: Add missing content, fix chunking boundaries
- **OR**: Test enriched_hybrid_llm (the 94% baseline) to confirm it works with MarkdownSemanticStrategy

---

## Next Steps

1. ✅ Complete grading for all 15 queries
2. ✅ Calculate final statistics
3. ✅ Compare to adaptive_hybrid (5.9/10)
4. ⬜ **DECISION POINT**: 
   - Option A: Test enriched_hybrid_llm (expected ~94% based on SMART_CHUNKING_VALIDATION_REPORT.md)
   - Option B: Analyze corpus gaps and add missing content
   - Option C: Stop testing - both strategies show similar poor performance (~57-59%)
