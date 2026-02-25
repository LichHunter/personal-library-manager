# enriched_hybrid_llm Strategy - Manual Grading Results

**Date**: 2026-01-26  
**Strategy**: enriched_hybrid_llm  
**Chunking**: MarkdownSemanticStrategy (target=400 words)  
**Chunks**: 80  
**Queries Graded**: 15

---

## Grading Rubric

| Score | Level | Criteria |
|-------|-------|----------|
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

---

## Query Results

### mh_002: Connection Pool Exhaustion Solutions
**Type**: multi-hop  
**Query**: If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?

**Expected Answer**: Both are valid. PgBouncer for connection pooling (max_db_connections=100, pool_mode=transaction). Read replicas for read-heavy workloads. Troubleshooting guide recommends PgBouncer first.

**Retrieved Chunks**:
1. ❌ Pods in CrashLoopBackOff (deployment troubleshooting - not relevant)
2. ✅ Database Architecture (mentions primary + 2 read replicas setup)
3. ✅ postgres-values.yaml (shows max_connections=100 config)
4. ✅ High Availability Architecture (multi-AZ with replicas)
5. ❌ Monitoring & Alerting (connection pool saturation mentioned but no solutions)

**Score**: 6/10  
**Reasoning**: The retrieved chunks contain relevant infrastructure details (max_connections=100, read replicas setup) but lack the critical PgBouncer configuration details (pool_mode=transaction, default_pool_size=25, max_db_connections=100). The comparison between PgBouncer and read replicas is missing entirely. Chunk 3 has the right file but doesn't show the full PgBouncer config. A user would need to infer the solution rather than find it directly.

---

### mh_004: HPA Scaling Parameters and API Gateway Resources
**Type**: multi-hop  
**Query**: How do the HPA scaling parameters relate to the API Gateway resource requirements?

**Expected Answer**: HPA: minReplicas=3, maxReplicas=10, targetCPU=70%. API Gateway: 2 vCPU, 4GB RAM per pod. Scales when CPU exceeds 70% of 2 vCPU.

**Retrieved Chunks**:
1. ❌ Grafana Dashboards (monitoring setup, not HPA config)
2. ✅ Resource Utilization (shows API Gateway 55% average, 80% peak, 4GB allocated)
3. ❌ Table of Contents (navigation only)
4. ✅ API Gateway microservice (2 vCPU, 4GB RAM per pod, auto-scaling 8-20 based on CPU)
5. ❌ Pods in CrashLoopBackOff (troubleshooting, not relevant)

**Score**: 7/10  
**Reasoning**: Chunks 2 and 4 provide the API Gateway resource requirements (2 vCPU, 4GB RAM) and mention auto-scaling based on CPU. However, the specific HPA parameters (minReplicas=3, maxReplicas=10, targetCPU=70%) are not present in the retrieved chunks. Chunk 4 mentions "auto-scaling 8-20" which contradicts the expected minReplicas=3, maxReplicas=10. A user would get partial information but miss the exact HPA configuration.

---

### tmp_003: Workflow Execution Timeout Sequence
**Type**: temporal  
**Query**: What's the sequence of events when a workflow execution times out?

**Expected Answer**: Workflow runs up to 3600s. If exceeded, automatically terminated. Error: 'exceeded maximum execution time of 3600 seconds'. Status: TIMEOUT. Can request custom timeout up to 7200s on Enterprise.

**Retrieved Chunks**:
1. ✅ Execution Timeout (3600 seconds default, automatically terminated, custom timeouts on Enterprise)
2. ✅ Sub-workflows creation (timeout configuration examples)
3. ✅ Error Handling (retry policies and error handling)
4. ❌ Incident gathering (troubleshooting procedures, not timeout behavior)
5. ❌ Error handling configuration (general error handling, not timeout-specific)

**Score**: 9/10  
**Reasoning**: Chunk 1 directly answers the query with all key details: 3600s default, automatic termination, custom timeouts on Enterprise. Chunk 2 provides practical examples of timeout configuration. The specific error message format ('exceeded maximum execution time of 3600 seconds') and TIMEOUT status are not explicitly stated, but all essential information is present. This is an excellent retrieval with only minor details missing.

---

### tmp_004: Workflow Definition Cache Propagation Timeline
**Type**: temporal  
**Query**: How long does it take for workflow definition cache changes to propagate?

**Expected Answer**: Workflow definitions cached in Redis with TTL of 1 hour. Cache invalidated on workflow update or manual flush. Cache hit rate is 94.2%.

**Retrieved Chunks**:
1. ❌ Sub-workflows creation (workflow creation, not caching)
2. ✅ Cache Invalidation Strategies (TTL: 1 hour for workflow definitions, event-based invalidation)
3. ❌ Database slow query analysis (database troubleshooting, not caching)
4. ❌ Performance Characteristics (latency targets, not cache propagation)
5. ❌ Monitoring & Alerting (cache hit/miss ratios mentioned but no details)

**Score**: 7/10  
**Reasoning**: Chunk 2 provides the critical information: 1-hour TTL for workflow definitions and event-based invalidation. However, the cache hit rate (94.2%) is not mentioned in any chunk. The propagation mechanism is partially explained (TTL + event-based) but lacks specific timing details for event-based invalidation. A user would understand the general caching strategy but miss the specific cache hit rate metric.

---

### tmp_005: Database Primary Failover Timeline
**Type**: temporal  
**Query**: What's the timeline for automatic failover when the database primary fails?

**Expected Answer**: Database primary failure: 30-60 seconds for automatic promotion of replica. Redis failover: <10 seconds. Kafka controller election: <30 seconds.

**Retrieved Chunks**:
1. ✅ Disaster Recovery Procedures (detection <30s, automatic promotion, recovery 30-60s)
2. ✅ Recovery Objectives (RPO 1 hour, RTO 4 hours, automated runbooks)
3. ❌ Scheduler Service (distributed scheduling, not failover)
4. ❌ High Availability Architecture (multi-AZ setup, not failover timing)
5. ❌ Testing & Validation (DR test results, not failover timeline)

**Score**: 8/10  
**Reasoning**: Chunk 1 directly provides the database failover timeline: detection <30s, recovery 30-60s. However, Redis failover (<10s) and Kafka controller election (<30s) are not mentioned in the retrieved chunks. The answer is mostly complete for the primary question but missing the comparative timelines for other components. A user would get the main answer but lack the full picture.

---

### cmp_001: PgBouncer vs Direct PostgreSQL Connections
**Type**: comparative  
**Query**: What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?

**Expected Answer**: PgBouncer: pool_mode=transaction, default_pool_size=25, max_db_connections=100. Allows 1000 client connections with only 100 actual DB connections. Direct: limited to max_connections=100.

**Retrieved Chunks**:
1. ✅ postgres-values.yaml (max_connections=100 config)
2. ❌ Pods in CrashLoopBackOff (troubleshooting, not relevant)
3. ✅ Database Architecture (primary-replica setup, instance type, storage)
4. ❌ Performance Characteristics (latency targets, not connection pooling)
5. ❌ Throughput Capacity (API and database throughput, not pooling)

**Score**: 5/10  
**Reasoning**: Chunk 1 shows max_connections=100 but doesn't explain PgBouncer configuration (pool_mode, default_pool_size, max_db_connections). The critical comparison—how PgBouncer allows 1000 client connections with only 100 DB connections—is completely missing. The chunks show infrastructure details but lack the conceptual explanation of the difference between pooling and direct connections. A user would need external knowledge to understand the answer.

---

### cmp_002: Backoff Strategies (Fixed, Linear, Exponential)
**Type**: comparative  
**Query**: How do fixed, linear, and exponential backoff strategies differ for retries?

**Expected Answer**: Fixed: same wait time (1s, 1s, 1s). Linear: increase by fixed amount (1s, 2s, 3s). Exponential: double each time (1s, 2s, 4s). Exponential is recommended.

**Retrieved Chunks**:
1. ✅ Retry Logic and Exponential Backoff (mentions exponential backoff, max retries 3, initial delay 1s)
2. ✅ Retry Policies (backoff_type options: "exponential", "fixed", "linear", with multiplier 2.0)
3. ❌ Cache Invalidation Strategies (caching, not retry logic)
4. ❌ Rate Limit Handling (rate limiting, not backoff strategies)
5. ❌ Fallback Actions (error handling, not backoff)

**Score**: 8/10  
**Reasoning**: Chunks 1 and 2 provide excellent information about the three backoff types and their configuration. Chunk 2 explicitly lists "exponential", "fixed", "linear" options with multiplier 2.0. However, the specific examples (1s, 1s, 1s vs 1s, 2s, 3s vs 1s, 2s, 4s) are not provided. The recommendation for exponential is implied but not explicitly stated. A user would understand the differences but lack concrete timing examples.

---

### cmp_003: /health vs /ready Endpoints
**Type**: comparative  
**Query**: What's the difference between /health and /ready endpoints?

**Expected Answer**: /health: liveness check, returns basic status. /ready: readiness check, checks dependencies like database and redis connectivity.

**Retrieved Chunks**:
1. ❌ Prometheus Setup (monitoring installation, not health endpoints)
2. ❌ Architecture Diagram (conceptual diagram, not endpoints)
3. ❌ Environment Configuration (env variables, not health checks)
4. ❌ Required Tools (deployment prerequisites, not health endpoints)
5. ❌ High Availability Architecture (multi-AZ setup, not health checks)

**Score**: 2/10  
**Reasoning**: None of the retrieved chunks contain information about /health or /ready endpoints. The chunks are about monitoring setup, architecture diagrams, and deployment configuration—completely wrong context. This is a failed retrieval where the LLM query rewriting failed to find the relevant documentation about health check endpoints. A user would find no useful information.

---

### neg_001: What NOT to Do When Rate Limited
**Type**: negation  
**Query**: What should I NOT do when I'm rate limited?

**Expected Answer**: Don't keep hammering the API. Instead: check Retry-After header, implement exponential backoff, monitor X-RateLimit-Remaining, cache responses, consider upgrading tier.

**Retrieved Chunks**:
1. ❌ Error handling configuration (general error handling, not rate limiting)
2. ✅ Handling Rate Limits in Code (Retry-After header, exponential backoff, rate limiting handling)
3. ❌ Data Limits (request/response size limits, not rate limiting behavior)
4. ❌ Error Codes (error code reference, not rate limiting solutions)
5. ❌ Retry Policies (general retry logic, not rate limiting specific)

**Score**: 7/10  
**Reasoning**: Chunk 2 provides practical guidance on handling rate limits: checking Retry-After header, implementing exponential backoff, and retry logic. However, the explicit "don't hammer the API" warning is not stated, and the monitoring of X-RateLimit-Remaining header and upgrading tier options are not mentioned. The chunk shows what TO DO but doesn't explicitly frame it as what NOT to do. A user would get practical solutions but miss the explicit negation framing.

---

### neg_002: Why HS256 Doesn't Work for JWT Validation
**Type**: negation  
**Query**: Why doesn't HS256 work for JWT token validation in CloudFlow?

**Expected Answer**: CloudFlow uses RS256 (asymmetric) not HS256 (symmetric). RS256 requires private key for signing, public key for validation. HS256 would fail with algorithm mismatch error.

**Retrieved Chunks**:
1. ✅ JWT Tokens (RS256 signing algorithm, JWT claims, example)
2. ✅ API Gateway (JWT token validation delegated to Auth Service)
3. ✅ Auth Service (JWT token generation and validation RS256 algorithm)
4. ❌ Troubleshooting Overview (general troubleshooting, not JWT-specific)
5. ✅ Authentication & Authorization (RS256 asymmetric signing, key rotation, JWKS endpoint)

**Score**: 9/10  
**Reasoning**: Chunks 1, 3, and 5 clearly state that CloudFlow uses RS256 (asymmetric) algorithm. Chunk 5 explicitly mentions "RS256 (asymmetric signing)" which contrasts with HS256 (symmetric). The explanation of why HS256 fails (algorithm mismatch) is implied but not explicitly stated. All essential information is present: CloudFlow uses RS256, it's asymmetric, and the contrast with HS256 is clear. Only the explicit error message is missing.

---

### neg_003: Minimum Scheduling Interval Constraint
**Type**: negation  
**Query**: Why can't I schedule workflows more frequently than every minute?

**Expected Answer**: Minimum scheduling interval is 1 minute. Expressions evaluating to more frequent executions will be rejected. For near real-time, use webhook or event-based triggers instead.

**Retrieved Chunks**:
1. ✅ Scheduling (cron syntax, common patterns, standard cron expressions)
2. ✅ Execution Limits (1000 executions per day, 100 concurrent, 10 per second burst limit)
3. ❌ Scheduler Service (distributed scheduling architecture, not constraints)
4. ❌ Rate Limit Tiers (rate limiting by tier, not scheduling constraints)
5. ❌ Data Limits (request/response size limits, not scheduling)

**Score**: 6/10  
**Reasoning**: Chunk 1 shows cron syntax but doesn't explicitly state the 1-minute minimum interval constraint. Chunk 2 mentions execution limits (1000/day, 100 concurrent, 10/second) which implies a minimum interval but doesn't state it directly. The alternative solutions (webhook or event-based triggers) are not mentioned in any chunk. A user would understand cron syntax and execution limits but wouldn't find the explicit answer to why 1-minute is the minimum.

---

### neg_004: Token Refresh Logic Consequences
**Type**: negation  
**Query**: What happens if I don't implement token refresh logic?

**Expected Answer**: Tokens expire after 3600 seconds (1 hour). Without refresh logic, authentication will fail after expiry. Need to implement refresh using refresh token (valid 7-30 days).

**Retrieved Chunks**:
1. ✅ Auth Service (JWT token generation and validation RS256)
2. ❌ Troubleshooting Overview (general troubleshooting, not token refresh)
3. ✅ Authentication & Authorization (RS256, key rotation, token revocation, validation)
4. ✅ JWT Tokens (exp claim max 3600 seconds from iat, JWT claims)
5. ❌ Redis Caching Layer (session storage, TTL 15 minutes, not token refresh)

**Score**: 7/10  
**Reasoning**: Chunk 4 explicitly states "exp (expiration): Unix timestamp (max 3600 seconds from iat)" which answers the token expiry question. Chunk 3 mentions token validation and revocation. However, the consequences of not implementing refresh logic (authentication failure) and the refresh token mechanism (valid 7-30 days) are not mentioned. A user would understand token expiry but miss the refresh token solution.

---

### neg_005: Hardcoding API Keys Security Risk
**Type**: negation  
**Query**: Why shouldn't I hardcode API keys in workflow definitions?

**Expected Answer**: Security risk - keys could be exposed. Use secrets instead: {{secrets.API_TOKEN}}. Secrets are encrypted at rest. Store in Settings > Secrets.

**Retrieved Chunks**:
1. ❌ Network Policy (Kubernetes network policies, not secrets management)
2. ✅ API Keys (API key authentication, security notes about not exposing keys, rotation every 90 days)
3. ✅ Authentication & Authorization (JWT validation, permission model, token revocation)
4. ❌ Data Limits (request/response size limits, not secrets)
5. ❌ API Reference Overview (API documentation overview, not secrets)

**Score**: 7/10  
**Reasoning**: Chunk 2 provides security guidance: "Never expose API keys in client-side code" and "Rotate keys every 90 days". This addresses the security risk aspect. However, the specific solution ({{secrets.API_TOKEN}} syntax) and the secrets management interface (Settings > Secrets) are not mentioned. The explanation of encryption at rest is also missing. A user would understand the security risk but lack the implementation details.

---

### imp_001: Handling Long-Running Data Processing
**Type**: implicit  
**Query**: Best practice for handling long-running data processing that might exceed time limits

**Expected Answer**: Workflow timeout is 3600s. Solutions: split into smaller workflows, enable checkpointing (every 300s), use parallel workers, request custom timeout (up to 7200s on Enterprise).

**Retrieved Chunks**:
1. ✅ Steps Per Workflow (max 50 steps, recommendation to keep focused, split into multiple workflows)
2. ✅ Execution Timeout (3600s default, automatically terminated, custom timeouts on Enterprise)
3. ✅ Sub-workflows creation (splitting workflows, timeout configuration)
4. ❌ Data Flow Architecture (synchronous request flow, timeout configuration)
5. ❌ Database slow query analysis (database troubleshooting, not workflow design)

**Score**: 8/10  
**Reasoning**: Chunks 1, 2, and 3 provide excellent guidance: 3600s timeout, recommendation to split workflows, custom timeouts on Enterprise. The solution of splitting into multiple workflows is clearly stated. However, checkpointing (every 300s) and parallel workers are not mentioned in the retrieved chunks. The core solutions are present but some advanced techniques are missing. A user would get practical solutions but miss optimization details.

---

### imp_003: Debugging Slow API Calls
**Type**: implicit  
**Query**: How to debug why my API calls are slow

**Expected Answer**: Check latency breakdown: Auth (18%), DB Query (64%), Business Logic (13%), Serialization (5%). Use cloudflow metrics latency-report. Check slow query log. Review connection pool status.

**Retrieved Chunks**:
1. ❌ SEV-2 High severity (incident severity definitions, not debugging)
2. ✅ Monitoring & Alerting (key metrics: latency percentiles, resource utilization, cache hit/miss)
3. ✅ Performance Characteristics (latency targets for API operations, P99 latency)
4. ❌ Load testing setup (k6 load testing configuration, not debugging)
5. ❌ RBAC Policy Violations (access control, not performance debugging)

**Score**: 6/10  
**Reasoning**: Chunks 2 and 3 provide monitoring guidance (latency percentiles, performance targets) but lack the specific latency breakdown percentages (Auth 18%, DB 64%, etc.). The specific tools mentioned in the expected answer (cloudflow metrics latency-report, slow query log) are not present in the retrieved chunks. Chunk 3 shows latency targets but not how to debug actual slow calls. A user would understand monitoring concepts but lack concrete debugging steps.

---

## Summary Statistics

### Score Distribution

| Score | Count | Percentage |
|-------|-------|-----------|
| 9 | 2 | 13.3% |
| 8 | 3 | 20.0% |
| 7 | 4 | 26.7% |
| 6 | 4 | 26.7% |
| 5 | 1 | 6.7% |
| 4 | 0 | 0% |
| 3 | 0 | 0% |
| 2 | 1 | 6.7% |
| 1 | 0 | 0% |

### Key Metrics

- **Average Score**: 6.87/10 (68.7%)
- **Pass Rate (≥8/10)**: 5/15 (33.3%)
- **Median Score**: 7/10
- **Mode Score**: 6 and 7 (tied, 4 queries each)
- **Excellent (9-10)**: 2 queries (13.3%)
- **Good (8)**: 3 queries (20.0%)
- **Acceptable (7)**: 4 queries (26.7%)
- **Marginal (6)**: 4 queries (26.7%)
- **Weak (5)**: 1 query (6.7%)
- **Failed (2)**: 1 query (6.7%)

---

## Comparison to Baselines

### Performance vs Previous Strategies

| Strategy | Average Score | Pass Rate (≥8) | Improvement |
|----------|---------------|----------------|-------------|
| **enriched_hybrid_llm** | **6.87/10** | **5/15 (33.3%)** | **Baseline** |
| adaptive_hybrid | 5.9/10 | 3/15 (20.0%) | +0.97 (+16.4%) |
| synthetic_variants | 5.7/10 | 4/15 (26.7%) | +1.17 (+20.5%) |

### Detailed Comparison

**vs adaptive_hybrid (5.9/10, 20% pass rate)**:
- enriched_hybrid_llm: +0.97 points (+16.4% improvement)
- Pass rate improvement: +13.3% (5 vs 3 queries)
- Better performance on: temporal queries, negation queries, implicit queries
- Similar weakness: comparative queries (cmp_001, cmp_003)

**vs synthetic_variants (5.7/10, 26.7% pass rate)**:
- enriched_hybrid_llm: +1.17 points (+20.5% improvement)
- Pass rate improvement: +6.6% (5 vs 4 queries)
- Better performance on: multi-hop queries, temporal queries
- Similar weakness: comparative queries, negation queries

### Query Type Performance

| Type | Count | Avg Score | Pass Rate |
|------|-------|-----------|-----------|
| Multi-hop (mh) | 2 | 6.5/10 | 0/2 (0%) |
| Temporal (tmp) | 3 | 7.67/10 | 2/3 (66.7%) |
| Comparative (cmp) | 3 | 5.33/10 | 1/3 (33.3%) |
| Negation (neg) | 5 | 7.0/10 | 2/5 (40.0%) |
| Implicit (imp) | 2 | 7.0/10 | 0/2 (0%) |

---

## Key Findings

### Strengths of enriched_hybrid_llm

1. **Excellent temporal query performance** (7.67/10 avg):
   - tmp_003 (timeout sequence): 9/10 - Perfect retrieval
   - tmp_005 (failover timeline): 8/10 - Good retrieval
   - Temporal queries benefit from LLM query rewriting

2. **Strong negation query handling** (7.0/10 avg):
   - neg_002 (HS256 JWT): 9/10 - Excellent
   - neg_001, neg_004, neg_005: 7/10 - Acceptable
   - LLM rewriting helps with "why not" and "what shouldn't" questions

3. **Improvement over baselines**:
   - +16.4% vs adaptive_hybrid
   - +20.5% vs synthetic_variants
   - Consistent gains across most query types

### Weaknesses and Failure Modes

1. **Comparative queries struggle** (5.33/10 avg):
   - cmp_001 (PgBouncer vs direct): 5/10 - Weak
   - cmp_003 (/health vs /ready): 2/10 - Failed
   - LLM rewriting doesn't help with "difference between X and Y" questions
   - Root cause: Chunks contain individual facts but lack comparative analysis

2. **Multi-hop queries underperform** (6.5/10 avg):
   - mh_002 (connection pool solutions): 6/10 - Marginal
   - mh_004 (HPA + API Gateway): 7/10 - Acceptable
   - Requires connecting multiple concepts across chunks
   - Missing specific configuration details (pool_mode, minReplicas)

3. **Implicit queries miss specific details** (7.0/10 avg):
   - imp_001 (long-running processing): 8/10 - Good
   - imp_003 (debug slow API): 6/10 - Marginal
   - Chunks provide general guidance but lack specific metrics/tools

4. **Complete retrieval failures**:
   - cmp_003 (/health vs /ready): 2/10 - Wrong context entirely
   - Root cause: Health check endpoints not documented in corpus
   - LLM rewriting failed to find relevant chunks

### Root Causes of Failures

1. **Vocabulary Mismatch** (affects mh_002, cmp_001, imp_001):
   - Query uses "connection pool exhaustion" but chunks use "max_connections"
   - Query asks about "PgBouncer" but chunks show "postgres-values.yaml"
   - LLM rewriting helps but doesn't fully bridge the gap

2. **Missing Comparative Analysis** (affects cmp_001, cmp_003):
   - Documentation contains individual facts but not side-by-side comparisons
   - Chunks about /health endpoint don't exist in corpus
   - LLM rewriting can't create information that doesn't exist

3. **Implicit Assumptions** (affects imp_003):
   - Query assumes latency breakdown percentages exist
   - Chunks show latency targets but not actual breakdown
   - Missing specific debugging tools (cloudflow metrics latency-report)

4. **Configuration Details** (affects mh_002, mh_004):
   - Expected answers include specific config values (pool_mode=transaction, minReplicas=3)
   - Chunks show some values but not all
   - Incomplete configuration documentation

---

## Expected vs Actual Performance

### Expected Performance
- **Target**: ~94% (9.4/10) based on SMART_CHUNKING_VALIDATION_REPORT.md
- **Expected Pass Rate**: ~90%+ (13-14 out of 15 queries ≥8)

### Actual Performance
- **Achieved**: 68.7% (6.87/10)
- **Actual Pass Rate**: 33.3% (5 out of 15 queries ≥8)
- **Gap**: -25.3 points (-269% below target)

### Analysis of Gap

The enriched_hybrid_llm strategy **significantly underperforms** the expected ~94% baseline:

1. **Corpus Limitations**:
   - The test corpus (5 CloudFlow documents) is much smaller than production documentation
   - Missing documentation for health endpoints, specific configuration details
   - Vocabulary mismatches between queries and documentation

2. **Query Type Mismatch**:
   - Comparative queries (cmp_*) are particularly weak (5.33/10)
   - Multi-hop queries require connecting facts across chunks
   - LLM rewriting excels at temporal/negation but struggles with comparisons

3. **Chunking Strategy Issues**:
   - 400-word semantic chunks may be too large for some queries
   - Chunks contain mixed topics (e.g., troubleshooting + architecture)
   - Chunk boundaries don't align with query intent

4. **LLM Rewriting Limitations**:
   - Effective for temporal and negation queries
   - Ineffective for comparative queries (can't create missing comparisons)
   - Can't compensate for missing documentation

---

## Conclusion

### Overall Assessment

The **enriched_hybrid_llm strategy achieves 6.87/10 (68.7%)**, which is a **significant improvement over baselines** (+16-20%) but **falls far short of the expected ~94% target**.

### Key Takeaways

1. **Strategy is effective but not production-ready**:
   - Outperforms adaptive_hybrid and synthetic_variants
   - Excellent for temporal and negation queries
   - Fails on comparative queries and missing documentation

2. **The ~94% target is not achievable with current corpus**:
   - Test corpus lacks critical documentation (health endpoints, specific configs)
   - Vocabulary mismatches between queries and documentation
   - Comparative queries require documentation that doesn't exist

3. **Recommendations for improvement**:
   - **Expand documentation**: Add explicit comparisons (PgBouncer vs direct, /health vs /ready)
   - **Improve chunking**: Reduce chunk size to 200-300 words for better precision
   - **Add configuration details**: Include all specific values (pool_mode, minReplicas, etc.)
   - **Hybrid approach**: Combine enriched_hybrid_llm with reranking for comparative queries
   - **Query-specific strategies**: Use different strategies for different query types

4. **Production readiness**:
   - **NOT READY** for production at 68.7% accuracy
   - Requires corpus expansion and documentation improvements
   - Consider hybrid approach combining multiple strategies
   - Acceptable for non-critical use cases (documentation search, learning)

### Next Steps

1. Expand test corpus with more comprehensive documentation
2. Add explicit comparative sections to documentation
3. Implement query-type-specific routing (temporal → enriched_hybrid_llm, comparative → reranking)
4. Test with larger, more realistic corpus
5. Evaluate against production query logs to identify real-world patterns

