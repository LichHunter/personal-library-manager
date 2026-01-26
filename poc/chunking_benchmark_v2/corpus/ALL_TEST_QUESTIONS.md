# All Test Questions - Comprehensive List

**Purpose**: Complete list of all questions used in retrieval testing  
**Date**: 2026-01-26

---

## Table of Contents

1. [94% Baseline Test Questions (10 questions)](#94-baseline-test-questions)
2. [68.7% Edge Case Test Questions (15 questions)](#687-edge-case-test-questions)
3. [90% Needle-Haystack Test (20 questions)](#90-needle-haystack-test)
4. [65% Adversarial Needle-Haystack Test (20 questions)](#65-adversarial-needle-haystack-test)
5. [Passing Queries (9 questions)](#passing-queries-not-tested-in-current-run)
6. [Summary Statistics](#summary-statistics)

---

## 94% Baseline Test Questions

**Source**: `results/manual_test_smart_chunking.md`  
**Test Date**: 2026-01-25  
**Strategy**: enriched_hybrid_llm  
**Chunking**: MarkdownSemanticStrategy  
**Average Score**: 9.4/10 (94%)

### Question 1: API Rate Limit
**Query**: How many requests per minute are allowed in the CloudFlow API?

**Expected Answer**: 100 requests per minute per authenticated user, 20 requests per minute for unauthenticated requests, with a burst allowance of 150 requests in a 10-second window

**Score**: 10/10  
**Difficulty**: Easy  
**Type**: Factual lookup

---

### Question 2: Rate Limit Exceeded
**Query**: What happens when I exceed the CloudFlow API rate limit?

**Expected Answer**: When you exceed the rate limit, you'll receive a `429 Too Many Requests` response with an error message indicating how long to wait before retrying

**Score**: 10/10  
**Difficulty**: Easy  
**Type**: Factual lookup

---

### Question 3: Handle Rate Limits
**Query**: How do I handle rate limit errors in my code?

**Expected Answer**: Implement exponential backoff, monitor `X-RateLimit-Remaining` header values, cache responses when appropriate, and consider upgrading to Enterprise tier for higher limits

**Score**: 10/10  
**Difficulty**: Easy  
**Type**: Procedural

---

### Question 4: Auth Methods
**Query**: What are the CloudFlow API authentication methods?

**Expected Answer**: CloudFlow supports three authentication methods: API Keys, OAuth 2.0, and JWT Tokens

**Score**: 10/10  
**Difficulty**: Medium (was 4/10 with fixed chunking - truncation issue)  
**Type**: Factual lookup

---

### Question 5: JWT Duration
**Query**: How long do JWT tokens last in CloudFlow?

**Expected Answer**: All tokens expire after 3600 seconds (1 hour). Implement token refresh logic in your application.

**Score**: 10/10  
**Difficulty**: Medium (was 3/10 with fixed chunking - wrong context issue)  
**Type**: Factual lookup

---

### Question 6: Workflow Timeout
**Query**: What should I do if my workflow exceeds the maximum execution timeout?

**Expected Answer**: Workflows have a default timeout of 3600 seconds (60 minutes). You can increase the timeout, optimize workflow steps, or split the workflow into smaller workflows

**Score**: 9/10  
**Difficulty**: Medium  
**Type**: Procedural

---

### Question 7: Restart Workflow
**Query**: How do I restart a failed workflow execution?

**Expected Answer**: View failed executions in the Dead Letter Queue, inspect the execution context and error details, and use the 'Retry' button to reprocess with the same input data

**Score**: 7/10  
**Difficulty**: Hard (vocabulary mismatch - "Dead Letter Queue" not well indexed)  
**Type**: Procedural

---

### Question 8: Troubleshooting
**Query**: What are the recommended steps for troubleshooting a workflow failure?

**Expected Answer**: Verify service health, check API connectivity, review recent deployments, inspect platform metrics, check logs, and follow the escalation procedure based on the severity level

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Procedural

---

### Question 9: DB Pooling
**Query**: How do I set up database connection pooling in CloudFlow?

**Expected Answer**: Use PgBouncer for connection pooling, configure CloudFlow to use PgBouncer, set connection pool modes, and add read replicas to optimize database performance

**Score**: 8/10  
**Difficulty**: Hard (was 2/10 with fixed chunking - PgBouncer section split)  
**Type**: Procedural

---

### Question 10: K8s Resources
**Query**: What Kubernetes resources are needed to deploy CloudFlow?

**Expected Answer**: Deploy an EKS cluster with managed node groups, configure storage with EBS CSI driver, set up a namespace with resource quotas, and use Helm charts for deployment

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Procedural

---

## 68.7% Edge Case Test Questions

**Source**: `corpus/edge_case_queries.json` (failed queries only)  
**Test Date**: 2026-01-26  
**Strategy**: enriched_hybrid_llm  
**Chunking**: MarkdownSemanticStrategy  
**Average Score**: 6.87/10 (68.7%)

### mh_002: Connection Pool Exhaustion Solutions
**Query**: If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?

**Expected Answer**: Both are valid. PgBouncer for connection pooling (max_db_connections=100, pool_mode=transaction). Read replicas for read-heavy workloads. Troubleshooting guide recommends PgBouncer first.

**Score**: 6/10  
**Difficulty**: Hard  
**Type**: Multi-hop  
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND  
**Issues**: PgBouncer configuration details missing, comparative analysis weak

---

### mh_004: HPA Scaling Parameters and API Gateway Resources
**Query**: How do the HPA scaling parameters relate to the API Gateway resource requirements?

**Expected Answer**: HPA: minReplicas=3, maxReplicas=10, targetCPU=70%. API Gateway: 2 vCPU, 4GB RAM per pod. Scales when CPU exceeds 70% of 2 vCPU.

**Score**: 7/10  
**Difficulty**: Hard  
**Type**: Multi-hop  
**Root Causes**: YAML_BLIND, EMBEDDING_BLIND  
**Issues**: HPA parameters not in retrieved chunks, only API Gateway resources present

---

### tmp_003: Workflow Execution Timeout Sequence
**Query**: What's the sequence of events when a workflow execution times out?

**Expected Answer**: Workflow runs up to 3600s. If exceeded, automatically terminated. Error: 'exceeded maximum execution time of 3600 seconds'. Status: TIMEOUT. Can request custom timeout up to 7200s on Enterprise.

**Score**: 9/10  
**Difficulty**: Medium  
**Type**: Temporal  
**Root Causes**: EMBEDDING_BLIND  
**Issues**: Missing exact error message format and TIMEOUT status

---

### tmp_004: Workflow Definition Cache Propagation Timeline
**Query**: How long does it take for workflow definition cache changes to propagate?

**Expected Answer**: Workflow definitions cached in Redis with TTL of 1 hour. Cache invalidated on workflow update or manual flush. Cache hit rate is 94.2%.

**Score**: 7/10  
**Difficulty**: Hard  
**Type**: Temporal  
**Root Causes**: EMBEDDING_BLIND  
**Issues**: Cache hit rate (94.2%) not documented, TTL present but incomplete

---

### tmp_005: Database Primary Failover Timeline
**Query**: What's the timeline for automatic failover when the database primary fails?

**Expected Answer**: Database primary failure: 30-60 seconds for automatic promotion of replica. Redis failover: <10 seconds. Kafka controller election: <30 seconds.

**Score**: 8/10  
**Difficulty**: Medium  
**Type**: Temporal  
**Root Causes**: EMBEDDING_BLIND  
**Issues**: Database failover present, but Redis and Kafka failover times missing

---

### cmp_001: PgBouncer vs Direct PostgreSQL Connections
**Query**: What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?

**Expected Answer**: PgBouncer: pool_mode=transaction, default_pool_size=25, max_db_connections=100. Allows 1000 client connections with only 100 actual DB connections. Direct: limited to max_connections=100.

**Score**: 5/10  
**Difficulty**: Very Hard  
**Type**: Comparative  
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND  
**Issues**: PgBouncer configuration incomplete, comparative analysis missing

---

### cmp_002: Backoff Strategies Comparison
**Query**: How do fixed, linear, and exponential backoff strategies differ for retries?

**Expected Answer**: Fixed: same wait time (1s, 1s, 1s). Linear: increase by fixed amount (1s, 2s, 3s). Exponential: double each time (1s, 2s, 4s). Exponential is recommended.

**Score**: 8/10  
**Difficulty**: Medium  
**Type**: Comparative  
**Root Causes**: EMBEDDING_BLIND  
**Issues**: All three strategies mentioned, but explicit comparison sequences not fully present

---

### cmp_003: /health vs /ready Endpoints
**Query**: What's the difference between /health and /ready endpoints?

**Expected Answer**: /health: liveness check, returns basic status. /ready: readiness check, checks dependencies like database and redis connectivity.

**Score**: 2/10  
**Difficulty**: Very Hard  
**Type**: Comparative  
**Root Causes**: EMBEDDING_BLIND  
**Issues**: Endpoints exist in YAML but explanation of difference not documented

---

### neg_001: What NOT to Do When Rate Limited
**Query**: What should I NOT do when I'm rate limited?

**Expected Answer**: Don't keep hammering the API. Instead: check Retry-After header, implement exponential backoff, monitor X-RateLimit-Remaining, cache responses, consider upgrading tier.

**Score**: 7/10  
**Difficulty**: Hard  
**Type**: Negation  
**Root Causes**: NEGATION_BLIND  
**Issues**: Positive guidance present ("do this") but negation framing weak ("don't do that")

---

### neg_002: Why HS256 Doesn't Work for JWT
**Query**: Why doesn't HS256 work for JWT token validation in CloudFlow?

**Expected Answer**: CloudFlow uses RS256 (asymmetric) not HS256 (symmetric). RS256 requires private key for signing, public key for validation. HS256 would fail with algorithm mismatch error.

**Score**: 9/10  
**Difficulty**: Medium  
**Type**: Negation  
**Root Causes**: NEGATION_BLIND  
**Issues**: RS256 usage clear, but explicit "HS256 doesn't work" statement missing

---

### neg_003: Why Can't Schedule More Frequently Than Every Minute
**Query**: Why can't I schedule workflows more frequently than every minute?

**Expected Answer**: Minimum scheduling interval is 1 minute. Expressions evaluating to more frequent executions will be rejected. For near real-time, use webhook or event-based triggers instead.

**Score**: 6/10  
**Difficulty**: Hard  
**Type**: Negation  
**Root Causes**: NEGATION_BLIND  
**Issues**: Scheduling info present but minimum interval constraint not explicit

---

### neg_004: No Token Refresh Consequences
**Query**: What happens if I don't implement token refresh logic?

**Expected Answer**: Tokens expire after 3600 seconds (1 hour). Without refresh logic, authentication will fail after expiry. Need to implement refresh using refresh token (valid 7-30 days).

**Score**: 7/10  
**Difficulty**: Hard  
**Type**: Negation  
**Root Causes**: NEGATION_BLIND  
**Issues**: Expiration time present, but consequences of not refreshing not explicit

---

### neg_005: Why Not Hardcode API Keys
**Query**: Why shouldn't I hardcode API keys in workflow definitions?

**Expected Answer**: Security risk - keys could be exposed. Use secrets instead: {{secrets.API_TOKEN}}. Secrets are encrypted at rest. Store in Settings > Secrets.

**Score**: 7/10  
**Difficulty**: Hard  
**Type**: Negation  
**Root Causes**: NEGATION_BLIND  
**Issues**: Security warning present, but workflow-specific syntax and UI location missing

---

### imp_001: Long-Running Data Processing Best Practices
**Query**: Best practice for handling long-running data processing that might exceed time limits

**Expected Answer**: Workflow timeout is 3600s. Solutions: split into smaller workflows, enable checkpointing (every 300s), use parallel workers, request custom timeout (up to 7200s on Enterprise).

**Score**: 8/10  
**Difficulty**: Medium  
**Type**: Implicit  
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND  
**Issues**: Timeout and Enterprise limits present, but checkpointing and splitting strategies incomplete

---

### imp_003: Debug Slow API Calls
**Query**: How to debug why my API calls are slow

**Expected Answer**: Check latency breakdown: Auth (18%), DB Query (64%), Business Logic (13%), Serialization (5%). Use cloudflow metrics latency-report. Check slow query log. Review connection pool status.

**Score**: 6/10  
**Difficulty**: Hard  
**Type**: Implicit  
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND  
**Issues**: Latency targets and slow query commands present, but specific percentage breakdown missing

---

## 90% Needle-Haystack Test

**Source**: `corpus/needle_questions_baseline.json`  
**Test Date**: 2026-01-26  
**Strategy**: enriched_hybrid_llm  
**Chunking**: MarkdownSemanticStrategy  
**Average Score**: 9.0/10 (90%)
**Pass Rate (≥8)**: 18/20 (90%)

*Note: This section contains 20 baseline needle-in-haystack questions testing retrieval of specific facts embedded in large documents. These questions establish a baseline for comparison with adversarial variants.*

---

## 65% Adversarial Needle-Haystack Test

**Source**: `corpus/needle_questions_adversarial.json`  
**Test Date**: 2026-01-26  
**Strategy**: enriched_hybrid_llm  
**Chunking**: MarkdownSemanticStrategy  
**Average Score**: 7.55/10 (65%)
**Pass Rate (≥7)**: 13/20 (65%)

### Test Overview

The adversarial needle-haystack test uses 20 challenging questions designed to expose weaknesses in semantic retrieval:

- **VERSION questions (5)**: Target version numbers in document metadata (frontmatter YAML, feature-state tags)
- **COMPARISON questions (5)**: Require comparing two concepts or policies
- **NEGATION questions (5)**: Use negative framing ("what's wrong with", "why can't")
- **VOCABULARY questions (5)**: Use synonyms or alternative phrasings for technical terms

### Category Breakdown

| Category | Passed | Total | Pass Rate | Avg Score | Key Finding |
|----------|--------|-------|-----------|-----------|------------|
| VERSION | 2 | 5 | 40% | 5.6 | Frontmatter metadata is adversarial - semantic embeddings miss YAML version numbers |
| COMPARISON | 5 | 5 | 100% | 9.8 | Semantic retrieval excels at finding and comparing related concepts |
| NEGATION | 3 | 5 | 60% | 7.6 | Mixed results - negation framing sometimes misleads retrieval |
| VOCABULARY | 3 | 5 | 60% | 7.2 | Synonym matching partially works - some vocabulary mismatches unresolved |

### Detailed Questions

#### VERSION Questions (5)

---

##### adv_v01: What's the minimum kubernetes version requirement for topology manager?
**Expected Answer**: v1.18

**Score**: 3/10  
**Difficulty**: Medium  
**Type**: Fact-lookup  
**Root Cause**: EMBEDDING_BLIND - Version requirement stored in YAML frontmatter, not in prose content

---

##### adv_v02: Which Kubernetes release made Topology Manager GA/stable?
**Expected Answer**: v1.27

**Score**: 3/10  
**Difficulty**: Hard  
**Type**: Fact-lookup  
**Root Cause**: EMBEDDING_BLIND - GA version in frontmatter feature-state shortcode, not in main content

---

##### adv_v03: When did the prefer-closest-numa-nodes option become generally available?
**Expected Answer**: Kubernetes 1.32

**Score**: 2/10  
**Difficulty**: Hard  
**Type**: Fact-lookup  
**Root Cause**: VOCABULARY_MISMATCH - "generally available" vs "GA" phrasing caused retrieval to favor unrelated API deprecation documents

---

##### adv_v04: In what k8s version did max-allowable-numa-nodes become GA?
**Expected Answer**: Kubernetes 1.35

**Score**: 10/10  
**Difficulty**: Hard  
**Type**: Fact-lookup  
**Retrieved**: "The `max-allowable-numa-nodes` option is GA since Kubernetes 1.35"

---

##### adv_v05: What's the default limit on NUMA nodes before kubelet refuses to start with topology manager?
**Expected Answer**: 8

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Fact-lookup  
**Retrieved**: "The maximum number of NUMA nodes that Topology Manager allows is 8"

---

#### COMPARISON Questions (5)

---

##### adv_c01: How does restricted policy differ from single-numa-node when pod can't get preferred affinity?
**Expected Answer**: restricted rejects any non-preferred; single-numa-node only rejects if >1 NUMA needed

**Score**: 10/10  
**Difficulty**: Hard  
**Type**: Conceptual  
**Retrieved**: Both policies clearly distinguished in retrieved chunks

---

##### adv_c02: What's the key difference between container scope and pod scope for topology alignment?
**Expected Answer**: container=individual alignment per container, no grouping; pod=groups all containers to common NUMA set

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Conceptual  
**Retrieved**: "container: there is no notion of grouping the containers to a specific set of NUMA nodes" vs "pod: grouping all containers in a pod to a common set of NUMA nodes"

---

##### adv_c03: Compare what happens with none policy vs best-effort policy when NUMA affinity can't be satisfied
**Expected Answer**: none=no alignment attempted; best-effort=stores non-preferred hint, admits pod anyway

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Conceptual  
**Retrieved**: Both behaviors explicitly stated in same chunk

---

##### adv_c04: How does topology manager behavior differ for Guaranteed QoS pods with integer CPU vs fractional CPU?
**Expected Answer**: integer CPU gets topology hints from CPU Manager; fractional CPU gets default hint only

**Score**: 9/10  
**Difficulty**: Hard  
**Type**: Conceptual  
**Retrieved**: Examples of both present but distinction requires inference

---

##### adv_c05: What's the difference between TopologyManagerPolicyBetaOptions and TopologyManagerPolicyAlphaOptions feature gates?
**Expected Answer**: Beta=enabled by default, Alpha=disabled by default; both control policy option visibility

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Conceptual  
**Retrieved**: "TopologyManagerPolicyBetaOptions default enabled... TopologyManagerPolicyAlphaOptions default disabled"

---

#### NEGATION Questions (5)

---

##### adv_n01: Why is using more than 8 NUMA nodes not recommended with topology manager?
**Expected Answer**: State explosion when enumerating NUMA affinities; use max-allowable-numa-nodes at own risk

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Conceptual  
**Retrieved**: "there will be a state explosion when trying to enumerate the possible NUMA affinities... is **not** recommended and is at your own risk"

---

##### adv_n02: What happens to a pod that fails topology affinity check with restricted policy? Can it be rescheduled?
**Expected Answer**: Pod enters Terminated state; scheduler will NOT reschedule; need ReplicaSet/Deployment

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Conceptual  
**Retrieved**: All three parts present verbatim in retrieved chunks

---

##### adv_n03: Why can't the Kubernetes scheduler prevent pods from failing on nodes due to topology?
**Expected Answer**: Scheduler is not topology-aware; this is a known limitation

**Score**: 6/10  
**Difficulty**: Hard  
**Type**: Conceptual  
**Root Cause**: CHUNKING_ISSUE - Relevant chunk exists but wasn't ranked high enough

---

##### adv_n04: What's wrong with using container scope for latency-sensitive applications?
**Expected Answer**: Containers may end up on different NUMA nodes since there's no grouping

**Score**: 2/10  
**Difficulty**: Hard  
**Type**: Conceptual  
**Root Cause**: VOCABULARY_MISMATCH - Negative framing didn't match positive framing in document

---

##### adv_n05: When does single-numa-node policy reject a pod that would be admitted by restricted?
**Expected Answer**: When pod needs resources from exactly 2+ NUMA nodes; restricted accepts any preferred, single-numa-node requires exactly 1

**Score**: 10/10  
**Difficulty**: Hard  
**Type**: Conceptual  
**Retrieved**: "a set containing more NUMA nodes - it results in pod rejection (because instead of one NUMA node, two or more NUMA nodes are required)"

---

#### VOCABULARY Questions (5)

---

##### adv_m01: How do I configure CPU placement policy in kubelet?
**Expected Answer**: --topology-manager-policy flag

**Score**: 10/10  
**Difficulty**: Medium  
**Type**: Fact-lookup  
**Retrieved**: "You can set a policy via a kubelet flag, `--topology-manager-policy`"

---

##### adv_m02: How do I enable NUMA awareness on Windows k8s nodes?
**Expected Answer**: Enable WindowsCPUAndMemoryAffinity feature gate

**Score**: 4/10  
**Difficulty**: Hard  
**Type**: Fact-lookup  
**Root Cause**: VOCABULARY_MISMATCH - "NUMA awareness on Windows" didn't match "Topology Manager support on Windows"

---

##### adv_m03: How does k8s coordinate resource co-location across multi-socket servers?
**Expected Answer**: Topology Manager acts as source of truth for CPU Manager and Device Manager

**Score**: 10/10  
**Difficulty**: Hard  
**Type**: Conceptual  
**Retrieved**: "The Topology Manager is a kubelet component, which acts as a source of truth so that other kubelet components can make topology aligned resource allocation choices"

---

##### adv_m04: What kubelet setting controls the granularity of resource alignment?
**Expected Answer**: topologyManagerScope (container or pod)

**Score**: 10/10  
**Difficulty**: Hard  
**Type**: Fact-lookup  
**Retrieved**: "The `scope` defines the granularity at which you would like resource alignment to be performed... setting the `topologyManagerScope` in the kubelet configuration file"

---

##### adv_m05: How do I optimize inter-process communication latency for pods?
**Expected Answer**: Use pod scope with single-numa-node policy to eliminate inter-NUMA overhead

**Score**: 2/10  
**Difficulty**: Hard  
**Type**: Conceptual  
**Root Cause**: VOCABULARY_MISMATCH - "inter-process communication latency" didn't match "applications that perform IPC"

---

### Adversarial Test Summary Statistics

#### Score Distribution

| Score | Count | Percentage |
|-------|-------|-----------|
| 10/10 | 11 | 55% |
| 9/10 | 1 | 5% |
| 6/10 | 1 | 5% |
| 4/10 | 1 | 5% |
| 3/10 | 2 | 10% |
| 2/10 | 4 | 20% |

#### Failure Analysis

| Failure Type | Count | Percentage | Questions |
|--------------|-------|-----------|-----------|
| EMBEDDING_BLIND | 2 | 10% | adv_v01, adv_v02 |
| VOCABULARY_MISMATCH | 4 | 20% | adv_v03, adv_n04, adv_m02, adv_m05 |
| CHUNKING_ISSUE | 1 | 5% | adv_n03 |

#### Key Findings

**Strengths**:
- COMPARISON questions excel (100% pass rate, 9.8 avg) - semantic retrieval is excellent at finding and comparing related concepts
- Policy explanations are well-retrieved - questions about policy behaviors consistently find relevant chunks
- Vocabulary mismatches often resolved - "CPU placement policy" → "topology manager policy", "resource co-location" → "topology aligned resource allocation"

**Weaknesses**:
- VERSION questions struggle (40% pass rate, 5.6 avg) - frontmatter metadata (YAML version numbers, feature-state shortcodes) is not captured by semantic embeddings
- Negation framing can fail - "What's wrong with X" queries sometimes miss content that explains "Y is better for this use case"
- Specialized vocabulary can fail - "IPC latency" didn't match "applications that perform IPC" despite semantic similarity

#### Comparison with Baseline Test

| Metric | Baseline (20 Q) | Adversarial (20 Q) | Delta |
|--------|-----------------|-------------------|-------|
| Average Score | 9.0/10 | 7.55/10 | -1.45 |
| Pass Rate (≥7) | 90% (18/20) | 65% (13/20) | -25% |
| Perfect Scores | 12 | 11 | -1 |
| Complete Failures (≤2) | 1 | 4 | +3 |

---

## Passing Queries (Not Tested in Current Run)

**Source**: `corpus/edge_case_queries.json` (passing queries)  
**Note**: These queries scored ≥8 in baseline testing and were not included in the 68.7% edge case test

### mh_001: JWT Expiration Consistency
**Query**: Compare JWT expiration in Auth Service vs the API documentation - are they consistent?

**Expected Answer**: Auth Service: 15-minute access token expiry. API docs: 3600 seconds (1 hour) max. These are different contexts - Auth Service internal tokens vs API JWT claims.

**Baseline Score**: 8/10  
**Type**: Multi-hop

---

### mh_003: Workflow Timeout vs Retry Backoff
**Query**: What's the relationship between workflow timeout (3600s) and the retry backoff strategy?

**Expected Answer**: Workflow timeout is 3600s max. Retry uses exponential backoff: 1s, 2s, 4s (max 3 retries). Total retry time ~7s, well within timeout. But long-running steps can still timeout.

**Baseline Score**: 9/10  
**Type**: Multi-hop

---

### mh_005: Scheduled Workflows During DR Failover
**Query**: What happens to scheduled workflows during a disaster recovery failover?

**Expected Answer**: RPO is 1 hour, RTO is 4 hours. Scheduler uses leader election with Redis. During failover, scheduled executions may be skipped (logged in audit trail). Kafka retention allows event replay.

**Baseline Score**: 8/10  
**Type**: Multi-hop

---

### tmp_001: DR Test Changes
**Query**: What changed between the last DR test and the current DR objectives?

**Expected Answer**: Last DR test (Dec 15, 2025): Actual RTO was 2h 23m (target 4h), RPO was 42m (target 1h). Both met objectives. DNS propagation issue was resolved.

**Baseline Score**: 9/10  
**Type**: Temporal

---

### tmp_002: Rotation Schedules
**Query**: How often should API keys be rotated and what's the certificate rotation schedule?

**Expected Answer**: API keys: rotate every 90 days. Certificates: rotate every 90 days (automated via cert-manager). Secrets in Vault also rotated every 90 days.

**Baseline Score**: 10/10  
**Type**: Temporal

---

### cmp_004: Authenticated vs Unauthenticated Rate Limits
**Query**: Compare the rate limits for authenticated vs unauthenticated API requests

**Expected Answer**: Authenticated: 100 requests/minute per user. Unauthenticated: 20 requests/minute. Burst allowance: 150 requests in 10-second window. Enterprise: 1000 req/min.

**Baseline Score**: 8/10  
**Type**: Comparative

---

### cmp_005: SEV-1 vs SEV-2 Incidents
**Query**: How do SEV-1 and SEV-2 incidents differ in response time and escalation?

**Expected Answer**: SEV-1: Immediate response (<15 min), page on-call immediately, complete outage. SEV-2: <1 hour response, create ticket and notify on-call, major functionality impaired.

**Baseline Score**: 9/10  
**Type**: Comparative

---

### imp_002: Survive Region Failure
**Query**: How to ensure my application survives a complete region failure

**Expected Answer**: Multi-AZ deployment across 3 AZs. Cross-region replication to us-west-2 (15-min lag). Manual failover procedure: update DNS, promote DR replica, scale up DR services. RTO: 2-4 hours.

**Baseline Score**: 8/10  
**Type**: Implicit

---

### imp_004: Production Monitoring Setup
**Query**: What monitoring should I set up for production workflows?

**Expected Answer**: Prometheus for metrics, Grafana for dashboards, Jaeger for distributed tracing. Key metrics: request rate, error rate, latency percentiles, cache hit ratios, Kafka consumer lag. Alerts via PagerDuty.

**Baseline Score**: 8/10  
**Type**: Implicit

---

## Summary Statistics

### By Test Set

| Test Set | Questions | Avg Score | Pass Rate (≥7) | Difficulty |
|----------|-----------|-----------|----------------|------------|
| **94% Baseline** | 10 | 9.4/10 (94%) | 90% (9/10) | 30% easy, 50% medium, 20% hard |
| **68.7% Edge Cases** | 15 | 6.87/10 (68.7%) | 33.3% (5/15) | 0% easy, 33% medium, 67% hard |
| **90% Needle-Haystack** | 20 | 9.0/10 (90%) | 90% (18/20) | Baseline needle-in-haystack |
| **65% Adversarial Needle-Haystack** | 20 | 7.55/10 (65%) | 65% (13/20) | Adversarial needle-in-haystack |
| **Passing Queries** | 9 | 8.56/10 (85.6%) | 100% (9/9) | Not tested in current run |
| **TOTAL** | 74 | 8.2/10 (82%) | 75.7% (56/74) | Mixed |

---

### By Question Type

| Type | Count | Avg Score (68.7% test) | Avg Score (94% test) |
|------|-------|------------------------|----------------------|
| **Factual Lookup** | 5 (94% test only) | N/A | 10.0/10 |
| **Procedural** | 5 (94% test only) | N/A | 8.8/10 |
| **Multi-hop** | 2 | 6.5/10 | N/A |
| **Temporal** | 3 | 8.0/10 | N/A |
| **Comparative** | 3 | 5.0/10 | N/A |
| **Negation** | 5 | 7.2/10 | N/A |
| **Implicit** | 2 | 7.0/10 | N/A |

---

### By Difficulty

| Difficulty | Count | Avg Score | Pass Rate |
|------------|-------|-----------|-----------|
| **Easy** | 3 | 10.0/10 | 100% |
| **Medium** | 13 | 8.5/10 | 69.2% |
| **Hard** | 16 | 6.6/10 | 31.3% |
| **Very Hard** | 2 | 3.5/10 | 0% |

---

### Root Causes (68.7% Edge Case Test)

| Root Cause | Frequency | Impact |
|------------|-----------|--------|
| **EMBEDDING_BLIND** | 13/15 (87%) | Semantic search fails to retrieve relevant chunks |
| **VOCABULARY_MISMATCH** | 6/15 (40%) | Query terms don't match document terms |
| **NEGATION_BLIND** | 5/15 (33%) | Negation framing weak in documentation |
| **YAML_BLIND** | 1/15 (7%) | YAML configuration not indexed well |

---

## Usage Notes

### For Testing New Strategies

1. **Start with 94% Baseline Questions** - These validate that the system works for typical queries
2. **Then test 68.7% Edge Cases** - These reveal weaknesses and corpus gaps
3. **Compare results** - Both tests should show improvement, but edge cases will always score lower

### For Corpus Improvement

**High Priority** (affects multiple hard queries):
- Add PgBouncer configuration details (affects mh_002, cmp_001)
- Add /health vs /ready explanation (affects cmp_003)
- Add HPA parameters (affects mh_004)
- Add cache hit rate metrics (affects tmp_004)

**Medium Priority**:
- Add comparative analysis sections
- Improve negation framing
- Add latency breakdown percentages
- Expand technical acronyms

### For Benchmarking

**Typical User Performance** (80% of queries):
- Use 94% Baseline Questions
- Expected: 90-95% accuracy

**Power User / Edge Case Performance** (20% of queries):
- Use 68.7% Edge Case Questions
- Expected: 65-70% accuracy

**Overall System Performance**:
- Use all 34 questions
- Expected: 80-85% accuracy (weighted average)

---

**File Version**: 1.1  
**Last Updated**: 2026-01-26  
**Total Questions**: 74 (10 baseline + 15 edge cases + 20 needle-haystack + 20 adversarial + 9 passing)
