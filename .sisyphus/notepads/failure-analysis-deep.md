# Deep Failure Analysis - enriched_hybrid_llm Pipeline

Analyzed: 2026-01-25
Total Failures: 16 (score ≤7)
Analysis Method: Stage-by-stage pipeline breakdown

## Pipeline Stages Reference

1. **Query Rewriting**: Claude Haiku rewrites query for better retrieval
2. **Domain Expansion**: Hardcoded keyword dictionary (DOMAIN_EXPANSIONS)
3. **Semantic Search**: BGE-base embeddings encode query and chunks
4. **BM25 Search**: Sparse keyword matching on enriched content
5. **RRF Fusion**: Combine semantic + BM25 with weighted scores
6. **Top-K Selection**: Return top 5 chunks

---

## Failure 1: mh_002 - Connection Pool Exhaustion / PgBouncer

**Query**: "If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?"

**Score**: 5/10

**Retrieved Chunks**:
1. [deployment_guide] - Pods in CrashLoopBackOff, High Memory Usage (tangential)
2. [architecture_overview] - PostgreSQL Primary Database with 1 primary + 2 read replicas
3. [deployment_guide] - postgres-values.yaml with max_connections = 100
4. [architecture_overview] - Multi-AZ Deployment showing PostgreSQL Primary/Replica
5. [architecture_overview] - Monitoring & Alerting with database connection pool saturation

**Correct Answer Location**:
- File: `deployment_guide.md`
- Section: `### PgBouncer Configuration` (lines 473-561)
- Content: `pool_mode = transaction`, `default_pool_size = 25`, `max_db_connections = 100`, `max_client_conn = 1000`
- Also: `troubleshooting_guide.md` lines 434-438 mentions PgBouncer for connection pooling

**Why Retrieval Failed**:
The PgBouncer configuration section in deployment_guide.md (lines 473-561) contains the detailed YAML configuration but:
1. The section header "### PgBouncer Configuration" doesn't contain "connection pool exhaustion"
2. The content is heavily YAML-formatted code blocks, not prose
3. BM25 would need exact keyword match on "PgBouncer" + "connection pool"
4. Query uses "connection pool exhaustion" but corpus uses "connection pooling"

**Pipeline Stage Analysis**:
- **Query Rewriting**: Likely preserved "PgBouncer" and "read replicas" - NEUTRAL
- **BM25 Stage**: FAILED - "connection pool exhaustion" ≠ "connection pooling". The word "exhaustion" doesn't appear near PgBouncer content
- **Semantic Stage**: PARTIAL - Semantic similarity between "connection pool exhaustion" and PgBouncer config is weak because config is YAML code, not descriptive prose
- **RRF Fusion**: Ranked tangential content higher because both BM25 and semantic missed the target

**Root Cause**: BM25_MISS + EMBEDDING_BLIND

**Evidence**: 
- "PgBouncer" appears 33 times in corpus but always in technical context (YAML, commands)
- "connection pool exhaustion" phrase doesn't exist in corpus
- Corpus uses "connection pooling" (noun) not "exhaustion" (problem state)
- The troubleshooting_guide.md section "Connection Pool Exhaustion" (line 380) exists but doesn't prominently mention PgBouncer as solution

---

## Failure 2: mh_004 - HPA Scaling Parameters

**Query**: "How do the HPA scaling parameters relate to the API Gateway resource requirements?"

**Score**: 6/10

**Retrieved Chunks**:
1. [deployment_guide] - Grafana dashboards (tangential)
2. [architecture_overview] - Resource Utilization: API Gateway 55% avg, 80% peak, 2.5GB/4GB
3. [architecture_overview] - Table of Contents (useless)
4. [architecture_overview] - API Gateway: 2 vCPU, 4GB RAM per pod, auto-scaling 8-20
5. [deployment_guide] - Pods in CrashLoopBackOff (tangential)

**Correct Answer Location**:
- File: `deployment_guide.md`
- Section: `### Horizontal Pod Autoscaling` (lines 804-857)
- Content: `minReplicas: 3`, `maxReplicas: 10`, `targetCPUUtilizationPercentage: 70`
- Also in values-production.yaml (lines 247-252): same HPA config

**Why Retrieval Failed**:
1. Query uses "HPA" acronym - DOMAIN_EXPANSIONS has "hpa" → "horizontal pod autoscaler HPA scaling replicas CPU utilization"
2. However, the HPA YAML section (lines 806-850) is mostly code, not prose
3. Retrieved API Gateway resources (chunk 4) but NOT the HPA config that controls scaling

**Pipeline Stage Analysis**:
- **Query Rewriting**: Should expand "HPA" - CHECK if expansion triggered
- **Domain Expansion**: "hpa" IS in DOMAIN_EXPANSIONS - should have helped
- **BM25 Stage**: PARTIAL - "HPA" keyword should match, but YAML code blocks may not tokenize well
- **Semantic Stage**: FAILED - Embeddings don't understand YAML structure; "minReplicas: 3" doesn't semantically relate to "scaling parameters"
- **RRF Fusion**: API Gateway resources ranked higher than HPA config

**Root Cause**: EMBEDDING_BLIND + YAML_CODE_BLOCK

**Evidence**:
- HPA section is 90% YAML code (lines 806-850)
- Semantic embeddings trained on prose, not Kubernetes YAML
- The prose description "CloudFlow is configured with Horizontal Pod Autoscaler (HPA)" (line 804) is only 1 line before 44 lines of YAML

---

## Failure 3: tmp_003 - Workflow Timeout Sequence

**Query**: "What's the sequence of events when a workflow execution times out?"

**Score**: 7/10

**Retrieved Chunks**:
1. [user_guide] - Execution Timeout: 3600 seconds, automatically terminated, Enterprise custom limits
2. [troubleshooting_guide] - Retry Logic and Exponential Backoff (tangential)
3. [user_guide] - Retry Policies (tangential)
4. [troubleshooting_guide] - RBAC Policy Violations (irrelevant)
5. [troubleshooting_guide] - Error handling config (tangential)

**Correct Answer Location**:
- File: `troubleshooting_guide.md`
- Section: `### Timeout Errors (3600 second limit)` (lines 518-587)
- Content: Error message "Workflow exceeded maximum execution time of 3600 seconds", Status: TIMEOUT, analysis commands
- Also: `user_guide.md` line 639-641 has timeout info

**Why Retrieval Failed**:
1. Retrieved user_guide timeout info (chunk 1) - GOOD
2. MISSED the troubleshooting_guide section with the exact error message and TIMEOUT status
3. Query asks for "sequence of events" but corpus describes it as error handling, not a sequence

**Pipeline Stage Analysis**:
- **Query Rewriting**: "sequence of events" may have been rewritten - could help or hurt
- **BM25 Stage**: PARTIAL - "timeout" matches but "sequence of events" doesn't appear in corpus
- **Semantic Stage**: PARTIAL - "sequence of events" is a temporal/procedural query; corpus uses declarative descriptions
- **RRF Fusion**: user_guide ranked higher than troubleshooting_guide

**Root Cause**: VOCABULARY_MISMATCH + QUERY_FRAMING

**Evidence**:
- Query: "sequence of events" - procedural framing
- Corpus: "Workflow exceeded maximum execution time" - error message framing
- The troubleshooting section (lines 518-587) has the detailed info but uses different vocabulary

---

## Failure 4: tmp_004 - Workflow Definition Cache TTL

**Query**: "How long does it take for workflow definition cache changes to propagate?"

**Score**: 4/10

**Retrieved Chunks**:
1. [architecture_overview] - Performance Characteristics: GET /workflows cache hit <50ms, miss <150ms
2. [architecture_overview] - Asynchronous Event Flow (irrelevant)
3. [troubleshooting_guide] - Slow query log (irrelevant)
4. [architecture_overview] - Monitoring & Alerting with cache hit/miss ratios
5. [user_guide] - Test Thoroughly (irrelevant)

**Correct Answer Location**:
- File: `architecture_overview.md`
- Section: `### Redis Caching Layer` → `4. **Workflow Definitions**:` (lines 489-493)
- Content: "Key pattern: `workflow:def:{workflow_id}`", "TTL: 1 hour", "Invalidation: On workflow update or manual flush"
- Also: Cache hit rate 94.2% (line 503)

**Why Retrieval Failed**:
1. Query asks about "cache changes to propagate" - this is about cache invalidation/TTL
2. Corpus has "TTL: 1 hour" and "Invalidation: On workflow update" but in a structured list format
3. Retrieved cache performance metrics but NOT the TTL configuration

**Pipeline Stage Analysis**:
- **Query Rewriting**: "propagate" may have been rewritten to "update" or "sync" - CHECK
- **BM25 Stage**: FAILED - "propagate" doesn't appear in corpus; "TTL" not in query
- **Semantic Stage**: FAILED - "cache changes propagate" semantically distant from "TTL: 1 hour"
- **RRF Fusion**: Performance metrics ranked higher than configuration

**Root Cause**: VOCABULARY_MISMATCH + EMBEDDING_BLIND

**Evidence**:
- Query uses "propagate" - corpus uses "invalidation", "TTL", "flush"
- The answer "TTL: 1 hour" is in a bullet-point list (line 491), not prose
- Semantic embeddings don't connect "propagate" with "TTL" or "invalidation"

---

## Failure 5: tmp_005 - Database Failover Timeline

**Query**: "What's the timeline for automatic failover when the database primary fails?"

**Score**: 7/10

**Retrieved Chunks**:
1. [architecture_overview] - DR Procedures: Database Primary Failure 30-60 seconds
2. [architecture_overview] - Recovery Objectives
3. [architecture_overview] - Scheduler Service with leader election
4. [architecture_overview] - Multi-AZ Deployment
5. [architecture_overview] - Architecture Diagram

**Correct Answer Location**:
- File: `architecture_overview.md`
- Section: `### Automatic Failover` (lines 970-974)
- Content: "Database: 30-60 seconds", "Redis: < 10 seconds", "Kafka: < 30 seconds"

**Why Retrieval Failed**:
1. Retrieved database failover (30-60s) - GOOD
2. MISSED Redis failover (<10s) and Kafka controller election (<30s)
3. All three are in the SAME section (lines 970-974) but chunking may have split them

**Pipeline Stage Analysis**:
- **Query Rewriting**: "database primary fails" is specific - should help
- **BM25 Stage**: GOOD - "database" + "failover" matched
- **Semantic Stage**: PARTIAL - Query about database, so Redis/Kafka not semantically connected
- **RRF Fusion**: Database-specific content ranked higher

**Root Cause**: FRAGMENTED + QUERY_SPECIFICITY

**Evidence**:
- Query specifically asks about "database primary" - retrieval correctly focused on database
- Redis and Kafka failover times are in same section but query didn't ask for them
- This is a partial success - got 1/3 of the answer because query was database-specific

---

## Failure 6: cmp_001 - PgBouncer vs Direct PostgreSQL

**Query**: "What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?"

**Score**: 2/10

**Retrieved Chunks**:
1. [deployment_guide] - Grafana dashboards (irrelevant)
2. [architecture_overview] - Architecture Diagram (irrelevant)
3. [deployment_guide] - Environment Configuration with DATABASE_URL
4. [troubleshooting_guide] - Slow query analysis (irrelevant)
5. [architecture_overview] - JWT Token Validation (irrelevant)

**Correct Answer Location**:
- File: `deployment_guide.md`
- Section: `### PgBouncer Configuration` (lines 473-561)
- Content: `pool_mode = transaction`, `default_pool_size = 25`, `max_db_connections = 100`, `max_client_conn = 1000`
- Also: `architecture_overview.md` line 447: "PgBouncer in transaction mode"

**Why Retrieval Failed**:
1. COMPLETE MISS - None of the retrieved chunks contain PgBouncer configuration
2. Query asks for "difference" - comparative query type
3. Corpus doesn't have explicit comparison between PgBouncer and direct connections

**Pipeline Stage Analysis**:
- **Query Rewriting**: "difference between" may have been preserved or expanded
- **BM25 Stage**: FAILED - "PgBouncer" should have matched but didn't rank high enough
- **Semantic Stage**: FAILED - "difference between X and Y" is a comparative pattern; embeddings don't handle comparisons well
- **RRF Fusion**: Completely wrong content ranked highest

**Root Cause**: BM25_MISS + COMPARATIVE_QUERY_FAILURE

**Evidence**:
- "PgBouncer" appears 33 times in corpus but retrieval returned 0 PgBouncer chunks
- Comparative queries ("difference between") require both concepts to be retrieved
- Corpus has PgBouncer config but NO explicit comparison with direct connections
- This is the worst failure (2/10) - complete retrieval miss

---

## Failure 7: cmp_002 - Backoff Strategies Comparison

**Query**: "How do fixed, linear, and exponential backoff strategies differ for retries?"

**Score**: 6/10

**Retrieved Chunks**:
1. [troubleshooting_guide] - Retry Logic: exponential backoff, 1s, 2s, 4s sequence
2. [user_guide] - Retry Policies: backoff_type "exponential" or "fixed", "linear"
3. [architecture_overview] - Cache Invalidation Strategies (irrelevant)
4. [troubleshooting_guide] - Rate limit retry logic
5. [user_guide] - Fallback Actions

**Correct Answer Location**:
- File: `user_guide.md`
- Section: `### Retry Policies` → `**Backoff Strategies**:` (lines 523-539)
- Content: Fixed (1s, 1s, 1s), Linear (1s, 2s, 3s), Exponential (1s, 2s, 4s) with explicit comparison

**Why Retrieval Failed**:
1. Retrieved mentions of backoff types but NOT the explicit comparison section
2. user_guide.md lines 523-539 has the exact comparison but wasn't in top 5
3. Retrieved chunk 2 mentions backoff_type options but not the detailed comparison

**Pipeline Stage Analysis**:
- **Query Rewriting**: "fixed, linear, exponential" should be preserved
- **BM25 Stage**: PARTIAL - Keywords matched but comparison section not ranked high
- **Semantic Stage**: PARTIAL - "differ" is comparative; embeddings may not capture comparison intent
- **RRF Fusion**: Got related content but not the best section

**Root Cause**: RANKING_ERROR + COMPARATIVE_QUERY

**Evidence**:
- The exact comparison exists at user_guide.md lines 525-539
- Retrieved tangential retry content instead of the comparison section
- Comparative queries need special handling - "differ" should boost comparison sections

---

## Failure 8: cmp_003 - /health vs /ready Endpoints

**Query**: "What's the difference between /health and /ready endpoints?"

**Score**: 5/10

**Retrieved Chunks**:
1. [architecture_overview] - Multi-AZ Deployment (irrelevant)
2. [troubleshooting_guide] - SEV-3/SEV-4 definitions (irrelevant)
3. [troubleshooting_guide] - (truncated)
4. [deployment_guide] - (not visible)
5. [architecture_overview] - (not visible)

**Correct Answer Location**:
- File: `deployment_guide.md`
- Section: `healthCheck:` in values-production.yaml (lines 277-290)
- Content: `livenessProbe: path: /health`, `readinessProbe: path: /ready`
- Also: Test commands at lines 385-390 show expected responses

**Why Retrieval Failed**:
1. Query asks for "difference" between two endpoints
2. Corpus has the endpoints defined but NO explicit explanation of what each checks
3. deployment_guide.md shows paths but not the semantic difference (liveness vs readiness)

**Pipeline Stage Analysis**:
- **Query Rewriting**: "/health" and "/ready" are URL paths - may not rewrite well
- **BM25 Stage**: PARTIAL - "/health" and "/ready" should match but are in YAML
- **Semantic Stage**: FAILED - URL paths don't have strong semantic meaning
- **RRF Fusion**: Irrelevant content ranked higher

**Root Cause**: CORPUS_GAP + EMBEDDING_BLIND

**Evidence**:
- Corpus has `/health` and `/ready` paths but NO explanation of what they check
- The expected responses (lines 386, 390) show structure but not purpose
- Query asks for conceptual difference; corpus only has configuration

---

## Failure 9: neg_001 - What NOT to Do When Rate Limited

**Query**: "What should I NOT do when I'm rate limited?"

**Score**: 6/10

**Retrieved Chunks**:
1. [troubleshooting_guide] - Rate limit handling code
2. [troubleshooting_guide] - Rate limit status
3. [user_guide] - (tangential)
4. [user_guide] - (tangential)
5. [api_reference] - Rate Limiting

**Correct Answer Location**:
- File: `api_reference.md`
- Section: `### Rate Limit Headers` → `**Best Practices**:` (lines 137-140)
- Content: "Monitor X-RateLimit-Remaining", "Implement exponential backoff", "Cache responses"
- Also: troubleshooting_guide.md has rate limit handling code

**Why Retrieval Failed**:
1. Query uses NEGATION framing: "what NOT to do"
2. Corpus describes what TO do, not what NOT to do
3. Retrieved rate limit content but framing doesn't match

**Pipeline Stage Analysis**:
- **Query Rewriting**: "NOT do" may have been rewritten to positive form - PROBLEMATIC
- **BM25 Stage**: PARTIAL - "rate limited" matched but "NOT" is a stop word
- **Semantic Stage**: FAILED - Negation semantics not captured; "NOT do X" ≈ "do X" in embedding space
- **RRF Fusion**: Retrieved positive guidance, not negative warnings

**Root Cause**: NEGATION_BLIND

**Evidence**:
- Query: "what should I NOT do" - negation framing
- Corpus: "Best Practices" - positive framing
- Embeddings don't distinguish "do X" from "don't do X"
- The answer is implicitly "don't keep hammering the API" but corpus says "implement backoff"

---

## Failure 10: neg_002 - Why HS256 Doesn't Work

**Query**: "Why doesn't HS256 work for JWT token validation in CloudFlow?"

**Score**: 7/10

**Retrieved Chunks**:
1. [api_reference] - JWT Tokens with RS256 signing algorithm
2. [troubleshooting_guide] - (tangential)
3. [architecture_overview] - JWT Token Validation: RS256
4. [architecture_overview] - Auth Service: RS256 algorithm
5. [architecture_overview] - (tangential)

**Correct Answer Location**:
- File: `api_reference.md`
- Section: `### JWT Tokens` (lines 59-101)
- Content: "RS256 signing algorithm" - implies HS256 doesn't work
- Also: architecture_overview.md line 124, 777 mention RS256

**Why Retrieval Failed**:
1. Query asks WHY HS256 doesn't work (negation + explanation)
2. Corpus only states RS256 is used, not WHY HS256 is excluded
3. Retrieved RS256 mentions but no explicit HS256 rejection reason

**Pipeline Stage Analysis**:
- **Query Rewriting**: "doesn't work" may have been rewritten
- **BM25 Stage**: PARTIAL - "HS256" doesn't appear in corpus at all
- **Semantic Stage**: PARTIAL - "HS256" and "RS256" are semantically related (both JWT algorithms)
- **RRF Fusion**: RS256 content ranked high but doesn't answer "why not HS256"

**Root Cause**: CORPUS_GAP + NEGATION_BLIND

**Evidence**:
- "HS256" doesn't appear anywhere in corpus (grep confirms)
- Corpus only states what IS used (RS256), not what ISN'T supported
- User can infer "RS256 only" but the "why" is not explained

---

## Failure 11: neg_003 - Scheduling Frequency Limit

**Query**: "Why can't I schedule workflows more frequently than every minute?"

**Score**: 7/10

**Retrieved Chunks**:
1. [troubleshooting_guide] - (tangential)
2. [user_guide] - Scheduling section
3. [api_reference] - (tangential)
4. [troubleshooting_guide] - (tangential)
5. [troubleshooting_guide] - (tangential)

**Correct Answer Location**:
- File: `user_guide.md`
- Section: `## Scheduling` (line 445)
- Content: "The minimum scheduling interval is **1 minute**. Expressions that evaluate to more frequent executions will be rejected."

**Why Retrieval Failed**:
1. Query uses negation: "can't schedule more frequently"
2. Corpus states the limit positively: "minimum interval is 1 minute"
3. Retrieved scheduling section but may not have the specific limit line

**Pipeline Stage Analysis**:
- **Query Rewriting**: "can't schedule" may have been rewritten to "schedule"
- **BM25 Stage**: PARTIAL - "schedule" + "minute" should match
- **Semantic Stage**: PARTIAL - Negation framing affects semantic matching
- **RRF Fusion**: Got scheduling content but specific limit may be in wrong chunk

**Root Cause**: NEGATION_BLIND + CHUNKING_BOUNDARY

**Evidence**:
- The answer is ONE line (line 445) in user_guide.md
- If chunking split this line from the scheduling section header, it may not rank high
- Query negation "can't" doesn't match corpus positive "minimum interval"

---

## Failure 12: neg_004 - Token Refresh Consequences

**Query**: "What happens if I don't implement token refresh logic?"

**Score**: 6/10

**Retrieved Chunks**:
1. [architecture_overview] - Auth Service token expiry
2. [troubleshooting_guide] - Token expiration troubleshooting
3. [architecture_overview] - (tangential)
4. [api_reference] - Token Expiration
5. [architecture_overview] - (tangential)

**Correct Answer Location**:
- File: `api_reference.md`
- Section: `### JWT Tokens` (line 103)
- Content: "All tokens expire after 3600 seconds (1 hour). Implement token refresh logic in your application."
- Also: troubleshooting_guide.md line 48, architecture_overview.md lines 142-143

**Why Retrieval Failed**:
1. Query asks about consequence of NOT implementing something
2. Corpus describes what to do, not consequences of not doing it
3. Retrieved token expiration info but not the consequence framing

**Pipeline Stage Analysis**:
- **Query Rewriting**: "don't implement" may have been rewritten to "implement"
- **BM25 Stage**: PARTIAL - "token refresh" matched
- **Semantic Stage**: FAILED - Conditional/consequence queries not well captured
- **RRF Fusion**: Got related content but not consequence-focused

**Root Cause**: NEGATION_BLIND + CONSEQUENCE_QUERY

**Evidence**:
- Query: "What happens if I don't..." - consequence of inaction
- Corpus: "Implement token refresh logic" - instruction
- The consequence (authentication fails) is not explicitly stated

---

## Failure 13: neg_005 - Why Not Hardcode API Keys

**Query**: "Why shouldn't I hardcode API keys in workflow definitions?"

**Score**: 5/10

**Retrieved Chunks**:
1. [api_reference] - API Keys security notes
2. [architecture_overview] - (tangential)
3. [api_reference] - (tangential)
4. [deployment_guide] - (tangential)
5. [user_guide] - Best Practices

**Correct Answer Location**:
- File: `user_guide.md`
- Section: `### 3. Use Secrets for Sensitive Data` (lines 723-737)
- Content: "Never hardcode API keys, passwords, or tokens in workflows", "Store secrets in **Settings** > **Secrets** with encryption at rest"

**Why Retrieval Failed**:
1. Query uses negation: "shouldn't hardcode"
2. Corpus has "Never hardcode" - similar but different phrasing
3. Retrieved API Keys section but not the Best Practices section with the answer

**Pipeline Stage Analysis**:
- **Query Rewriting**: "shouldn't hardcode" may have been rewritten
- **BM25 Stage**: PARTIAL - "hardcode" should match "Never hardcode"
- **Semantic Stage**: PARTIAL - Negation affects matching
- **RRF Fusion**: API Keys section ranked higher than Best Practices

**Root Cause**: RANKING_ERROR + NEGATION_BLIND

**Evidence**:
- user_guide.md line 723: "Never hardcode API keys" - exact answer
- api_reference.md line 27: "Never expose API keys in client-side code" - related but different
- Best Practices section (lines 685-837) has the complete answer but wasn't top-ranked

---

## Failure 14: imp_001 - Long-Running Data Processing

**Query**: "Best practice for handling long-running data processing that might exceed time limits"

**Score**: 6/10

**Retrieved Chunks**:
1. [user_guide] - Workflow Limits section
2. [user_guide] - (tangential)
3. [user_guide] - (tangential)
4. [troubleshooting_guide] - Timeout errors
5. [architecture_overview] - (tangential)

**Correct Answer Location**:
- File: `troubleshooting_guide.md`
- Section: `### Timeout Errors (3600 second limit)` → `#### Solutions` (lines 549-587)
- Content: "Increase workflow timeout", "Enable parallel processing", "Add checkpointing", "Split workflow into smaller workflows"
- Also: user_guide.md lines 636, 679 mention splitting workflows

**Why Retrieval Failed**:
1. Query uses implicit vocabulary: "long-running data processing"
2. Corpus uses explicit vocabulary: "timeout", "3600 seconds", "checkpointing"
3. Retrieved timeout info but not the solutions section

**Pipeline Stage Analysis**:
- **Query Rewriting**: "long-running" may have been expanded to "timeout"
- **BM25 Stage**: PARTIAL - "time limits" ≠ "timeout" exactly
- **Semantic Stage**: PARTIAL - "long-running data processing" is implicit; corpus is explicit
- **RRF Fusion**: Got limits but not solutions

**Root Cause**: VOCABULARY_MISMATCH + IMPLICIT_QUERY

**Evidence**:
- Query: "long-running data processing" - user vocabulary
- Corpus: "Workflow exceeded maximum execution time of 3600 seconds" - system vocabulary
- Solutions (parallel workers, checkpointing, split workflows) are in troubleshooting_guide but not retrieved

---

## Failure 15: imp_003 - Debug Slow API Calls

**Query**: "How to debug why my API calls are slow"

**Score**: 7/10

**Retrieved Chunks**:
1. [architecture_overview] - Performance Characteristics
2. [troubleshooting_guide] - Slow Query Performance
3. [architecture_overview] - (tangential)
4. [troubleshooting_guide] - Latency Breakdown Analysis
5. [deployment_guide] - (tangential)

**Correct Answer Location**:
- File: `troubleshooting_guide.md`
- Section: `### High API Latency` → `#### Latency Breakdown Analysis` (lines 255-295)
- Content: Auth 18%, DB Query 64%, Business Logic 13%, Serialization 5%
- Also: `cloudflow metrics latency-report` command

**Why Retrieval Failed**:
1. Retrieved latency breakdown (chunk 4) - GOOD
2. But the specific percentages may not be in the retrieved chunk
3. Query is implicit ("slow") vs corpus explicit ("latency", "P95", "P99")

**Pipeline Stage Analysis**:
- **Query Rewriting**: "slow" may have been expanded to "latency", "performance"
- **BM25 Stage**: PARTIAL - "slow" matches "Slow Query Performance"
- **Semantic Stage**: GOOD - "slow API calls" semantically relates to latency
- **RRF Fusion**: Got relevant content but may have missed specific percentages

**Root Cause**: CHUNKING_BOUNDARY + PARTIAL_RETRIEVAL

**Evidence**:
- Retrieved Latency Breakdown Analysis section
- The specific percentages (Auth 18%, DB 64%) may be in a different chunk
- This is a near-success - got the right section but possibly wrong chunk boundary

---

## Failure 16: imp_004 - Production Monitoring Setup

**Query**: "What monitoring should I set up for production workflows?"

**Score**: 8/10 (borderline pass, included for completeness)

**Retrieved Chunks**:
1. [deployment_guide] - Prometheus Setup
2. [deployment_guide] - Grafana Dashboards
3. [architecture_overview] - Monitoring stack
4. [architecture_overview] - Key Metrics
5. [deployment_guide] - (tangential)

**Correct Answer Location**:
- File: `deployment_guide.md`
- Section: `## Monitoring and Observability` (lines 581-661)
- Content: Prometheus, Grafana, key metrics (request rate, error rate, latency)
- Also: architecture_overview.md lines 918-937 has monitoring details

**Why Retrieval Failed** (borderline):
1. Retrieved Prometheus and Grafana - GOOD
2. May have missed Jaeger for distributed tracing
3. May have missed PagerDuty alerts configuration

**Pipeline Stage Analysis**:
- **Query Rewriting**: "monitoring" + "production" should be preserved
- **BM25 Stage**: GOOD - "monitoring" + "production" matched
- **Semantic Stage**: GOOD - Monitoring is well-represented in corpus
- **RRF Fusion**: Good ranking overall

**Root Cause**: PARTIAL_RETRIEVAL (minor)

**Evidence**:
- This is actually a near-success (8/10)
- Retrieved core monitoring content
- Minor gaps in Jaeger/PagerDuty details

---

## Summary

### Root Cause Distribution

| Root Cause | Count | % | Description |
|------------|-------|---|-------------|
| NEGATION_BLIND | 5 | 31% | Embeddings don't distinguish "do X" from "don't do X" |
| VOCABULARY_MISMATCH | 4 | 25% | Query uses different words than corpus |
| EMBEDDING_BLIND | 4 | 25% | Embeddings fail on YAML/code/structured content |
| BM25_MISS | 3 | 19% | Keyword matching failed due to vocabulary gap |
| RANKING_ERROR | 3 | 19% | Correct content exists but ranked too low |
| CORPUS_GAP | 3 | 19% | Information not explicitly in corpus |
| COMPARATIVE_QUERY | 2 | 13% | "Difference between X and Y" queries fail |
| CHUNKING_BOUNDARY | 2 | 13% | Answer split across chunk boundaries |
| YAML_CODE_BLOCK | 2 | 13% | YAML/code content not well indexed |

*Note: Some failures have multiple root causes*

### Pipeline Stage Failures

| Stage | Failures | % | Notes |
|-------|----------|---|-------|
| Semantic Stage | 12 | 75% | Embeddings struggle with negation, code, comparisons |
| BM25 Stage | 8 | 50% | Vocabulary mismatch, stop words, code tokenization |
| RRF Fusion | 6 | 38% | Wrong content ranked higher than correct content |
| Query Rewriting | 4 | 25% | Negation may be lost in rewriting |
| Chunking | 2 | 13% | Answers split across boundaries |

### Key Insights

1. **Negation is the #1 failure mode**: 5 of 16 failures (31%) involve negation queries ("what NOT to do", "why doesn't X work"). Embeddings treat "do X" and "don't do X" as semantically similar.

2. **YAML/Code content is invisible to embeddings**: PgBouncer config (YAML), HPA config (YAML), and health check paths are in code blocks that don't embed well. This affects 4 failures.

3. **Vocabulary mismatch is pervasive**: Users say "propagate", corpus says "TTL". Users say "long-running", corpus says "timeout". Users say "exhaustion", corpus says "pooling".

4. **Comparative queries need special handling**: "Difference between X and Y" queries require retrieving BOTH concepts and finding comparison content. Current pipeline doesn't handle this.

5. **PgBouncer is a blind spot**: 2 of 16 failures (mh_002, cmp_001) specifically fail to retrieve PgBouncer content despite it appearing 33 times in corpus. The content is in YAML code blocks.

### Recommendations

1. **Add negation-aware query expansion**: When query contains "not", "don't", "shouldn't", "can't", expand to include positive forms AND add negation-specific keywords.

2. **Enrich YAML/code blocks with prose descriptions**: Add natural language summaries to code blocks so embeddings can capture meaning.

3. **Add synonym expansion for common vocabulary mismatches**: "propagate" → "TTL, invalidation, update", "exhaustion" → "pooling, limit, max connections".

4. **Implement comparative query detection**: When query contains "difference", "compare", "vs", retrieve both concepts and boost comparison sections.

5. **Improve PgBouncer indexing**: Add "connection pool", "pool exhaustion", "connection limit" keywords to PgBouncer chunks.
