# Gem Strategy Test Results

**Strategy**: adaptive_hybrid
**Date**: 2026-01-26T08:03:52.959460
**Queries Tested**: 9

## Query: mh_001
**Type**: multi-hop

**Query**: Compare JWT expiration in Auth Service vs the API documentation - are they consistent?

**Expected Answer**: Auth Service: 15-minute access token expiry. API docs: 3600 seconds (1 hour) max. These are different contexts - Auth Service internal tokens vs API JWT claims.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 8/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: mh_003
**Type**: multi-hop

**Query**: What's the relationship between workflow timeout (3600s) and the retry backoff strategy?

**Expected Answer**: Workflow timeout is 3600s max. Retry uses exponential backoff: 1s, 2s, 4s (max 3 retries). Total retry time ~7s, well within timeout. But long-running steps can still timeout.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 9/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: mh_005
**Type**: multi-hop

**Query**: What happens to scheduled workflows during a disaster recovery failover?

**Expected Answer**: RPO is 1 hour, RTO is 4 hours. Scheduler uses leader election with Redis. During failover, scheduled executions may be skipped (logged in audit trail). Kafka retention allows event replay.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 8/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: tmp_001
**Type**: temporal

**Query**: What changed between the last DR test and the current DR objectives?

**Expected Answer**: Last DR test (Dec 15, 2025): Actual RTO was 2h 23m (target 4h), RPO was 42m (target 1h). Both met objectives. DNS propagation issue was resolved.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 9/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: tmp_002
**Type**: temporal

**Query**: How often should API keys be rotated and what's the certificate rotation schedule?

**Expected Answer**: API keys: rotate every 90 days. Certificates: rotate every 90 days (automated via cert-manager). Secrets in Vault also rotated every 90 days.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: cmp_004
**Type**: comparative

**Query**: Compare the rate limits for authenticated vs unauthenticated API requests

**Expected Answer**: Authenticated: 100 requests/minute per user. Unauthenticated: 20 requests/minute. Burst allowance: 150 requests in 10-second window. Enterprise: 1000 req/min.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 8/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: cmp_005
**Type**: comparative

**Query**: How do SEV-1 and SEV-2 incidents differ in response time and escalation?

**Expected Answer**: SEV-1: Immediate response (<15 min), page on-call immediately, complete outage. SEV-2: <1 hour response, create ticket and notify on-call, major functionality impaired.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 9/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: imp_002
**Type**: implicit

**Query**: How to ensure my application survives a complete region failure

**Expected Answer**: Multi-AZ deployment across 3 AZs. Cross-region replication to us-west-2 (15-min lag). Manual failover procedure: update DNS, promote DR replica, scale up DR services. RTO: 2-4 hours.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 8/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: imp_004
**Type**: implicit

**Query**: What monitoring should I set up for production workflows?

**Expected Answer**: Prometheus for metrics, Grafana for dashboards, Jaeger for distributed tracing. Key metrics: request rate, error rate, latency percentiles, cache hit ratios, Kafka consumer lag. Alerts via PagerDuty.

**Retrieved Chunks**:
1. [doc_id: ???, chunk_id: ???]
   > (chunk content will appear here after running retrieval)

**Baseline Score**: 8/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---
