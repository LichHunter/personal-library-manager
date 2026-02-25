# Gem Strategy Test Results

**Strategy**: test_strategy
**Date**: 2026-01-26T08:04:33.383152
**Queries Tested**: 2

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
