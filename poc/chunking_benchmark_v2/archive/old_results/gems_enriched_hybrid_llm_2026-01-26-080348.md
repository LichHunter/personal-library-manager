# Gem Strategy Test Results

**Strategy**: enriched_hybrid_llm
**Date**: 2026-01-26T08:03:48.482995
**Queries Tested**: 2

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
