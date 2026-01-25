# Problems - Precision Investigation

## [2026-01-25] Current Blockers
- None yet - proceeding with Task 3 (Root Cause Analysis)


## [2026-01-25 12:10] Query Expansion Limitation Discovered

### Problem
Query expansion achieves only 79.2% coverage, cannot reach 85% target.

### Root Cause
- BM25 with expansion: RPO/RTO chunk ranks #1 ✅
- Semantic with expansion: RPO/RTO chunk ranks #26 ❌
- RRF final rank: #9 (outside top-5)
- **Embedding model (BGE) lacks domain-specific knowledge**

### Implication
Query expansion helps keyword matching but not semantic understanding.
Need different approach to improve semantic component.

### Blocker
Cannot achieve 95% target with current approach.
Need Oracle consultation for strategic guidance.

