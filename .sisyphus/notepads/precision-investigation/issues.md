# Issues - Precision Investigation

## [2026-01-25] Known Issues
- RPO/RTO facts: 0% coverage across ALL queries
- JWT expiry details: Consistently missed across variant queries
- TTL facts: Mixed results - some queries find, others miss
- Short queries perform worse: "database stack", "api gateway resources", "RPO RTO" all 0% coverage


## [2026-01-25 12:00] Task 6: Query Expansion Below Target

### Issue
Query expansion achieved 79.2% coverage, below 85% target (expected 86.8%).

### Root Cause
- Query expansion works for BM25 (RPO/RTO chunk ranks #1)
- Semantic embedding of original query still fails
- RRF gives equal weight to both signals
- Relevant chunks ranked outside top-5 due to poor semantic score

### Impact
Only +1.9% improvement on original queries (77.4% → 79.2%)
Better improvement on casual (+5.7%) and contextual (+3.8%) queries

### Next Steps
Need to iterate with improved approach - see Task 7

