# Decisions - Precision Investigation

## [2026-01-25] Logging Infrastructure
- Decision: Add trace logging to enriched_hybrid.py and run_benchmark.py
- Rationale: Need visibility into retrieval ranking to diagnose precision loss
- Outcome: Successfully implemented, committed as 7dd6b70


## [2026-01-25 11:15] Task 3.1: Oracle Consultation - Solution Prioritization

### Oracle Recommendations

**Priority 1: Query Expansion with Domain Dictionary** (Quick: 2-4h)
- Addresses: VOCABULARY_MISMATCH (4) + ACRONYM_GAP (1) = 5 facts
- Expected gain: +5 facts → 86.8% coverage
- Implementation: Add domain expansion dictionary to retrieval
- Key insight: VOCABULARY_MISMATCH and ACRONYM_GAP are the same root cause

**Priority 2: Negation-Aware Query Rewriting** (Short: 4-8h)
- Addresses: INDIRECT_QUERY (2 facts)
- Expected gain: +2 facts → 90.6% coverage
- Implementation: Pattern-based query rewriting + multi-query retrieval

**Priority 3: Skip for Now**
- FACT_BURIED (1 fact): Extraction problem, not retrieval
- GRANULARITY (1 fact): Solved by Priority 1 dictionary
- PHRASING_MISMATCH (1 fact): Low value for 1 fact

### Testing Strategy
- Incremental stacking with isolated measurement
- Baseline → Query expansion → Query expansion + negation
- Measure cumulative progress toward 95% target

### Acceptable Thresholds
- 90%: Excellent for production
- 85%: Good, acceptable gaps
- 80%: Minimum viable

### Estimated Timeline
- 1 day to reach ~90% coverage
- Query expansion: 2h implementation + 1h testing
- Negation rewriting: 4h implementation + 30m testing


## [2026-01-25 11:40] Task 5.1: Oracle Consultation (Skipped - Already Complete)

Oracle consultation was completed in Task 3.1 and already validated the solution strategy:
- Priority 1: Query Expansion
- Priority 2: Negation Rewriting
- Testing strategy: Incremental stacking

No additional consultation needed. Proceeding to implementation.


## [2026-01-25 12:15] Task 7.1: Oracle Consultation - Weighted RRF Strategy

### Oracle Recommendation

**Don't abandon query expansion** - it's working for BM25. Problem is RRF fusion diluting BM25's gains.

**Solution: Weighted RRF with Expansion-Aware Boosting**

1. Weight BM25 higher when query expansion fires (1.5x)
2. Dampen semantic when expansion triggers (0.7x)
3. Increase candidate pool for expanded queries (2x multiplier)
4. Lower RRF k-constant for expanded queries (60 → 30)

**Expected Result**: 83-87% coverage (~3 hours effort)

### Rationale

Current RPO/RTO example:
- BM25 rank #1 → RRF: 1/(60+0) = 0.0167
- Semantic rank #26 → RRF: 1/(60+25) = 0.0118
- Combined: 0.0285 (loses to balanced ranks)

With weighted RRF:
- BM25: 1.5/(60+0) = 0.025
- Semantic: 0.7/(60+25) = 0.0082
- Combined: 0.0332 (competitive with top results)

### Escalation Path

If weighted RRF < 85%:
- HyDE for remaining hard queries
- Query-type classification (route technical queries to BM25-heavy path)

