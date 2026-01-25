# Learnings - Precision Investigation

## [2026-01-25] Session Start
- Tasks 1-2 completed successfully
- Comprehensive trace logging implemented
- Baseline benchmark run completed
- Ready for root cause analysis


## [2026-01-25 11:00] Task 3: Root Cause Analysis Complete

### Key Findings
- Analyzed enriched_hybrid_fast strategy (77.4% coverage, 12 missed facts)
- Root causes categorized:
  - VOCABULARY_MISMATCH (4 facts): Query terms don't match document terms
  - ACRONYM_GAP (1 fact): RPO/RTO poorly embedded - 100% miss rate
  - INDIRECT_QUERY (2 facts): Problem-style queries don't match factual content
  - TECHNICAL_JARGON (1 fact): "iat" JWT terminology
  - GRANULARITY (1 fact): Specific config values not linked to concepts
  - FACT_BURIED (1 fact): Fact in retrieved chunk but not extracted
  - REDUNDANT_PHRASING (1 fact): Same info, different phrasing
  - PHRASING_MISMATCH (1 fact): Markdown formatting differences

### Critical Insights
- Negation queries show 0% improvement over semantic baseline (54.7%)
- RPO/RTO facts: 100% miss rate across all 6 queries
- Enriched hybrid helps most with synonym (+11.3%) and original (+9.5%) queries
- BM25 + semantic fusion doesn't solve indirect/negation query patterns

### High-Impact Solutions Identified
1. Query expansion/rewriting (addresses VOCABULARY_MISMATCH)
2. Acronym dictionary (addresses ACRONYM_GAP)
3. Negation-aware retrieval (addresses INDIRECT_QUERY)
4. Fact extraction post-processing (addresses FACT_BURIED)


## [2026-01-25 11:30] Task 4: Solution Research Complete

### Solutions Identified

6 solutions researched and ranked by ROI:
1. Query Expansion (1.67 facts/hour) - PRIORITY 1
2. Chunk Overlap (0.67 facts/hour)
3. Negation Rewriting (0.33 facts/hour) - PRIORITY 2
4. Multi-Query (0.30 facts/hour)
5. HyDE (0.25 facts/hour)
6. Fact Extraction (0.19 facts/hour)

### Key Insights

- Query expansion addresses both VOCABULARY_MISMATCH and ACRONYM_GAP (same root cause)
- Highest ROI solution is simplest: domain dictionary with ~50 lines of code
- Negation rewriting complements expansion (different failure mode)
- Complex solutions (HyDE, fact extraction) have lower ROI
- Estimated 1 day to reach 90%+ coverage with top 2 solutions


## [2026-01-25 11:35] Task 5: Testable Hypotheses Formulated

### Hypotheses

**H1: Query Expansion** - 77.4% → 86.8% (+5 facts)
- Test procedure: Implement domain dictionary, run benchmark
- Success: >= 85% coverage

**H2: Negation Rewriting** - 86.8% → 90.6% (+2 facts)
- Test procedure: Implement pattern rewriting, use multi-query
- Success: >= 89% coverage

**H3: Stacked Solution** - 77.4% → 90.6% (+7 facts)
- Test procedure: Enable both solutions, validate cumulative effect
- Success: >= 90% coverage

### Testing Strategy

Phase 1: Baseline validation ✅
Phase 2: Individual solution testing (H1, then H2)
Phase 3: Stacked solution validation (H3)
Phase 4: Iteration if needed

### Success Thresholds

- 90%+: SUCCESS
- 85-90%: GOOD
- 80-85%: ACCEPTABLE
- <80%: FAILURE

