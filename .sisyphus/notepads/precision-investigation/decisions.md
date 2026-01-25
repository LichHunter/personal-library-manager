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


## [2026-01-25 12:35] Task 8.1: Oracle Final Validation

### Oracle Assessment

**Production Readiness**: ✅ ACCEPTABLE

**Coverage**: 83% within acceptable 80-85% range
**Performance**: ✅ <500ms latency maintained
**Complexity**: ✅ Low (~120 lines, no dependencies)
**Maintainability**: ✅ Good (dictionary-based, testable)

### Recommendations

**Ship current solution** - ROI on further non-LLM improvements is diminishing.

**Monitoring**:
- Track expansion trigger rate (target: 5-15%)
- Log zero-result queries for dictionary expansion
- Monitor latency percentiles (alert if p95 > 400ms)

**Maintenance**:
- Weekly review of zero-result queries
- Add to dictionary when 3+ similar patterns found
- Keep dictionary under 50 entries

**Escalation Triggers** (for LLM solutions):
- Coverage drops below 80%
- Users report "can't find X" for existing facts
- Business requires >90% coverage

**What NOT to pursue**:
- Further dictionary expansion (diminishing returns)
- Negation rewriting (only 3 facts affected)
- Fact extraction (requires LLM, save for v2)

### Action Plan

1. ✅ Merge to main
2. ✅ Add basic logging
3. ✅ Document dictionary maintenance
4. ⏸️ Park negation handling for future

**Final verdict**: Ship it. Monitor. Move on.


## [2026-01-25 13:00] Research Phase: BMX and LLM Integration

### Research Findings

**BMX Algorithm**:
- New BM25 successor by Mixedbread (Aug 2024)
- Entropy-weighted similarity + semantic enhancement
- Python library: `baguetter` (mixedbread-ai/baguetter)
- Claims to outperform BM25 on BEIR benchmarks

**Existing LLM Infrastructure**:
- AnthropicProvider ready (supports Claude Haiku/Sonnet)
- HyDE, Multi-Query, LOD_LLM strategies exist
- call_llm() function available

### Oracle Strategic Guidance

**Key Insight**: "Your BM25 isn't the bottleneck - it's vocabulary mismatch. BMX won't help."

**Recommended Approach**: LLM Query Rewriting with Claude Haiku

**Why**:
- Addresses 5-6 of 9 missed facts (vocabulary mismatch, technical jargon, indirect queries)
- Lowest complexity, acceptable latency (+200-300ms)
- Haiku sufficient for simple reformulation task

**Skip BMX**: Not worth testing - doesn't address actual failure modes

### Test Order (Per Oracle)

1. **Phase 1**: Query Rewriting with Claude Haiku (2-4h)
   - Expected: 4-6 additional facts → 89-94% coverage
   
2. **Phase 2**: Targeted Fact Extraction (if needed)
   - Only if Phase 1 doesn't hit 95%

3. **Skip BMX**: Unless LLM approaches fail

### Acceptable Tradeoffs

- Latency: 500-700ms total (vs 200ms now) - ACCEPTABLE
- Cost: ~$0.000025/query with Haiku - NEGLIGIBLE

