# Task 0.1: Create Edge Case Test Dataset - COMPLETED

**Date**: 2026-01-26
**Status**: ✅ COMPLETED

## Summary

Successfully created `poc/chunking_benchmark_v2/corpus/edge_case_queries.json` with all 24 queries extracted from the failure dataset.

## File Details

- **Location**: `poc/chunking_benchmark_v2/corpus/edge_case_queries.json`
- **Size**: 15KB, 236 lines
- **Format**: Valid JSON (verified with `json.load()`)

## Content Breakdown

### Metadata
- Source: `.sisyphus/notepads/failure-dataset.md`
- Generated: 2026-01-26
- Total queries: 24
- Failed queries: 15 (score ≤7)
- Passing queries: 9 (score >7)

### Failed Queries (15)

All queries include:
- `id`: Query identifier (e.g., mh_002)
- `type`: Query type (multi-hop, temporal, comparative, negation, implicit)
- `query`: The actual query text
- `expected_answer`: What the correct answer should be
- `baseline_score`: Current retrieval score (≤7)
- `target_score`: Target improvement score (typically 8)
- `root_causes`: List of failure patterns (e.g., VOCABULARY_MISMATCH, EMBEDDING_BLIND, NEGATION_BLIND, YAML_BLIND)
- `failure_notes`: Detailed explanation of why retrieval failed

**Failed Query Types**:
- Multi-hop: mh_002, mh_004 (2)
- Temporal: tmp_003, tmp_004, tmp_005 (3)
- Comparative: cmp_001, cmp_002, cmp_003 (3)
- Negation: neg_001, neg_002, neg_003, neg_004, neg_005 (5)
- Implicit: imp_001, imp_003 (2)

### Passing Queries (9)

All queries include:
- `id`: Query identifier
- `type`: Query type
- `query`: The actual query text
- `expected_answer`: What the correct answer should be
- `baseline_score`: Current retrieval score (>7)
- `must_not_regress`: Boolean flag (always true) - these queries must maintain their score

**Passing Query Types**:
- Multi-hop: mh_001, mh_003, mh_005 (3)
- Temporal: tmp_001, tmp_002 (2)
- Comparative: cmp_004, cmp_005 (2)
- Implicit: imp_002, imp_004 (2)

## Data Quality Notes

### Discrepancy Found
The source file's summary table header claims "16 failed, 8 passing" but the raw scores table shows "15 failed, 9 passing". The raw scores table is authoritative and was used for this extraction.

**Raw Scores Verification**:
- Counted all 24 queries from raw scores table (lines 498-526)
- Verified each query ID and score
- Confirmed no duplicates
- All 24 queries accounted for

### Root Cause Mapping

Queries are tagged with root causes from the failure analysis:

| Root Cause | Count | Queries |
|-----------|-------|---------|
| VOCABULARY_MISMATCH | 4 | mh_002, cmp_001, imp_001, imp_003 |
| EMBEDDING_BLIND | 11 | mh_002, mh_004, tmp_003, tmp_004, tmp_005, cmp_002, cmp_003, imp_001, imp_003 |
| NEGATION_BLIND | 5 | neg_001, neg_002, neg_003, neg_004, neg_005 |
| YAML_BLIND | 1 | mh_004 |

## Acceptance Criteria Met

✅ File created at `poc/chunking_benchmark_v2/corpus/edge_case_queries.json`
✅ Valid JSON (parses without errors)
✅ All 15 failed queries included with:
  - expected_answer
  - baseline_score
  - root_causes
  - failure_notes
✅ All 9 passing queries included with:
  - must_not_regress: true
✅ Verification command works:
  ```
  python -c "import json; d=json.load(open('poc/chunking_benchmark_v2/corpus/edge_case_queries.json')); print(f'Failed: {len(d[\"failed_queries\"])}, Passing: {len(d[\"passing_queries\"])}')"
  ```
  Output: `Failed: 15, Passing: 9`

## Next Steps

This dataset is ready for use in:
1. Task 0.2: Create shared utilities module (gem_utils.py)
2. Task 0.3: Create manual test runner (test_gems.py)
3. Phase 1: Strategy implementation and testing

## Files Modified

- ✅ Created: `poc/chunking_benchmark_v2/corpus/edge_case_queries.json`
- ✅ Created: `.sisyphus/notepads/task-0-1-completion.md` (this file)

