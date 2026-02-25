# Test Set Preparation - Completed

**Date**: 2026-02-21
**Status**: COMPLETE
**Task**: `todo/01_prepare_test_set.md`

## Summary

Created a labeled test query dataset with 229 queries for evaluating adaptive retrieval approaches on Kubernetes documentation.

## Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| Test query file | `benchmarks/datasets/test_queries.json` | Created |
| Schema documentation | `benchmarks/datasets/SCHEMA.md` | Created |
| Dataset README | `benchmarks/datasets/README.md` | Created |
| Consolidation script | `scripts/consolidate_queries.py` | Created |
| Build script | `scripts/build_test_set.py` | Created |

## Query Counts

| Query Type | Count | Required | Status |
|------------|-------|----------|--------|
| Factoid | 54 | 50 | PASS |
| Procedural | 55 | 50 | PASS |
| Explanatory | 53 | 50 | PASS |
| Comparison | 33 | 30 | PASS |
| Troubleshooting | 34 | 30 | PASS |
| **Total** | **229** | **210** | **PASS** |

## Sources

### Existing Queries (65 total)
- `needle_questions.json`: 20 queries (Topology Manager)
- `needle_questions_adversarial.json`: 20 queries (VERSION, COMPARISON, NEGATION, VOCABULARY)
- `informed_questions.json`: 25 queries (kubefix diverse topics)

### Generated Queries (164 total)
- Generated to fill gaps in minimum requirements
- Cover diverse Kubernetes topics beyond Topology Manager
- Include proper schema fields and expected answers

## Verification

### Minimum Count Verification
- [x] Factoid >= 50: 54 queries
- [x] Procedural >= 50: 55 queries
- [x] Explanatory >= 50: 53 queries
- [x] Comparison >= 30: 33 queries
- [x] Troubleshooting >= 30: 34 queries

### Schema Compliance
- [x] All queries have required fields (id, query, query_type, expected_answer)
- [x] All queries have optimal_granularity
- [x] All queries have source attribution
- [x] Existing queries have relevant_doc_ids from original datasets

### PLM Search Validation
- [x] Sample queries tested against PLM search service
- [x] Queries return relevant results
- [x] Schema supports chunk_id labeling

## Notes

1. **Chunk labeling**: `relevant_chunk_ids` are empty and will be populated during baseline measurement when running all queries through PLM search.

2. **Oracle performance**: Will be established in `02_baseline.md` by testing each query at chunk, heading, and document granularities.

3. **Topic diversity**: Generated queries cover API server, etcd, deployments, services, RBAC, networking, storage, and more - beyond the Topology Manager focus of existing queries.

## Next Step

Proceed to: `todo/02_baseline.md` - Measure current PLM performance
