
## [2026-01-28] Golden Test Retrieval Mismatch

**Status**: OPEN - Needs investigation

**Issue**: Modular implementation produces slightly different retrieval results despite identical enrichment and embeddings.

**Evidence**:
- ✅ Enrichment: IDENTICAL for all chunks
- ✅ Embeddings: IDENTICAL (np.allclose passes)
- ❌ Retrieval: Different chunk IDs in different order

**Example Query**: "What is the Topology Manager?"
- Original: `['..._probes_1', '..._logs_6', '..._leases_4', '..._leases_2', '..._logs_0']`
- Modular:  `['..._logs_6', '..._leases_1', '..._probes_1', '..._leases_2', '..._leases_4']`

**Verified Identical**:
- Query rewriting: Mocked to return original query
- Query expansion: Uses same DOMAIN_EXPANSIONS dictionary and logic
- Enrichment: Verified byte-for-byte match
- Embeddings: Verified with np.allclose
- RRF order: Semantic FIRST, BM25 SECOND (matches reference)
- Adaptive weights: 3.0/0.3/10 vs 1.0/1.0/60 (matches reference)

**Potential Causes**:
1. **Float precision**: Subtle differences in np.dot() or BM25 scores causing tie-breaking differences
2. **RRF tie-breaking**: When scores are equal, Python's sorted() is stable but depends on insertion order
3. **BM25 tokenization**: Possible difference in how `.lower().split()` is applied
4. **Random state**: Some component might have non-deterministic behavior

**Next Steps**:
1. Add detailed logging to both implementations' retrieve() methods
2. Compare intermediate scores (semantic, BM25, RRF) for each chunk
3. Identify exact point where results diverge
4. Check if scores are identical or just very close (float precision)

