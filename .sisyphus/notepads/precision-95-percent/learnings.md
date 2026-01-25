## Task 0: Baguetter Installation - Environment Fix

### Problem
Baguetter (BMX library) installation failed with ImportError for system libraries:
- `libz.so.1` (zlib) - required by numpy
- `libstdc++.so.6` (C++ standard library) - required by faiss-cpu

### Root Cause
Nix provides isolated system libraries in `/nix/store/`, but Python packages installed via pip/uv expect standard Linux library paths. The venv created by `uv` didn't have access to required Nix-provided system libraries.

### Solution
Updated `flake.nix` to include required system libraries in `LD_LIBRARY_PATH`:

```nix
LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
  "/run/opengl-driver"
  pkgs.zlib                # For numpy (libz.so.1)
  pkgs.stdenv.cc.cc.lib    # For faiss (libstdc++.so.6)
];
```

Also added explicit export in shellHook to ensure the path is available.

### Verification
After fix, baguetter works in `nix develop` shell:
```bash
cd /home/fujin/Code/personal-library-manager
nix develop
cd poc/chunking_benchmark_v2
source .venv/bin/activate
python -c "from baguetter.indices import BMXSparseIndex; idx = BMXSparseIndex(); print('✓ Works!')"
```

### Key Learnings

1. **Nix Environment Management**: 
   - Nix provides isolated, versioned system libraries
   - Python C extensions (numpy, faiss) need explicit library paths
   - `LD_LIBRARY_PATH` must include Nix store paths for system libs

2. **Direnv vs Nix Develop**:
   - `direnv` caches environment and may not pick up changes immediately
   - `nix develop` always uses fresh environment from flake
   - For testing flake changes, use `nix develop` directly

3. **Python Package Dependencies**:
   - Baguetter → faiss-cpu → numpy → libz.so.1 (zlib)
   - Baguetter → faiss-cpu → libstdc++.so.6 (C++ stdlib)
   - Pre-built wheels (manylinux) expect standard Linux paths

4. **Debugging Process**:
   - Check what libraries are missing: Read ImportError messages
   - Find Nix packages: `nix-store -q --references $(which python3.12) | grep zlib`
   - Test flake values: `nix eval .#devShells.x86_64-linux.default.LD_LIBRARY_PATH`
   - Verify in clean shell: `nix develop --command bash -c 'echo $LD_LIBRARY_PATH'`

### Files Modified
- `flake.nix`: Added zlib and libstdc++ to LD_LIBRARY_PATH
- `poc/chunking_benchmark_v2/pyproject.toml`: Added baguetter>=0.1.1
- `poc/chunking_benchmark_v2/uv.lock`: Updated with baguetter dependencies

### Commit
- `493911c`: chore(benchmark): add baguetter dependency for BMX
- `0dd12fb`: fix(nix): add zlib and libstdc++ to LD_LIBRARY_PATH for Python C extensions

### Time Spent
~2.5 hours debugging environment issues

### Next Steps
- Task 0 complete: Baguetter installed and verified ✓
- Proceed to Task 1: Investigate BMX API for RRF compatibility

---

## Task 1: BMX API Investigation for RRF Compatibility

### Objective
Investigate whether BMX (Baguetter's sparse index) can be used as a drop-in replacement for BM25 in our RRF (Reciprocal Rank Fusion) implementation.

### Test Script
Created `poc/chunking_benchmark_v2/test_bmx_api.py` to empirically test:
1. BMX API structure and return types
2. Whether BMX returns scores for ALL documents
3. Score scale comparison (BMX vs BM25)
4. RRF fusion compatibility

### Key Findings

#### 1. BMX API Structure
- **Return Type**: `baguetter.indices.base.SearchResults` (dataclass)
- **Attributes**: `keys`, `scores`, `normalized`, `to_dict()`
- **Not subscriptable**: Cannot use dict-like access (e.g., `results['scores']`)
- **Access pattern**: Use attribute access (e.g., `results.scores`)

#### 2. Full Corpus Score Availability ⚠️ CRITICAL
**BMX does NOT return scores for all documents by default:**
- When `top_k >= n_docs`: Returns scores for ALL documents ✓
- When `top_k < n_docs`: Returns ONLY top-k scores ✗

**Impact on RRF**: This is a BREAKING INCOMPATIBILITY
- Current RRF implementation expects score arrays for ALL documents
- BMX requires explicit `top_k=len(corpus)` to get full scores
- **Solution**: Always call `idx.search(query, top_k=len(corpus))` to get all scores

#### 3. Score Scale Comparison
Both BMX and BM25 use similar scales:
- **BM25 range**: 0.0 to 3.69 (mean: 0.604, std: 1.245)
- **BMX range**: 0.0 to 4.55 (mean: 0.741, std: 1.528)
- **Scale ratio**: BMX scores are ~1.22x larger than BM25
- **Implication**: Scores are NOT directly comparable, but ranks are

#### 4. RRF Fusion Compatibility ✓
RRF fusion works with BMX scores:
- Both BM25 and BMX produce rank-based fusion results
- RRF uses reciprocal ranks, not absolute scores
- Score scale differences don't matter for RRF (only ranks matter)
- **Verified**: Successfully computed RRF with BMX + BM25 scores

#### 5. BMX API Methods
Key methods for our use case:
```python
# Indexing
idx.add_many(keys: list[str], values: list[str]) -> BMXSparseIndex

# Searching
idx.search(query: str, *, top_k: int = 100) -> SearchResults
# Returns: SearchResults with .keys, .scores, .normalized attributes

# Batch operations
idx.search_many(queries: list[str], *, top_k: int = 100) -> list[SearchResults]
```

### Answers to Investigation Questions

**Q: Can BMX return scores for ALL documents?**
A: Yes, but only when `top_k >= corpus_size`. Default behavior returns only top-k.

**Q: What is BMX score scale vs BM25?**
A: BMX scores are ~1.22x larger than BM25, but both use similar distributions. Ranks are comparable.

**Q: How to adapt RRF fusion for BMX?**
A: 
1. Always call `idx.search(query, top_k=len(corpus))` to get full score array
2. Convert SearchResults to numpy array: `np.array(results.scores)`
3. Use existing RRF logic unchanged (RRF uses ranks, not absolute scores)

### Implementation Strategy for Task 2

**Minimal changes required:**
1. Replace BM25Okapi initialization with BMXSparseIndex
2. Change indexing: `bm25.get_scores(tokens)` → `bmx.search(query, top_k=len(corpus)).scores`
3. Convert BMX scores to numpy array for RRF computation
4. No changes to RRF fusion logic (it's rank-based)

**Code pattern:**
```python
# Old (BM25)
bm25_scores = self.bm25.get_scores(query.lower().split())

# New (BMX)
bmx_results = self.bmx_idx.search(query, top_k=len(self.chunks))
bm25_scores = np.array(bmx_results.scores)  # Convert to numpy for RRF
```

### Potential Issues & Mitigations

1. **Performance**: BMX uses FAISS under the hood (sparse vectors)
   - Mitigation: Benchmark against BM25 in Task 2

2. **Tokenization**: BMX has built-in tokenization
   - Current: Manual tokenization for BM25
   - BMX: Automatic tokenization
   - Mitigation: Use BMX's tokenizer for consistency

3. **Score interpretation**: BMX scores are sparse (many zeros)
   - Observed: 7/10 documents had score 0.0 for "JWT authentication" query
   - BM25: Also produces zeros for non-matching documents
   - Implication: Behavior is similar, no issue for RRF

### Files Created
- `poc/chunking_benchmark_v2/test_bmx_api.py` - Investigation script

### Time Spent
~1 hour (test script creation + execution + analysis)

### Next Steps
- Task 1 complete: BMX API investigation finished ✓
- Proceed to Task 2: Implement BMX in enriched_hybrid.py and test RRF fusion

---

## Task 2: BMX-based Hybrid Retrieval Strategy Implementation

### Objective
Create a new retrieval strategy `EnrichedHybridBMXRetrieval` by adapting `EnrichedHybridRetrieval` to use BMX (Baguetter's sparse index) instead of BM25.

### Implementation Summary

#### File Created
- `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_bmx.py` (340 lines)

#### Key Changes from EnrichedHybridRetrieval

1. **Import Changes**
   - Removed: `from rank_bm25 import BM25Okapi`
   - Added: `from baguetter.indices import BMXSparseIndex`

2. **Initialization**
   - Changed: `self.bm25: Optional[BM25Okapi] = None` → `self.bmx: Optional[BMXSparseIndex] = None`
   - Preserved: All enrichment, caching, and embedder initialization

3. **Indexing Method**
   - Preserved: Enrichment pipeline (same as before)
   - Preserved: Embedding generation (same as before)
   - Changed: BM25 initialization to BMX initialization
   ```python
   # Old: self.bm25 = BM25Okapi(tokenized)
   # New:
   self.bmx = BMXSparseIndex()
   chunk_ids = [chunk.id for chunk in chunks]
   self.bmx.add_many(chunk_ids, enriched_contents)
   ```

4. **Retrieval Method**
   - Preserved: Query expansion (expand_query function)
   - Preserved: Weighted RRF logic (bm25_weight, sem_weight, rrf_k adjustments)
   - Preserved: Semantic search (embeddings + dot product)
   - Changed: BM25 scoring to BMX search
   ```python
   # Old: bm25_scores = self.bm25.get_scores(expanded_query.lower().split())
   # New:
   bmx_results = self.bmx.search(expanded_query, top_k=len(self.chunks))
   bm25_scores = np.array(bmx_results.scores)
   ```
   - Preserved: RRF fusion logic (unchanged - rank-based, not score-based)

5. **Statistics Method**
   - Changed: `bm25_avg_doc_len` → `bmx_index: "BMXSparseIndex"`
   - Preserved: All other stats

#### Preserved Components
- ✓ DOMAIN_EXPANSIONS dictionary (exact copy)
- ✓ expand_query() function (exact copy)
- ✓ Enrichment pipeline (FastEnricher, caching)
- ✓ Semantic search (embeddings)
- ✓ Weighted RRF logic (all parameters and calculations)
- ✓ Debug tracing (all _trace_log calls)
- ✓ Logging and statistics

#### Registration
- Added import: `from .enriched_hybrid_bmx import EnrichedHybridBMXRetrieval`
- Added to RETRIEVAL_STRATEGIES: `"enriched_hybrid_bmx": EnrichedHybridBMXRetrieval`
- Added to __all__: `"EnrichedHybridBMXRetrieval"`

### Verification Results

✓ Strategy registered in RETRIEVAL_STRATEGIES
✓ Class imports successfully
✓ No LSP errors in enriched_hybrid_bmx.py
✓ All required methods present (index, retrieve, get_index_stats)

### Key Design Decisions

1. **BMX API Usage**: Always call `idx.search(query, top_k=len(self.chunks))` to get full score array for RRF compatibility (from Task 1 findings)

2. **Score Conversion**: Convert BMX SearchResults to numpy array for RRF computation: `np.array(bmx_results.scores)`

3. **Tokenization**: BMX handles tokenization internally (unlike BM25 which required manual tokenization)

4. **RRF Unchanged**: RRF fusion logic is rank-based, not score-based, so it works identically with BMX scores

### Potential Next Steps

1. **Benchmarking**: Compare BMX vs BM25 performance in hybrid retrieval
2. **Parameter Tuning**: Adjust rrf_k, weights for BMX-specific characteristics
3. **Integration Testing**: Test with full benchmark suite

### Files Modified
- `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_bmx.py` - Created
- `poc/chunking_benchmark_v2/retrieval/__init__.py` - Updated imports and registry

### Time Spent
~30 minutes (file creation + registration + verification)

### Next Steps
- Task 2 complete: BMX-based hybrid retrieval strategy implemented ✓
- Ready for benchmarking and integration testing
