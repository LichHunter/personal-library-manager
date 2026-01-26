# ⚠️ RETRIEVAL GEMS PROJECT - INVALID RESULTS

**Date**: 2026-01-26  
**Status**: ❌ **INVALID - DO NOT USE**

---

## Critical Flaw

The entire "Retrieval Gems Implementation" project (retrieval-gems-implementation-v2) produced **invalid results** due to a fundamental testing error.

### The Problem

All gem strategies were tested with **FixedSizeStrategy(512 tokens)** chunking, when the production system uses **MarkdownSemanticStrategy** chunking.

**Evidence**: `test_gems.py` line 78-82:
```python
def chunk_documents(
    documents: list[Document], chunk_size: int = 512, overlap: int = 0
) -> list[Chunk]:
    strategy = FixedSizeStrategy(chunk_size=chunk_size, overlap=overlap)
    return strategy.chunk_many(documents)
```

### Impact on Results

| Configuration | Chunking | Retrieval | Manual Score |
|---------------|----------|-----------|--------------|
| **Production (Correct)** ✅ | MarkdownSemanticStrategy | enriched_hybrid_llm | **94%** |
| Gems: Synthetic Variants ❌ | FixedSizeStrategy(512) | synthetic_variants | 6.6/10 (66%) |
| Gems: Adaptive Hybrid ❌ | FixedSizeStrategy(512) | adaptive_hybrid | 6.0/10 (60%) |
| Gems: All others ❌ | FixedSizeStrategy(512) | various | 5.1-6.0/10 |

**The 28-point regression (94% → 66%) is entirely due to wrong chunking, not retrieval strategy.**

### Previous Validation

From `SMART_CHUNKING_VALIDATION_REPORT.md`:

> "The retrieval strategy (`enriched_hybrid_llm`) was never broken - **the chunking was**. Switching from fixed-size (512 tokens) to semantic markdown chunking increased accuracy from 54% to 94%"

### Invalid Files

All results in this directory with `gems_*` prefix are invalid:
- `results/gems_adaptive_hybrid_*.md` (15 queries graded)
- `results/gems_negation_aware_*.md` (15 queries graded)
- `results/gems_synthetic_variants_*.md` (15 queries graded)
- `results/gems_bm25f_hybrid_*.md` (15 queries graded)
- `results/gems_contextual_*.md` (15 queries graded)
- `results/gems_hybrid_gems_*.md` (24 queries graded)

**Total**: 99 manually graded test cases - all invalid due to wrong chunking.

### Invalid Documentation

All notepad files in `.sisyphus/notepads/retrieval-gems-implementation-v2/`:
- `strategy-baselines.md` (384 lines) - comparisons invalid
- `gems-implementation-results.md` - conclusions invalid
- `production-recommendations.md` - recommendations invalid
- `ab-comparison.md` - comparison invalid
- All other analysis files - invalid

---

## Correct Production Configuration

**Use this instead:**

```python
# Chunking
from strategies import MarkdownSemanticStrategy
chunker = MarkdownSemanticStrategy(
    max_heading_level=4,
    target_chunk_size=400,
    min_chunk_size=50,
    max_chunk_size=800,
    overlap_sentences=1
)

# Retrieval
from retrieval.enriched_hybrid_llm import EnrichedHybridLLMRetrieval
strategy = EnrichedHybridLLMRetrieval(name="enriched_hybrid_llm")
```

**Performance**: 94% manual grading pass rate (9.4/10 average)

**Validation**: See `SMART_CHUNKING_VALIDATION_REPORT.md` for correct results.

---

## Lessons Learned

1. **Always verify test configuration** - A single wrong parameter invalidated 2 days of work
2. **Chunking matters more than retrieval** - 94% → 66% regression from chunking alone
3. **Read previous validation reports** - The answer was already documented
4. **Test assumptions early** - Should have verified chunking strategy on day 1

---

## Action Items

- [x] Mark all gem results as INVALID
- [x] Document the flaw
- [ ] Re-run gem strategies with MarkdownSemanticStrategy (if needed)
- [ ] Update test_gems.py to use correct chunking
- [ ] Add validation check to prevent this in future

---

**DO NOT USE THESE RESULTS FOR PRODUCTION DECISIONS**

Use `enriched_hybrid_llm` + `MarkdownSemanticStrategy` instead (94% validated).
