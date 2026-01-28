# Modular Retrieval Pipeline - Learnings

## Task 1: Immutable Type System (2026-01-28)

### What We Built
Created `poc/modular_retrieval_pipeline/types.py` with 7 frozen dataclasses forming a transformation chain:
- **Query**: Original user query
- **RewrittenQuery**: Query after LLM reformulation (preserves original)
- **ExpandedQuery**: Query after synonym/term expansion (preserves rewritten)
- **EmbeddedQuery**: Query after vectorization (preserves expanded)
- **ScoredChunk**: Retrieved chunk with provenance tracking (source: bm25|semantic|rrf)
- **FusionConfig**: RRF configuration (k, weights)
- **PipelineResult**: Complete result with full transformation history

### Design Decisions

#### 1. Immutability via @dataclass(frozen=True)
- All dataclasses use `frozen=True` to prevent attribute modification after creation
- Verified with test: `q.text = "modified"` raises `AttributeError: cannot assign to field 'text'`
- Ensures data integrity throughout pipeline

#### 2. Immutable Collections
- Used `tuple` instead of `list` for all collections (immutable by default)
- Examples:
  - `metadata: tuple = field(default_factory=tuple)` instead of `dict`
  - `expansions: tuple[str, ...] = field(default_factory=tuple)` instead of `list[str]`
  - `scored_chunks: tuple[ScoredChunk, ...] = field(default_factory=tuple)` instead of `list[ScoredChunk]`
- Verified tuple immutability: `result.scored_chunks[0] = ...` raises `TypeError: 'tuple' object does not support item assignment`

#### 3. Transformation Chain Preservation
- Each type preserves the previous type in the chain:
  - `RewrittenQuery.original: Query`
  - `ExpandedQuery.query: RewrittenQuery`
  - `EmbeddedQuery.query: ExpandedQuery`
- Enables full history reconstruction: Query → Rewritten → Expanded → Embedded
- `PipelineResult` captures all stages for debugging

#### 4. Provenance Tracking
- `ScoredChunk.source: Literal["bm25", "semantic", "rrf"]` tracks which signal produced the score
- Enables understanding which retrieval method contributed to each result
- Critical for debugging and understanding fusion behavior

#### 5. Optional Intermediate Results
- `PipelineResult` uses `Optional[...]` for intermediate stages
- Allows partial pipelines (e.g., skip rewriting, skip expansion)
- Flexibility for different pipeline configurations

### Key Patterns

#### Pattern 1: Metadata as Immutable Tuples
```python
metadata: tuple = field(default_factory=tuple)
```
- Replaces mutable `dict` with immutable `tuple`
- Allows storing arbitrary metadata without mutation risk
- Can be extended with structured fields as needed

#### Pattern 2: Literal Types for Provenance
```python
source: Literal["bm25", "semantic", "rrf"]
```
- Type-safe enumeration of signal sources
- Enables static analysis and IDE autocomplete
- Better than string literals for maintainability

#### Pattern 3: Transformation Chain
```python
@dataclass(frozen=True)
class ExpandedQuery:
    query: RewrittenQuery  # Preserves previous stage
    expanded: str          # New data
    expansions: tuple[str, ...] = field(default_factory=tuple)
```
- Each type wraps the previous type
- Enables full history without duplication
- Supports Unix pipe philosophy (output of N → input of N+1)

### Verification Results

All tests passed:
- ✓ Query immutability: Cannot modify `text` field
- ✓ RewrittenQuery creation and chaining
- ✓ ExpandedQuery with multiple expansions
- ✓ EmbeddedQuery with vector embedding
- ✓ ScoredChunk with provenance tracking
- ✓ FusionConfig with RRF parameters
- ✓ PipelineResult with full history
- ✓ Tuple immutability: Cannot modify `scored_chunks`
- ✓ Transformation history preserved through all stages
- ✓ Python syntax validation passed

### Next Steps

This type system is foundational for:
1. Query rewriting component (uses Query → RewrittenQuery)
2. Query expansion component (uses RewrittenQuery → ExpandedQuery)
3. Embedding component (uses ExpandedQuery → EmbeddedQuery)
4. Retrieval component (produces ScoredChunk)
5. Fusion component (uses FusionConfig to combine signals)
6. Pipeline orchestration (uses PipelineResult for full history)

### Technical Notes

- All dataclasses are frozen (immutable)
- All collections use tuples (immutable)
- Type hints are complete and precise
- No circular dependencies
- Follows enrichment/base.py pattern from chunking_benchmark_v2
- Supports Unix pipe philosophy for modular components
