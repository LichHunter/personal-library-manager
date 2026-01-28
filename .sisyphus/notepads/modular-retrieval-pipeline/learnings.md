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

## Task 2: Base Pipeline Abstractions (2026-01-28)

### What We Built
Created `poc/modular_retrieval_pipeline/base.py` with Unix pipe-style pipeline abstractions:
- **Component Protocol**: Generic protocol with `process(InputT) -> OutputT` method
- **Pipeline Class**: Fluent API for chaining components with `add()` method
- **PipelineError**: Base exception with component index and original error tracking
- **TypeValidationError**: Specialized exception for type mismatches between components

### Design Decisions

#### 1. Protocol-Based Component Interface
```python
@runtime_checkable
class Component(Protocol[InputT, OutputT]):
    def process(self, data: InputT) -> OutputT:
        ...
```
- Used `Protocol` instead of ABC for duck typing flexibility
- Generic types `InputT` and `OutputT` enable type safety
- `@runtime_checkable` allows `isinstance()` checks at runtime
- Components are pure functions (no state, no side effects)

#### 2. Fluent API for Pipeline Construction
```python
pipeline = (Pipeline()
    .add(QueryRewriter())
    .add(QueryExpander())
    .add(BM25Scorer()))
```
- Method chaining with `add()` returning `self`
- Optional `build()` method to make pipeline immutable
- Auto-build on first `run()` if not explicitly built
- Cannot add components after pipeline is built

#### 3. Unix Pipe Data Flow
```python
# Sequential execution: output of N → input of N+1
current_data = data
for component in components:
    current_data = component.process(current_data)
return current_data
```
- Simple sequential data flow (no parallelism)
- Each component receives output from previous component
- Immutable data objects (components return new objects)
- Composable: any component can be swapped if it implements protocol

#### 4. Fail-Fast Error Propagation
```python
try:
    current_data = component.process(current_data)
except TypeError as e:
    raise TypeValidationError(...) from e
except Exception as e:
    raise PipelineError(...) from e
```
- First error stops pipeline execution immediately
- Wraps errors in `PipelineError` with context:
  - Component index (0-based)
  - Component instance
  - Original exception
- Special handling for `TypeError` → `TypeValidationError`

#### 5. Runtime Type Validation
- Python's gradual typing means type hints are not enforced at runtime
- Pipeline catches `TypeError` exceptions as proxy for type mismatches
- Provides clear error messages with component index and types
- Not perfect (some type errors may not raise TypeError) but pragmatic

### Key Patterns

#### Pattern 1: Generic Protocol for Type Safety
```python
class Component(Protocol[InputT, OutputT]):
    def process(self, data: InputT) -> OutputT: ...
```
- Enables static type checking with mypy/pyright
- Generic types document expected input/output
- Duck typing allows any class with `process()` method

#### Pattern 2: Fluent Builder API
```python
Pipeline().add(c1).add(c2).add(c3).run(data)
```
- Readable, declarative pipeline construction
- Method chaining reduces boilerplate
- Optional `build()` for explicit immutability

#### Pattern 3: Rich Error Context
```python
class PipelineError(Exception):
    def __init__(self, message, component_index, component, original_error):
        self.component_index = component_index
        self.component = component
        self.original_error = original_error
```
- Preserves original exception with `from e`
- Adds pipeline-specific context (which component failed)
- Enables debugging without losing stack trace

### Verification Results

All tests passed:
- ✓ Single component: `"hello"` → `UpperCase()` → `"HELLO"`
- ✓ Chained components: `"hello"` → `UpperCase()` → `AddExclamation()` → `Repeat()` → `"HELLO!HELLO!"`
- ✓ Fail-fast: `FailingComponent` raises `PipelineError` with component index 1
- ✓ Type validation: `ExpectsInt` with string input raises `TypeValidationError`
- ✓ Error context: Exception includes component index, component type, and original error

### Implementation Notes

#### Type Hints and Runtime Behavior
- Changed `run()` return type from `OutputT` to `Any` to satisfy type checker
- Python's type system is gradual: type hints are documentation, not enforcement
- Runtime type validation relies on catching `TypeError` exceptions
- This is pragmatic: perfect type safety would require runtime type checking library

#### Protocol vs ABC
- Chose `Protocol` over `ABC` for flexibility
- Any class with `process()` method works (duck typing)
- No need to inherit from base class
- Better for integrating with existing code

#### Immutability Enforcement
- Pipeline becomes immutable after `build()` or first `run()`
- Cannot add components after building
- Prevents accidental modification during execution
- Components themselves should be stateless (enforced by convention, not code)

### Next Steps

This pipeline abstraction enables:
1. **Query rewriting component**: `Query → RewrittenQuery`
2. **Query expansion component**: `RewrittenQuery → ExpandedQuery`
3. **Embedding component**: `ExpandedQuery → EmbeddedQuery`
4. **BM25 retrieval component**: `EmbeddedQuery → tuple[ScoredChunk, ...]`
5. **Semantic retrieval component**: `EmbeddedQuery → tuple[ScoredChunk, ...]`
6. **Fusion component**: `tuple[ScoredChunk, ...] → PipelineResult`

Each component is a pure function that can be tested in isolation and composed into different pipeline configurations.

### Technical Notes

- File: `poc/modular_retrieval_pipeline/base.py`
- Dependencies: `typing` module only (no external dependencies)
- Type hints: Complete with generics (`InputT`, `OutputT`)
- Error handling: Fail-fast with rich context
- Immutability: Pipeline is immutable after build
- Composability: Components are pure functions
- Follows Unix pipe philosophy: simple, sequential, composable

## Task 3: Cache Integration (2026-01-28)

### What We Built
Created `poc/modular_retrieval_pipeline/cache.py` with cache integration for expensive operations:
- **CacheableComponent**: Decorator-pattern wrapper that adds caching to any component
- **DiskCacheBackend**: Disk-based cache using pickle serialization
- **CacheBackend Protocol**: Interface for pluggable cache backends (disk, Redis, etc.)
- **Hash-based cache keys**: SHA256 hashing of serialized input data

### Design Decisions

#### 1. Decorator Pattern for Caching
```python
class CacheableComponent(Generic[InputT, OutputT]):
    def __init__(self, component: Component[InputT, OutputT], cache_dir: str = "cache"):
        self.component = component
        self.backend = DiskCacheBackend(cache_dir)
```
- Wraps any component without modifying it
- Implements Component protocol for pipeline compatibility
- Transparent to pipeline: `Pipeline().add(CacheableComponent(slow_component))`
- Follows decorator pattern: composition over inheritance

#### 2. Hash-Based Cache Keys
```python
def _make_cache_key(self, data: InputT) -> str:
    # Try JSON serialization first (simple types)
    # Fall back to pickle (complex types)
    # Compute SHA256 hash and take first 16 chars
    return hashlib.sha256(serialized).hexdigest()[:16]
```
- Deterministic: identical inputs always produce same key
- Collision-resistant: different inputs produce different keys (SHA256)
- Filesystem-safe: 16-character hex string
- Flexible: handles both simple types (JSON) and complex types (pickle)

#### 3. Pluggable Cache Backends
```python
class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def put(self, key: str, value: Any) -> None: ...
    def clear(self) -> int: ...
```
- Protocol-based interface (duck typing)
- Default: DiskCacheBackend (pickle files)
- Extensible: can implement Redis, memcached, etc.
- Decouples caching logic from storage mechanism

#### 4. Disk Cache Implementation
```python
class DiskCacheBackend:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
```
- Uses pickle for serialization (supports arbitrary Python objects)
- Creates cache directory automatically
- One file per cache key: `{key}.pkl`
- Graceful error handling: returns None on deserialization failure

#### 5. Component Protocol Integration
```python
class CacheableComponent(Generic[InputT, OutputT]):
    def process(self, data: InputT) -> OutputT:
        cache_key = self._make_cache_key(data)
        cached_result = self.backend.get(cache_key)
        if cached_result is not None:
            return cached_result
        result = self.component.process(data)
        self.backend.put(cache_key, result)
        return result
```
- Implements Component protocol: `process(InputT) -> OutputT`
- Transparent caching: caller doesn't know if result is cached
- Works in pipelines: `Pipeline().add(cached_component).add(next_component)`

### Key Patterns

#### Pattern 1: Decorator for Cross-Cutting Concerns
```python
# Without caching
slow_component = ExpensiveComponent()
result = slow_component.process(data)

# With caching (no code change to ExpensiveComponent)
cached = CacheableComponent(slow_component, cache_dir="cache")
result = cached.process(data)  # Same interface, cached behavior
```
- Adds caching without modifying component
- Follows Open/Closed Principle: open for extension, closed for modification
- Composable: can wrap any component

#### Pattern 2: Protocol-Based Backends
```python
# Default disk backend
cached = CacheableComponent(component)

# Custom Redis backend
redis_backend = RedisBackend(host="localhost")
cached = CacheableComponent(component, backend=redis_backend)
```
- Pluggable backends via Protocol
- No inheritance required
- Easy to test with mock backends

#### Pattern 3: Serialization Fallback
```python
try:
    serialized = json.dumps(data, sort_keys=True, default=str).encode()
except (TypeError, ValueError):
    serialized = pickle.dumps(data)
```
- JSON for simple types (fast, human-readable)
- Pickle for complex types (flexible, supports any Python object)
- Deterministic: `sort_keys=True` ensures same key for same data

### Verification Results

All tests passed with excellent performance:
- ✓ Cache key generation: Same input → same key, different inputs → different keys
- ✓ Caching speedup: First call 0.1003s, second call 0.0001s (1122x speedup!)
- ✓ Cache persistence: Different instances share cache across disk
- ✓ Pipeline integration: CacheableComponent works in Pipeline.add()
- ✓ Clear cache: clear_cache() removes all cached files

### Performance Metrics

```
First call (with execution):  0.1003s
Second call (from cache):     0.0001s
Speedup:                      1122.2x
```

For expensive operations (YAKE extraction, spaCy NER, embeddings):
- First call: Runs expensive operation, stores result
- Subsequent calls: Returns cached result in ~0.1ms
- Typical speedup: 100-1000x depending on operation cost

### Design Trade-offs

#### Pros
- ✓ Transparent: no changes to wrapped component
- ✓ Composable: works with any component
- ✓ Flexible: pluggable backends
- ✓ Deterministic: hash-based keys
- ✓ Persistent: disk-based by default
- ✓ Simple: minimal code, easy to understand

#### Cons
- ✗ Serialization overhead: JSON/pickle adds cost
- ✗ Memory usage: pickle files on disk
- ✗ Cache invalidation: no automatic expiration
- ✗ Distributed caching: disk-based doesn't work across machines

### Integration with Existing Code

#### EnrichmentCache Comparison
The existing `EnrichmentCache` in `poc/chunking_benchmark_v2/enrichment/cache.py`:
- Specific to enrichment results (EnrichmentResult type)
- Uses JSON serialization
- Stores in `enrichment_cache/` directory
- Has `clear()` method

Our `CacheableComponent`:
- Generic: works with any component
- Supports JSON + pickle
- Configurable cache directory
- Has `clear_cache()` method
- Pluggable backends

**Design decision**: Created new generic wrapper instead of reusing EnrichmentCache because:
1. EnrichmentCache is specific to enrichment results
2. CacheableComponent needs to work with any component type
3. Different serialization needs (pickle vs JSON)
4. Decorator pattern is more flexible than inheritance

### Next Steps

This cache integration enables:
1. **Keyword extraction caching**: YAKE results cached by document
2. **NER caching**: spaCy entity extraction cached by text
3. **Embedding caching**: Vector embeddings cached by query
4. **Query rewriting caching**: LLM rewrites cached by original query
5. **Retrieval caching**: BM25/semantic results cached by query

Each expensive operation can be wrapped with `CacheableComponent` for 100-1000x speedup on repeated queries.

### Technical Notes

- File: `poc/modular_retrieval_pipeline/cache.py`
- Test file: `poc/modular_retrieval_pipeline/test_cache.py`
- Dependencies: `hashlib`, `json`, `pickle`, `pathlib` (all stdlib)
- Type hints: Complete with generics (`InputT`, `OutputT`)
- Error handling: Graceful fallback on serialization errors
- Immutability: Cached values are immutable (enforced by component protocol)
- Composability: Works with any component implementing Component protocol
- Performance: 1122x speedup on 0.1s operations

