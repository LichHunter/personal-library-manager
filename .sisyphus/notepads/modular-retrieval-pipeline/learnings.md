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

## Task 4: Query Rewriter Component (2026-01-28)

### What We Built
Created `poc/modular_retrieval_pipeline/components/query_rewriter.py` with QueryRewriter component:
- **QueryRewriter class**: Wraps existing rewrite_query function from enriched_hybrid_llm
- **Component protocol**: Implements `process(Query) -> RewrittenQuery` interface
- **Timeout handling**: Configurable timeout (default 5.0 seconds)
- **Stateless design**: No stored LLM client, pure function interface
- **Provenance tracking**: Preserves original query and tracks model used

### Design Decisions

#### 1. Wrapping Existing Function (Don't Reimplement)
```python
def process(self, data: Query) -> RewrittenQuery:
    rewritten_text = rewrite_query(data.text, timeout=self.timeout)
    return RewrittenQuery(
        original=data,
        rewritten=rewritten_text,
        model="claude-3-haiku",
    )
```
- Reuses existing `rewrite_query()` from `poc/chunking_benchmark_v2/retrieval/query_rewrite.py`
- No reimplementation of LLM logic
- Focuses on type transformation and Component protocol adaptation
- Leverages proven query rewriting strategy

#### 2. Stateless Component Design
```python
def __init__(self, timeout: float = 5.0):
    self.timeout = timeout
```
- Only stores timeout parameter (configuration)
- No LLM client stored as instance variable
- Each `process()` call independently invokes `rewrite_query()`
- Enables safe concurrent usage and testing

#### 3. Type Transformation
```python
# Input: Query (original user query)
# Output: RewrittenQuery (preserves original + adds rewritten version)
```
- Accepts Query objects (immutable dataclass)
- Returns RewrittenQuery objects (immutable dataclass)
- Preserves original query for provenance tracking
- Tracks which model performed rewriting ("claude-3-haiku")

#### 4. Timeout Configuration
```python
rewriter_5s = QueryRewriter(timeout=5.0)
rewriter_10s = QueryRewriter(timeout=10.0)
```
- Configurable timeout per instance
- Default 5.0 seconds (matches enriched_hybrid_llm strategy)
- Passed to `rewrite_query()` function
- Handles LLM timeout gracefully (returns original query on timeout)

#### 5. Package Structure
```
poc/modular_retrieval_pipeline/
├── __init__.py                    # Package marker
├── types.py                       # Immutable types
├── base.py                        # Component protocol
├── cache.py                       # Caching wrapper
├── components/
│   ├── __init__.py               # Package marker
│   └── query_rewriter.py         # QueryRewriter component
├── test_query_rewriter.py        # Tests
```
- Created `components/` subdirectory for component implementations
- Added `__init__.py` files to make proper Python packages
- Avoids naming conflicts with built-in `types` module

### Key Patterns

#### Pattern 1: Component Wrapper for Existing Functions
```python
class QueryRewriter(Component[Query, RewrittenQuery]):
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    def process(self, data: Query) -> RewrittenQuery:
        result = rewrite_query(data.text, timeout=self.timeout)
        return RewrittenQuery(original=data, rewritten=result, model="claude-3-haiku")
```
- Wraps existing function without modifying it
- Adapts function signature to Component protocol
- Adds type safety and provenance tracking
- Enables composition in pipelines

#### Pattern 2: Preserving Provenance
```python
return RewrittenQuery(
    original=data,           # Preserve original for history
    rewritten=rewritten_text,  # New data
    model="claude-3-haiku",  # Track which model
)
```
- Each transformation preserves previous stage
- Enables full history reconstruction
- Supports debugging and understanding data flow
- Follows immutable transformation chain pattern

#### Pattern 3: Configuration via Constructor
```python
rewriter = QueryRewriter(timeout=5.0)
```
- Timeout is configuration, not state
- Immutable after construction
- Enables different timeout strategies
- Supports dependency injection

### Verification Results

All tests passed:
- ✓ Component protocol: QueryRewriter implements Component[Query, RewrittenQuery]
- ✓ Type transformation: Query → RewrittenQuery works correctly
- ✓ Timeout configuration: Configurable timeout parameter
- ✓ Immutability: RewrittenQuery is frozen dataclass
- ✓ Stateless: No stored LLM client, pure function interface
- ✓ Provenance: Original query preserved in result
- ✓ Model tracking: Model field set to "claude-3-haiku"

### Test Results

```
Test 1: Component protocol implementation
✓ QueryRewriter implements Component protocol

Test 2: Type transformation
✓ Query → RewrittenQuery transformation works
  Original: why does my token expire
  Rewritten: token authentication expiration mechanism lifetime management
  Model: claude-3-haiku

Test 3: Timeout configuration
✓ Timeout parameter is configurable

Test 4: Immutability of RewrittenQuery
✓ RewrittenQuery is immutable (frozen dataclass)

Test 5: Pure function interface
✓ Pure function interface (stateless)

Test 6: No stored state
✓ Component is stateless (no stored LLM client)
```

### Integration with Pipeline

The QueryRewriter component fits into the pipeline as the first transformation:

```python
pipeline = (Pipeline()
    .add(QueryRewriter(timeout=5.0))      # Query → RewrittenQuery
    .add(QueryExpander())                 # RewrittenQuery → ExpandedQuery
    .add(EmbeddingComponent())            # ExpandedQuery → EmbeddedQuery
    .add(BM25Retriever())                 # EmbeddedQuery → ScoredChunks
    .add(SemanticRetriever())             # EmbeddedQuery → ScoredChunks
    .add(FusionComponent())               # ScoredChunks → PipelineResult
)
```

### Design Trade-offs

#### Pros
- ✓ Reuses proven query rewriting logic
- ✓ Stateless: safe for concurrent usage
- ✓ Configurable timeout
- ✓ Preserves provenance
- ✓ Type-safe with Component protocol
- ✓ Immutable output
- ✓ Simple, focused implementation

#### Cons
- ✗ Depends on external rewrite_query function
- ✗ LLM calls are non-deterministic
- ✗ Timeout handling is implicit (returns original on timeout)
- ✗ No caching (could wrap with CacheableComponent if needed)

### Next Steps

This component enables:
1. **Query expansion**: RewrittenQuery → ExpandedQuery
2. **Embedding**: ExpandedQuery → EmbeddedQuery
3. **Retrieval**: EmbeddedQuery → ScoredChunks
4. **Fusion**: Combine BM25 + semantic signals
5. **Full pipeline**: Query → PipelineResult with full history

### Technical Notes

- File: `poc/modular_retrieval_pipeline/components/query_rewriter.py`
- Test file: `poc/modular_retrieval_pipeline/test_query_rewriter.py`
- Dependencies: `modular_retrieval_pipeline.types`, `modular_retrieval_pipeline.base`, `chunking_benchmark_v2.retrieval.query_rewrite`
- Type hints: Complete with generics (Component[Query, RewrittenQuery])
- Error handling: Propagates exceptions from rewrite_query (fail-fast)
- Immutability: Output is frozen dataclass
- Composability: Works in pipelines with other components
- Performance: Depends on LLM latency (typically 0.5-2s per query)


## Task 5: Query Expander Component (2026-01-28)

### What We Built
Created `poc/modular_retrieval_pipeline/components/query_expander.py` with QueryExpander component:
- **QueryExpander class**: Expands queries with domain-specific terms to address vocabulary mismatch
- **Component protocol**: Implements `process(RewrittenQuery) -> ExpandedQuery` interface
- **DOMAIN_EXPANSIONS dict**: Ported from enriched_hybrid_llm with 14 expansion rules
- **Expansion tracking**: Records which expansion keys were applied (tuple of strings)
- **Stateless design**: DOMAIN_EXPANSIONS as class constant, pure function interface
- **Case-insensitive matching**: Matches expansion terms regardless of case
- **Deduplication**: Avoids adding terms already in query (case-insensitive)

### Design Decisions

#### 1. Porting DOMAIN_EXPANSIONS Dictionary
```python
DOMAIN_EXPANSIONS = {
    "rpo": "recovery point objective RPO data loss backup",
    "token": "JWT authentication token iat exp issued claims expiration",
    "database": "PostgreSQL Redis Kafka storage data layer",
    "monitoring": "Prometheus Grafana Jaeger observability metrics tracing",
    # ... 10 more expansions
}
```
- Copied from `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:20-42`
- 14 expansion rules covering critical vocabulary gaps
- Addresses VOCABULARY_MISMATCH (59% failure rate in realistic benchmarks)
- Covers acronyms (RPO, RTO, JWT, HPA) and technical stacks (database, monitoring)

#### 2. Expansion Tracking
```python
expansions: tuple[str, ...] = field(default_factory=tuple)
```
- Stores which expansion keys were applied (e.g., `("token",)` or `("database", "monitoring")`)
- Immutable tuple (frozen dataclass)
- Enables debugging: understand which expansions helped/hurt retrieval
- Different from expansion text: tracks keys, not full expansion strings

#### 3. Case-Insensitive Matching
```python
query_lower = query_text.lower()
for term, expansion in self.DOMAIN_EXPANSIONS.items():
    if term in query_lower:
        expansions_applied.append((term, expansion))
```
- Matches expansion keys case-insensitively
- Handles "TOKEN", "Token", "token" identically
- Improves recall: catches variations in user input

#### 4. Case-Insensitive Deduplication
```python
# Use lowercase comparison to catch case-insensitive duplicates
query_terms = set(query_lower.split())
expansion_terms_lower = {term.lower() for term in expansion_terms}
new_terms_lower = expansion_terms_lower - query_terms

# Map back to original case from expansion_terms
new_terms = {term for term in expansion_terms if term.lower() in new_terms_lower}
```
- Prevents "JWT" from being added if "jwt" already in query
- Preserves original case from expansion dictionary
- Critical for correctness: "JWT" in expansion, "jwt" in query should deduplicate

#### 5. Stateless Component Design
```python
class QueryExpander(Component[RewrittenQuery, ExpandedQuery]):
    DOMAIN_EXPANSIONS = DOMAIN_EXPANSIONS  # Class constant, not instance state
    
    def process(self, data: RewrittenQuery) -> ExpandedQuery:
        # Pure function: no state mutation, deterministic
```
- DOMAIN_EXPANSIONS is class constant (shared across instances)
- No instance variables except inherited from Component protocol
- Deterministic: same input always produces same output
- Safe for concurrent usage

#### 6. Type Transformation
```python
# Input: RewrittenQuery (from QueryRewriter)
# Output: ExpandedQuery (preserves RewrittenQuery + adds expanded text)
```
- Accepts RewrittenQuery objects (immutable dataclass)
- Returns ExpandedQuery objects (immutable dataclass)
- Preserves full provenance chain: Query → Rewritten → Expanded
- Tracks transformation method: "domain_specific"

### Key Patterns

#### Pattern 1: Domain-Specific Expansion Dictionary
```python
DOMAIN_EXPANSIONS = {
    "token": "JWT authentication token iat exp issued claims expiration",
    "rpo": "recovery point objective RPO data loss backup",
    # ...
}
```
- Maps short terms to expansion phrases
- Expansion phrases are space-separated terms
- Addresses vocabulary gaps in specific domains
- Easy to extend: add new term → expansion pairs

#### Pattern 2: Expansion Tracking
```python
expansions_applied = []
for term, expansion in DOMAIN_EXPANSIONS.items():
    if term in query_lower:
        expansions_applied.append((term, expansion))

expansion_keys = tuple(term for term, _ in expansions_applied)
```
- Collect which keys matched
- Store as tuple for immutability
- Enables debugging: which expansions were applied?
- Different from storing full expansion text

#### Pattern 3: Deduplication with Case Handling
```python
# Lowercase for comparison
query_terms = set(query_lower.split())
expansion_terms_lower = {term.lower() for term in expansion_terms}
new_terms_lower = expansion_terms_lower - query_terms

# Preserve original case
new_terms = {term for term in expansion_terms if term.lower() in new_terms_lower}
```
- Compares lowercase versions
- Preserves original case from expansion dictionary
- Avoids duplicates while maintaining readability

### Verification Results

All 12 tests passed:
- ✓ Component protocol: QueryExpander implements Component[RewrittenQuery, ExpandedQuery]
- ✓ Type transformation: RewrittenQuery → ExpandedQuery works correctly
- ✓ Token expansion: "token" query expands with JWT-related terms
- ✓ No matching expansions: Queries with no matches return unchanged
- ✓ Multiple expansions: "monitoring and database" expands both terms
- ✓ Case-insensitive matching: "TOKEN AUTHENTICATION" matches "token" key
- ✓ Deduplication: "JWT" appears once even if in both query and expansion
- ✓ Immutability: ExpandedQuery is frozen dataclass
- ✓ Input immutability: Input RewrittenQuery not modified
- ✓ Pure function: Same input always produces same output
- ✓ Provenance tracking: Full chain preserved (Query → Rewritten → Expanded)
- ✓ DOMAIN_EXPANSIONS constant: Properly defined as class constant

### Test Results

```
Test 1: Component protocol implementation
✓ QueryExpander implements Component protocol

Test 2: Type transformation
✓ RewrittenQuery → ExpandedQuery transformation works
  Input: token auth
  Expanded: token auth JWT authentication claims exp expiration iat issued
  Expansions: ('token',)

Test 3: Token expansion
✓ Token expansion works correctly
  Expansions applied: ('token',)

Test 4: No matching expansions
✓ No matching expansions handled correctly

Test 5: Multiple expansions
✓ Multiple expansions work correctly
  Expansions applied: ('database', 'monitoring')

Test 6: Case-insensitive matching
✓ Case-insensitive matching works

Test 7: Deduplication
✓ Deduplication works correctly
  Expanded: token JWT JSON authentication claims exp expiration iat issued web

Test 8: Immutability of ExpandedQuery
✓ ExpandedQuery is immutable (frozen dataclass)

Test 9: Input immutability
✓ Input RewrittenQuery is not modified

Test 10: Pure function interface
✓ Pure function interface (stateless and deterministic)

Test 11: Provenance tracking
✓ Provenance tracking works
  Original: token auth
  Rewritten: token auth
  Expanded: token auth JWT authentication claims exp expiration iat issued

Test 12: DOMAIN_EXPANSIONS as class constant
✓ DOMAIN_EXPANSIONS is properly defined as class constant
  Total expansions: 14
```

### Integration with Pipeline

The QueryExpander component fits into the pipeline as the second transformation:

```python
pipeline = (Pipeline()
    .add(QueryRewriter(timeout=5.0))      # Query → RewrittenQuery
    .add(QueryExpander())                 # RewrittenQuery → ExpandedQuery
    .add(EmbeddingComponent())            # ExpandedQuery → EmbeddedQuery
    .add(BM25Retriever())                 # EmbeddedQuery → ScoredChunks
    .add(SemanticRetriever())             # EmbeddedQuery → ScoredChunks
    .add(FusionComponent())               # ScoredChunks → PipelineResult
)
```

### Design Trade-offs

#### Pros
- ✓ Addresses vocabulary mismatch (59% failure rate in realistic benchmarks)
- ✓ Stateless: safe for concurrent usage
- ✓ Deterministic: same input always produces same output
- ✓ Preserves provenance: full transformation chain tracked
- ✓ Type-safe with Component protocol
- ✓ Immutable output
- ✓ Case-insensitive matching improves recall
- ✓ Deduplication prevents bloat
- ✓ Easy to extend: add new term → expansion pairs

#### Cons
- ✗ Fixed expansion dictionary (not learned from data)
- ✗ No weighting: all expansions treated equally
- ✗ No context awareness: same expansion for all queries
- ✗ Potential query bloat: many new terms added
- ✗ No feedback loop: can't learn which expansions help

### Performance Characteristics

- **Time complexity**: O(n*m) where n = query terms, m = expansion keys
- **Space complexity**: O(k) where k = total expansion terms
- **Typical performance**: <1ms for most queries
- **Bottleneck**: Deduplication set operations (negligible for typical query sizes)

### Next Steps

This component enables:
1. **Embedding**: ExpandedQuery → EmbeddedQuery
2. **Retrieval**: EmbeddedQuery → ScoredChunks
3. **Fusion**: Combine BM25 + semantic signals
4. **Full pipeline**: Query → PipelineResult with full history

### Technical Notes

- File: `poc/modular_retrieval_pipeline/components/query_expander.py`
- Test file: `poc/modular_retrieval_pipeline/test_query_expander.py`
- Dependencies: `modular_retrieval_pipeline.types`, `modular_retrieval_pipeline.base`
- Type hints: Complete with generics (Component[RewrittenQuery, ExpandedQuery])
- Error handling: None (pure function, no external dependencies)
- Immutability: Output is frozen dataclass
- Composability: Works in pipelines with other components
- Performance: <1ms per query (negligible overhead)

### Key Insight: Vocabulary Mismatch is Critical

The realistic benchmark shows 59% failure rate due to vocabulary mismatch:
- Users describe problems with natural language
- Documentation uses technical terms
- Query expansion bridges this gap by adding synonyms and related terms
- Example: "token" → "JWT authentication iat exp issued claims expiration"

This component directly addresses the #1 failure mode in retrieval systems.
