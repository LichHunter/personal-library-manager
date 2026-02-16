## CLI Implementation (Task 3)

### V6 Pipeline Integration
- `ground_candidates()` expects `dict[str, list[str]]` mapping source names to term lists
- Returns `dict[str, dict]` with normalized keys and metadata (term, sources, source_count)
- Must extract terms from grounded dict before passing to `filter_noise()`
- `classify_candidates_llm()` signature: `(candidates: list[str], text: str, model: str)`

### Chunking Configuration
- `HeadingChunker` accepts `min_tokens` and `max_tokens` in constructor
- Cannot set attributes on base `Chunker` class - must instantiate concrete class
- Use direct import for parameterized chunkers: `from plm.extraction.chunking.heading import HeadingChunker`

### Vocabulary Loading
- `load_negatives()` expects JSON with keys: safe_1000, safe_500, safe_200, safe_100
- Each key contains list of dicts with "word" field
- Returns empty set if file doesn't exist (graceful degradation)

### CLI Design Patterns
- Environment variables with defaults for Docker deployment
- CLI flags override env vars for local testing
- Signal handlers (SIGTERM/SIGINT) for graceful shutdown
- Process-once mode for testing without watch loop
- Dry-run mode for validation without side effects

