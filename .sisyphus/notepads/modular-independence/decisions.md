# Decisions: Modular Independence

## Architectural Choices

### Logger Strategy
- **Decision**: Copy BenchmarkLogger verbatim from chunking_benchmark_v2
- **Rationale**: Battle-tested, has all needed levels (TRACE/DEBUG/INFO/WARN/ERROR)
- **Alternative rejected**: Implement minimal logger (too much work, reinventing wheel)

### LLM Provider Scope
- **Decision**: Anthropic only (exclude OllamaProvider)
- **Rationale**: Not used in production, reduces copy size
- **Alternative rejected**: Copy both providers (unnecessary complexity)

### Auth Path
- **Decision**: Keep hardcoded to `~/.local/share/opencode/auth.json`
- **Rationale**: Standard OpenCode location, no need for configurability yet
- **Alternative rejected**: Make configurable (YAGNI)

### Error Handling
- **Decision**: Silent fallback (return original query on LLM failure)
- **Rationale**: Matches existing behavior, ensures retrieval never fails
- **Alternative rejected**: Raise exceptions (breaks retrieval flow)

## Implementation Choices

(Will be populated with specific implementation decisions during tasks)
