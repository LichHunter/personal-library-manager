# Architectural Decisions - Realistic Questions Benchmark

## [2026-01-27] Model Selection
- **Decision**: Use Claude 3.5 Haiku for transformation
- **Rationale**: Cheap, fast, good enough for simple transformations
- **Alternative considered**: GPT-4o-mini (rejected: more expensive)

## [2026-01-27] Quality Evaluation
- **Decision**: Use 5 automated heuristics instead of ML-based evaluation
- **Rationale**: Simple, fast, interpretable, no training data needed
- **Threshold**: Overall score ≥0.5 and ≤2 issues = pass

## [2026-01-27] Path Mapping
- **Decision**: Simple string replace, skip non-matching docs
- **Rationale**: 97.6% coverage is sufficient, edge cases not worth complexity
