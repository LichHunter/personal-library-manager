# Architectural Decisions

## Class-Based Design
**Decision**: Use classes (`RetrievalGrader`, `RetrievalMetrics`) instead of standalone functions  
**Rationale**: Follows existing codebase pattern (QueryRewriter), allows for future configuration, cleaner imports

## Position Weights (Total Score Formula)
```python
POSITION_WEIGHTS = {
    1: 1.0,      # Full credit - best possible
    2: 0.95,     # Small penalty
    3: 0.95,     # Small penalty
    4: 0.85,     # Moderate penalty
    5: 0.85,     # Moderate penalty
    None: 0.6,   # Not found - significant penalty but credit good content
}
```

**Rationale**: Graduated penalties incentivize top rankings while still crediting quality content

## Pass Rate Thresholds
- 8.0+ = Excellent
- 7.0+ = Good (PRIMARY threshold)
- 6.5+ = Acceptable

**Rationale**: 7.0 means "user could likely solve their problem" - the key quality bar

---

