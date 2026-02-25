# Learnings: Semantic-Router (Aurelio Labs)

## Core Architecture: 4-Stage Pipeline

```
1. Encode → 2. Retrieve → 3. Score → 4. Filter
   (10ms)     (5ms)        (1ms)      (instant)
```

**Key Insight**: Separate routing (fast, semantic) from execution (slow, LLM-based)

## Pattern 1: Score Aggregation

**Problem**: Each route has multiple utterances. Top-k returns mixed results.

**Solution**: Group scores by route, then aggregate
```python
def score_route(route_name, top_k_results):
    route_scores = [s for r, s in top_k_results if r == route_name]
    return np.mean(route_scores)  # or max, or sum
```

**Aggregation methods**:
- **Mean** (default): Balanced, robust
- **Max**: "Best match wins" — good for distinct routes
- **Sum**: More evidence = higher confidence — good for overlapping routes

## Pattern 2: Per-Route Thresholds

**Don't use global threshold!**
```python
# BAD: threshold = 0.5 for all
# GOOD: Each route has its own
routes = {
    "politics": {"threshold": 0.25, ...},
    "chitchat": {"threshold": 0.30, ...},
}
```

**Why**: Different routes have different similarity distributions

## Pattern 3: Threshold Optimization (fit())

**Random search is surprisingly effective**:
```python
def optimize_thresholds(X_train, y_train, n_iter=500):
    best_acc = evaluate(current_thresholds)
    
    for _ in range(n_iter):
        # Random perturbation ±0.8
        candidate = perturb(current_thresholds, range=0.8)
        acc = evaluate(candidate)
        if acc > best_acc:
            best_acc, best_thresholds = acc, candidate
    
    return best_thresholds
```

**Results**: 34.85% → 90.91% accuracy after fit()

**Critical**: Include negative examples (`y=None`) to prevent overly low thresholds

## Pattern 4: Lazy LLM Execution

```python
# Fast path: Semantic routing only
if route_matches_semantically(query):
    if route.needs_parameters:
        # Slow path: LLM only when needed
        params = llm.extract(query, schema)
        return RouteChoice(route, params)
    else:
        return RouteChoice(route, None)
```

**Key**: LLM is lazy-loaded, only called AFTER semantic routing succeeds

## Pattern 5: Hybrid Routing (Dense + Sparse)

```python
def hybrid_score(query, route, alpha=0.3):
    dense = cosine_similarity(embed(query), embed(route.utterances))
    sparse = bm25_score(query, route.utterances)
    return alpha * dense + (1 - alpha) * sparse
```

**alpha values**:
- 0.3 (default): 70% keyword, 30% semantic
- 0.7: More semantic
- 0.1: Heavily keyword-focused (medical/legal)

## Pattern 6: Observability

**Always return metadata for debugging**:
```python
@dataclass
class RouteDecision:
    route_name: Optional[str]
    similarity_score: float
    threshold: float
    matched: bool
    top_k_candidates: List[Tuple[str, float]]
```

**Monitor**:
- Score distribution per route (detect drift)
- Near-miss queries (score just below threshold)
- Ambiguous queries (multiple routes with similar scores)

## Pattern 7: Graceful Failure

```python
if len(passed_routes) == 0:
    return RouteChoice()  # Empty choice, not an error
```

**Check with**: `if route_choice.name is None:`

## Documentation Excellence

1. **Jupyter notebooks as docs** — every feature is executable
2. **Progressive complexity** — quickstart → intermediate → advanced
3. **Real examples** — not foo/bar, actual use cases
4. **Performance metrics** — before/after numbers for every optimization

## Key Takeaways

1. **Speed through separation**: Routing (fast) vs execution (slow)
2. **Per-route thresholds**: One size does NOT fit all
3. **Fit, don't guess**: Random search beats manual tuning
4. **Include negatives**: Train with `None` examples
5. **Monitor scores**: Log for drift detection
6. **Fail gracefully**: Empty choice, not exceptions

---
*Source: Investigation of aurelio-labs/semantic-router, 1.6K stars*
