
## [2026-01-30] Informed vs Realistic Benchmark Results

**Hypothesis**: Using technical Kubernetes terminology would improve retrieval by 30-50%.

**Results**: NO IMPROVEMENT (0.0% delta)

| Metric | Realistic | Informed | Delta |
|--------|-----------|----------|-------|
| Hit@5  | 36.0%     | 36.0%    | 0.0%  |
| Hit@1  | 24.0%     | 24.0%    | 0.0%  |
| MRR    | 0.280     | 0.280    | 0.000 |

**Interpretation**:
The vocabulary mismatch hypothesis appears to be INCORRECT or INCOMPLETE. Using proper K8s terminology (e.g., "SubjectAccessReview", "kube-proxy", "ABAC") did NOT improve retrieval accuracy.

**Possible explanations**:
1. The "realistic" questions already contained enough technical terms
2. Semantic embeddings bridged the vocabulary gap
3. The root cause is NOT vocabulary but something else (e.g., chunking, query expansion, fusion weights)
4. Both question sets may be targeting the same 9 documents that succeed

**Next steps for investigation**:
- Analyze which specific questions succeeded/failed in both benchmarks
- Check if it's the same documents being retrieved
- Examine BM25 vs semantic contributions to ranking
- Consider if query expansion is already solving vocabulary mismatch

