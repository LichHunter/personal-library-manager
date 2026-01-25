# Problems: BMX Optimal Strategy

## Unresolved Blockers

(Subagents will append findings here)

## [2026-01-25] BLOCKER: Baguetter Package Not Available

### Issue
All three BMX strategies fail at runtime because `baguetter` package cannot be imported in the current environment.

### Error
```
TypeError: 'NoneType' object is not callable
  self.bmx = BMXSparseIndex()
```

### Root Cause
- `baguetter` is listed in `poc/chunking_benchmark_v2/pyproject.toml`
- Package has complex C dependencies (faiss, numpy with specific library paths)
- Nix environment doesn't have libz.so.1 in the right location for the .venv
- Silent import failure (try/except sets BMXSparseIndex = None)

### Impact
- **Task 5 (Sanity Check)**: BLOCKED - Cannot verify strategies work
- **Task 6 (Full Benchmark)**: BLOCKED - Cannot run 120-query benchmark
- **Task 7 (Analysis)**: BLOCKED - No results to analyze

### Possible Solutions
1. **Add baguetter to flake.nix** - Requires nix expertise, may have version conflicts
2. **Use alternative sparse index** - Replace BMX with BM25 (defeats purpose of investigation)
3. **Fix .venv library paths** - Complex, may require LD_LIBRARY_PATH manipulation
4. **Run outside nix** - Loses reproducibility guarantees

### Recommendation
**User decision required**: This investigation was specifically about BMX performance. Without baguetter, we cannot test BMX strategies. User should decide whether to:
- Invest time in fixing the environment
- Pivot to testing with BM25 instead (already works)
- Defer this investigation until environment is resolved
