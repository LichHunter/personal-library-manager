# Archived Plans: Never Executed

**Archived:** 2026-02-18

These plans were prepared during the research phase but never executed. They are preserved for historical reference and potential future revival.

## Reason for Archival

| Plan | Status | Reason |
|------|--------|--------|
| `poc-1b-scale-testing.md` | Never started | POC-1b experiment was implemented but never run at scale |
| `poc-2-confidence-scoring.md` | Partially started | POC exists but plan's phases never executed |
| `strategy-v5-implementation.md` | Superseded | V5 replaced by V6 (POC-1c) before completion |
| `benchmark-llm-grading.md` | Never started | Benchmark tooling deprioritized |
| `benchmark-question-loader-refactor.md` | Never started | Refactoring deprioritized |
| `docker-redis-cache.md` | POC-only | Redis cache remained in POC, not promoted |
| `enriched-hybrid-no-llm.md` | Never started | Experimental strategy not pursued |
| `informed-benchmark-run.md` | Never started | Benchmark run deprioritized |
| `modular-benchmark-exact-replication.md` | Never started | Replication testing deprioritized |
| `modular-independence.md` | Never started | Module testing deprioritized |
| `modular-retrieval-pipeline.md` | Superseded | Replaced by `rag-retrieval-system.md` |

## Revival Notes

If reviving any of these plans:
1. Review against current codebase state (significant changes since creation)
2. Update file paths and imports (src/ structure changed)
3. Re-run Metis review for current context
4. Update effort estimates (tooling has improved)

## Active Plans Retained

The following plans remain active in `.sisyphus/plans/`:
- `pipeline-service.md` - Currently active (query rewriting, HTTP service)
- `rag-retrieval-system.md` - Executed (search system in production)
- `poc-1c-scalable-ner.md` - Executed (V6 pipeline in production)
- `slow-extraction-docker.md` - Executed (Docker packaging done)
- `fast-extraction-gliner.md` - Executed (fast extraction done)
- `src-structure-implementation.md` - Executed (src/plm/ structure done)
