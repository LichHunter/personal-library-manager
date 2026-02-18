# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-18  
**Commit:** d96da8d  
**Branch:** master

## OVERVIEW

Local-first RAG system for personal document corpus (NotebookLM-like). Extracts technical terms, builds hybrid search index (BM25 + semantic + RRF fusion), answers queries with citations. Python 3.10+, Anthropic Claude, FAISS, SQLite.

## STRUCTURE

```
./
├── src/plm/                    # Production package (55 files, ~10k lines)
│   ├── extraction/fast/        # Heuristic extraction (GLiNER, YAKE, regex)
│   ├── extraction/slow/        # V6 LLM pipeline (Claude, 11-module hybrid_ner/)
│   ├── search/                 # HybridRetriever, BM25, semantic, RRF
│   └── shared/llm/             # LLM provider abstraction (Anthropic, Gemini)
├── poc/                        # Isolated experiments (each has own pyproject.toml)
├── tests/                      # pytest: unit/, integration/, search/
├── docker/                     # Docker Compose for search service
├── data/vocabularies/          # auto_vocab.json, train_documents.json
└── slow-extraction/            # Runtime I/O (input/, output/, logs/)
```

## ENTRY POINTS

| Entry Point | File | Invocation |
|-------------|------|------------|
| Search CLI | `src/plm/search/service/cli.py` | `plm-query "query"` |
| FastAPI | `src/plm/search/service/app.py` | `uvicorn plm.search.service.app:app` |
| Fast extraction | `src/plm/extraction/fast/cli.py` | `python -m plm.extraction.fast.cli` |
| Slow extraction | `src/plm/extraction/slow/cli.py` | Docker only, env-var configured |

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add search feature | `src/plm/search/retriever.py` | `HybridRetriever` orchestrates all search |
| Modify RRF weights | `src/plm/search/retriever.py:73-84` | `DEFAULT_*` and `EXPANDED_*` constants |
| Add extraction stage | `src/plm/extraction/slow/hybrid_ner/pipeline.py` | V6 pipeline: extract→ground→filter→validate→postprocess |
| Add LLM provider | `src/plm/shared/llm/` | Subclass `LLMProvider` from `base.py` |
| Add query component | `src/plm/search/components/` | BM25, semantic, enricher, expander, rrf |
| Test retrieval | `tests/search/test_retriever.py` | Integration tests for HybridRetriever |

## BUILD & RUN

```bash
# Dev environment (Nix + direnv)
direnv allow                    # Auto-creates .venv, runs uv sync

# Build Docker images (Nix builds, not docker build)
nix build .#fast-extraction-docker && docker load < result
nix build .#slow-extraction-docker && docker load < result
nix build .#search-service-docker && docker load < result

# Run search service
docker compose -f docker/docker-compose.search.yml up search-service

# Run tests
pytest tests/
```

## CONVENTIONS

- **Package manager**: `uv` exclusively (not pip/poetry)
- **Type checker**: `basedpyright` (no config, all defaults)
- **No linter/formatter**: Style is unenforced; follow existing patterns
- **Build backend**: `hatchling` (exception: one POC uses setuptools)
- **PyTorch**: CPU-only from custom index
- **POCs are isolated**: Each has own `pyproject.toml` + `.venv`, not shared

## ANTI-PATTERNS

| Pattern | Why Forbidden | Location |
|---------|---------------|----------|
| `as any`, `@ts-ignore` | N/A (Python) | — |
| GLiNER for software NER | Rejected in POC-1c, produces garbage | `src/plm/extraction/fast/gliner.py` is dead code |
| `calculate_token_metrics()` | DEPRECATED, broken for multi-doc | Use `calculate_token_metrics_per_doc` |
| Retrieval Gems results | Tested with wrong chunking strategy | `poc/chunking_benchmark_v2/archive/` |
| Skip ground-truth filter | `NEVER filter GT terms` | Test code invariant |

## CRITICAL INVARIANTS

- **RRF order**: Semantic FIRST, BM25 SECOND — changing breaks POC parity
- **RRF accumulation**: `dict.get(idx, 0)` pattern required
- **Embeddings normalized**: EmbeddingEncoder normalizes, cosine = dot product
- **V6 pipeline stages**: Extract → Ground → Filter → Validate → Postprocess (order matters)

## UNIQUE PATTERNS

- **Nix sub-flakes inside src/**: Each service (`fast/`, `slow/`, `search/`) has own `flake.nix` imported by root
- **OpenCode OAuth**: Services authenticate via `~/.local/share/opencode/auth.json`, not API key
- **Runtime I/O at root**: `slow-extraction/` directory is live data exchange, not in `src/`
- **POC → Production promotion**: `src/plm/extraction/slow/hybrid_ner/` is direct copy of POC-1c V6

## COMMANDS

```bash
# Development
direnv allow                              # Enter dev shell
uv sync                                   # Sync dependencies
pytest tests/                             # Run tests
basedpyright                              # Type check

# Docker (from repo root)
nix build .#search-service-docker && docker load < result
docker compose -f docker/docker-compose.search.yml up

# Extraction pipeline
docker compose -f src/plm/extraction/slow/docker-compose.yml up slow-extraction
```

## NOTES

- **No CI pipeline** — All testing is manual via `pytest` and `scripts/*.py`
- **POC-1b never executed** — Strategies E/F/G/H/I implemented but 0 test results
- **RAPTOR integration risky** — 5 known bugs in upstream code (see `docs/research/raptor/`)
- **Project pivoted** — Name says "book metadata" but it's Kubernetes docs RAG
