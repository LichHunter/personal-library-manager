# PLM - Personal Library Manager

A local-first RAG system for querying personal document corpora with grounded, cited answers. Think NotebookLM, but self-hosted.

## Features

- **Hybrid Search**: BM25 lexical + semantic embeddings with RRF fusion
- **Entity Extraction**: Automatic technical term extraction (heuristic + LLM pipelines)
- **Citation-Grounded**: Every answer backed by source references
- **Local-First**: Runs on consumer hardware (32GB RAM recommended)
- **Multiple Interfaces**: CLI, REST API, MCP server

## Prerequisites

| Requirement | Version | Install |
|-------------|---------|---------|
| **Nix** | 2.18+ | [nix-installer](https://github.com/DeterminateSystems/nix-installer) (recommended) or [nixos.org](https://nixos.org/download/) |
| **direnv** | 2.32+ | [direnv.net](https://direnv.net/docs/installation.html) |
| **Docker** | 24+ | [docker.com](https://docs.docker.com/engine/install/) (only for containerized deployment) |

Python 3.12 is provided by Nix - do not install manually.

### Enable Nix Flakes

```bash
# Add to ~/.config/nix/nix.conf (create if doesn't exist)
experimental-features = nix-command flakes
```

If using the Determinate Systems installer, flakes are enabled by default.

### Anthropic API Key

Required for LLM features (slow extraction, query rewriting):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd personal-library-manager

# Enter development environment (creates .venv, installs dependencies)
direnv allow

# Or manually:
nix develop
uv sync
```

The Nix shell provides Python 3.12, uv, and all system dependencies. First run downloads ~2GB of dependencies.

## Quick Start

```bash
# 1. Extract entities from documents
python -m plm.extraction.fast.cli \
  --input ./my-docs \
  --output ./extracted \
  --workers 4

# 2. Start search service
INDEX_PATH=./index WATCH_DIR=./extracted \
uv run uvicorn plm.search.service.app:app --port 8000

# 3. Query
plm-query "How does X work?" --k 5
```

## Services

| Service | Purpose | Entry Point |
|---------|---------|-------------|
| **Fast Extraction** | Heuristic entity extraction (regex, YAKE) | `python -m plm.extraction.fast.cli` |
| **Slow Extraction** | LLM-powered extraction (Claude) | `python -m plm.extraction.slow.cli` |
| **Search Service** | Hybrid retrieval API | `uvicorn plm.search.service.app:app` |
| **MCP Server** | Model Context Protocol interface | `plm-mcp` |

## CLI Reference

### Fast Extraction

```bash
python -m plm.extraction.fast.cli \
  --input ./documents \
  --output ./extracted \
  --workers 8 \
  --pattern "**/*.md,**/*.txt"
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | required | Input directory |
| `--output` | required | Output directory for JSON |
| `--workers` | 1 | Parallel threads |
| `--pattern` | `**/*.md,**/*.txt` | File glob patterns |

### Search CLI

```bash
plm-query "your question" --url http://localhost:8000 --k 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | `http://localhost:8000` | Search service URL |
| `--k` | 5 | Number of results |
| `--rewrite` | false | Use LLM query rewriting |

## Docker Deployment

Two build methods are available:
- **Nix** (recommended): Reproducible, minimal images
- **Dockerfile**: Standard Docker builds for CI/CD compatibility

### Option 1: Nix Builds (Recommended)

Nix produces reproducible, minimal images without Dockerfiles.

```bash
nix build .#<target>    # Creates ./result symlink to image tarball
docker load < result    # Imports tarball into Docker daemon
```

**Important**: Each `nix build` overwrites the `result` symlink. You must `docker load` before building the next image.

### Build Single Image

```bash
# Search service (most common)
nix build .#search-service-docker && docker load < result
# Loaded: search-service:latest

# Fast extraction
nix build .#fast-extraction-docker && docker load < result
# Loaded: fast-extraction:latest

# Slow extraction (LLM pipeline)
nix build .#slow-extraction-docker && docker load < result
# Loaded: slow-extraction:latest
```

### Build All Images

```bash
for target in fast-extraction-docker slow-extraction-docker search-service-docker; do
  echo "Building $target..."
  nix build .#$target && docker load < result
done

# Verify
docker images | grep -E "(fast|slow|search)"
```

### Available Targets

| Nix Target | Docker Image | Description |
|------------|--------------|-------------|
| `.#fast-extraction-docker` | `fast-extraction:latest` | Heuristic extraction (regex, YAKE) |
| `.#slow-extraction-docker` | `slow-extraction:latest` | LLM extraction (Claude) |
| `.#search-service-docker` | `search-service:latest` | Hybrid retrieval API |

### Option 2: Dockerfile Builds

Standard Docker builds for environments without Nix.

```bash
# Build from repo root
docker build -f docker/Dockerfile.search -t plm-search:latest .
docker build -f docker/Dockerfile.fast-extraction -t plm-fast-extraction:latest .
docker build -f docker/Dockerfile.slow-extraction -t plm-slow-extraction:latest .
```

| Dockerfile | Image | Description |
|------------|-------|-------------|
| `docker/Dockerfile.search` | `plm-search:latest` | Hybrid retrieval API |
| `docker/Dockerfile.fast-extraction` | `plm-fast-extraction:latest` | Heuristic extraction |
| `docker/Dockerfile.slow-extraction` | `plm-slow-extraction:latest` | LLM extraction |

### Run with Docker Compose

```bash
# Search service standalone
docker compose -f docker/docker-compose.search.yml up

# Full pipeline with Redis queue
docker compose -f docker/docker-compose.full.yml --profile extraction up
```

## API Reference

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Kubernetes?", "k": 5}'
```

Response:
```json
{
  "results": [
    {
      "content": "...",
      "doc_id": "kubernetes-intro.md",
      "score": 0.85,
      "heading": "Overview"
    }
  ]
}
```

### GET /health

Returns `{"status": "ready"}` when service is operational.

### GET /status

Returns index statistics (document count, chunk count, etc.).

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEX_PATH` | `/data/index` | Search index directory |
| `WATCH_DIR` | none | Directory to watch for new extractions |
| `ANTHROPIC_API_KEY` | none | Claude API key |
| `PLM_LLM_MODEL` | `claude-haiku` | LLM for query rewriting |
| `QUEUE_ENABLED` | `false` | Enable Redis Streams mode |
| `QUEUE_URL` | `redis://localhost:6379` | Redis connection |
| `PLM_LOG_LEVEL` | `INFO` | Log level: TRACE, DEBUG, INFO, WARN, ERROR |
| `PLM_LOG_DIR` | `/data/logs` | Directory for log files |
| `PLM_LOG_TO_FILE` | `true` | Enable file logging (trace + info) |

### Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| Anthropic | `claude-haiku`, `sonnet`, `opus` | Default, requires API key |
| Ollama | `llama3.2`, `mistral`, `qwen2.5` | Local, auto-downloads |
| Gemini | `gemini-2.5-flash`, `gemini-2.5-pro` | Requires OAuth |

## Project Structure

```
personal-library-manager/
├── src/plm/
│   ├── extraction/
│   │   ├── fast/          # Heuristic extraction
│   │   └── slow/          # LLM extraction pipeline
│   ├── search/
│   │   ├── retriever.py   # HybridRetriever (BM25 + semantic)
│   │   ├── components/    # BM25, semantic, RRF fusion
│   │   └── service/       # FastAPI, CLI, MCP
│   ├── shared/
│   │   └── llm/           # LLM provider abstraction
│   └── benchmark/         # Retrieval benchmark suite
├── docker/                # Docker Compose files
├── tests/                 # pytest test suite
├── docs/                  # Architecture & research docs
└── poc/archive/           # Archived proof-of-concept experiments
```

## Development

```bash
# Run tests
pytest tests/

# Type checking
basedpyright

# Run specific test
pytest tests/search/test_retriever.py -v
```

## Benchmark

End-to-end retrieval benchmark for measuring search quality against ground-truth question-answer pairs.

```bash
# Run benchmark against running search service
python -m plm.benchmark.cli --datasets needle

# With service trace integration
python -m plm.benchmark.cli --trace-log /path/to/search_trace.log
```

**Datasets**: `needle` (20 precise questions), `realistic` (400 broad queries), `informed` (25 expert queries)

**Metrics**: Hit@k, MRR (Mean Reciprocal Rank), latency percentiles

**Features**:
- Auto-detects trace logs from Docker containers
- Shows full retrieval pipeline traces for failed queries
- Outputs JSON results for programmatic analysis

See [`src/plm/benchmark/README.md`](./src/plm/benchmark/README.md) for detailed usage.

## Documentation

- [Architecture](./docs/architecture/) - System design documents
- [Research](./docs/RESEARCH.md) - RAG research findings
- [Requirements](./docs/REQUIREMENTS.md) - Project requirements

## License

MIT
