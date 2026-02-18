# Personal Knowledge Assistant

A local-first, NotebookLM-like system for querying a personal document corpus with grounded, cited answers.

## Vision

Build a trustworthy personal knowledge assistant that:
- Handles 1000+ documents with complex queries
- Provides synthesized answers with proper citations
- **No hallucinations** - critical requirement
- Runs locally on consumer hardware (32GB RAM, 8GB VRAM)

## Project Status

**Current Phase**: Research & Proof of Concept

| Phase | Status | Notes |
|-------|--------|-------|
| Requirements | Complete | See [REQUIREMENTS.md](./docs/REQUIREMENTS.md) |
| Architecture Design | In Progress | See [DESIGN_PLAN.md](./docs/DESIGN_PLAN.md) |
| RAG Pipeline Research | In Progress | See [RAG_PIPELINE_ARCHITECTURE.md](./docs/architecture/RAG_PIPELINE_ARCHITECTURE.md) |
| Term Extraction Research | **Complete** | POC-1 series done. Target 95/95/5 not achieved; max ~80/90/20. See [POC-1 Results](#poc-1-llm-term-extraction) |

---

## Key Research Findings

### POC-1: LLM Term Extraction (Complete)

**Status**: Complete (POC-1, POC-1b, POC-1c) | **Original Target**: 95% precision / 95% recall / 5% hallucination

The POC-1 series investigated LLM-based technical term extraction, testing vocabulary-assisted and vocabulary-free approaches.

#### Final Results

| Approach | Precision | Recall | Hallucination | F1 | Vocabulary |
|----------|-----------|--------|---------------|-----|------------|
| V6 (best with vocab) | 90.7% | 95.8% | 9.3% | 0.932 | 176 terms |
| V6 @ 50 docs | 84.2% | 92.8% | 15.8% | 0.883 | 176 terms |
| Retrieval few-shot | 81.6% | 80.6% | 18.4% | 0.811 | **0 terms** |
| SLIMER zero-shot | 84.9% | 66.0% | 15.1% | 0.743 | **0 terms** |

#### Key Findings

1. **95/95/5 NOT achievable** — benchmark ceiling is ~P=94%, R=96%, H=6% due to ground truth annotation gaps
2. **Vocabulary-free maximum: ~80/90/20** — retrieval few-shot eliminates maintenance with ~10% F1 drop
3. **GLiNER rejected** — produces garbage results for software entities, no usable signal
4. **Heuristic patterns work** — CamelCase, backticks, ALL_CAPS, dot.paths viable for fast extraction

#### Recommendation

Use **retrieval few-shot** for production (zero vocabulary maintenance). For RAG, recall matters more than matching benchmark conventions. The precision/hallucination tradeoff is acceptable.

#### Production Package

The `src/plm/extraction/` package implements:
- **Fast system**: Heuristic-based extraction (regex patterns, confidence scoring)
- **Slow system**: V6 LLM pipeline (5 stages: Extract → Ground → Filter → Validate → Postprocess)

**Full Details**: 
- [POC-1c Results](./poc/poc-1c-scalable-ner/RESULTS.md)
- [V6 Analysis](./poc/poc-1c-scalable-ner/docs/V6_RESULTS.md)
- [POC-1b Results](./poc/poc-1b-llm-extraction-improvements/RESEARCH_RESULTS.md)

---

## Project Structure

```
personal-library-manager/
├── README.md                 # This file
├── flake.nix                 # Nix configuration
├── pyproject.toml            # Python project config
│
├── src/plm/                  # Production package
│   ├── extraction/
│   │   ├── fast/             # GLiNER + YAKE extraction (cli.py, watcher.py)
│   │   └── slow/             # V6 LLM pipeline (cli.py, hybrid_ner/)
│   ├── search/
│   │   ├── retriever.py      # HybridRetriever (BM25 + semantic + RRF)
│   │   ├── components/       # BM25, semantic, embedder, enricher, expander
│   │   ├── storage/          # SQLite storage backend
│   │   └── service/          # FastAPI app, CLI, watcher, queue consumer
│   └── shared/
│       ├── llm/              # LLM providers (Anthropic, Gemini)
│       └── queue/            # Redis Streams integration
│
├── docker/
│   ├── docker-compose.full.yml   # Full pipeline with Redis
│   └── docker-compose.search.yml # Search service standalone
│
├── tests/                    # Test suite (unit, integration, search, queue)
├── data/vocabularies/        # auto_vocab.json, train_documents.json
│
├── fast-extraction/          # Runtime I/O for fast extraction
│   ├── input/                # Place documents here
│   └── output/               # JSON results appear here
│
├── slow-extraction/          # Runtime I/O for slow extraction
│   ├── input/                # Place documents here
│   ├── output/               # JSON results appear here
│   └── logs/                 # Low-confidence logs
│
├── poc/                      # Proof of Concept experiments
│   ├── poc-1c-scalable-ner/  # Scalable NER (Complete) - V6 source
│   └── ...                   # Other POCs
│
└── docs/                     # Documentation
    ├── REQUIREMENTS.md       # System requirements
    ├── DESIGN_PLAN.md        # Design phase tracker
    └── architecture/         # Architecture documents
```

## Quick Links

### Research Results

| POC | Status | Key Finding | Location |
|-----|--------|-------------|----------|
| POC-1 (1, 1b, 1c) | **Complete** | Target 95/95/5 not achieved; max ~80/90/20 vocab-free | [poc/poc-1c-scalable-ner/](./poc/poc-1c-scalable-ner/) |
| Retrieval | In Progress | Strategy comparison | [poc/retrieval_benchmark/](./poc/retrieval_benchmark/) |

### Design Documents

- [Requirements](./docs/REQUIREMENTS.md) - System requirements and success criteria
- [Design Plan](./docs/DESIGN_PLAN.md) - Development phase tracker
- [RAG Architecture](./docs/architecture/RAG_PIPELINE_ARCHITECTURE.md) - Detailed pipeline design
- [Research Notes](./docs/RESEARCH.md) - Research findings and decisions

---

## How to Run

### Prerequisites

- Nix (with flakes enabled)
- Python 3.11+
- Anthropic API access (for LLM features)
- Docker (optional, for containerized deployment)

### Development Setup

```bash
# Enter development shell (auto-creates .venv, runs uv sync)
cd personal-library-manager
direnv allow  # or: nix develop

# Verify installation
pytest tests/
```

### Running Services

The system has three services that can run **standalone** (file-based) or in **queue mode** (Redis Streams):

| Service | Purpose | Standalone | Queue Mode |
|---------|---------|------------|------------|
| Fast Extraction | GLiNER + YAKE entity extraction | Writes JSON files | Publishes to Redis |
| Slow Extraction | LLM V6 pipeline extraction | Writes JSON files | Publishes to Redis |
| Search Service | Hybrid retrieval (BM25 + semantic) | Watches directory | Consumes from Redis |

---

### Option 1: Direct Python (Development)

#### Fast Extraction (Batch)

```bash
python -m plm.extraction.fast.cli \
  --input ./documents \
  --output ./extraction-output \
  --workers 8 \
  --pattern "**/*.md,**/*.txt"
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input directory containing documents |
| `--output` | (required) | Output directory for JSON results |
| `--workers` | `1` | Parallel threads (recommended: 8) |
| `--pattern` | `**/*.md,**/*.txt` | Comma-separated glob patterns |
| `--log-file` | none | INFO+ log file |
| `--trace-file` | none | TRACE+ log file (per-chunk detail) |
| `--low-confidence-dir` | none | Copy flagged documents here |
| `--confidence-threshold` | `0.7` | Document confidence threshold |
| `--extraction-threshold` | `0.3` | GLiNER entity threshold |

#### Fast Extraction (Watch Mode)

```bash
python -m plm.extraction.fast.watcher \
  --input ./watch-input \
  --output ./extraction-output \
  --poll-interval 5 \
  --process-existing
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Directory to watch |
| `--output` | (required) | Output directory for JSON files |
| `--pattern` | `*.md,*.txt` | Glob patterns (non-recursive by default) |
| `--poll-interval` | `5.0` | Seconds between directory scans |
| `--process-existing` | false | Process existing files on startup |

#### Slow Extraction

Slow extraction is environment-variable configured (designed for Docker):

```bash
INPUT_DIR=./documents \
OUTPUT_DIR=./slow-output \
VOCAB_PATH=./data/vocabularies/auto_vocab.json \
TRAIN_DOCS_PATH=./data/vocabularies/train_documents.json \
PROCESS_ONCE=true \
python -m plm.extraction.slow.cli
```

#### Search Service

```bash
INDEX_PATH=./search-index \
WATCH_DIR=./extraction-output \
uv run uvicorn plm.search.service.app:app --port 8000
```

Query the service:

```bash
# CLI tool
plm-query "What is Kubernetes?" --url http://localhost:8000 --k 5

# Or direct HTTP
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Kubernetes?", "k": 5}'
```

---

### Option 2: Docker Compose (Production)

#### Build Docker Images

```bash
# Build all images (from repo root)
nix build .#fast-extraction-docker && docker load < result
nix build .#slow-extraction-docker && docker load < result
nix build .#search-service-docker && docker load < result
```

#### Standalone Mode (File-Based)

Each service has its own `docker-compose.yml` for standalone operation:

```bash
# Fast extraction (batch processing)
mkdir -p fast-extraction/input fast-extraction/output
cp my-documents/*.md fast-extraction/input/
docker compose -f src/plm/extraction/fast/docker-compose.yml up

# Slow extraction
mkdir -p slow-extraction/input slow-extraction/output
cp my-documents/*.md slow-extraction/input/
docker compose -f src/plm/extraction/slow/docker-compose.yml up

# Search service (with directory watcher)
docker compose -f docker/docker-compose.search.yml up
```

#### Queue Mode (Full Pipeline)

Use `docker/docker-compose.full.yml` for the complete pipeline with Redis:

```bash
# Start Redis + Search service (queue consumer mode)
docker compose -f docker/docker-compose.full.yml up -d redis search-service

# Start with extraction services
docker compose -f docker/docker-compose.full.yml --profile extraction up -d

# Process documents through fast extraction
cp document.md fast-extraction/input/

# Watch the pipeline flow
docker compose -f docker/docker-compose.full.yml logs -f
```

---

### Environment Variables Reference

#### Queue Configuration (All Services)

| Variable | Default | Description |
|----------|---------|-------------|
| `QUEUE_ENABLED` | `false` | Enable Redis Streams integration |
| `QUEUE_URL` | `redis://localhost:6379` | Redis connection URL |
| `QUEUE_STREAM` | `plm:extraction` | Stream name for messages |

#### Fast Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `QUEUE_ENABLED` | `false` | Publish to queue instead of files |

#### Slow Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/data/input` | Input directory to watch |
| `OUTPUT_DIR` | `/data/output` | Output directory for JSON |
| `LOG_DIR` | `/data/logs` | Log directory |
| `VOCAB_PATH` | `/data/vocabularies/auto_vocab.json` | Vocabulary file path |
| `TRAIN_DOCS_PATH` | `/data/vocabularies/train_documents.json` | Training docs for FAISS |
| `POLL_INTERVAL` | `30` | Watch interval in seconds |
| `PROCESS_ONCE` | `false` | Process existing and exit |
| `DRY_RUN` | `false` | Skip writing output |

#### Search Service

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEX_PATH` | `/data/index` | SQLite + BM25 index directory |
| `WATCH_DIR` | none | Directory to watch (standalone mode) |
| `PROCESS_EXISTING` | `false` | Process existing files on start |

#### Authentication

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Direct API key for Claude |
| `OPENCODE_AUTH_PATH` | Path to OpenCode OAuth `auth.json` |

---

### API Endpoints (Search Service)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Execute search query |
| `/health` | GET | Service health (`ready`/`starting`) |
| `/status` | GET | Detailed index statistics |

**POST /query** request body:
```json
{
  "query": "What is Kubernetes?",
  "k": 5,
  "use_rewrite": false
}
```

---

### Example Workflows

#### Workflow 1: Quick Local Test

```bash
# 1. Extract entities from documents
python -m plm.extraction.fast.cli \
  --input ./my-docs --output ./extracted --workers 4

# 2. Start search service watching extracted output
INDEX_PATH=./index WATCH_DIR=./extracted \
uv run uvicorn plm.search.service.app:app --port 8000

# 3. Query
plm-query "deployment strategies" --k 10
```

#### Workflow 2: Continuous Pipeline (Docker)

```bash
# 1. Start full pipeline
docker compose -f docker/docker-compose.full.yml --profile extraction up -d

# 2. Drop documents into input directory
cp new-document.md fast-extraction/input/

# 3. Document flows: fast-extraction → Redis queue → search-service → indexed

# 4. Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "new document topic", "k": 5}'
```

---

## Running POCs

Each POC has its own README with setup and execution instructions. See [poc/README.md](./poc/README.md) for guidelines.

```bash
# Navigate to specific POC
cd poc/poc-1c-scalable-ner
uv sync
source .venv/bin/activate
```

---

## Core Principles

1. **No Hallucinations** - Critical requirement, especially no fake citations
2. **Trustworthy Answers** - Must be reliable enough to use without verifying everything
3. **Local-First** - Designed for consumer hardware
4. **Grounded Responses** - Every claim backed by source citations

---

*Last Updated: 2026-02-18*
