# Fast Extraction System

Zero-LLM entity and keyword extraction using GLiNER (zero-shot NER) and YAKE (keyword extraction). Processes documents locally on CPU with no API calls.

## Pipeline

1. **Chunking**: Split document by headings, then pack sentences into chunks of ≤200 GLiNER tokens (via `pysbd` sentence segmentation)
2. **Entity extraction**: GLiNER `urchade/gliner_medium-v2.1` zero-shot NER with 8 entity labels (library, framework, programming language, software tool, API, database, protocol, technology)
3. **Keyword extraction**: YAKE unsupervised keyword extraction (top 10, bigrams, dedup threshold 0.9)
4. **Confidence scoring**: Average entity confidence per document, low-confidence flagging

## Quick Start

```bash
# Build Docker image (from repo root)
nix build .#fast-extraction-docker
docker load < result

# Process documents (from repo root)
docker compose -f src/plm/extraction/fast/docker-compose.yml up fast-extraction
```

## Building

```bash
# Build Docker image
nix build .#fast-extraction-docker
docker load < result
# Output: Loaded image: plm-fast-extraction:0.1.0

# Build Python venv only (for development)
nix build .#fast-extraction
```

## Usage

### Docker Compose

Place documents in `fast-extraction/input/` (from repo root):

```bash
# Default: 8 workers, full logging, low-confidence flagging
docker compose -f src/plm/extraction/fast/docker-compose.yml up fast-extraction
# Results appear in fast-extraction/output/

# Single worker mode (for debugging)
docker compose -f src/plm/extraction/fast/docker-compose.yml --profile single up fast-extraction-single
```

### CLI Directly

```bash
python -m plm.extraction.fast.cli \
  --input /path/to/documents \
  --output /path/to/output \
  --workers 8 \
  --pattern "**/*.md,**/*.txt" \
  --log-file extraction.log \
  --trace-file extraction.trace.log \
  --low-confidence-dir /path/to/flagged
```

### Docker Run

```bash
docker run --rm \
  -v ./documents:/data/input:ro \
  -v ./output:/data/output \
  plm-fast-extraction:0.1.0 \
  --input=/data/input --output=/data/output --workers=8
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input directory containing documents |
| `--output` | (required) | Output directory for JSON results |
| `--workers` | `1` | Parallel threads (recommended: 8, see below) |
| `--pattern` | `**/*.md,**/*.txt` | Comma-separated glob patterns |
| `--log-file` | none | INFO+ log file (readable summary) |
| `--trace-file` | none | TRACE+ log file (per-chunk detail) |
| `--low-confidence-dir` | none | Copy flagged documents + JSON here |
| `--confidence-threshold` | `0.7` | Below this avg confidence, flag document |
| `--extraction-threshold` | `0.3` | GLiNER entity confidence threshold |

## Threading Model

GLiNER inference is ~97% of processing time and partially releases the GIL during PyTorch ops. Threading helps:

| Workers | Torch Threads | Speed (200 K8s docs, 16-core CPU) |
|---------|---------------|-------------------------------------|
| 1 | 16 | 3.2 s/doc (baseline) |
| 4 | 4 | 2.1 s/doc (1.5x) |
| 8 | 2 | 2.0 s/doc (1.6x, best) |

Torch threads are auto-set to `max(1, cpu_count // workers)`. RAM: ~1.3GB base (model) + ~25MB per extra worker. Diminishing returns beyond 8 workers.

## Output Format

Each document produces a JSON file:

```json
{
  "source_file": "/data/input/concepts_architecture_leases.md",
  "headings": [
    {
      "heading": "## Leases",
      "level": 2,
      "chunks": [
        {
          "text": "Chunk text content...",
          "terms": ["Kubernetes", "coordination.k8s.io"],
          "entities": [
            {"text": "Kubernetes", "label": "technology", "score": 0.85, "start": 0, "end": 10}
          ],
          "keywords": ["concept weight", "API Group", "Leases api"],
          "start_char": 0,
          "end_char": 412
        }
      ]
    }
  ],
  "avg_confidence": 0.72,
  "total_entities": 15,
  "is_low_confidence": false,
  "error": null
}
```

## Feeding Output to Search

The search system ingests fast extraction output directly:

```python
from plm.search.adapters import load_extraction_directory
from plm.search.retriever import HybridRetriever

documents = load_extraction_directory("/path/to/extraction/output")
retriever = HybridRetriever("index.db", "bm25_index")
retriever.batch_ingest(documents, batch_size=256)
```

Or via the search service's directory watcher, which auto-ingests new JSON files.

## GLiNER Model

- **Model**: `urchade/gliner_medium-v2.1` (~500MB)
- **First run**: Downloads model from HuggingFace (cached for subsequent runs)
- **Docker**: Model cache persisted via `plm-gliner-models` Docker volume
- **Offline mode**: Set `HF_HUB_OFFLINE=1` (default in Docker) after first download

## Directory Structure

```
fast/
├── cli.py                 # CLI entrypoint (batch processing, threading)
├── document_processor.py  # Document → chunks → entities + keywords
├── gliner.py              # GLiNER wrapper (lazy loading, dedup, truncation detection)
├── heuristic.py           # Regex-based extractors (CamelCase, backticks, etc.)
├── confidence.py          # Confidence scoring for routing
├── flake.nix              # Nix build definition
├── docker-compose.yml     # Docker Compose services
└── README.md              # This file
```
