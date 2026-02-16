# Slow Extraction System

LLM-powered technical term extraction using the V6 pipeline from POC-1c. This is an **exact replica** of the POC-1c V6 extraction strategy, achieving 91-93% precision/recall on the SO NER benchmark.

## V6 Pipeline Overview

The V6 strategy uses retrieval-augmented few-shot prompting with FAISS indexing:

1. **Retrieval**: Find similar documents from 741 training examples using sentence-transformers
2. **Few-shot extraction**: Use retrieved examples as context for LLM extraction
3. **Candidate verification**: Validate extracted terms against document context
4. **Span expansion**: Expand partial matches to full term boundaries
5. **Deduplication**: Remove overlapping and redundant terms

## Quick Start

```bash
# Build Docker image (from repo root)
nix build .#slow-extraction-docker
docker load < result

# Process documents once (from repo root)
docker compose -f src/plm/extraction/slow/docker-compose.yml up slow-extraction

# Or watch mode (continuous processing)
docker compose -f src/plm/extraction/slow/docker-compose.yml --profile watch up slow-extraction-watch
```

## Building

### Prerequisites

- Nix with flakes enabled
- Docker

### Build Commands

```bash
# Build Docker image (creates result symlink)
# IMPORTANT: Run from repo root, not from this directory
nix build .#slow-extraction-docker

# Load into Docker
docker load < result
# Output: Loaded image: plm-slow-extraction:0.1.0

# Build Python venv only (for development)
nix build .#slow-extraction
```

## Usage

### Docker Compose (Recommended)

The `docker-compose.yml` in this directory defines two services:

**Process Once Mode** - Process existing files and exit:
```bash
# Place documents in slow-extraction/input/ (from repo root)
docker compose -f src/plm/extraction/slow/docker-compose.yml up slow-extraction
# Results appear in slow-extraction/output/
```

**Watch Mode** - Continuously monitor for new documents:
```bash
docker compose -f src/plm/extraction/slow/docker-compose.yml --profile watch up slow-extraction-watch
```

### Using API Key Instead of OpenCode Auth

Edit `docker-compose.yml` to use an API key:

```yaml
environment:
  - ANTHROPIC_API_KEY=sk-ant-...
  # Remove OPENCODE_AUTH_PATH
```

Or override at runtime:
```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose -f src/plm/extraction/slow/docker-compose.yml up slow-extraction
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/data/input` | Directory to watch for documents |
| `OUTPUT_DIR` | `/data/output` | Output directory for JSON results |
| `LOG_DIR` | `/data/logs` | Log directory |
| `VOCAB_PATH` | `/data/vocabularies/auto_vocab.json` | Path to auto_vocab.json |
| `TRAIN_DOCS_PATH` | `/data/vocabularies/train_documents.json` | Path to training documents for FAISS index |
| `PROCESS_ONCE` | `false` | Process existing files and exit |
| `DRY_RUN` | `false` | Dry run mode (no output written) |
| `POLL_INTERVAL` | `30` | Polling interval in seconds (watch mode) |

### Authentication

**Option 1: OpenCode OAuth** (default)
- Mount auth file: `-v ~/.local/share/opencode:/auth:ro`
- Set path: `-e OPENCODE_AUTH_PATH=/auth/auth.json`

**Option 2: API Key**
- Set environment variable: `-e ANTHROPIC_API_KEY=sk-ant-...`

### Volume Mounts

| Mount Point | Description | Mode |
|-------------|-------------|------|
| `/data/input` | Input documents (plain text) | read-only |
| `/data/output` | Output JSON results | read-write |
| `/data/logs` | Log files | read-write |
| `/data/vocabularies` | Vocabulary + training data | read-only |
| `/auth` | OpenCode auth directory | read-only |

### Required Data Files

The following files must be present in `/data/vocabularies`:

| File | Description |
|------|-------------|
| `auto_vocab.json` | Seed vocabulary with entity terms |
| `train_documents.json` | 741 training documents for FAISS retrieval |

## Output Format

Each processed document produces a JSON file:

```json
{
  "file": "document.md",
  "processed_at": "2026-02-16T12:00:00Z",
  "chunks": [
    {
      "text": "Document text with\nmultiline content",
      "chunk_index": 0,
      "heading": "Section Title",
      "terms": [
        {
          "term": "Kubernetes",
          "confidence": 0.95,
          "level": "HIGH",
          "sources": ["taxonomy", "heuristic"]
        }
      ]
    }
  ],
  "stats": {
    "total_chunks": 1,
    "total_terms": 5,
    "high_confidence": 3,
    "medium_confidence": 2,
    "low_confidence": 0
  }
}
```

### Confidence Levels

| Level | Confidence Range | Description |
|-------|------------------|-------------|
| HIGH | ≥ 0.8 | Strong signal from multiple sources |
| MEDIUM | 0.5 - 0.8 | Moderate confidence, validated by LLM |
| LOW | < 0.5 | Weak signal, logged separately |

### Low-Confidence Log

Terms below threshold are logged to `/data/logs/low_confidence.jsonl`:

```jsonl
{"file": "doc.md", "term": "ambiguous", "confidence": 0.3, "level": "LOW", "context": "..."}
```

## V6 Pipeline Details

The V6 strategy (`strategy_v6` in `hybrid_ner.py`) implements:

1. **FAISS Index Building**: Embeds 741 training documents using `all-MiniLM-L6-v2`
2. **Retrieval**: For each input document, retrieves k=3 most similar training examples
3. **Few-shot Extraction**: Uses retrieved examples as context for Claude LLM extraction
4. **Candidate Verification**: `use_candidate_verify=True` validates each term in context
5. **Term Index Scoring**: Uses entity frequency statistics from training data
6. **Span Processing**: Expansion, deduplication, subspan suppression

### Performance (SO NER Benchmark)

| Metric | Value |
|--------|-------|
| Precision | 91.3% |
| Recall | 93.3% |
| F1 Score | 0.923 |

## Directory Structure

```
slow-extraction/
├── input/           # Place documents here (plain text, not markdown)
├── output/          # JSON results appear here
└── logs/            # Log files

data/vocabularies/
├── auto_vocab.json           # Seed terms vocabulary (3276 terms)
├── train_documents.json      # Training corpus (741 docs) for FAISS
└── term_index.json           # Entity frequency index (9877 terms)
```

## Development

### Verify Extraction

```bash
# Test with a document from the SO NER dataset (from repo root)
python3 -c "
import json
with open('poc/poc-1c-scalable-ner/artifacts/test_documents.json') as f:
    docs = json.load(f)
with open('slow-extraction/input/test.txt', 'w') as f:
    f.write(docs[0]['text'])
"

# Run extraction
docker compose -f src/plm/extraction/slow/docker-compose.yml up slow-extraction

# Check results
cat slow-extraction/output/test.json
```

## Troubleshooting

### Vocabulary files not found

```
Fatal error: vocab_path not found: /data/vocabularies/auto_vocab.json
```

**Solution**: Ensure `/data/vocabularies` is mounted and contains:
- `auto_vocab.json`
- `train_documents.json`

### No terms extracted

Check:
1. Document contains technical content (code, library names, etc.)
2. Input is plain text (not markdown with code blocks)
3. LLM API is accessible (check auth/API key)

### Container exits immediately

For watch mode, set `PROCESS_ONCE=false`. Default is `true` (process and exit).

### Model loading slow on first run

The first run downloads the `all-MiniLM-L6-v2` model (~90MB). Subsequent runs use the cached model.

## Implementation Notes

This Docker image contains an **exact replica** of the POC-1c V6 extraction pipeline:
- `src/plm/extraction/v6/hybrid_ner.py` - Main extraction logic
- `src/plm/extraction/v6/retrieval_ner.py` - FAISS retrieval
- `src/plm/extraction/v6/scoring.py` - Term scoring
- `src/plm/extraction/cli_v6.py` - Docker entrypoint

The code was copied from `poc/poc-1c-scalable-ner/` with import paths adjusted for the package structure. Results are verified to be **100% identical** to POC-1c when using the same input text.
