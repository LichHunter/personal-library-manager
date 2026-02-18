# Draft: Fast Extraction System with GLiNER

## Requirements (confirmed)

### Input
- **Source**: Configurable path (CLI argument)
- **Formats**: Mixed - Markdown (.md) and plain text (.txt)
- **Processing**: Read all files in folder recursively

### Output
- **Format**: One JSON file per document
- **Location**: Configurable path (CLI argument)
- **Structure**:
```json
{
  "source_file": "path/to/doc.md",
  "headings": [
    {
      "heading": "# Introduction",
      "level": 1,
      "chunks": [
        {
          "text": "chunk text...",
          "terms": ["React", "TypeScript", "API"],
          "start_char": 0,
          "end_char": 250
        }
      ]
    }
  ]
}
```

### Extraction
- **Method**: GLiNER only (no heuristics)
- **Model**: `urchade/gliner_medium-v2.1` (zero-shot)
- **Labels**: library, framework, programming language, software tool, API, database, protocol, technology

### Chunking Strategy
- **Max chunk size**: 200-250 words (safe under GLiNER's 384 word limit)
- **Overlap**: 25-50 words at chunk boundaries to catch split entities
- **Heading awareness**: Chunks grouped under their parent heading
- **Plain text handling**: Treat entire document as one "section" if no headings

### Verification
- **Unit tests**: Chunking logic, GLiNER wrapper, JSON output structure
- **Manual**: Spot-check on real documents

## Technical Decisions

### GLiNER Token Safety
- GLiNER's `max_len=384` is in WORDS, not characters
- Safe chunk size: 200-250 words provides margin for subword expansion
- Must detect truncation warnings and fail loudly (not silently lose entities)

### Existing Code to Leverage
- `src/plm/extraction/chunking/HeadingChunker` - adapt for 200-word max
- `poc/poc-2-confidence-scoring/ner_models.py` - GLiNER usage patterns (but fix the wrong 512-char truncation!)

### New Code to Create
- `src/plm/extraction/fast/gliner.py` - GLiNER wrapper with proper chunking
- `src/plm/extraction/cli.py` - CLI for batch processing
- Tests for chunking and extraction

## Scope Boundaries

### INCLUDE
- Read markdown and plain text files
- Chunk documents respecting GLiNER limits
- Extract entities using GLiNER zero-shot
- Group chunks by heading hierarchy
- Output JSON per document
- CLI with configurable input/output paths
- Unit tests for core logic

### EXCLUDE
- Fine-tuned GLiNER model (using zero-shot)
- Heuristic extractors (GLiNER only per user request)
- Confidence scoring / routing logic
- Integration with slow extraction pipeline
- Web UI or API server
- Entity deduplication across documents

## Open Questions
- None remaining - all clarified during interview
