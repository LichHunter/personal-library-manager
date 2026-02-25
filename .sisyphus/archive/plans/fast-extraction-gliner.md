# Fast Extraction System with GLiNER

## TL;DR

> **Quick Summary**: Build a production fast extraction pipeline that reads documents from a folder, chunks them safely for GLiNER (≤200 words), extracts software entities, outputs structured JSON grouped by document headings, and flags low-confidence documents for review.
> 
> **Deliverables**:
> - GLiNER wrapper with proper chunking and truncation detection
> - Batch processing CLI with configurable input/output paths
> - JSON output per document with heading → chunk → terms hierarchy
> - Low-confidence document filtering (copies docs with avg confidence <70% to separate folder)
> - Unit tests for chunking and extraction logic
> 
> **Estimated Effort**: Medium (2-3 days)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (chunker) → Task 2 (GLiNER wrapper) → Task 3 (document processor) → Task 4 (CLI) → Task 5 (tests)

---

## Context

### Original Request
Build a fast extraction system that:
1. Reads all files from a configurable input folder
2. Processes documents through GLiNER with proper chunking
3. Ensures no tokens are dropped (detect and handle GLiNER warnings)
4. Outputs hierarchical JSON: document → heading → chunk (text + terms)

### Interview Summary
**Key Discussions**:
- Input: Mixed markdown (.md) and plain text (.txt) files
- Output: One JSON file per document with heading hierarchy
- Model: GLiNER zero-shot (`urchade/gliner_medium-v2.1`)
- Extraction: GLiNER only (no heuristics)
- Labels: library, framework, programming language, software tool, API, database, protocol, technology
- Verification: Unit tests + manual spot-check

**Research Findings**:
- GLiNER `max_len=384` is in **words** (GLiNER's regex tokenizer), not Python words
- GLiNER regex `r"\w+(?:[-_]\w+)*|\S"` counts punctuation separately
- Technical text with punctuation expands ~30-50% → safe limit is **200 words**
- Truncation emits `UserWarning`, then silently drops tail tokens
- Must use GLiNER's `WordsSplitter` for accurate token counting

### Metis Review
**Identified Gaps** (addressed):
- GLiNER was rejected in POC-1c → User acknowledged, proceeding anyway
- Token counting must use GLiNER's tokenizer → Plan includes this
- Heading length eats into chunk budget → Will subtract heading from limit
- Chunk boundary entities → Will use 25-word overlap
- Error handling → Fail loudly on truncation warnings

**Known Limitation**: GLiNER zero-shot achieved only F1=0.518 on software entities in POC-2. User is aware and proceeding.

---

## Tokenization Reference (from POC-2)

### GLiNER's Two-Stage Tokenization

GLiNER uses a two-stage tokenization pipeline (critical for correct chunking):

**Stage 1 — Word splitting** (what `max_len=384` counts):
```python
# GLiNER's WordsSplitter regex
r"\w+(?:[-_]\w+)*|\S"
```
This produces **word-level tokens**:
- `hello-world` → 1 token (hyphenated words stay together)
- `hello world` → 2 tokens
- `"Hello,"` → 3 tokens: `"`, `Hello`, `,` (punctuation is separate!)

**Stage 2 — Subword tokenization** by DeBERTa (internal to GLiNER):
- Each word from Stage 1 may expand to multiple subword tokens
- This happens inside `DataCollator` during training
- For inference, `predict_entities()` handles this internally

### Why 200 Words is Safe

| Text Type | Python `split()` | GLiNER Tokens | Expansion |
|-----------|------------------|---------------|-----------|
| English prose | 300 words | ~350 tokens | +17% |
| **Technical markdown** | 200 words | ~280 tokens | **+40%** |
| Code-heavy text | 150 words | ~250 tokens | +67% |

**Rule**: For technical documentation, use **200 Python words max** to stay safely under 384 GLiNER tokens.

### Correct Token Counting

```python
from gliner.data_processing.tokenizer import WordsSplitter

def count_gliner_tokens(text: str) -> int:
    """Count tokens the way GLiNER counts them."""
    splitter = WordsSplitter(splitter_type="whitespace")
    return len(list(splitter(text)))

# WRONG: len(text.split())  — undercounts punctuation
# RIGHT: count_gliner_tokens(text)
```

### Training Data Format (for reference)

If fine-tuning GLiNER later, the training format is:
```json
{"tokenized_text": ["word1", "word2", ...], "ner": [[start, end, "type"], ...]}
```
- `tokenized_text`: Word-level tokens (whitespace split)
- `ner`: `[start_idx, end_idx, entity_type]` — indices are **INCLUSIVE** on both ends

---

## Chunking Strategy: Sentence-Aware Packing

### Why Sentence-Aware (Not Just Word-Based)

POC-2 training used **sentence-level samples** where each BIO sentence became one training sample. The model learned on sentence boundaries. For inference, we should respect this by:

1. **Never splitting mid-sentence** — entities rarely cross sentence boundaries
2. **Packing sentences** into chunks until hitting the token limit
3. **Preserving semantic coherence** — each chunk is a logical unit

### Algorithm: Sentence-Aware Chunking with Overlap

```python
def chunk_text_by_sentences(
    text: str, 
    max_gliner_tokens: int = 200,
    heading: str | None = None
) -> list[Chunk]:
    """
    Pack complete sentences into chunks without exceeding token limit.
    Uses sentence-level overlap to catch entities at boundaries.
    Never splits mid-sentence unless a single sentence exceeds the limit.
    """
    # Step 1: Split into sentences using pysbd
    sentences = split_into_sentences(text)
    
    # Step 2: Subtract heading budget from limit
    effective_limit = max_gliner_tokens
    if heading:
        heading_tokens = count_gliner_tokens(heading)
        effective_limit = max_gliner_tokens - heading_tokens
    
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    last_sentence_of_prev_chunk = None  # For overlap
    
    for sentence in sentences:
        sentence_tokens = count_gliner_tokens(sentence)
        
        # Case 1: Single sentence exceeds limit → force split (rare)
        if sentence_tokens > effective_limit:
            # Flush current chunk first
            if current_chunk_sentences:
                chunks.append(join_sentences(current_chunk_sentences))
                last_sentence_of_prev_chunk = current_chunk_sentences[-1]
                current_chunk_sentences = []
                current_token_count = 0
            # Word-based fallback for oversized sentence
            chunks.extend(word_based_split(sentence, effective_limit))
            last_sentence_of_prev_chunk = None  # Can't overlap word-split
            continue
        
        # Case 2: Adding this sentence would exceed limit → start new chunk
        if current_token_count + sentence_tokens > effective_limit:
            chunks.append(join_sentences(current_chunk_sentences))
            last_sentence_of_prev_chunk = current_chunk_sentences[-1]
            
            # OVERLAP: Start new chunk with last sentence of previous chunk
            if last_sentence_of_prev_chunk:
                overlap_tokens = count_gliner_tokens(last_sentence_of_prev_chunk)
                current_chunk_sentences = [last_sentence_of_prev_chunk, sentence]
                current_token_count = overlap_tokens + sentence_tokens
            else:
                current_chunk_sentences = [sentence]
                current_token_count = sentence_tokens
        else:
            # Case 3: Sentence fits → add to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(join_sentences(current_chunk_sentences))
    
    return chunks
```

### Why Sentence-Level Overlap?

Entities can span sentence boundaries (rare but possible):
```
Chunk 1: "...configure PostgreSQL." [END]
Chunk 2: [START] "This ensures data integrity..."
```

With overlap, Chunk 2 starts with "...configure PostgreSQL. This ensures..." catching any cross-boundary entities.

**Deduplication**: After extraction, dedupe entities with identical `(text, start, end)` tuples.

### Sentence Splitting

**Use `pysbd` library** (rule-based, handles edge cases robustly):

```python
from pysbd import Segmenter

# Initialize once
_segmenter = Segmenter(language="en", clean=False)

def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using pysbd (rule-based).
    Handles: URLs, abbreviations (Dr., e.g., i.e.), code paths, numbers.
    """
    return _segmenter.segment(text)
```

**Why pysbd over regex?**
| Pattern | Regex `(?<=[.!?])\s+(?=[A-Z])` | pysbd |
|---------|-------------------------------|-------|
| `github.com. See...` | ❌ Splits mid-URL | ✅ Correct |
| `e.g. React` | ❌ Incorrectly splits | ✅ Correct |
| `Dr. Smith said...` | ❌ Splits after Dr. | ✅ Correct |
| `src/main.py. This...` | ❌ Splits after .py | ✅ Correct |

**Dependency**: Add `pysbd` to pyproject.toml (MIT license, ~2KB, no model download)

### Comparison: POC-2 vs Production

| Aspect | POC-2 (Training) | Production (Inference) |
|--------|-----------------|------------------------|
| Input | BIO file with blank-line delimiters | Raw markdown/text documents |
| Strategy | 1 sentence = 1 sample | Pack sentences into chunks |
| Boundary | Blank line | `.?!` + whitespace + capital |
| Max tokens | ~92 (natural) | 200 (configured limit) |
| Truncation | None needed | Start new chunk if limit exceeded |

---

## Low Confidence Document Copying

### Purpose
Documents where GLiNER produces low-confidence extractions (<70% average confidence) should be flagged for human review or slow-system processing.

### Behavior
- Calculate **average entity confidence** per document
- If avg confidence < 70%, copy the source document to `--low-confidence-dir`
- Also save the extraction JSON to the low-confidence directory
- Log which documents were flagged

### CLI Flag
```bash
python -m plm.extraction.cli \
  --input ./docs \
  --output ./extracted \
  --low-confidence-dir ./needs-review \
  --confidence-threshold 0.7
```

### Output Structure
```
./needs-review/
├── doc1.md              # Original document (copied)
├── doc1.json            # Extraction result (for context)
├── doc2.txt
├── doc2.json
└── manifest.json        # List of flagged files with their avg confidence
```

---

## Work Objectives

### Core Objective
Create a robust fast extraction pipeline that processes documents through GLiNER without losing any tokens, outputting structured JSON grouped by heading hierarchy.

### Concrete Deliverables
- `src/plm/extraction/chunking/gliner_chunker.py` — Sentence-aware chunker for GLiNER
- `src/plm/extraction/fast/gliner.py` — GLiNER wrapper with truncation detection
- `src/plm/extraction/fast/document_processor.py` — Document → structured output
- `src/plm/extraction/fast/cli.py` — CLI for batch processing (in fast/ folder)
- `tests/extraction/test_gliner_chunking.py` — Unit tests
- JSON output files in user-specified directory
- Low-confidence documents copied to separate directory

### Definition of Done
- [ ] `python -m plm.extraction.cli --input ./docs --output ./extracted` runs without errors
- [ ] No truncation warnings during processing (checked programmatically)
- [ ] Output JSON has correct structure: document → headings → chunks → terms
- [ ] Low-confidence documents copied to separate directory with manifest.json
- [ ] All unit tests pass

### Must Have
- Chunk size ≤200 words (GLiNER tokenizer count)
- Truncation detection — fail loudly if GLiNER truncates
- Heading hierarchy preserved in output
- Character offsets in output for each chunk
- CLI with `--input` and `--output` arguments
- Low-confidence document filtering with `--low-confidence-dir` option
- Manifest file tracking flagged documents with their confidence scores

### Must NOT Have (Guardrails)
- NO heuristic extractors (GLiNER only per user request)
- NO confidence scoring or routing logic
- NO integration with slow extraction pipeline
- NO fine-tuned model (using zero-shot)
- NO entity deduplication across documents
- NO web UI or API server
- NO streaming/incremental output

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES (pytest available)
- **Automated tests**: YES (Tests-after)
- **Framework**: pytest

### Agent-Executed QA Scenarios (MANDATORY)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Python modules** | Bash (pytest) | Run test file, assert pass |
| **CLI** | Bash | Run CLI with test files, check exit code and output |
| **JSON output** | Bash (jq) | Validate schema and structure |
| **Truncation safety** | Bash (Python) | Process docs, assert no warnings caught |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation):
└── Task 1: Create GLiNER-aware chunker

Wave 2 (After Task 1):
├── Task 2: Create GLiNER extraction wrapper
└── Task 3: Create document processor

Wave 3 (After Wave 2):
└── Task 4: Create CLI

Wave 4 (After Wave 3):
└── Task 5: Write unit tests + integration test

Critical Path: Task 1 → Task 2 → Task 3 → Task 4 → Task 5
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | None |
| 2 | 1 | 3 | - |
| 3 | 1, 2 | 4 | - |
| 4 | 3 | 5 | - |
| 5 | 4 | None | - |

---

## TODOs

- [ ] 1. Create Sentence-Aware GLiNER Chunker

  **What to do**:
  - Create `src/plm/extraction/chunking/gliner_chunker.py`
  - **Sentence-aware chunking** (NOT word-based overlap):
    1. Split text into sentences using `pysbd` library (handles URLs, abbreviations, code paths)
    2. Pack sentences into chunks until hitting 200 GLiNER tokens
    3. If next sentence doesn't fit → start new chunk (never split mid-sentence)
    4. If single sentence exceeds 200 tokens → fallback to word-based split for that sentence only
    5. Include sentence-level overlap (last sentence of previous chunk starts new chunk)
  - Use GLiNER's `WordsSplitter` to count tokens accurately
  - For markdown: split by headings first, then sentence-chunk within sections
  - For plain text: treat entire doc as one section, then sentence-chunk
  - Subtract heading length from chunk budget when prepending headings
  - Register as `@register_chunker("gliner")`

  **Must NOT do**:
  - Do not use Python `split()` for word counting — use GLiNER's WordsSplitter
  - Do not split mid-sentence (except for oversized sentences)
  - Do not use word-based overlap — use sentence packing instead
  - Do not modify existing HeadingChunker (create new one)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core logic with tricky token counting requirements
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (foundation)
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/plm/extraction/chunking/heading.py` — Existing HeadingChunker pattern to follow
  - `src/plm/extraction/chunking/base.py` — Chunker ABC and registration

  **External References**:
  - GLiNER WordsSplitter: `from gliner.data_processing.tokenizer import WordsSplitter`
  - Regex pattern: `r"\w+(?:[-_]\w+)*|\S"`

  **Acceptance Criteria**:

  - [ ] `gliner_chunker.py` exists with `GLiNERChunker` class
  - [ ] Uses `WordsSplitter(splitter_type="whitespace")` for token counting
  - [ ] `get_chunker("gliner")` returns the new chunker
  - [ ] No chunk exceeds 200 GLiNER tokens (verified by test)
  - [ ] Chunks respect sentence boundaries (no mid-sentence splits)
  - [ ] Sentences are packed greedily (fit as many as possible per chunk)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Chunker produces valid-sized chunks
    Tool: Bash
    Preconditions: Package installed
    Steps:
      1. python -c "
         from plm.extraction.chunking import get_chunker
         from gliner.data_processing.tokenizer import WordsSplitter
         chunker = get_chunker('gliner')
         # Text with multiple sentences
         text = 'This is sentence one. ' * 50 + 'This is sentence two. ' * 50
         chunks = chunker.chunk(text, filename='test.txt')
         splitter = WordsSplitter(splitter_type='whitespace')
         for c in chunks:
             token_count = len(list(splitter(c.text)))
             assert token_count <= 200, f'Chunk too large: {token_count}'
         print(f'PASS: {len(chunks)} chunks, all <= 200 GLiNER tokens')
         "
      2. Assert: exit code 0
      3. Assert: output contains "PASS"
    Expected Result: All chunks within limit
    Evidence: Terminal output

  Scenario: Chunks respect sentence boundaries
    Tool: Bash
    Steps:
      1. python -c "
         from plm.extraction.chunking import get_chunker
         chunker = get_chunker('gliner')
         text = 'First sentence here. Second sentence here. Third sentence here. Fourth sentence here.'
         chunks = chunker.chunk(text, filename='test.txt')
         for c in chunks:
             # Each chunk should end with sentence-ending punctuation (or be the last chunk)
             text_stripped = c.text.strip()
             if text_stripped:
                 assert text_stripped[-1] in '.!?' or c == chunks[-1], f'Chunk does not end at sentence boundary: {text_stripped[-20:]}'
         print('PASS: All chunks respect sentence boundaries')
         "
    Expected Result: No mid-sentence splits
    Evidence: Terminal output

  Scenario: Sentences are packed greedily
    Tool: Bash
    Steps:
      1. python -c "
         from plm.extraction.chunking import get_chunker
         from gliner.data_processing.tokenizer import WordsSplitter
         chunker = get_chunker('gliner')
         splitter = WordsSplitter(splitter_type='whitespace')
         # 4 sentences of ~40 tokens each = ~160 tokens total, should fit in 1-2 chunks
         text = 'Word ' * 39 + 'end. ' + 'Word ' * 39 + 'end. ' + 'Word ' * 39 + 'end. ' + 'Word ' * 39 + 'end.'
         chunks = chunker.chunk(text, filename='test.txt')
         # Should pack multiple sentences per chunk, not 1 sentence per chunk
         assert len(chunks) <= 2, f'Expected 1-2 chunks for 160 tokens, got {len(chunks)}'
         print(f'PASS: Greedy packing works, got {len(chunks)} chunks')
         "
    Expected Result: Multiple sentences packed per chunk
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(chunking): add sentence-aware GLiNER chunker`
  - Files: `src/plm/extraction/chunking/gliner_chunker.py`
  - Pre-commit: `python -c "from plm.extraction.chunking import get_chunker; get_chunker('gliner')"`

---

- [ ] 2. Create GLiNER Extraction Wrapper

  **What to do**:
  - Create `src/plm/extraction/fast/gliner.py`
  - **Lazy singleton pattern** for model loading (not module-level):
    ```python
    _model = None
    
    def get_model() -> GLiNER:
        global _model
        if _model is None:
            _model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        return _model
    ```
  - Define labels constant: `["library", "framework", "programming language", "software tool", "API", "database", "protocol", "technology"]`
  - Implement `extract_entities(text: str, model: GLiNER | None = None) -> list[ExtractedEntity]`
    - Accept optional `model` parameter for dependency injection (testability)
    - Default to `get_model()` if not provided
  - Catch truncation warnings using `warnings.catch_warnings(record=True)`
  - **FAIL LOUDLY** if truncation warning detected (raise `TruncationError`)
  - Return entities with: text, label, confidence score, start/end offsets
  - **Deduplicate** entities with identical `(text, start, end)` tuples (for overlap handling)

  **Must NOT do**:
  - Do not load model at module level (use lazy singleton)
  - Do not truncate input (rely on chunker to provide safe-sized input)
  - Do not silently ignore warnings

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Wrapper around existing library
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `poc/poc-2-confidence-scoring/ner_models.py:185-220` — GLiNER usage pattern (but fix the char truncation!)
  - `src/plm/extraction/fast/heuristic.py` — Fast extraction module structure

  **External References**:
  - GLiNER API: `model.predict_entities(text, labels, threshold=0.3)`

  **Acceptance Criteria**:

  - [ ] `gliner.py` exists with `extract_entities()` function
  - [ ] Model loaded once at module level (not per call)
  - [ ] Returns list of entities with text, label, score, start, end
  - [ ] Raises `TruncationError` if GLiNER warns about truncation
  - [ ] Threshold set to 0.3 (configurable via constant)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Extract entities from safe-sized text
    Tool: Bash
    Steps:
      1. python -c "
         from plm.extraction.fast.gliner import extract_entities
         text = 'We use React and PostgreSQL for our TypeScript application.'
         entities = extract_entities(text)
         print(f'Found {len(entities)} entities')
         for e in entities:
             print(f'  {e.text}: {e.label} ({e.score:.2f})')
         "
      2. Assert: exit code 0
      3. Assert: output mentions at least one of: React, PostgreSQL, TypeScript
    Expected Result: Entities extracted successfully
    Evidence: Entity list in output

  Scenario: Raises on truncated input
    Tool: Bash
    Steps:
      1. python -c "
         from plm.extraction.fast.gliner import extract_entities
         try:
             text = 'word ' * 500  # Too long, should trigger warning
             extract_entities(text)
             print('FAIL: Should have raised TruncationError')
         except Exception as e:
             if 'truncat' in str(e).lower():
                 print('PASS: TruncationError raised correctly')
             else:
                 print(f'FAIL: Wrong exception: {e}')
         "
      2. Assert: output contains "PASS"
    Expected Result: Exception raised for oversized input
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(extraction): add GLiNER wrapper with truncation detection`
  - Files: `src/plm/extraction/fast/gliner.py`
  - Pre-commit: `python -c "from plm.extraction.fast.gliner import extract_entities"`

---

- [ ] 3. Create Document Processor

  **What to do**:
  - Create `src/plm/extraction/fast/document_processor.py`
  - Implement `process_document(filepath: Path, confidence_threshold: float = 0.7) -> DocumentResult`
  - **File error handling**: Wrap file reads in try/except
    - Handle `UnicodeDecodeError` (try UTF-8, fallback to latin-1)
    - Handle `FileNotFoundError`, `PermissionError`
    - Log errors and return `DocumentResult` with `error` field set
  - Read file content (handle markdown and plain text)
  - Detect headings using regex `^(#{1,6})\s+(.+?)$` for markdown
  - Use GLiNERChunker to split document into chunks
  - Extract entities from each chunk using GLiNER wrapper
  - Build hierarchical structure: headings → chunks → terms
  - **Calculate average confidence** across all entities in document:
    ```python
    # Handle zero-entity case (avoid ZeroDivisionError)
    avg_confidence = sum(e.score for e in entities) / len(entities) if entities else 0.0
    ```
  - Set `is_low_confidence = True` if avg_confidence < confidence_threshold
  - Define `DocumentResult`, `HeadingSection`, `ChunkResult` dataclasses

  **Output structure**:
  ```python
  @dataclass
  class ChunkResult:
      text: str
      terms: list[str]  # deduplicated entity texts
      entities: list[ExtractedEntity]  # full entity objects
      start_char: int
      end_char: int

  @dataclass
  class HeadingSection:
      heading: str
      level: int
      chunks: list[ChunkResult]

  @dataclass
  class DocumentResult:
      source_file: str
      headings: list[HeadingSection]
      avg_confidence: float  # Average confidence across all entities
      total_entities: int    # Total entity count
      is_low_confidence: bool  # True if avg_confidence < threshold
  ```

  **Must NOT do**:
  - Do not deduplicate entities across documents (only within chunk)
  - Do not handle directories (single file only)
  - Do not write output (just return structure)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Data transformation, moderate complexity
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 4
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `src/plm/extraction/chunking/heading.py:_build_heading_map()` — Heading detection pattern

  **Acceptance Criteria**:

  - [ ] `document_processor.py` exists
  - [ ] `process_document()` returns `DocumentResult`
  - [ ] Headings correctly detected from markdown
  - [ ] Plain text files processed as single section
  - [ ] Chunks grouped under correct headings
  - [ ] Each chunk has terms list and character offsets
  - [ ] `avg_confidence` calculated correctly (mean of all entity scores)
  - [ ] `is_low_confidence` flag set correctly based on threshold

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Process markdown with headings
    Tool: Bash
    Steps:
      1. Create test file: echo -e "# Intro\n\nWe use React.\n\n## Setup\n\nInstall PostgreSQL." > /tmp/test.md
      2. python -c "
         from pathlib import Path
         from plm.extraction.fast.document_processor import process_document
         result = process_document(Path('/tmp/test.md'))
         print(f'Sections: {len(result.headings)}')
         for h in result.headings:
             print(f'  {h.heading} (level {h.level}): {len(h.chunks)} chunks')
         assert len(result.headings) >= 2, 'Should have at least 2 sections'
         print('PASS')
         "
    Expected Result: Headings parsed correctly
    Evidence: Section structure in output

  Scenario: Process plain text without headings
    Tool: Bash
    Steps:
      1. Create test file: echo "This is plain text about Python and Docker." > /tmp/test.txt
      2. python -c "
         from pathlib import Path
         from plm.extraction.fast.document_processor import process_document
         result = process_document(Path('/tmp/test.txt'))
         assert len(result.headings) == 1, 'Should have 1 section (root)'
         print(f'Terms found: {result.headings[0].chunks[0].terms}')
         print('PASS')
         "
    Expected Result: Processed as single section
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(extraction): add document processor with heading hierarchy`
  - Files: `src/plm/extraction/fast/document_processor.py`
  - Pre-commit: `python -c "from plm.extraction.fast.document_processor import process_document"`

---

- [ ] 4. Create Batch Processing CLI with Low-Confidence Filtering

  **What to do**:
  - Create `src/plm/extraction/fast/cli.py` (in fast/ folder, not connecting to slow system yet)
  - Use `argparse` for CLI arguments:
    - `--input PATH` (required): Input directory
    - `--output PATH` (required): Output directory
    - `--low-confidence-dir PATH` (optional): Directory for low-confidence documents
    - `--confidence-threshold FLOAT` (optional, default 0.7): Threshold for low-confidence flagging
    - `--pattern GLOB` (optional, default `"**/*.md,**/*.txt"`): File patterns
    - `--extraction-threshold FLOAT` (optional, default 0.3): GLiNER entity confidence threshold
  - Recursively find all matching files
  - Process each document using `process_document()`
  - Convert `DocumentResult` to JSON and save to output directory
  - Use same filename with `.json` extension
  - **Low-confidence filtering**:
    - Calculate average entity confidence per document
    - If avg confidence < `--confidence-threshold` AND `--low-confidence-dir` is set:
      - Copy original source file to low-confidence directory
      - Also save extraction JSON there
      - Add entry to `manifest.json` in low-confidence directory
  - Print progress: `Processing file X of Y: filename`
  - Print summary: `Processed X files, extracted Y total entities, Z flagged as low-confidence`

  **Must NOT do**:
  - Do not process in parallel (keep it simple)
  - Do not overwrite existing JSON without warning
  - Do not continue silently on errors (log and fail)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard CLI scaffolding
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - Standard argparse patterns

  **Acceptance Criteria**:

  - [ ] `cli.py` exists with `main()` function
  - [ ] `--input` and `--output` are required arguments
  - [ ] `--low-confidence-dir` and `--confidence-threshold` arguments work
  - [ ] Creates output directory if it doesn't exist
  - [ ] Outputs one `.json` file per input document
  - [ ] Low-confidence docs copied to separate directory with manifest
  - [ ] Progress and summary printed to stdout (including low-confidence count)
  - [ ] Exit code 0 on success, 1 on error

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: CLI processes test directory
    Tool: Bash
    Steps:
      1. mkdir -p /tmp/test_input /tmp/test_output
      2. echo "# Test\n\nUsing React and Node.js" > /tmp/test_input/doc1.md
      3. echo "Python with Flask" > /tmp/test_input/doc2.txt
      4. python -m plm.extraction.fast.cli --input /tmp/test_input --output /tmp/test_output
      5. ls /tmp/test_output/
      6. Assert: doc1.json and doc2.json exist
    Expected Result: JSON files created
    Evidence: Directory listing

  Scenario: JSON structure is correct
    Tool: Bash
    Steps:
      1. cat /tmp/test_output/doc1.json | jq '.headings[0].heading'
      2. Assert: output is "# Test"
      3. cat /tmp/test_output/doc1.json | jq '.headings[0].chunks[0].terms'
      4. Assert: output is array (may be empty or have entities)
    Expected Result: Valid JSON structure
    Evidence: jq output

  Scenario: Low-confidence documents flagged
    Tool: Bash
    Steps:
      1. mkdir -p /tmp/test_input /tmp/test_output /tmp/needs_review
      2. echo "Random text with no recognizable entities xyz abc" > /tmp/test_input/lowconf.txt
      3. python -m plm.extraction.fast.cli --input /tmp/test_input --output /tmp/test_output --low-confidence-dir /tmp/needs_review --confidence-threshold 0.7
      4. ls /tmp/needs_review/
      5. Assert: lowconf.txt or manifest.json exists (if flagged)
    Expected Result: Low-confidence docs copied
    Evidence: Directory listing

  Scenario: Manifest contains flagged documents
    Tool: Bash
    Steps:
      1. cat /tmp/needs_review/manifest.json | jq '.documents | length'
      2. Assert: Returns number >= 0
      3. cat /tmp/needs_review/manifest.json | jq '.documents[0].avg_confidence'
      4. Assert: Returns float < 0.7 (if any flagged)
    Expected Result: Manifest tracks low-confidence docs
    Evidence: jq output
  ```

  **Commit**: YES
  - Message: `feat(fast): add batch extraction CLI with low-confidence filtering`
  - Files: `src/plm/extraction/fast/cli.py`
  - Pre-commit: `python -m plm.extraction.fast.cli --help`

---

- [ ] 5. Write Unit Tests

  **What to do**:
  - Create `tests/extraction/test_gliner_chunking.py`
  - Test GLiNERChunker:
    - Chunks don't exceed 200 GLiNER words
    - Overlap exists between consecutive chunks
    - Heading budget subtracted correctly
    - Empty input returns empty list
  - Test GLiNER wrapper:
    - Returns entities for valid input
    - Raises on oversized input
  - Test document processor:
    - Markdown headings detected
    - Plain text processed as single section
  - Test CLI (integration):
    - Processes test directory correctly
    - Output JSON has correct structure

  **Must NOT do**:
  - Do not mock GLiNER model (test with real model)
  - Do not require external test data (create inline)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard test writing
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `tests/` — Existing test patterns

  **Acceptance Criteria**:

  - [ ] Test file exists at `tests/extraction/test_gliner_chunking.py`
  - [ ] `pytest tests/extraction/test_gliner_chunking.py` passes
  - [ ] Coverage of chunker, wrapper, processor, and CLI
  - [ ] At least 10 test cases

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: All tests pass
    Tool: Bash
    Steps:
      1. pytest tests/extraction/test_gliner_chunking.py -v
      2. Assert: exit code 0
      3. Assert: output contains "passed"
    Expected Result: All tests green
    Evidence: pytest output
  ```

  **Commit**: YES
  - Message: `test(extraction): add unit tests for GLiNER extraction`
  - Files: `tests/extraction/test_gliner_chunking.py`
  - Pre-commit: `pytest tests/extraction/test_gliner_chunking.py`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(chunking): add sentence-aware GLiNER chunker` | chunking/gliner_chunker.py |
| 2 | `feat(extraction): add GLiNER wrapper with truncation detection` | fast/gliner.py |
| 3 | `feat(extraction): add document processor with heading hierarchy` | fast/document_processor.py |
| 4 | `feat(fast): add batch extraction CLI with low-confidence filtering` | fast/cli.py |
| 5 | `test(extraction): add unit tests for GLiNER extraction` | test_gliner_chunking.py |

---

## Success Criteria

### Verification Commands
```bash
# Install dependencies
cd personal-library-manager
uv sync

# Run CLI on test documents
python -m plm.extraction.fast.cli --input ./test_docs --output ./test_output

# Verify output
ls test_output/  # Should have .json files
cat test_output/example.json | jq '.headings[0].chunks[0].terms'

# Run tests
pytest tests/extraction/test_gliner_chunking.py -v
```

### Final Checklist
- [ ] CLI runs without errors on mixed markdown/text input
- [ ] No truncation warnings during processing
- [ ] JSON output has correct hierarchical structure
- [ ] All unit tests pass
- [ ] Model loads once (not per file)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GLiNER produces poor results | HIGH | Medium | User acknowledged; this is a known limitation |
| Memory issues with large docs | Low | Medium | Chunking keeps memory bounded |
| Slow processing | Medium | Low | Single-threaded is acceptable for now |

---

## Dependencies

### Infrastructure Dependencies

| Dependency | Purpose | Setup Instructions |
|------------|---------|-------------------|
| gliner==0.2.17 | Entity extraction (pin exact version) | `uv add gliner==0.2.17` |
| torch | GLiNER backend | Installed with gliner |
| pysbd | Sentence boundary detection | `uv add pysbd` |

### Data Dependencies

| Data | Source | Availability |
|------|--------|--------------|
| Test documents | User-provided | Required for manual testing |

---

*Plan Version: 1.1*
*Created: 2026-02-17*
*Updated: 2026-02-17 — Applied all Oracle review fixes (pysbd, overlap, zero-entity, lazy loading, file error handling, version pinning)*
