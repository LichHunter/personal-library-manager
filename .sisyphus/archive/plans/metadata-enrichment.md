# Metadata Enrichment for MarkdownSemanticStrategy

## Context

### Original Request
Implement metadata enrichment for the chunking strategy to improve retrieval accuracy from 94% to 96-98%. Focus on structural metadata (NOT LLM-based enrichment).

### Interview Summary
**Key Discussions**:
- User tested chunk overlap - no improvement observed
- User said "metadata enrichment looks promising"
- User "hesitates" on parent document retriever - defer for now
- Goal is basic structural metadata capture during chunking

**Research Findings**:
- Pattern exists in `heading_paragraph.py:84-90` for proper heading stack management
- Current `markdown_semantic.py` sets `heading_path=[section.heading]` (single item, not full breadcrumb)
- Corpus metadata in `corpus_metadata_realistic.json` already has `category` field
- Current retrieval strategies DON'T use chunk metadata in scoring - metadata is captured but not leveraged

### Metis Review
**Identified Gaps** (addressed):
- Breadcrumb format: Will use `" > "` delimiter for string format (searchable)
- Position edge cases: Single-chunk documents will be "only"; split sections keep same breadcrumb
- Empty heading handling: Use document title as fallback
- Metadata will be added to `chunk.metadata` dict (not Chunk dataclass fields)

---

## Work Objectives

### Core Objective
Enrich chunks with structural metadata (heading breadcrumb, document context, position tracking) to provide better context for retrieval and debugging.

### Concrete Deliverables
- Modified `MarkdownSemanticStrategy` with full heading path tracking
- Document-level metadata (`doc_title`, `doc_type`) propagated to chunks
- Position metadata (`chunk_index`, `total_chunks`, `chunk_position`) added
- Verification that benchmark runs successfully with new metadata

### Definition of Done
- [x] `python run_benchmark.py --config config_realistic.yaml` completes without errors
- [x] Chunks contain `heading_path` as full breadcrumb (not single item)
- [x] Chunks contain `doc_title`, `doc_type` from document/corpus metadata
- [x] Chunks contain `chunk_index`, `total_chunks`, `chunk_position`

### Must Have
- Full heading path tracking following `heading_paragraph.py:84-90` pattern
- Document title and type propagation
- Position tracking (index, total, first/middle/last)
- Backward compatibility with existing benchmark

### Must NOT Have (Guardrails)
- NO LLM-based metadata extraction
- NO modifications to `Chunk` dataclass in `base.py` (use metadata dict)
- NO changes to retrieval strategies (scope is chunking only)
- NO changes to other chunking strategies (only `MarkdownSemanticStrategy`)
- NO content modification (metadata is additive only)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (benchmark framework with `run_benchmark.py`)
- **User wants tests**: Manual verification via benchmark
- **Framework**: Existing benchmark + manual inspection

### Manual Execution Verification

Each TODO includes verification procedures to run in terminal.

---

## Task Flow

```
Task 1 (heading stack) → Task 2 (doc metadata) → Task 3 (position tracking) → Task 4 (verify benchmark)
```

## Parallelization

| Task | Depends On | Reason |
|------|------------|--------|
| 1 | None | Foundation for other changes |
| 2 | 1 | Needs chunk creation to add doc metadata |
| 3 | 2 | Needs chunk collection for position tracking |
| 4 | 3 | Verifies all changes |

---

## TODOs

- [x] 1. Implement Full Heading Path Tracking

  **What to do**:
  - Add `heading_stack` list to track heading hierarchy during `_extract_sections()`
  - When encountering a heading, pop from stack while stack's top level >= current heading level
  - Push current heading to stack
  - Build `heading_path` as `" > ".join([h['title'] for h in heading_stack])`
  - Update `Section` dataclass to store full `heading_path` (list of titles)
  - Update `_create_chunk()` to use full `heading_path` from section
  
  **Must NOT do**:
  - Do NOT modify `Chunk` dataclass in `base.py`
  - Do NOT change the signature of public methods

  **Parallelizable**: NO (foundation task)

  **References**:

  **Pattern References**:
  - `poc/chunking_benchmark_v2/strategies/heading_paragraph.py:84-90` - Heading stack management pattern (pop while level >= current, then push)
  - `poc/chunking_benchmark_v2/strategies/heading_paragraph.py:89-90` - Building heading_path from stack

  **Current Implementation**:
  - `poc/chunking_benchmark_v2/strategies/markdown_semantic.py:134-189` - `_extract_sections()` method where heading extraction happens
  - `poc/chunking_benchmark_v2/strategies/markdown_semantic.py:23-29` - `Section` dataclass definition
  - `poc/chunking_benchmark_v2/strategies/markdown_semantic.py:321-345` - `_create_chunk()` where heading_path is set

  **Issue to Fix**:
  - `markdown_semantic.py:334` - Currently sets `heading_path=[section.heading]` (single item) - should be full breadcrumb

  **Acceptance Criteria**:

  **Manual Verification**:
  - [ ] Using interactive Python:
    ```bash
    cd poc/chunking_benchmark_v2
    source .venv/bin/activate
    python -c "
    from strategies import Document
    from strategies.markdown_semantic import MarkdownSemanticStrategy
    
    # Test document with nested headings
    doc = Document(
        id='test',
        title='Test Doc',
        content='''# Main
    ## Sub Section
    ### Deep Section
    Some content here.
    ## Another Sub
    More content.
    '''
    )
    strategy = MarkdownSemanticStrategy()
    chunks = strategy.chunk(doc)
    for c in chunks:
        print(f'{c.heading}: heading_path={c.heading_path}')
        print(f'  metadata.heading_path_str={c.metadata.get(\"heading_path_str\", \"NOT SET\")}')
    "
    ```
  - [ ] Expected: Chunk for "Deep Section" should have `heading_path=['Main', 'Sub Section', 'Deep Section']` or `metadata['heading_path_str']='Main > Sub Section > Deep Section'`

  **Commit**: YES
  - Message: `feat(chunking): add full heading path tracking to MarkdownSemanticStrategy`
  - Files: `poc/chunking_benchmark_v2/strategies/markdown_semantic.py`
  - Pre-commit: Manual verification above

---

- [x] 2. Add Document-Level Metadata Propagation

  **What to do**:
  - Modify `chunk()` method to extract document metadata from `Document` object
  - Pass `doc_title` (from `document.title`) to `_create_chunk()`
  - Pass `doc_type` (from `document.metadata.get('category', 'unknown')`) to `_create_chunk()`
  - Add these to `chunk.metadata` dict in `_create_chunk()`
  - Handle fallback if `document.metadata` doesn't have `category`

  **Must NOT do**:
  - Do NOT modify `Document` dataclass
  - Do NOT modify corpus metadata files
  - Do NOT add required fields that break backward compatibility

  **Parallelizable**: NO (depends on Task 1)

  **References**:

  **Pattern References**:
  - `poc/chunking_benchmark_v2/strategies/base.py:8-15` - Document dataclass with `metadata` field
  - `poc/chunking_benchmark_v2/corpus/corpus_metadata_realistic.json` - Corpus metadata with `category` field

  **Current Implementation**:
  - `poc/chunking_benchmark_v2/strategies/markdown_semantic.py:107-132` - `chunk()` method that receives Document
  - `poc/chunking_benchmark_v2/strategies/markdown_semantic.py:321-345` - `_create_chunk()` where metadata is set
  - `poc/chunking_benchmark_v2/run_benchmark.py:127-134` - Where Document is created (metadata comes from corpus)

  **Data Flow**:
  - `corpus_metadata_realistic.json` has `category` field
  - `run_benchmark.py:load_corpus()` creates Document but doesn't populate `document.metadata`
  - Need to either: (a) populate Document.metadata in run_benchmark.py, OR (b) pass category separately

  **Acceptance Criteria**:

  **Manual Verification**:
  - [ ] Using interactive Python:
    ```bash
    cd poc/chunking_benchmark_v2
    source .venv/bin/activate
    python -c "
    from strategies import Document
    from strategies.markdown_semantic import MarkdownSemanticStrategy
    
    doc = Document(
        id='test',
        title='CloudFlow API Reference',
        content='# API\\n## Auth\\nContent here.',
        metadata={'category': 'api'}
    )
    strategy = MarkdownSemanticStrategy()
    chunks = strategy.chunk(doc)
    for c in chunks:
        print(f'doc_title={c.metadata.get(\"doc_title\", \"NOT SET\")}')
        print(f'doc_type={c.metadata.get(\"doc_type\", \"NOT SET\")}')
    "
    ```
  - [ ] Expected: Each chunk should have `metadata['doc_title']='CloudFlow API Reference'` and `metadata['doc_type']='api'`

  **Commit**: YES
  - Message: `feat(chunking): propagate document title and type to chunk metadata`
  - Files: `poc/chunking_benchmark_v2/strategies/markdown_semantic.py`
  - Pre-commit: Manual verification above

---

- [x] 3. Add Position Tracking Metadata

  **What to do**:
  - After all chunks are created, iterate through and add position metadata
  - Add `chunk_index`: 0-based position in document
  - Add `total_chunks`: total number of chunks in document
  - Add `chunk_position`: "first", "middle", "last", or "only" (for single-chunk documents)
  - Handle the case of split sections (multiple chunks from same section)

  **Must NOT do**:
  - Do NOT add prev_chunk_id/next_chunk_id (defer to parent retriever)
  - Do NOT modify chunk IDs

  **Parallelizable**: NO (depends on Task 2)

  **References**:

  **Current Implementation**:
  - `poc/chunking_benchmark_v2/strategies/markdown_semantic.py:128-131` - Post-processing loop where chunk_idx is already assigned
  - This is the ideal location to add position metadata

  **Pattern Reference**:
  - `poc/chunking_benchmark_v2/strategies/heading_paragraph.py:133-137` - Shows metadata assignment pattern

  **Acceptance Criteria**:

  **Manual Verification**:
  - [ ] Using interactive Python:
    ```bash
    cd poc/chunking_benchmark_v2
    source .venv/bin/activate
    python -c "
    from strategies import Document
    from strategies.markdown_semantic import MarkdownSemanticStrategy
    
    # Multi-section document
    doc = Document(
        id='test',
        title='Test Doc',
        content='''# Section 1
    Content for section 1.
    
    # Section 2
    Content for section 2.
    
    # Section 3
    Content for section 3.
    '''
    )
    strategy = MarkdownSemanticStrategy()
    chunks = strategy.chunk(doc)
    for c in chunks:
        print(f'{c.id}: chunk_index={c.metadata.get(\"chunk_index\", \"NOT SET\")} '
              f'total_chunks={c.metadata.get(\"total_chunks\", \"NOT SET\")} '
              f'chunk_position={c.metadata.get(\"chunk_position\", \"NOT SET\")}')
    "
    ```
  - [ ] Expected: First chunk has `chunk_position='first'`, last has `'last'`, middle has `'middle'`
  - [ ] Single-chunk documents should have `chunk_position='only'`

  **Commit**: YES
  - Message: `feat(chunking): add position tracking metadata (index, total, position)`
  - Files: `poc/chunking_benchmark_v2/strategies/markdown_semantic.py`
  - Pre-commit: Manual verification above

---

- [x] 4. Verify Full Benchmark Runs Successfully

  **What to do**:
  - Run the full benchmark with realistic config
  - Verify no errors or regressions
  - Optionally inspect output to confirm metadata is captured
  - Document any changes in retrieval accuracy (informational)

  **Must NOT do**:
  - Do NOT modify benchmark code
  - Do NOT change config files

  **Parallelizable**: NO (depends on all previous tasks)

  **References**:

  **Commands**:
  - `poc/chunking_benchmark_v2/README.md` - Benchmark usage instructions
  - Config file: `poc/chunking_benchmark_v2/config_realistic.yaml`

  **Acceptance Criteria**:

  **Manual Verification**:
  - [ ] Run benchmark:
    ```bash
    cd /home/fujin/Code/personal-library-manager
    nix develop  # Or ensure proper environment
    cd poc/chunking_benchmark_v2
    source .venv/bin/activate
    python run_benchmark.py --config config_realistic.yaml
    ```
  - [ ] Expected: Benchmark completes without errors
  - [ ] Expected: Results comparable to baseline (no major regression)

  - [ ] Verify metadata in chunks (optional inspection):
    ```bash
    python -c "
    import json
    from pathlib import Path
    from strategies import Document
    from strategies.markdown_semantic import MarkdownSemanticStrategy
    
    # Load a real corpus document
    docs_dir = Path('corpus/realistic_documents')
    metadata = json.loads(Path('corpus/corpus_metadata_realistic.json').read_text())
    
    doc_meta = metadata[0]  # First document
    doc = Document(
        id=doc_meta['id'],
        title=doc_meta['title'],
        content=(docs_dir / doc_meta['filename']).read_text(),
        metadata={'category': doc_meta['category']}
    )
    
    strategy = MarkdownSemanticStrategy()
    chunks = strategy.chunk(doc)
    
    print(f'Total chunks: {len(chunks)}')
    print(f'\\nFirst chunk metadata:')
    for k, v in chunks[0].metadata.items():
        print(f'  {k}: {v}')
    print(f'\\nLast chunk metadata:')
    for k, v in chunks[-1].metadata.items():
        print(f'  {k}: {v}')
    "
    ```
  - [ ] Expected: Metadata includes `heading_path_str`, `doc_title`, `doc_type`, `chunk_index`, `total_chunks`, `chunk_position`

  **Commit**: NO (verification only, no code changes)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(chunking): add full heading path tracking to MarkdownSemanticStrategy` | markdown_semantic.py | Manual Python test |
| 2 | `feat(chunking): propagate document title and type to chunk metadata` | markdown_semantic.py | Manual Python test |
| 3 | `feat(chunking): add position tracking metadata (index, total, position)` | markdown_semantic.py | Manual Python test |
| 4 | (no commit) | - | Full benchmark run |

---

## Success Criteria

### Verification Commands
```bash
# Run benchmark
cd poc/chunking_benchmark_v2
source .venv/bin/activate
python run_benchmark.py --config config_realistic.yaml
# Expected: Completes without errors

# Verify metadata structure
python -c "
from strategies import Document
from strategies.markdown_semantic import MarkdownSemanticStrategy

doc = Document(id='t', title='Title', content='# A\n## B\nText', metadata={'category': 'test'})
chunks = MarkdownSemanticStrategy().chunk(doc)
print(chunks[0].metadata)
"
# Expected: Contains heading_path_str, doc_title, doc_type, chunk_index, total_chunks, chunk_position
```

### Final Checklist
- [x] All "Must Have" present (heading path, doc metadata, position tracking)
- [x] All "Must NOT Have" absent (no LLM, no base.py changes, no retrieval changes)
- [x] Benchmark runs successfully
- [x] Backward compatibility maintained
