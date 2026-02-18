# Draft: Production Pipeline Service + Manual Testing

## Requirements (confirmed)

### Core Architecture
- **Pipeline as a RUNNING PROCESS** - HybridRetriever wrapped in HTTP service
- **Communication**: HTTP API (FastAPI)
- **Data flow**: Pipeline PULLS from watched directory (fast system writes output there)
- **Query interface**: CLI tool sends HTTP requests to running pipeline

### Flow
```
[Fast System] 
    → process_document() → writes JSON to watched directory

[Pipeline Service] (HTTP API + directory watcher)
    → watches directory, auto-ingests new files
    → builds/updates indexes (BM25 + semantic)
    → serves queries via HTTP POST /query

[CLI Tool]
    → sends HTTP requests to pipeline service
    → displays results for manual inspection
```

### Testing Approach
- **NOT a pytest** - Manual testing workflow
- Use full corpus (200 K8s docs) + needle questions (20 questions)
- No accuracy assertions - just report results for inspection
- **CRITICAL**: Call production code directly - no mocking

## Research Findings

### Critical Gap Identified
The existing regression test (`tests/search/test_regression_accuracy.py`) uses:
- `MarkdownSemanticStrategy` (POC chunking)
- `spaCy` NER

But PRODUCTION uses:
- `GLiNERChunker` 
- `GLiNER` NER
- `document_processor.process_document()`

**There is NO test exercising the TRUE production path:**
```
process_document(filepath) 
  → document_result_to_chunks(doc) 
  → HybridRetriever.batch_ingest(docs) 
  → HybridRetriever.retrieve(query)
```

### Reference Test Pattern (from modular POC)
The `test_exact_replication.py` does:
1. Mock query rewriting for deterministic results
2. Load Kubernetes sample docs (first 3 from `kubernetes_sample_200/`)
3. Chunk with `MarkdownSemanticStrategy`
4. Index with both original and modular implementations
5. Compare: enriched contents, embeddings, retrieval results

### Production Code Path
1. **Fast System**: `src/plm/extraction/fast/document_processor.py`
   - `process_document(filepath)` → `DocumentResult`
   - Uses `GLiNERChunker`, YAKE keywords, GLiNER entities

2. **Adapter**: `src/plm/search/adapters/gliner_adapter.py`
   - `document_result_to_chunks(doc)` → `list[dict]`

3. **Main Pipeline**: `src/plm/search/retriever.py`
   - `HybridRetriever.batch_ingest(documents)`
   - `HybridRetriever.retrieve(query, k=5)`

### Test Data Available
- Corpus: `poc/chunking_benchmark_v2/corpus/kubernetes_sample_200/` (200 .md files)
- Needle questions: `poc/chunking_benchmark_v2/corpus/needle_questions.json` (20 questions)
- Informed questions: `poc/modular_retrieval_pipeline/corpus/informed_questions.json` (50 questions)

### Existing Script Reference
`scripts/run_production_regression.py` - This IS the canonical production path but as a script, not a test.

## Decisions Made

### Architecture
- **HTTP API**: FastAPI service wrapping HybridRetriever
- **Directory watching**: Pipeline watches directory for fast system output
- **Auto-ingestion**: New files automatically ingested and indexed
- **Query endpoint**: POST /query for search requests

### Test Data
- **Corpus**: Full 200 K8s docs from `kubernetes_sample_200/`
- **Questions**: 20 needle questions from `needle_questions.json` (for manual testing)

### No Middleware - CONFIRMED
- Direct calls to production classes (`HybridRetriever`)
- Real embeddings (bge-base-en-v1.5)
- Real BM25 index
- Real queries via HTTP - no mocking

## Technical Decisions

### Service Stack
- FastAPI for HTTP API
- watchdog (or similar) for directory watching
- JSON format for inter-process communication

### Endpoints Needed
- POST /query - Send query, get results
- GET /status - Check pipeline health, index stats
- POST /ingest (optional) - Manual ingestion trigger

### Data Format
Fast system output format (what pipeline watches for):
- **DocumentResult serialized to JSON** (existing format from process_document())
- Pipeline uses existing gliner_adapter.py to convert
- **NO CHANGES to fast system output**

### Persistence
- **Index persisted to disk/Docker volume** (not in-memory)
- Survives restarts
- Uses existing SQLite + BM25 index persistence in HybridRetriever

## CRITICAL GAPS IDENTIFIED

### 1. Query Rewriting - MISSING FROM PRODUCTION (MUST FIX)
- POC has `QueryRewriter` component using Claude Haiku for query enrichment
- Production `HybridRetriever` only has `QueryExpander` (rule-based, NOT LLM)
- `QueryRewriter` is mentioned in pipeline.py docstring but NEVER IMPLEMENTED!
- **MUST DO**: Port QueryRewriter to `src/plm/search/components/query_rewriter.py`
- **MUST DO**: Use shared LLM module (`src/plm/shared/llm/base.py:call_llm()`)
- **MUST DO**: Make query rewriting OPTIONAL (configurable flag to enable/disable)
- **MUST DO**: Integrate into HybridRetriever.retrieve() with optional parameter

### 2. LLM Module Usage - CONFIRMED
- POC uses local `poc/.../utils/llm_provider.py`
- Production MUST use shared `src/plm/shared/llm/base.py:call_llm()`
- Same function signature - drop-in replacement

### 3. Test Strategy - Direct Comparison with POC
- **Comparison method**: Same chunk IDs (order can differ)
- Call API directly, get results, compare with POC output
- For deterministic comparison: mock query rewriting (same as POC test does)
- Report differences with explanation

## Scope Boundaries

### INCLUDE
1. **Port QueryRewriter to production** (from POC, using shared LLM module)
2. **Make Haiku query enrichment optional** (flag to enable/disable)
3. FastAPI service wrapping HybridRetriever
4. Directory watcher for auto-ingestion
5. Query endpoint for manual testing (with/without query rewriting)
6. CLI tool to send queries and display results
7. **Direct comparison test with POC** (same chunk IDs verification)
8. Integration with existing fast system output

### EXCLUDE
- Authentication/authorization
- Multi-user support
- Production deployment concerns (just local testing)
- Automated test assertions (manual inspection)
