# Hierarchical Storage Migration Plan

## TL;DR

> **Problem**: Structured metadata (keywords, entities) is serialized to string and lost. No document/heading hierarchy.
>
> **Solution**: Add hierarchy (documents → headings → chunks) with proper JSON metadata columns. Migrate existing data without re-ingestion.
>
> **Estimated Effort**: Medium
> **Breaking Changes**: None (default behavior unchanged)

---

## Current State (Problems)

### 1. Metadata Loss
```python
# enricher.py - keywords/entities become a string
enriched_content = "kubernetes, autoscaler | Kubernetes\n\nOriginal content"
# Structured data is GONE after this point
```

### 2. Flat Storage
```
documents
└── chunks (no grouping by heading)
```

### 3. No Aggregation
- Can't search at document level
- Can't search at heading/section level
- Only chunk-level retrieval

---

## Target State

### Hierarchy
```
documents
├── id, source_file, created_at
├── embedding (aggregated from all chunks)
├── keywords_json (union of all chunk keywords)
└── entities_json (union of all chunk entities)
    │
    └── headings
        ├── id, doc_id, heading_text, heading_level
        ├── embedding (aggregated from child chunks)
        ├── keywords_json (union of child keywords)
        └── entities_json (union of child entities)
            │
            └── chunks
                ├── id, heading_id, content, enriched_content
                ├── embedding
                ├── keywords_json (structured, not string)
                ├── entities_json (structured, not string)
                ├── start_char, end_char
                └── chunk_index (order within heading)
```

### Key Changes
1. **New `headings` table** - intermediate level
2. **JSON columns** - `keywords_json`, `entities_json` on all three tables
3. **Aggregated embeddings** - mean of child embeddings at each level
4. **Preserve `enriched_content`** - for backward compatibility

---

## Schema Migration

### New Tables

```sql
-- New headings table
CREATE TABLE headings (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    heading_text TEXT NOT NULL,
    heading_level INTEGER,  -- 1 for #, 2 for ##, etc.
    start_char INTEGER,
    end_char INTEGER,
    embedding BLOB,
    keywords_json TEXT,     -- JSON array
    entities_json TEXT      -- JSON object {label: [texts]}
);

-- Add to existing tables
ALTER TABLE documents ADD COLUMN embedding BLOB;
ALTER TABLE documents ADD COLUMN keywords_json TEXT;
ALTER TABLE documents ADD COLUMN entities_json TEXT;

ALTER TABLE chunks ADD COLUMN heading_id TEXT REFERENCES headings(id);
ALTER TABLE chunks ADD COLUMN keywords_json TEXT;
ALTER TABLE chunks ADD COLUMN entities_json TEXT;
ALTER TABLE chunks ADD COLUMN chunk_index INTEGER;
```

---

## Migration Strategy (Recover from Source)

**Key insight**: Don't parse `enriched_content` - recover from original source data.

### Data Sources (in priority order)

1. **Redis Streams** (`plm:extraction`) - if messages still exist
2. **JSON output files** - if fast-extraction saved them
3. **Reprocess** - send documents through fast-extraction again (NO lossy fallback)

### Queue Message Structure (Full Data Available!)

```python
class HeadingSection(TypedDict):
    heading: str
    level: int              # ← HEADING LEVEL!
    chunks: list[Chunk]

class Chunk(TypedDict):
    text: str
    terms: list[str]        # ← TERMS
    entities: list[Entity]  # ← FULL ENTITIES
    keywords: list[str]     # ← KEYWORDS
    start_char: int
    end_char: int

class Entity(TypedDict):
    text: str
    label: str              # ← ORG, PRODUCT, etc. PRESERVED!
    score: float
```

### Where Data Gets Lost (Bug to Fix)

```python
# retriever.py:batch_ingest_documents() - line ~243
keywords = chunk.get("keywords", [])   # ✓ available
entities = chunk.get("entities", {})   # ✓ available

enriched = self.enricher.process({...})  # → string

self.storage.add_chunk(
    enriched_content=enriched,  # ← only string stored
    # keywords NOT passed
    # entities NOT passed
)
```

### Migration Steps

**Step 0: Backup original data**
```bash
# Backup SQLite database
cp /path/to/index.db /path/to/index.db.backup.$(date +%Y%m%d_%H%M%S)

# Backup Redis stream to file
docker exec plm-redis redis-cli XRANGE plm:extraction - + > redis_extraction_backup.txt

# Or dump full Redis RDB
docker exec plm-redis redis-cli BGSAVE
docker cp plm-redis:/data/dump.rdb ./dump.rdb.backup
```

**Step 1: Check Redis for existing messages**
```python
# Check if messages exist in Redis stream
redis_client.xlen("plm:extraction")
redis_client.xrange("plm:extraction", "-", "+", count=5)
```

**Step 2: Re-process from source (if available)**
- Read messages from Redis or JSON files
- Already have full structured data with entity labels
- Match to existing chunks by (doc_id, start_char, end_char)
- Update chunks with keywords_json, entities_json

**Step 3: Build heading records (with deduplication)**
```python
def create_heading_from_source(doc_id: str, section: HeadingSection) -> dict:
    chunks = section["chunks"]
    
    # Aggregate keywords (dedupe via set, case-insensitive)
    all_keywords: set[str] = set()
    seen_keywords_lower: set[str] = set()
    for c in chunks:
        for kw in c.get("keywords", []):
            if kw.lower() not in seen_keywords_lower:
                all_keywords.add(kw)
                seen_keywords_lower.add(kw.lower())
    
    # Aggregate entities (dedupe by label + text, case-insensitive)
    all_entities: dict[str, set[str]] = defaultdict(set)
    seen_entities: dict[str, set[str]] = defaultdict(set)  # label -> lowercase texts
    for c in chunks:
        for entity in c.get("entities", []):
            label = entity["label"]
            text = entity["text"]
            if text.lower() not in seen_entities[label]:
                all_entities[label].add(text)
                seen_entities[label].add(text.lower())
    
    return {
        'id': f"{doc_id}_h{hash(section['heading'])}",
        'doc_id': doc_id,
        'heading_text': section["heading"],
        'heading_level': section.get("level", 0),
        'keywords_json': json.dumps(sorted(all_keywords)),
        'entities_json': json.dumps({k: sorted(v) for k, v in all_entities.items()}),
    }
```

**Step 4: Aggregate to document level (with deduplication)**
```python
def create_document_aggregates(doc_id: str, headings: list[dict]) -> dict:
    # Aggregate keywords from all headings (dedupe)
    all_keywords: set[str] = set()
    seen_keywords_lower: set[str] = set()
    for h in headings:
        for kw in json.loads(h['keywords_json']):
            if kw.lower() not in seen_keywords_lower:
                all_keywords.add(kw)
                seen_keywords_lower.add(kw.lower())
    
    # Aggregate entities from all headings (dedupe by label + text)
    all_entities: dict[str, set[str]] = defaultdict(set)
    seen_entities: dict[str, set[str]] = defaultdict(set)
    for h in headings:
        for label, texts in json.loads(h['entities_json']).items():
            for text in texts:
                if text.lower() not in seen_entities[label]:
                    all_entities[label].add(text)
                    seen_entities[label].add(text.lower())
    
    # Aggregate embeddings (mean of heading embeddings)
    embeddings = [h['embedding'] for h in headings if h.get('embedding') is not None]
    agg_embedding = np.mean(embeddings, axis=0) if embeddings else None
    
    return {
        'embedding': agg_embedding,
        'keywords_json': json.dumps(sorted(all_keywords)),
        'entities_json': json.dumps({k: sorted(v) for k, v in all_entities.items()}),
    }
```

### No Lossy Fallback - Reprocess Instead

**If source data unavailable (Redis empty, no JSON files):**
- Do NOT parse enriched_content (loses entity labels)
- Instead: reprocess documents through fast-extraction pipeline
- This ensures 100% data fidelity

```python
def get_documents_needing_reprocess(storage) -> list[str]:
    """Find documents without structured metadata."""
    docs = storage.get_all_documents()
    missing = []
    for doc in docs:
        # Check if any chunk lacks keywords_json
        chunks = storage.get_chunks_by_doc(doc['id'])
        if any(c.get('keywords_json') is None for c in chunks):
            missing.append(doc['source_file'])
    return missing

def trigger_reprocess(source_files: list[str]):
    """Send documents back through fast-extraction."""
    for source_file in source_files:
        # Copy to fast-extraction input directory
        # Pipeline will reprocess and reingest with full metadata
        shutil.copy(source_file, FAST_EXTRACTION_INPUT_DIR)
```

---

## TODOs

- [ ] 0. **Backup original data** (before any changes)
  - Copy SQLite database file
  - Export Redis stream to file
  - Verify backups are readable

- [ ] 1. **Fix the data loss bug FIRST** (before migration)
  - Update `HybridRetriever.ingest_document()` to pass keywords/entities to storage
  - Update `SQLiteStorage.add_chunk()` to accept and store `keywords_json`, `entities_json`
  - This ensures NEW ingestions preserve structured data

- [ ] 2. **Schema migration** `src/plm/search/storage/migrations/001_hierarchical.py`
  - Add `headings` table
  - Add `keywords_json`, `entities_json` columns to chunks
  - Add `embedding`, `keywords_json`, `entities_json` to documents
  - Add `heading_id` to chunks (nullable for backward compat)

- [ ] 3. **Data migration script**
  - Check Redis Streams for source messages (`plm:extraction`)
  - If available: recover full structured data, update chunks
  - If not available: trigger full reingestion (no lossy fallback)
  - Build heading records from grouped chunks
  - Aggregate embeddings/metadata to document level

- [ ] 4. **Update ingestion pipeline**
  - Store keywords_json, entities_json on chunks
  - Create/update heading records on ingest
  - Aggregate embeddings/metadata to heading and document levels

- [ ] 5. **Add retrieval modes to HybridRetriever**
  - `retrieve(mode="chunks")` - current behavior (default)
  - `retrieve(mode="headings")` - return heading-level results  
  - `retrieve(mode="documents")` - return document-level results

- [ ] 6. **Update MCP server**
  - Add `mode` parameter to search tool
  - Or separate `search_sections` tool

---

## Pre-Migration Check

**Status: ✅ PASSED**
```bash
docker exec plm-redis redis-cli XLEN plm:extraction
# Result: 1571 messages (matches ~1569 indexed documents)
```

**Conclusion**: Full structured data available in Redis. No reprocessing needed.

## Open Questions

1. **Embedding aggregation** - Mean vs. weighted mean vs. separate index? Mean is simple but may lose nuance for large documents.

2. **Heading level from existing data** - Current `heading` column has text like "## What is a Pod?" - need to parse `#` count. Source data has `level` field directly.

3. **Reprocessing strategy** - If Redis empty, do we:
   - Clear DB and reingest everything fresh?
   - Or keep existing chunks, add new columns, mark as needing reprocess?

---

## Success Criteria

- [ ] Migration runs without re-ingesting documents
- [ ] Default search behavior unchanged (chunks mode)
- [ ] New modes available: headings, documents
- [ ] Keywords/entities stored as structured JSON going forward
- [ ] Aggregated embeddings at heading and document levels

---

## Risks

| Risk | Mitigation |
|------|------------|
| Migration corrupts data | Backup before migration, run on copy first |
| Redis data missing | Reprocess through fast-extraction (no lossy fallback) |
| Performance regression | Index heading_id FK, test query plans |
| Duplicate entities/keywords | Case-insensitive deduplication during aggregation |
