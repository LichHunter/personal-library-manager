# Learnings: R2R (SciPhi) — Production RAG Features

## What Production RAG Actually Needs

Despite complexity, R2R shows what real users want:

### 1. Observability
- Dashboards, logs, workflow tracking
- Visual entity browser
- Real-time status updates

### 2. Retry Mechanisms
- Extraction fails often (LLM timeouts, rate limits)
- Save failed jobs, allow manual retry
- Workflow tracking with status

### 3. Concurrency Control
- Separate limits for ingestion vs queries
- Don't let batch jobs block interactive search

### 4. Batch Processing
- Handle 100k+ documents
- Progress tracking: `logger.info(f"{i}/{total} processed")`
- Resume from failure

## Knowledge Graph Patterns

### Entity Storage
```python
# Dual-store pattern
documents_entities  # Per-document extraction
graphs_entities     # Collection-level graph

# Extract once, reuse many times
client.documents.extract(document_id)  # Expensive
client.graphs.pull(collection_id)       # Cheap
```

### LLM-Enriched Descriptions
- Not just entity names — semantic descriptions
- Embed descriptions for semantic graph search
- Batch processing (256 entities at a time)

### Storage: Postgres + pgvector
- Single database for everything
- JSONB for flexible metadata
- Cascade deletes for data integrity
- **No separate graph database needed**

## Agentic RAG Patterns

### Tool Registry
```python
class ToolRegistry:
    def discover_tools(self, path="~/.plm/tools/"):
        for file in glob(f"{path}/*.py"):
            tool = import_tool(file)
            self.register(tool)
```
- Plugin architecture
- User-defined tools without code changes
- Context injection (search_settings, methods)

### Two Modes
1. **RAG Mode**: Search + retrieval tools
2. **Research Mode**: Reasoning + computation tools

## What Users Actually Value

1. **Docker one-liner**: `docker compose up`
2. **RESTful API**: Language-agnostic
3. **Knowledge graphs**: Relationship-aware search
4. **Hybrid search**: Semantic + keyword
5. **Visual feedback**: See entities, workflows

## Patterns to Adopt

### Immediate (Low Effort)
- Batch processing with progress logs
- Separate extraction from indexing
- Configurable prompts (JSON/YAML, not hardcoded)
- Streaming search API

### Medium-Term
- Tool registry for extensibility
- Simple dashboard (FastAPI + HTMX)
- Retry mechanism for failed extractions

### Long-Term
- Orchestration layer (Hatchet/Temporal)
- Multi-tenant collections
- Knowledge graph integration

## What NOT to Copy

- ❌ 2,913 lines for graph storage
- ❌ Dual configuration systems (TOML + env + runtime)
- ❌ 76+ dependencies
- ❌ 11 Docker services for basic RAG

## Key Takeaway

R2R's real value isn't in their architecture — it's in **production-ready features**:
- Observability
- Fault tolerance
- Extensibility
- Operational simplicity

**Focus on reliability more than features.**

---
*Source: Investigation of SciPhi-AI/R2R @ 9c5a94d*
