# Plan: Full Request Traceability in Search Service

## Goal

Every request to the search service must be fully traceable. Given a `request_id`, we can see:
- When request was received
- Original query text
- Query rewriting (before/after, timing)
- Query expansion (terms added)
- BM25/SPLADE search (candidates found, scores, timing)
- Semantic search (candidates found, scores, timing)
- RRF fusion (per-candidate scores from each source, final fused scores)
- Reranking (input scores vs output scores, score changes)
- Final results returned
- Total request time

## Architecture

### 1. Expand Existing PipelineLogger

Use `src/plm/shared/logger.py` `PipelineLogger` - do NOT create new logger.

Add request-aware logging method:

```python
# In PipelineLogger class
def request(self, request_id: str, stage: str, msg: str) -> None:
    elapsed = time.perf_counter() - self._start
    self._emit("TRACE", f"[{request_id}] [{stage}] {msg}")
```

### 2. Request Context

Simple dataclass passed through retrieval:

```python
@dataclass
class RequestContext:
    request_id: str
    log: PipelineLogger
```

## Changes Required

### Phase 1: Extend PipelineLogger

#### Task 1.1: Add request-aware logging to PipelineLogger

**File**: `src/plm/shared/logger.py`

Add method for request-scoped logging:

```python
def request(self, request_id: str, stage: str, msg: str) -> None:
    """Log a request-scoped trace message."""
    self._emit("TRACE", f"[{request_id}] [{stage}] {msg}")
```

#### Task 1.2: Add log level configuration

**File**: `src/plm/search/service/app.py`

Environment variables:

```python
LOG_DIR = os.environ.get("PLM_LOG_DIR", "/data/logs")
LOG_LEVEL = os.environ.get("PLM_LOG_LEVEL", "INFO")  # INFO, DEBUG, TRACE
LOG_TO_FILE = os.environ.get("PLM_LOG_TO_FILE", "true").lower() == "true"
```

Initialize logger at startup:

```python
# In lifespan():
log_dir = Path(LOG_DIR)
log_dir.mkdir(parents=True, exist_ok=True)

app.state.logger = PipelineLogger(
    log_file=log_dir / "search_info.log" if LOG_TO_FILE else None,
    trace_file=log_dir / "search_trace.log" if LOG_TO_FILE else None,
    console=True,
    min_level=LOG_LEVEL,
)
```

---

### Phase 2: Propagate Logger and Request ID

#### Task 2.1: Update HybridRetriever.__init__()

**File**: `src/plm/search/retriever.py`

Accept logger in constructor:

```python
def __init__(
    self,
    db_path: str,
    bm25_index_path: str,
    rewrite_timeout: float = 5.0,
    config: RetrievalConfig | None = None,
    logger: PipelineLogger | None = None,  # NEW
) -> None:
    self.log = logger or PipelineLogger()  # Fallback to default
```

#### Task 2.2: Update HybridRetriever.retrieve()

**File**: `src/plm/search/retriever.py`

Add `request_id` parameter:

```python
def retrieve(
    self,
    query: str,
    k: int = 5,
    use_rewrite: bool = False,
    use_rerank: bool = False,
    explain: bool = False,
    request_id: str | None = None,  # NEW
) -> list[dict] | tuple[list[dict], dict]:
    rid = request_id or str(uuid.uuid4())
    self.log.request(rid, "receive", f"query={query!r} k={k} rewrite={use_rewrite} rerank={use_rerank}")
```

#### Task 2.3: Update app.py to pass logger and request_id

**File**: `src/plm/search/service/app.py`

```python
# In lifespan():
app.state.retriever = HybridRetriever(
    db_path=str(db_path),
    bm25_index_path=INDEX_PATH,
    logger=app.state.logger,  # Pass logger
)

# In query endpoint:
request_id = http_request.state.request_id

result = retriever.retrieve(
    query=request.query,
    k=request.k,
    use_rewrite=request.use_rewrite,
    use_rerank=request.use_rerank,
    explain=request.explain,
    request_id=request_id,  # Pass request_id
)

# ALWAYS return request_id in response (not just when explain=True)
return QueryResponse(
    ...
    request_id=request_id,  # Always included
)
```

#### Task 2.4: Update QueryResponse to always include request_id

**File**: `src/plm/search/service/app.py`

Change from:
```python
request_id: str | None = Field(None, description="Request correlation ID (only when explain=True)")
```

To:
```python
request_id: str = Field(..., description="Request correlation ID for log tracing")
```

---

### Phase 3: Add Detailed Trace Points

#### Task 3.1: Query Receive

```python
self.log.request(rid, "receive", f"query={query!r} k={k} rewrite={use_rewrite} rerank={use_rerank}")
```

#### Task 3.2: Query Rewriting

```python
self.log.request(rid, "rewrite:start", f"input={query!r}")
rewritten_query = self._query_rewriter.process(original_query)
self.log.request(rid, "rewrite:end", f"output={rewritten_query.rewritten!r} model={rewritten_query.model}")
```

#### Task 3.3: Query Expansion

```python
self.log.request(rid, "expand:start", f"input={rewritten_query.rewritten!r}")
expanded = self.expander.process(rewritten_query)
self.log.request(rid, "expand:end", f"output={expanded.expanded!r} added={list(expanded.expansions)}")
```

#### Task 3.4: Sparse Search (BM25/SPLADE)

```python
self.log.request(rid, "sparse:start", f"query={expanded_query[:80]!r} k={n_candidates}")
sparse_results = self.sparse_retriever.search(expanded_query, k=n_candidates)
self.log.request(rid, "sparse:end", f"found={len(sparse_results)}")

for i, r in enumerate(sparse_results[:5]):
    item = item_index[r["index"]]
    self.log.request(rid, "sparse:hit", f"rank={i} score={r['score']:.4f} id={item['id']}")
```

#### Task 3.5: Semantic Search

```python
self.log.request(rid, "semantic:start", f"items={len(all_items)}")
sem_scores = np.dot(embeddings, query_emb)
sem_ranks = np.argsort(sem_scores)[::-1]
self.log.request(rid, "semantic:end", f"computed={len(sem_scores)}")

for i, idx in enumerate(sem_ranks[:5]):
    item = item_index[idx]
    self.log.request(rid, "semantic:hit", f"rank={i} score={sem_scores[idx]:.4f} id={item['id']}")
```

#### Task 3.6: RRF Fusion (DETAILED)

```python
self.log.request(rid, "rrf:start", f"rrf_k={rrf_k} sparse_w={sparse_weight} sem_w={sem_weight}")

for idx in top_indices[:10]:
    item = item_index[idx]
    sp_rank = sparse_rank_by_idx.get(idx, -1)
    se_rank = sem_rank_by_idx.get(idx, -1)
    sp_score = sparse_score_by_idx.get(idx, 0)
    se_score = float(sem_scores[idx]) if idx < len(sem_scores) else 0
    sp_contrib = sparse_weight / (rrf_k + sp_rank) if sp_rank >= 0 else 0
    se_contrib = sem_weight / (rrf_k + se_rank) if se_rank >= 0 else 0
    
    self.log.request(rid, "rrf:hit", 
        f"id={item['id']} "
        f"sparse[r={sp_rank} s={sp_score:.3f} c={sp_contrib:.4f}] "
        f"sem[r={se_rank} s={se_score:.3f} c={se_contrib:.4f}] "
        f"final={rrf_scores[idx]:.4f}"
    )

self.log.request(rid, "rrf:end", f"fused={len(rrf_scores)} returned={len(results)}")
```

#### Task 3.7: Reranking (DETAILED)

```python
self.log.request(rid, "rerank:start", f"input={len(input_results)} model={self._reranker.model_name}")

for i, r in enumerate(input_results[:10]):
    self.log.request(rid, "rerank:before", f"rank={i} score={r['score']:.4f} id={r['id']}")

reranked = self._reranker.rerank(query, input_results, top_k=k)

for i, r in enumerate(reranked):
    old_rank = next((j for j, rr in enumerate(input_results) if rr['id'] == r['id']), -1)
    self.log.request(rid, "rerank:after", 
        f"rank={i} score={r.get('rerank_score', 0):.4f} old={old_rank} id={r['id']}"
    )

self.log.request(rid, "rerank:end", f"output={len(reranked)}")
```

#### Task 3.8: Complete

```python
self.log.request(rid, "complete", f"results={len(results)} elapsed_ms={elapsed_ms:.1f}")
```

---

### Phase 4: Environment Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PLM_LOG_DIR` | `/data/logs` | Directory for log files |
| `PLM_LOG_LEVEL` | `INFO` | Min level for console: TRACE, DEBUG, INFO, WARN, ERROR |
| `PLM_LOG_TO_FILE` | `true` | Enable file logging (trace + info files) |

---

## Example Output

### Trace Log (full detail)

```
[a1b2c3] [receive] query="How do I scale a deployment?" k=10 rewrite=True rerank=False
[a1b2c3] [rewrite:start] input="How do I scale a deployment?"
[a1b2c3] [rewrite:end] output="kubernetes deployment scaling replicas HPA autoscaler" model=claude-haiku
[a1b2c3] [expand:start] input="kubernetes deployment scaling replicas HPA autoscaler"
[a1b2c3] [expand:end] output="kubernetes deployment scaling replicas HPA autoscaler horizontal pod" added=["horizontal pod"]
[a1b2c3] [sparse:start] query="kubernetes deployment scaling replicas HPA autoscaler horiz..." k=100
[a1b2c3] [sparse:end] found=100
[a1b2c3] [sparse:hit] rank=0 score=14.2341 id=concepts_workloads_controllers_deployment_4a8f2b
[a1b2c3] [sparse:hit] rank=1 score=13.8921 id=tasks_run-application_horizontal-pod-autoscale_7c3e1a
[a1b2c3] [sparse:hit] rank=2 score=12.4532 id=reference_kubectl_kubectl-scale_2d9f8c
[a1b2c3] [semantic:start] items=20801
[a1b2c3] [semantic:end] computed=20801
[a1b2c3] [semantic:hit] rank=0 score=0.8921 id=concepts_workloads_controllers_deployment_4a8f2b
[a1b2c3] [semantic:hit] rank=1 score=0.8745 id=tasks_run-application_horizontal-pod-autoscale_7c3e1a
[a1b2c3] [semantic:hit] rank=2 score=0.8234 id=concepts_workloads_controllers_replicaset_9b2d4e
[a1b2c3] [rrf:start] rrf_k=60 sparse_w=1.0 sem_w=1.0
[a1b2c3] [rrf:hit] id=concepts_..._4a8f2b sparse[r=0 s=14.234 c=0.0164] sem[r=0 s=0.892 c=0.0164] final=0.0328
[a1b2c3] [rrf:hit] id=tasks_..._7c3e1a sparse[r=1 s=13.892 c=0.0161] sem[r=1 s=0.875 c=0.0161] final=0.0322
[a1b2c3] [rrf:end] fused=150 returned=10
[a1b2c3] [complete] results=10 elapsed_ms=287.4
```

### With Reranking

```
[d4e5f6] [rerank:start] input=50 model=cross-encoder/ms-marco-MiniLM-L-6-v2
[d4e5f6] [rerank:before] rank=0 score=0.0328 id=concepts_..._4a8f2b
[d4e5f6] [rerank:before] rank=1 score=0.0322 id=tasks_..._7c3e1a
[d4e5f6] [rerank:before] rank=2 score=0.0298 id=concepts_..._9b2d4e
[d4e5f6] [rerank:after] rank=0 score=0.9842 old=1 id=tasks_..._7c3e1a
[d4e5f6] [rerank:after] rank=1 score=0.9234 old=0 id=concepts_..._4a8f2b
[d4e5f6] [rerank:after] rank=2 score=0.8123 old=2 id=concepts_..._9b2d4e
[d4e5f6] [rerank:end] output=10
```

---

## Implementation Order

1. **Task 1.1**: Add `request()` method to `PipelineLogger`
2. **Task 1.2**: Add env var configuration in `app.py`
3. **Task 2.1**: Add `logger` param to `HybridRetriever.__init__()`
4. **Task 2.2**: Add `request_id` param to `HybridRetriever.retrieve()`
5. **Task 2.3**: Wire up logger and request_id in `app.py`
6. **Task 2.4**: Always return `request_id` in response (not just explain mode)
7. **Task 3.1-3.8**: Add all trace points in `retriever.py`

## Files Modified

| File | Changes |
|------|---------|
| `src/plm/shared/logger.py` | Add `request()` method |
| `src/plm/search/service/app.py` | Add logger init, env vars, pass request_id |
| `src/plm/search/retriever.py` | Add logger param, request_id param, all trace points |

## Verification

1. Start service with `PLM_LOG_LEVEL=TRACE`
2. Run benchmark: `python -m plm.benchmark.cli --datasets needle --limit 5`
3. Check trace log file for full request details
4. Grep for specific request_id to see complete lifecycle

### Example: Trace a Single Request

```bash
# Find all logs for request a1b2c3
grep "\[a1b2c3\]" search_trace.log

# See what chunks were considered in RRF
grep "\[a1b2c3\].*\[rrf:detail\]" search_trace.log

# See reranking score changes
grep "\[a1b2c3\].*\[rerank:" search_trace.log

# Look up specific chunk content in database
sqlite3 /data/index/index.db "SELECT content FROM chunks WHERE id='concepts_workloads_controllers_deployment_4a8f2b'"
```

### IDs in Logs

| Field | Purpose |
|-------|---------|
| `request_id` | Correlate all logs for one request |
| `id` | Entity ID (chunk, heading, or document) - look up in DB if needed |

### Response Always Includes request_id

```json
{
  "query": "How do I scale a deployment?",
  "k": 10,
  "results": [...],
  "elapsed_ms": 287.4,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### Tracing Workflow

```bash
# 1. Make request, get request_id from response
curl -s http://localhost:8000/query -d '{"query": "scale deployment"}' | jq .request_id
# "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# 2. Search logs for that request
grep "a1b2c3d4" /data/logs/search_trace.log

# 3. Look up specific entity in database if needed
sqlite3 /data/index/index.db "SELECT content FROM chunks WHERE id='concepts_..._4a8f2b'"
```
