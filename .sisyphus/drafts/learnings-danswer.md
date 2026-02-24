# Learnings: Danswer (Onyx) Enterprise Search

## Key Architectural Decisions

### 1. Vespa Migration — Unified Hybrid Search
**Problem**: Separate vector + keyword engines couldn't be weighted together properly
**Solution**: Migrated to Vespa for unified normalization across search types

**Why it matters**:
- Internal names like "Meechum" or "Foundry" had no general English representation
- Unified hybrid search beats separate engines at scale
- **Lesson**: Don't use separate engines — unify them

### 2. Time Decay for Relevance
**Problem**: Stale documents pollute search results (enterprises don't clean up)
**Solution**: `time_last_touched` with custom decay curves in Vespa

**Adoptable Pattern**:
```python
# Decay score based on document age
def decay_score(base_score, last_modified, decay_rate=0.1):
    age_days = (now - last_modified).days
    return base_score * exp(-decay_rate * age_days)
```
**Lesson**: Time decay is non-negotiable for enterprise search

### 3. Multi-Vector Per Document
**Innovation**: Different context sizes for same document
- Small chunks: Precise retrieval
- Large chunks: Context preservation
- Prevents document duplication

**Lesson**: Index same doc with multiple chunk sizes, don't duplicate

## What They Deprecated (and Why)

### Intent Model — Removed
- Built DistilBERT classifier (Keyword/Semantic/QA)
- Became unnecessary with Vespa's unified ranking
- **Lesson**: Don't build query routers if your search engine handles all types natively

### Separate Reranking — Simplified
- Vespa's ranking expressions handle it in one pass
- Reranking adds latency
- **Lesson**: Integrated ranking > post-processing reranking

## Connector Architecture — 40+ Connectors

### Three Connector Types
| Type | Purpose | Frequency |
|------|---------|-----------|
| Load | Bulk index (snapshot) | Initial sync, reindex |
| Poll | Incremental updates | Background job, keep fresh |
| Slim | Fetch IDs only | Pruning, existence checks |

**Key Pattern**:
```python
class PollConnector:
    def poll_source(self, start: float, end: float) -> Iterator[Document]:
        """Only fetch docs modified in time range"""

class SlimConnector:
    def retrieve_all_slim_docs(self) -> Iterator[SlimDocument]:
        """Only IDs + metadata, no content"""
```

**Lesson**: Separate "what" (config) from "how to access" (credentials)

## Production Learnings

### Scale
- Tens of millions of documents per customer
- Self-hosted requirement for data security
- Resource efficiency unlocks smaller customers

### Enterprise Requirements
- Permission mirroring (users see only what they can access)
- Multi-tenancy from day one (not bolted on)
- SSO (OIDC/SAML/OAuth2)

## What Didn't Work

1. **Separate search engines**: Couldn't weight results together
2. **Post-processing time decay**: Accuracy degraded past millions of docs
3. **Intent classification**: Unnecessary with unified search

## Actionable Recommendations

### Adopt
- ✅ Time decay for relevance
- ✅ Multi-vector indexing (different chunk sizes)
- ✅ Connector patterns (Load/Poll/Slim)
- ✅ Standalone component testing

### Avoid
- ❌ Intent classification (let search handle it)
- ❌ Separate reranking stages
- ❌ Multiple search engines

---
*Source: Investigation of danswer-ai/danswer (now Onyx), 17.5K stars*
