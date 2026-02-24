# Learnings: Haystack (deepset) — Pipeline Architecture

## Core Architecture: Graph-Based Execution

**Key Insight**: Pipelines are directed multigraphs (NetworkX), not linear chains
- **Branching**: Multiple parallel execution paths
- **Loops**: Iterative refinement
- **Conditional routing**: Dynamic path selection

```python
self.graph = networkx.MultiDiGraph()
```

**Why MultiDiGraph?**
- Multi: Multiple edges between same nodes (different sockets)
- Directed: Enforces data flow direction
- Graph: Enables complex topologies

## Component Interface: Decorator Pattern

```python
@component
class MyRetriever:
    @component.output_types(documents=list[Document])
    def run(self, query: str, top_k: int = 10) -> dict[str, Any]:
        return {"documents": [...]}
```

**Features**:
- Auto-registration in global registry
- Validation at decoration time
- Socket introspection from method signatures
- No manual socket declaration needed

## Router Components

### ConditionalRouter — Jinja2-Based
```python
routes = [
    {"condition": "{{streams|length > 2}}", "output_name": "enough"},
    {"condition": "{{streams|length <= 2}}", "output_name": "insufficient"},
]
router = ConditionalRouter(routes)
```

**Why Jinja2?**
- Serializable routing rules
- Non-programmers can modify logic
- Runtime introspection of required variables

### TransformersZeroShotTextRouter — ML-Based
```python
router = TransformersZeroShotTextRouter(
    labels=["passage", "query"],
    model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
)
# Creates output sockets dynamically based on labels
```

**Pattern**: Dynamic output socket creation based on config

## Production Patterns

### Serialization
- Pipelines serialize to YAML/JSON
- Components implement `to_dict()` and `from_dict()`
- Enables version control, GitOps, A/B testing

### Warm-up Pattern
```python
class EmbeddingEncoder:
    def warm_up(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
```
- Lazy model loading
- Faster startup
- Better resource management

### Observability
- OpenTelemetry integration
- Semantic tags: `haystack.component.input`, `haystack.component.output`
- Pipeline visualization (Mermaid)

## Documentation Excellence

1. **Jupyter notebooks as docs** — every feature executable
2. **Progressive disclosure** — quickstart → concepts → advanced
3. **Real examples** — not foo/bar
4. **Performance metrics** — before/after for every optimization

## Patterns to Adopt

### Immediate
- Component decorator system (auto-validate types)
- Jinja2-based routing (serializable, user-friendly)
- Warm-up pattern (lazy loading)

### Future
- Pipeline serialization (YAML configs)
- Async execution (parallel BM25 + semantic)
- Type validation at connection time

## What to Avoid
- Over-engineering early (don't need MultiDiGraph for linear pipelines)
- Too much magic (introspection can confuse)
- Premature abstraction

---
*Source: Investigation of deepset-ai/haystack @ 9b57ffb*
