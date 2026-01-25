# Enhanced Chunking with Dual Storage & Enriched Search

## Decision Log

### Interview Session (2024-01-24)

**User Requirements:**
1. Split documents into variable-length paragraph-based chunks
2. Keep two pairs: original data + enhanced data with metadata/keywords
3. Search by enhanced data to improve retrieval
4. Add "pointers or sugar" like keywords, metadata comparison
5. Don't just take top-N, but narrow down documents/paragraphs first

**Decisions Made:**

| Question | Decision | Reasoning |
|----------|----------|-----------|
| Enrichment types | ALL 5: keywords, contextual prefix, questions, summary, entities | Test all and compare to find best |
| Search strategy | Dual embedding | Embed enhanced content, return original |
| Narrowing approach | Doc → Section → Chunk | Like existing LOD strategy |
| LLM usage | Local Ollama, multiple sizes | Test 5 models for quality vs speed |
| Benchmark structure | Extend existing benchmark | Add to chunking_benchmark_v2 |
| Chunk granularity | Hierarchical (multiple sizes) | Enrich at doc, section, chunk levels |
| Storage | Parallel arrays (like ReverseHyDE) | Proven pattern in codebase |
| Testing | Benchmark comparison only | No unit tests needed |
| Metrics | Coverage, precision, latency, index time | Comprehensive comparison |

**Models to Test:**
- llama3.2:1b (smallest/fastest)
- llama3.2:3b (current default)
- llama3.1:8b (larger)
- mistral:7b (alternative architecture)
- qwen2.5:7b (strong reasoning)

---

## Research Findings

### Anthropic's Contextual Retrieval
- **Core idea**: Prepend chunk-specific context to each chunk before embedding
- **Implementation**: Use LLM to generate situating context like:
  > "This chunk is from the API Reference section, discussing authentication..."
- **Results**: 35% reduction in retrieval failures, 67% with reranking
- **Dual storage**: Original chunk + contextualized chunk

### LlamaIndex Patterns
- **HierarchicalNodeParser**: Creates multi-level nodes (2048→512→128 tokens)
- **AutoMergingRetriever**: Searches small chunks, merges to parent if threshold met
- **Small-to-Big**: Index small chunks, return larger context
- **MetadataFilters**: Pre-filter by structured metadata before vector search

### Existing Codebase Patterns
- **ReverseHyDE** (`reverse_hyde.py`): Generates questions at index time, stores separately
- **RAPTOR** (`raptor.py`): Hierarchical summaries tree
- **LOD** (`lod.py`): Doc→Section→Chunk narrowing
- **Key insight**: All enrichments use parallel storage, NOT Chunk modification

### Enrichment Strategies

| Strategy | What's Stored | Search Improvement |
|----------|--------------|-------------------|
| **Keywords/Tags** | Extracted key terms | BM25 boost, metadata filter |
| **Generated Questions** | Q&A pairs | Better query matching |
| **Contextual Prefix** | LLM-generated context | Embedding similarity |
| **Summary** | Condensed version | Quick filtering |
| **Entity Extraction** | Named entities | Structured filtering |

---

## Metis Gap Analysis

### Identified Gaps (addressed)

1. **Storage pattern conflict**: User said "extend Chunk" but existing code uses parallel storage
   - Resolution: Use parallel storage pattern (proven in codebase)

2. **LLM budget concern**: 5 enrichments × 3 levels × chunks = thousands of calls
   - Resolution: Persist enrichments to disk, reuse across runs

3. **Combination explosion**: 31 possible enrichment combinations
   - Resolution: Two-phase benchmark approach

4. **Baseline definition unclear**
   - Resolution: Use existing semantic + hybrid as baseline

5. **Empty/failed enrichments**
   - Resolution: Fallback to original content

### Guardrails Set

**MUST DO:**
- Follow existing parallel storage pattern (like ReverseHyDE)
- Persist enrichments to disk
- Use existing `call_ollama()` helper
- Track enrichment generation time separately

**MUST NOT DO:**
- Modify existing `Chunk` dataclass
- Modify existing retrieval strategies
- Change benchmark evaluation logic
- Add new dependencies
- Test all 25 combinations (5×5)

---

## Work Objectives

### Core Objective
Create an enrichment pipeline that generates multiple types of enhanced content per chunk, stores them alongside originals, and uses them for improved hierarchical retrieval - then benchmark to determine which enrichments and which model sizes provide the best results.

### Concrete Deliverables
- `poc/chunking_benchmark_v2/enrichment/` module with 5 enrichment generators
- `poc/chunking_benchmark_v2/enrichment/cache/` for persisted enrichments
- `poc/chunking_benchmark_v2/retrieval/enriched_lod.py` - new retrieval strategy
- `config_enriched.yaml` for benchmark configuration
- Benchmark results comparing all strategies

### Definition of Done
- [ ] All 5 enrichment types generate valid output for >95% of chunks
- [ ] Enriched retrieval achieves measurable improvement over baseline
- [ ] Phase 1 benchmark: Compare 5 enrichment types (using 3b model)
- [ ] Phase 2 benchmark: Compare 5 model sizes (using best enrichment)
- [ ] Results document: best enrichment type + optimal model size recommendation

---

## Two-Phase Benchmark Strategy

### Phase 1: Find Best Enrichment Type
- **Test**: keywords, contextual, questions, summary, entities
- **Model**: llama3.2:3b (baseline)
- **Output**: Winning enrichment type
- **Configs**: 5 (one per enrichment) + baseline

### Phase 2: Find Optimal Model Size
- **Test**: llama3.2:1b, llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b
- **Enrichment**: Winner from Phase 1
- **Output**: Quality vs speed tradeoff analysis
- **Configs**: 5 (one per model)

---

## TODO Plan

### Task Flow

```
1 (Setup) → 2 (Keyword) → 3 (Context) → 4 (Questions) → 5 (Summary) → 6 (Entities)
                                    ↓
                            7 (EnrichedLOD Retrieval)
                                    ↓
                            8 (Benchmark Config)
                                    ↓
                            9 (Run Benchmark)
                                    ↓
                            10 (Analyze Results)
```

### Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| A | 2, 3, 4, 5, 6 | Independent enrichment generators after setup |

| Task | Depends On | Reason |
|------|------------|--------|
| 2-6 | 1 | Need base module structure |
| 7 | 2-6 | Need all enrichers before retrieval |
| 8 | 7 | Need strategy before config |
| 9 | 8 | Need config before run |
| 10 | 9 | Need results before analysis |

---

### Task 1: Setup enrichment module structure

**What to do**:
- Create `poc/chunking_benchmark_v2/enrichment/__init__.py` with base `Enricher` class
- Create `poc/chunking_benchmark_v2/enrichment/cache.py` for disk persistence
- Define common interface: `enrich(content: str, context: dict, model: str) -> dict`
- Add `call_ollama()` helper (copy from reverse_hyde.py or import)
- Support configurable LLM model parameter for all enrichers
- Models to support: llama3.2:1b, llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b

**Must NOT do**:
- Don't modify any existing files
- Don't add dependencies

**References**:
- `retrieval/reverse_hyde.py:18-32` - `call_ollama()` helper
- `retrieval/base.py:9-50` - Base class pattern
- `strategies/__init__.py:1-24` - Module structure

**Acceptance Criteria**:
```bash
ls -la poc/chunking_benchmark_v2/enrichment/
# Expected: __init__.py, cache.py
```
```python
from enrichment import Enricher
print(Enricher.__abstractmethods__)
# Expected: {'enrich'}
```

---

### Task 2: Implement keyword extraction enricher

**What to do**:
- Create `enrichment/keywords.py`
- Use LLM to extract 5-10 keywords per chunk
- Format as comma-separated list for BM25 matching
- Include fallback: if LLM fails, use TF-IDF top terms

**Must NOT do**:
- Don't add spaCy or heavy dependencies
- Don't extract more than 10 keywords

**References**:
- `retrieval/reverse_hyde.py:38-43` - LLM prompt pattern
- `retrieval/reverse_hyde.py:62-79` - Response parsing

**Acceptance Criteria**:
```python
from enrichment.keywords import KeywordEnricher
k = KeywordEnricher()
result = k.enrich("CloudFlow uses OAuth 2.0 for authentication with JWT tokens.")
# Expected: {"keywords": ["OAuth", "authentication", "JWT", ...]}
```

---

### Task 3: Implement contextual prefix enricher (Anthropic style)

**What to do**:
- Create `enrichment/contextual.py`
- Generate situating context: "This chunk is from [section] of [document]..."
- Prepend context to original content for enhanced embedding
- Accept document/section metadata as context input

**Must NOT do**:
- Don't exceed 100 words for context prefix
- Don't call LLM if metadata already provides sufficient context

**References**:
- `strategies/paragraph_heading.py:101-105` - Heading prepending
- Anthropic: https://www.anthropic.com/news/contextual-retrieval

**Acceptance Criteria**:
```python
from enrichment.contextual import ContextualEnricher
c = ContextualEnricher()
result = c.enrich(
    "Set the API_KEY environment variable.",
    context={"doc_title": "Deployment Guide", "section": "Environment Setup"}
)
# Expected: enhanced_content starts with context prefix
```

---

### Task 4: Implement question generation enricher

**What to do**:
- Create `enrichment/questions.py`
- Generate 3 questions per chunk that it answers
- Reuse prompt pattern from ReverseHyDE
- Store questions as list for multi-vector matching

**Must NOT do**:
- Don't duplicate ReverseHyDE retrieval logic
- Don't generate more than 5 questions per chunk

**References**:
- `retrieval/reverse_hyde.py:38-43` - EXACT prompt to reuse
- `retrieval/reverse_hyde.py:62-79` - Question parsing

**Acceptance Criteria**:
```python
from enrichment.questions import QuestionEnricher
q = QuestionEnricher()
result = q.enrich("To reset your password, click Forgot Password on the login page.")
# Expected: {"questions": ["How do I reset my password?", ...]}
```

---

### Task 5: Implement summary enricher

**What to do**:
- Create `enrichment/summary.py`
- Generate 1-2 sentence summary of chunk content
- Reuse summarization pattern from RAPTOR
- Handle long chunks by truncating input

**Must NOT do**:
- Don't generate summaries longer than 50 words
- Don't summarize chunks already very short (<30 words)

**References**:
- `retrieval/raptor.py:14-42` - `create_ollama_summarizer()`
- `retrieval/raptor.py:17-24` - Summarization prompt

**Acceptance Criteria**:
```python
from enrichment.summary import SummaryEnricher
s = SummaryEnricher()
result = s.enrich("CloudFlow is a platform..." * 50)
# Expected: summary is 10-50 words
```

---

### Task 6: Implement entity extraction enricher

**What to do**:
- Create `enrichment/entities.py`
- Extract: API endpoints, function names, config keys, technical terms
- Use LLM (more flexible than spaCy for technical docs)
- Format: `{"apis": [], "functions": [], "configs": [], "terms": []}`

**Must NOT do**:
- Don't add spaCy or NER dependencies
- Don't extract person names or generic nouns

**References**:
- `retrieval/reverse_hyde.py:62-79` - Response parsing

**Acceptance Criteria**:
```python
from enrichment.entities import EntityEnricher
e = EntityEnricher()
result = e.enrich("Call /api/v1/users with your API_KEY header.")
# Expected: {"apis": ["/api/v1/users"], "configs": ["API_KEY"], ...}
```

---

### Task 7: Create EnrichedLOD retrieval strategy

**What to do**:
- Create `retrieval/enriched_lod.py`
- Combine enrichment pipeline with LOD hierarchical search
- Index: Generate enrichments, embed enhanced content, store original
- Search: Doc→Section→Chunk narrowing using enhanced embeddings
- Return: Original chunk content (not enhanced)
- Store enrichments in parallel arrays (ReverseHyDE pattern)

**Must NOT do**:
- Don't modify existing LOD or other strategies
- Don't embed at search time
- Don't return enhanced content in results

**References**:
- `retrieval/lod.py:11-177` - Full LOD implementation
- `retrieval/reverse_hyde.py:81-136` - Parallel storage pattern
- `retrieval/base.py:119-133` - Index/retrieve interface

**Acceptance Criteria**:
```python
from retrieval.enriched_lod import EnrichedLODRetrieval
strategy = EnrichedLODRetrieval()
assert hasattr(strategy, 'index')
assert hasattr(strategy, 'retrieve')
# After retrieval, chunks should NOT contain context prefix
```

---

### Task 8: Create benchmark configuration

**What to do**:
- Create `config_enriched.yaml`
- Define strategies:
  - Baseline: semantic, hybrid
  - Phase 1: 5 enrichment types × 3b model
  - Phase 2: best enrichment × 5 models
- Configure enrichment settings (model list, cache path)

**Must NOT do**:
- Don't test all 25 combinations
- Don't modify existing config files

**References**:
- `config_realistic.yaml` - Existing config format
- `run_benchmark.py:372-383` - Config parsing

**Acceptance Criteria**:
```bash
python -c "import yaml; yaml.safe_load(open('config_enriched.yaml'))"
# Expected: No error
```

---

### Task 9: Run enrichment benchmark

**What to do**:
- **Phase 1**: Run enrichment type comparison (5 enrichments × 1 model)
- **Phase 2**: Run model size comparison (1 enrichment × 5 models)
- Capture: coverage, precision, latency, index time, enrichment time
- Save to `results/enriched_phase1_*/` and `results/enriched_phase2_*/`

**Must NOT do**:
- Don't run more than 3 iterations per strategy
- Don't skip baseline strategies

**Acceptance Criteria**:
```bash
python run_benchmark.py --config config_enriched.yaml
ls results/enriched_*/benchmark_results.json
# Expected: Results files exist
```

---

### Task 10: Analyze results and document findings

**What to do**:
- Generate comparison reports for both phases
- Phase 1 Analysis: Which enrichment type works best?
- Phase 2 Analysis: Which model size is optimal? (quality vs speed)
- Update `README.md` with findings
- Include recommendation: best enrichment + optimal model

**Must NOT do**:
- Don't claim significance without >5% improvement
- Don't recommend without cost/benefit analysis

**Acceptance Criteria**:
```bash
python generate_report.py results/enriched_*/
grep "Enrichment Results" README.md
# Expected: Section exists with comparison
```

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(benchmark): add enrichment module structure` | enrichment/__init__.py, cache.py |
| 2 | `feat(enrichment): add keyword extraction` | keywords.py |
| 3 | `feat(enrichment): add contextual prefix` | contextual.py |
| 4 | `feat(enrichment): add question generation` | questions.py |
| 5 | `feat(enrichment): add summary enricher` | summary.py |
| 6 | `feat(enrichment): add entity extraction` | entities.py |
| 7 | `feat(retrieval): add EnrichedLOD strategy` | enriched_lod.py |
| 8 | `feat(benchmark): add enriched config` | config_enriched.yaml |
| 10 | `docs(benchmark): document enrichment results` | README.md |

---

## Success Criteria

### Verification Commands
```bash
# All enrichers work
python -c "from enrichment import *; print('OK')"

# Benchmark runs
python run_benchmark.py --config config_enriched.yaml

# Report generates
python generate_report.py results/enriched_*/
```

### Final Checklist
- [ ] All 5 enrichment types implemented and working
- [ ] EnrichedLOD retrieval strategy integrates with benchmark
- [ ] Enrichers support configurable model parameter
- [ ] Phase 1 benchmark completes (enrichment type comparison)
- [ ] Phase 2 benchmark completes (model size comparison)
- [ ] Results show quality vs speed tradeoff across models
- [ ] README updated with: best enrichment + optimal model recommendation
- [ ] No existing code modified (guardrails respected)
