# Draft: src/ Structure Design for PLM

## Requirements Confirmed

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package name | `plm` | Short, standard Python convention |
| GLiNER integration | YES | Trainable NER per RAG_PIPELINE_ARCHITECTURE.md |
| Storage backend | JSON files | Simple, matches POC artifacts/ pattern |
| Implementation scope | Both fast AND slow systems | Complete V6 pipeline |

---

## Source Analysis

### From POC-1c (V6 Strategy)

**5-Stage Pipeline**:
1. **Extraction** (maximize recall): candidate_verify + taxonomy + heuristics + seeds
2. **Grounding**: verify_span, normalize, dedup, track sources
3. **Noise Filter**: rule-based rejection (stop words, negatives, patterns)
4. **Confidence Routing**: HIGH/MEDIUM/LOW → validation or auto-keep
5. **Post-processing**: span expansion, suppression, final dedup

**Performance**: F1=0.932 @ 10 docs, F1=0.883 @ 50 docs

### From RAG_PIPELINE_ARCHITECTURE.md

**Fast System Components**:
- YAKE (statistical keywords) - ~100-200ms/doc
- GLiNER (trainable NER) - ~100-300ms/doc
- Term Graph Matcher
- Confidence Calculator

**Slow System Components**:
- LLM Deep Extraction (novel terms, synonyms)
- Tiered Review (LLM auto-review → Human review)

**Key Principle**: Slow system corrections → training data for fast system

### From External Research

**Best Practices (2026)**:
- `src/` layout standard for ML/NLP projects
- Confidence-based routing between fast/slow
- Modular component design with clear interfaces
- Configuration-driven architecture

---

## Key Mapping: V6 Stages → Module Structure

| V6 Stage | Fast/Slow | Module | Key Components |
|----------|-----------|--------|----------------|
| Stage 1: Extraction | Both | `extraction/` | `fast/heuristic.py`, `slow/candidate_verify.py`, `slow/taxonomy.py` |
| Stage 2: Grounding | Fast | `extraction/grounding.py` | `verify_span()`, `normalize_term()`, source tracking |
| Stage 3: Noise Filter | Fast | `extraction/noise_filter.py` | Stop words, negatives, patterns |
| Stage 4: Routing | Bridge | `extraction/fast/confidence.py` | HIGH/MEDIUM/LOW classification |
| Stage 4: Validation | Slow | `extraction/slow/validation.py` | Term-retrieval context validation |
| Stage 5: Post-process | Fast | `extraction/postprocess.py` | Span expansion, suppression |

---

## Portable from POC-1c

| POC File | Target Location | Changes Needed |
|----------|-----------------|----------------|
| `scoring.py` | `plm/scoring/` | Split into `matching.py` + `normalization.py` |
| `hybrid_ner.py` (Stage logic) | `plm/extraction/pipeline.py` | Refactor into modular stages |
| `retrieval_ner.py` | `plm/retrieval/` | Clean import paths |
| `poc/shared/llm/` | `plm/llm/` | Already modular, minimal changes |
| `artifacts/auto_vocab.json` | `data/vocabularies/` | Move as seed data |
| `artifacts/tech_domain_negatives.json` | `data/vocabularies/` | Move as reference |

---

## Open Design Questions

1. **Should heuristic extraction be a spaCy component?**
   - Option A: Standalone Python module (current POC approach)
   - Option B: spaCy custom component (per spaCy-LLM research)
   - **Recommendation**: Start standalone, migrate to spaCy later for pipeline integration

2. **How to handle GLiNER model management?**
   - Model weights storage location
   - Retraining workflow
   - Version control for models

3. **Async vs sync for slow system?**
   - Batch processing (daily) vs on-demand
   - Queue management

---

## Next Step

Generate work plan for implementation once structure is confirmed.
