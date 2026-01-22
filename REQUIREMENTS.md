# Personal Knowledge Assistant - Requirements

## 1. Core Purpose
A local-first, NotebookLM-like system for querying a personal document corpus with grounded, cited answers.

## 2. Document Handling

| Requirement | Details |
|-------------|---------|
| **Formats** | All formats, converted to Markdown as canonical format |
| **Scale** | 1000+ documents, each ~500+ words minimum |
| **Images** | Text-first; image understanding deferred (design should not preclude it) |
| **Updates** | Mostly static; manual trigger for file/folder/full reindex |
| **Change tracking** | Hash-based detection, compare DB vs disk |
| **Cross-references** | Track document links; relation-aware reindexing (discussed later) |

## 3. Query Types Supported

| Type | Example |
|------|---------|
| Simple lookup | "What does function X do?" |
| Multi-doc synthesis | "What is the architecture of module Y?" |
| Comparison | "Compare error handling in service A vs B" |
| Exhaustive search | "What are all places we use Redis?" |
| Decision/rationale | "Why did we decide to implement feature X?" |
| Full system understanding | "How does data flow from user to backend?" |

## 4. Answer Requirements

| Requirement | Details |
|-------------|---------|
| **Format** | Synthesized summary combining multiple sources |
| **Citations** | Required for key claims (not every word) |
| **Grounding** | Important - user should be able to verify claims |
| **Transparency** | Must explicitly state: not found / conflicts / partial info |
| **Conflicts** | Present both sources: "Doc A says X, Doc B says Y" |
| **Gaps** | "Found X and Y, but couldn't find info about Z" |

## 5. Conversation

| Requirement | Details |
|-------------|---------|
| **Follow-ups** | Yes - within same topic ("tell me more about X you mentioned") |
| **Full memory** | Not required (session-based context sufficient) |
| **Meta-questions** | Nice to have ("which doc had most info?", "any contradictions?") |

## 6. Performance

| Requirement | Details |
|-------------|---------|
| **Speed** | Flexible - up to 5 minutes acceptable for complex queries |
| **Adaptive effort** | System gauges query complexity, adjusts search depth |
| **Manual override** | Option to force "quick" vs "deep research" mode |
| **Indexing time** | Target: under 15-30 min for 1000 docs |

## 7. Accuracy (Non-negotiable)

| Requirement | Priority |
|-------------|----------|
| No hallucinations | **Critical** - especially no fake citations |
| No missing important info | **Critical** |
| Trustworthy enough to use without verifying everything | **Critical** |

## 8. Technical Constraints

| Constraint | Details |
|------------|---------|
| **Hardware baseline** | 32GB RAM, 8GB VRAM |
| **LLM size** | ~7B comfortable, 13B with quantization |
| **Model runtime** | Model-agnostic (Ollama, llama.cpp, online APIs later) |
| **Scalability** | Design for stronger hardware later (24GB VRAM) |

## 9. Interface

| Requirement | Details |
|-------------|---------|
| **Initial** | API-first with dev scripts |
| **Users** | Single user |
| **Future** | TUI/Web UI built on top of API |

## 10. Success Criteria

**Success:**
- Complex questions get well-sourced answers
- Would have taken 30+ minutes to compile manually
- Trustworthy enough to use directly in work

**Failure:**
- Hallucinations (especially with seemingly valid citations)
- Missing important information that exists in docs
- Too slow for conversational use (>5 min consistently)

---

## Backlog (Discuss Later)

1. **Document change tracking & relations**
   - Hash-based change detection
   - Cross-document link tracking  
   - Relation-aware re-indexing

2. **Image/diagram understanding**
   - Vision model integration
   - Extract entities from architecture diagrams

3. **Stronger hardware optimization**
   - Larger models (24GB VRAM)
   - Online model fallback

---

## Test Corpus

**Synthetic "SaaS Company" documentation:**
- ~50 documents
- Types: Architecture overview, module design docs, API specs, ADRs, how-to guides, meeting notes/RFCs, runbooks
- Cross-references between documents
- To be generated after requirements finalized

---
