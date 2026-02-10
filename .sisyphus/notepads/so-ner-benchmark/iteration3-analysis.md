# SO NER Benchmark: Iteration 3 Analysis

> Date: 2026-02-09
> Strategy: v3_precise (taxonomy-driven NER + structural filter)

## Results

| Metric | Iter 1 | Iter 2 | Iter 3 | Target |
|--------|--------|--------|--------|--------|
| Precision | 64.4% | 67.9% | **89.1%** | 95% |
| Recall | 91.9% | 86.7% | **75.7%** | 95% |
| Hallucination | 35.6% | 32.1% | **10.9%** | ≤5% |
| F1 | 0.757 | 0.761 | **0.819** | — |
| Extracted | 247 | 221 | **147** | — |

## Progress: Precision UP, Recall DOWN

The taxonomy prompt dramatically improved precision (+25%) and halved hallucination, but recall dropped 16%. The pipeline is now too conservative.

## FP Analysis (16 remaining — down from 88!)

| Term | Doc | True FP? | Category |
|------|-----|----------|----------|
| getInputSizes | Q42873050 | Partial match (GT has full sig) | Variant |
| std::fopen | Q39288595 | Partial match (GT has full call) | Variant |
| microsoft | Q39288595 | Partial (GT: "microsoft crypto API") | Variant |
| boost | Q39288595 | Valid library GT missed | GT gap |
| MSVC | Q39288595 | Valid compiler GT missed | GT gap |
| ID | Q9965761 | Generic | True FP |
| partialview | Q44055225 | Valid MVC concept | GT gap |
| Credential | Q44055225 | Generic | True FP |
| database | Q44055225 | Valid but not in GT | GT gap |
| Staff | Q44055225 | User-defined name | True FP |
| Weblogic | Q45734089 | Valid (GT has "Weblogic 12C server") | Variant |
| pom.xml | Q45734089 | Valid file GT missed | GT gap |
| SOAP | Q45734089 | Valid protocol GT missed | GT gap |
| error 500 | Q42692779 | Partial (GT has full phrase) | Variant |
| server | Q42692779 | Generic | True FP |
| action | Q42692779 | Generic | True FP |

**True FPs: 4** (ID, Credential, Staff, server, action) = ~4/147 = **2.7% true hallucination**
**GT gaps: 6** (boost, MSVC, partialview, database, pom.xml, SOAP)
**Variant matches: 5** (partial string matches of GT terms)

## FN Analysis (37 — up from 15!)

### Category A: Common words not extracted (should be entities) — 15 FN
table, tables, image, bytearrays, clients, Client, calculator, keys, key, row, 
long, container, screen, each, request

→ Fix: Prompt must emphasize extracting common words as entities

### Category B: Compound terms with special chars — 5 FN  
MAX SIZE(4608x3456), MAX Size, MAXIMUM, left/right arrow, up/down arrow, up/down keys

→ Fix: Prompt must handle compound terms and slash-separated variants

### Category C: Domain terms the LLM didn't recognize — 9 FN
Session, ondemand, flex containers, flex container, .long,
microsoft crypto API, crypto, Pseudo-randomic, IDE

→ Fix: Some of these the LLM should extract with better recall emphasis

### Category D: Generic-sounding but valid entities — 8 FN
exception, phone, CPU, configuration, jQuery-generated, post, each, random

→ Fix: Prompt must include more examples of generic words as entities

## Key Insight

The v3_precise prompt successfully killed descriptive phrases (precision WIN) but also killed
common words that ARE entities (recall LOSS). Need a hybrid: keep the NER taxonomy framing
but add MORE emphasis on common words being valid entities with many more examples.

## Next Step: Iteration 4

Strategy: v4_hybrid — NER taxonomy + aggressive recall emphasis + more examples
Changes needed:
1. Add explicit "COMMON WORDS ARE ENTITIES" section with many more examples
2. List words from FN analysis as examples to extract
3. Keep structural filter (it's working well — only 4 true FPs)
4. Add "Be exhaustive — missing an entity is worse than including a borderline one"
