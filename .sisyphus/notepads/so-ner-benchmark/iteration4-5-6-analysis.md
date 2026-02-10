# SO NER Benchmark: Iterations 4-6 Analysis (Revised)

> Date: 2026-02-09
> Strategy: v4_hybrid (iter 4), v5_contextual (iter 5-6)

## Results Summary

| Metric | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | **Iter 6** | Target |
|--------|--------|--------|--------|--------|--------|------------|--------|
| Precision | 64.4% | 67.9% | 89.1% | 78.3% | 81.8% | **89.1%** | 95% |
| Recall | 91.9% | 86.7% | 75.7% | 96.0% | 96.0% | **94.8%** | 95% |
| Hallucination | 35.6% | 32.1% | 10.9% | 21.7% | 18.2% | **10.9%** | ≤5% |
| F1 | 0.757 | 0.761 | 0.819 | 0.862 | 0.883 | **0.919** | — |
| Extracted | 247 | 221 | 147 | 212 | 203 | **184** | — |
| GT terms | 173 | 173 | 173 | 173 | 173 | **173** | — |
| FPs | 88 | 71 | 16 | 46 | 37 | **20** | — |
| FNs | 15 | 23 | 37 | 11 | 10 | **12** | — |

## Iteration 4-6 Strategy Evolution

**Iter 4 (v4_hybrid)**: Aggressive recall emphasis + common-word-as-entity examples. Recall=96% but precision=78.3%.
**Iter 5 (v5_contextual v1)**: Added URL filter + context-aware Sonnet classifier for ambiguous common words. -9 FPs.
**Iter 6 (v5_contextual v2)**: Tightened context validator with explicit GENERIC word list. Best F1=0.919. -17 more FPs.

## HONEST FP Analysis: The GT Is NOT Wrong

After examining the actual BIO annotations in test.txt and the SO NER Annotation Guidelines (Tabassum et al.), the GT makes **deliberate, principled decisions** about what constitutes an entity. My earlier claim of "GT errors" was wrong.

### The 20 Remaining FPs — Properly Categorized

#### Category A: TRUE False Positives — Pipeline Over-Extracts (13)

These terms are **correctly not annotated** per the SO NER guidelines:

| FP | Doc | Why GT is right |
|----|-----|-----------------|
| LEVEL-3 | Q42873050 | Not a recognized named entity. An API constant value, but not annotated as such. |
| Target | Q42873050 | Generic English word in context, not a specific API class name to annotators. |
| ZSL | Q42873050 | Acronym for a feature concept, not a named entity per SO NER typology. |
| boost | Q39288595 | **Context-dependent**. GT labels "boost" as Library in train.txt but O in this test context. The annotators judged it's used generically here, not as a library reference. |
| MSVC | Q39288595 | GT deliberately does not label compiler names. Application type is for "software names that are NOT code libraries." MSVC is a compiler, falls between categories. |
| SOAP | Q45734089 | Protocols are not consistently annotated. The guidelines list UDP under Algorithm, but SOAP is not annotated here. Annotator judgment. |
| DSL | Q45734089 | Context-dependent. GT labels "DSL" as Language elsewhere but O here — used as a generic concept ("Java DSL"), not a specific language name. |
| gallery | Q9965761 | Generic English word, not a software entity. |
| items | Q9965761 | Part of a Value span ("Slide #1 of 3 items"), not an entity. |
| partial view | Q44055225 | Descriptive phrase, not a named entity per GT. |
| partialview | Q44055225 | Framework concept but not annotated as a named entity. |
| service class | Q45734089 | Descriptive phrase, not an entity name. |
| web-service | Q45734089 | Generic compound term. GT does not annotate hyphenated descriptors. |

#### Category B: PARTIAL MATCH — Pipeline Extracts Wrong Span (5)

Our pipeline extracts a **substring** of the GT entity, creating both a FP and a FN:

| FP (what we extract) | GT (what we should extract) | Type |
|----------------------|---------------------------|------|
| getInputSizes | getInputSizes(ImageFormat.YUV_420_888) | Library_Function |
| send() | NetStream.send() | Library_Function |
| std::fopen | std::fopen("/dev/urandom") | Library_Function |
| Weblogic | Weblogic 12C server | Device |
| error 500 | server internal error 500 / internal server error 500 | Error_Name |

**This is a real pipeline deficiency**: the GT follows a "maximum span" annotation rule — always annotate the longest valid span. Our pipeline truncates these.

#### Category C: SCORING SETUP Issue — We Excluded GT Types (2)

| FP | GT Type | Why excluded |
|----|---------|-------------|
| pom.xml | File_Name | We put File_Name in EXCLUDED_ENTITY_TYPES |
| ID | Variable_Name | We put Variable_Name in EXCLUDED_ENTITY_TYPES |

These are annotated entities that we chose to exclude from scoring. `pom.xml` IS labeled as B-File_Name. `ID` IS labeled as B-Variable_Name. Our choice to exclude these types causes them to appear as FPs.

### What This Means

**13 TRUE FPs (7.1% hallucination rate)**: The pipeline genuinely over-extracts terms the GT correctly does not label. Key patterns:
- Protocol/compiler names (SOAP, MSVC, DSL) — GT's entity typology doesn't cover these well
- Context-dependent words (boost) — GT annotators use context to decide
- Generic compound terms (web-service, partial view, service class)
- Generic nouns (gallery, items, Target)

**5 PARTIAL MATCHES (span errors)**: The pipeline extracts substrings of GT entities. This is a real extraction quality issue — we need better handling of:
- Qualified function names: `NetStream.send()` not just `send()`
- Function calls with arguments: `getInputSizes(ImageFormat.YUV_420_888)` not just `getInputSizes`
- Multi-word named entities: "Weblogic 12C server" not just "Weblogic"

**2 EXCLUDED TYPE**: Our scoring setup incorrectly excludes File_Name (163 entities in test set!) and Variable_Name. At minimum, File_Name should be in RELEVANT_ENTITY_TYPES.

## FN Analysis (12 Remaining)

| FN | GT Type | Root cause |
|----|---------|------------|
| tables, table | Data_Structure | LLM non-determinism (extracted in some runs, not others) |
| MAX SIZE(4608x3456) | Library_Variable | Unusual compound with parens — LLM doesn't extract |
| MAX Size | Library_Variable | Unusual casing pattern |
| MAXIMUM | Library_Variable | All-caps constant — LLM doesn't recognize as entity |
| up/down arrow | Keyboard_IP | Slash-separated variant not handled |
| ondemand | Library_Variable | Run-together compound |
| std::fopen("/dev/urandom") | Library_Function | Full function call with path argument — LLM truncates |
| exception | Library_Class | Context validator likely rejected as generic |
| Pseudo-randomic | Algorithm | Unusual hyphenated technical term |
| jQuery-generated | Library | Hyphenated compound with library prefix |
| configuration | Library_Function | Context validator likely rejected as generic |

## Key Insights

### 1. The GT is principled, not wrong
The SO NER annotations follow clear rules: maximum span length, context-dependent labeling, specific entity typology. Terms like "boost" being O is not an error — it's the annotator's context judgment.

### 2. Our pipeline has three real weaknesses
1. **Span extraction**: We extract partial spans (getInputSizes vs getInputSizes(ImageFormat.YUV_420_888)). The GT expects maximum spans.
2. **Over-extraction of near-entity terms**: We extract SOAP, MSVC, DSL, boost which are real software concepts but fall outside the GT's entity typology.
3. **Excluded type scoring**: File_Name exclusion is questionable — pom.xml is clearly a technical term.

### 3. The 95/95/5 target is unreachable without architectural changes
- **Precision ceiling with current approach**: ~89%. The remaining 11% FPs are real over-extractions of terms the GT correctly doesn't label.
- **Recall ceiling**: ~95%. The remaining FNs are genuinely hard (compound terms, context-dependent).
- To reach 95% precision, the pipeline needs to **not extract** terms like SOAP, boost, MSVC — even though they're valid software concepts. This requires understanding the GT's entity typology more precisely.

### 4. Fixing the partial match problem would help both metrics
If we extracted full spans instead of truncated ones:
- 5 FPs become 0 (getInputSizes→full form, send()→NetStream.send(), etc.)
- 5 FNs become 0 (the corresponding GT entries get matched)
- Adjusted: TP=169, FP=15, FN=7 → P=91.8%, R=96.0%, H=8.2%, F1=0.938

### 5. Fixing excluded types would further help
Adding File_Name back to relevant types: pom.xml becomes TP instead of FP.

## Recommendation

The benchmark reveals **real pipeline deficiencies**, not GT problems:

1. **Fix span extraction** (biggest win): Extract maximum spans for function calls and multi-word entities
2. **Add File_Name to relevant types**: It's a valid term extraction target
3. **Accept that some over-extraction is inherent**: The pipeline's concept of "technical term" is broader than SO NER's entity typology — terms like SOAP, MSVC, boost ARE useful for RAG but aren't SO NER entities
4. **Report dual metrics**: Raw benchmark scores AND "RAG-relevant" scores where near-miss FPs are acknowledged

Best achievable with current architecture: **~92% precision, ~96% recall** (after span fix + type fix).
True targets require a fundamentally different approach to match the GT's strict entity typology.
