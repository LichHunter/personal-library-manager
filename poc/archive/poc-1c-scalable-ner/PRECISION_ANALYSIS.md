# V5.3 Precision Failure Analysis & Action Plan

## Current State

| Scale | P | R | F1 | TP | FP | FN |
|-------|---|---|-----|----|----|-----|
| 10 docs | 90.7% | 94.6% | 0.926 | 194 | 20 | 11 |
| 50 docs | 79.5% | 93.1% | 0.858 | 727 | 187 | 54 |

**Target: P >= 95%, R >= 90%**

---

## Root Cause Analysis

### FP Categorization (187 FPs from 50 docs)

| Category | Count | % | Examples |
|----------|-------|---|---------|
| Seed/bypass auto-keeps | 51 | 27% | html(3), server(2), xml(2), php(2), java, json, c++, android |
| Common-word-rule victims | 39 | 21% | event(4), controller(3), key(2), page(2), closure, middleware, await |
| Structural auto-keeps | 33 | 18% | ActiveModel, GUID, IBM, DTC, M3, Answer_to_Question_ID |
| Other single-word | 29 | 16% | cascade, boost, configuration, height, width, Resources |
| Multi-word phrases | 26 | 14% | "stack trace", "debug info", "media queries", "key event" |
| URLs/paths | 9 | 5% | .net/meo/yKgSf, docs.spring.io, pic.dhe.ibm.com |

### Root Cause #1: haiku_simple extractor (P=60.5%)

Individual extractor benchmark (10 docs):

| Extractor | P | R | F1 | FPs | TPs |
|-----------|---|---|-----|-----|-----|
| haiku_simple | 60.5% | 71.4% | 0.655 | 88 | 135 |
| sonnet_taxonomy | 75.2% | 93.2% | 0.832 | 77 | 233 |
| haiku_fewshot | 85.6% | 81.6% | 0.836 | 27 | 160 |
| heuristic | 84.3% | 28.9% | 0.431 | 11 | 59 |
| seeds | 76.7% | 23.2% | 0.357 | 14 | 46 |

**haiku_simple is our worst extractor by a wide margin.** In doc Q3639039, it extracted 37 terms against 13 GT (P=29.7%), dumping string, int, long, boolean, var, list, server, browser, console, key. Its recall (71.4%) is also LOWER than sonnet_taxonomy (93.2%), so its only contribution is as a diversity vote — but most of its unique extractions are FPs.

### Root Cause #2: Seed/bypass auto-keep mechanism (51 FPs)

Terms with entity_ratio >= 0.50 in training data get added to seeds/bypass lists and auto-keep without validation. But entity_ratio is a STATISTICAL property — "html" has ratio 0.92 (annotated 59/64 times in training), yet the GT doesn't annotate it in 3 test docs. The bypass mechanism can't handle contextual annotation.

Key FP seed/bypass terms with their entity ratios:
- html: 0.92, xml: 0.95, php: 0.95, java: 0.95, c++: 1.00, json: 1.00 (high ratio but GT skips)
- server: 0.57, window: 0.67, client: 0.57, table: 0.82 (moderate ratio, contextual)
- route: 0.50, assembly: 0.50, decimal: 0.50 (borderline)

### Root Cause #3: Common-Word Rule in prompts

The SHARED_ENTITY_DEFINITION says:
> "string" = Data_Type, "button" = UI_Element, "server" = Application

This trains extractors to ALWAYS extract these words. But GT annotation is contextual — "server" is only an entity when it's the FOCUS of discussion, not an incidental mention. The validation prompt uses the same definition, so it ALSO approves them.

**Both Oracle consultations independently identified this as the fundamental prompt design flaw.**

---

## Action Plan (v5.4)

### Phase 1: Quick Wins (expected: -60-80 FPs, P: 79% -> 87-89%)

#### Fix 1: Drop or Replace haiku_simple

**Option A (recommended)**: Replace haiku_simple with a PRECISION-FOCUSED Haiku prompt:

```
HAIKU_PRECISE_PROMPT:
"Extract ONLY terms that name SPECIFIC technical things in this text.
DO NOT extract:
- Generic terms like 'server', 'button', 'key' unless they name something SPECIFIC
- Common words that could be replaced by a synonym without changing meaning
- Terms used descriptively rather than as named references

ONLY extract: Named libraries, specific class/function names, programming languages,
specific tools/products, error names, file formats, API endpoints, specific versions."
```

**Option B**: Drop haiku_simple entirely. sonnet_taxonomy (R=93.2%) covers its recall with +15pp precision.

#### Fix 2: Remove seed/bypass auto-keep, route through validation

Currently: seeds in bypass set skip validation entirely.
Change: ALL terms go through Sonnet validation, regardless of seed/bypass status. The entity_ratio evidence in validation prompt is sufficient calibration.

This targets 51 FPs (27% of total).

#### Fix 3: Fix URL/path filtering

9 FPs from URLs that survive post-processing. Tighten regex:
- Catch `.net/path/` patterns (currently only catches `http://`)
- Catch `pic.domain.com`, `docs.domain.io` patterns
- Catch `*.aspx`, `*.html` suffixes in non-code context

### Phase 2: Prompt Redesign (expected: additional -30-40 FPs, P: 87% -> 91-93%)

#### Fix 4: Rewrite SHARED_ENTITY_DEFINITION to be contextual

Replace unconditional Common-Word Rule with contextual test:

```
OLD: "string" = Data_Type (always)
NEW: "string" = Data_Type ONLY when the text discusses string operations,
     string methods, or String class. NOT when "string" appears incidentally.

THE CONTEXTUAL TEST:
Would removing this term break understanding of the technical solution?
- "use String.format()" -> removing "String" breaks understanding -> ENTITY
- "parse the string" -> replacing with "text" doesn't break meaning -> NOT ENTITY
```

#### Fix 5: Add contrastive negative examples to extraction prompts

Show extractors WHAT NOT TO EXTRACT:

```
THESE ARE NOT ENTITIES (generic usage):
- "the server returned an error" -> 'server' = generic, NOT entity
- "click the button" -> 'button' = generic UI word, NOT entity
- "parse the xml" -> 'xml' = incidental mention, NOT entity

THESE ARE ENTITIES (specific referent):
- "Configure Apache server" -> 'Apache' = specific product, ENTITY
- "The Button component" -> 'Button' = specific React component, ENTITY
- "using lxml library" -> 'lxml' = specific library, ENTITY
```

#### Fix 6: Rewrite validation prompt with salience test

Replace current "is this an entity?" with "is this SALIENT here?":

```
"For each term, apply the SALIENCE TEST:
Would a StackOverflow reader need to know specifically what this term
refers to in order to understand the answer?
- YES: ENTITY (it's a specific tool/library/type being discussed)
- NO: GENERIC (any programmer knows this generic concept)"
```

### Phase 3: Advanced Techniques (expected: additional -20-30 FPs, P: 91% -> 93-95%)

#### Fix 7: Self-consistency voting for extraction

Run each LLM extractor 3x at temperature=0.7, keep only terms appearing in >= 2/3 runs. True entities are consistently extracted; one-off FPs are noise that won't replicate.

Cost: 3x Haiku calls ≈ 1x Sonnet call. Can replace sonnet_taxonomy with 3x haiku_taxonomy to stay budget-neutral.

#### Fix 8: Structural auto-keep guard tightening

33 FPs from structural patterns (CamelCase, ALL_CAPS, dot-paths). Currently entity_ratio=0 guard exists but ALL_CAPS with high ratio still auto-keep.

Add: ALL structural auto-keeps require at least 1 LLM vote confirmation OR entity_ratio >= 0.7.

#### Fix 9: Multi-word phrase filter tightening

26 FPs from multi-word phrases. Add negative patterns:
- "X trace" where X in {stack, diagnostic}
- "X info" where X in {debug, ...}
- "X queries" where X in {media, ...}
- Phrases where ALL words are common English (no technical markers)

---

## Haiku Prompt Variant Designs

### Variant 1: HAIKU_PRECISE (replace haiku_simple)

Focus: HIGH PRECISION, moderate recall. Only extract clear-cut entities.

```python
HAIKU_PRECISE_PROMPT = """\
Extract software named entities from this StackOverflow text.

ONLY extract terms that are SPECIFIC named things:
- Named libraries/frameworks: React, jQuery, NumPy, boost
- Class/function/method names: ArrayList, querySelector(), .map()
- Programming languages: Java, Python, C#, JavaScript, SQL
- Specific tools/products: Docker, Chrome, Visual Studio, Android
- Named file formats/types: JSON, XML, CSV, WSDL, .xaml
- Specific error names: NullPointerException, error 500
- Named OS/platforms: Linux, Windows, macOS, iOS
- Specific versions: v3.2, ES6, Silverlight 4

DO NOT extract:
- Generic words even if technically valid: server, button, string, table,
  key, page, event, controller, list, form, screen, view, model
- Descriptive phrases: "vertical orientation", "debug info"
- Process words: "serializing", "loading"
- Common programming concepts: interface, repository, command, handler

When in doubt, SKIP the term. Missing a borderline term is better than
including a generic word.

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
```

### Variant 2: HAIKU_CONTEXTUAL (alternative to haiku_simple)

Focus: Context-sensitive extraction using the salience test.

```python
HAIKU_CONTEXTUAL_PROMPT = """\
Extract software named entities from this StackOverflow text.

An entity is a term that names a SPECIFIC technical thing that the reader
needs to understand to follow the solution. Apply the SALIENCE TEST:

Could you replace this term with a generic synonym without losing meaning?
- "import React" -> can't replace "React" -> ENTITY
- "the server crashed" -> "the machine crashed" works -> NOT entity
- "ArrayList<String>" -> can't replace -> ENTITY
- "store in a table" -> "store in a structure" works -> NOT entity

Extract: specific library/tool/class/function names, programming languages,
named APIs, file formats used specifically, error types, versions, keyboard keys.

Skip: generic technical vocabulary (server, button, key, controller, event,
model, view, handler) unless the text explicitly discusses that specific type
(e.g., "the Button component" = entity, "click the button" = not entity).

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
```

### Variant 3: HAIKU_TYPE_SPECIALIST_CODE

Focus: Only code identifiers (classes, functions, variables).

```python
HAIKU_CODE_IDENTIFIERS_PROMPT = """\
Extract ONLY code identifiers from this StackOverflow text.

Code identifiers are programmer-defined names that appear in source code:
- Class names: ArrayList, HttpClient, WebClient, IEnumerable
- Function/method names: querySelector(), recv(), map(), send()
- Variable/constant names: MAX_SIZE, isEnabled, userData
- Module/package paths: System.out, R.layout, com.ibm.websphere

DO NOT extract:
- Programming language names (Java, Python, etc.)
- Generic type names (string, int, boolean, array, list)
- Tool/product names (Docker, Chrome, etc.)
- Concepts or descriptions

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
```

### Variant 4: HAIKU_CONSERVATIVE_FEWSHOT (replace haiku_fewshot prompt)

Focus: Add negative examples to few-shot prompt to teach what GT skips.

```python
RETRIEVAL_PROMPT_WITH_NEGATIVES = """\
You are extracting software named entities from StackOverflow text.

Here are examples of correct annotation from similar posts.
Note both what IS and what IS NOT annotated:

{examples_block}

TERMS THAT WERE NOT ANNOTATED (even though they appear in the text):
{negative_examples_block}

---

Now extract ALL entities from this text. Follow the annotation STYLE of the
examples above. Notice what the annotators chose NOT to annotate — generic
terms like "server", "controller", "event" are typically NOT entities unless
they name something specific.

TEXT: {text}

Return ONLY a JSON array of entity strings. No explanations.
ENTITIES:"""
```

---

## Expected Results by Phase

| Phase | Changes | Expected P | Expected R | FPs Eliminated |
|-------|---------|-----------|-----------|----------------|
| Current (v5.3) | - | 79.5% | 93.1% | - |
| Phase 1 | Drop haiku_simple, remove seed bypass, fix URLs | 87-89% | 91-93% | ~70 |
| Phase 2 | Contextual prompts, negative examples, salience validation | 91-93% | 90-92% | ~35 |
| Phase 3 | Self-consistency, structural guard, phrase filter | 93-95% | 90-91% | ~25 |

---

## Implementation Order

1. **v5.4a**: Drop haiku_simple + remove seed/bypass auto-keep + fix URLs
2. **v5.4b**: Replace SHARED_ENTITY_DEFINITION with contextual version
3. **v5.4c**: Add HAIKU_PRECISE or HAIKU_CONTEXTUAL prompt variant
4. **v5.4d**: Add negative few-shot to retrieval prompt
5. **v5.4e**: Salience-based validation prompt
6. **v5.4f**: Self-consistency voting

Benchmark at each step on 10 docs (cheap), then validate winner on 50 docs.

---

## Risk Assessment

- **Recall risk**: Removing seed bypass and making prompts conservative may drop recall 2-4pp. Monitor closely.
- **The 51 seed/bypass FPs include terms with entity_ratio 0.92-1.00** (html, xml, php, java, json). These ARE real entities in most contexts — removing bypass will cause Sonnet validation to sometimes reject them incorrectly. We may need a high-entity-ratio safety net.
- **GT annotation gaps**: ~32% of current FPs (60/187) are terms the pipeline correctly identifies as entities but GT doesn't annotate. Even with perfect policy alignment, the measured precision ceiling is ~88-90% on this benchmark. The true precision (on genuinely wrong terms) is already ~85%.
