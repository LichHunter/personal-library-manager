# Hidden Gems: First Principles Analysis

Generated: 2026-01-25
Fundamental Questions Explored: 7
Novel Hypotheses Generated: 5

---

## Fundamental Questions

### Q1: What does "semantic similarity" actually measure? Is it the right metric?

**Analysis**:
Semantic similarity (cosine similarity of embeddings) measures:
- **What it captures**: General topic similarity, paraphrase detection, conceptual relatedness
- **What it misses**: 
  - Negation ("do X" ≈ "don't do X" in embedding space)
  - Technical terms (rare words get poor embeddings)
  - Code structure (syntax matters, not just semantics)
  - Exact matches (BM25 better for "PgBouncer" exact term)

**Evidence from our failures**:
- 31% of failures are NEGATION_BLIND - embeddings can't distinguish "do" vs "don't"
- 25% are EMBEDDING_BLIND - technical terms, code, YAML not captured
- BM25 catches 50% of what embeddings miss

**Hypothesis 1**: **Semantic similarity is the WRONG metric for negation queries and code.**

**Testable Experiment**:
- For negation queries: Use BM25 ONLY (weight=1.0), disable semantic
- For code queries: Use structure-based retrieval (language maps)
- Measure: Does accuracy improve for these query types?

---

### Q2: Are chunks the right unit? What if we retrieve sentences? Paragraphs? Sections?

**Analysis**:
Current: 512-token chunks (MarkdownSemanticStrategy)

**Alternatives**:
1. **Sentences**: Too granular, loses context
2. **Paragraphs**: Better semantic unit, but may be too small
3. **Sections** (heading + content): Preserves full context, but may be too large
4. **Propositions** (atomic facts): Anthropic's approach - each fact is a unit

**Evidence from our failures**:
- 19% are FRAGMENTED - answer spread across chunks
- 25% are WRONG_SECTION - right doc, wrong section
- Smart chunking (MarkdownSemantic) already improved from 54% to 94%

**Hypothesis 2**: **Sections (heading + all content under heading) might be better than chunks for our corpus.**

**Testable Experiment**:
- Create "section-level" chunks (entire section under each h2/h3)
- Compare retrieval accuracy vs current 512-token chunks
- Trade-off: Larger context window usage vs better context preservation

---

### Q3: Is embedding ALL content the right approach? What if we embed differently?

**Analysis**:
Current: Embed entire chunk text (heading + body + code)

**Alternatives**:
1. **Embed heading separately**: Headings are semantic anchors
2. **Embed code separately**: Code has different semantics than prose
3. **Embed questions**: Generate questions the chunk answers, embed those
4. **Embed summaries**: Summarize chunk, embed summary

**Evidence from our failures**:
- 25% are EMBEDDING_BLIND - code, YAML don't embed well
- Headings are strong signals but get diluted in full chunk embedding

**Hypothesis 3**: **Embedding headings separately and giving them higher weight would improve "wrong section" failures.**

**Testable Experiment**:
- Create TWO embeddings per chunk: heading_embedding + body_embedding
- At retrieval: `score = 0.7 * sim(query, heading) + 0.3 * sim(query, body)`
- Measure: Does this reduce WRONG_SECTION failures?

---

### Q4: What information is LOST in the embedding? Can we preserve it?

**Analysis**:
Embeddings are lossy compression (768 dimensions from thousands of tokens).

**What's lost**:
1. **Exact keywords**: "PgBouncer" becomes a vector, loses exact match
2. **Negation**: "not", "don't", "avoid" get embedded away
3. **Structure**: Code indentation, YAML hierarchy, list order
4. **Emphasis**: Bold, italics, ALL_CAPS lost

**Evidence from our failures**:
- 50% have BM25_MISS - exact keywords not matched
- 31% have NEGATION_BLIND - negation lost in embedding

**Hypothesis 4**: **Preserving exact keywords alongside embeddings (hybrid search) is essential, but we need to weight BM25 HIGHER for queries with exact technical terms.**

**Testable Experiment**:
- Detect technical terms in query (camelCase, snake_case, ALL_CAPS, etc.)
- If >30% of query is technical: BM25 weight = 0.7, semantic weight = 0.3
- If <30% technical: BM25 weight = 0.4, semantic weight = 0.6
- Measure: Does adaptive weighting reduce EMBEDDING_BLIND failures?

---

### Q5: What does the user ACTUALLY need? Not just similar text, but answers.

**Analysis**:
Current: Retrieve chunks similar to query

**What user actually needs**:
1. **Answer to question**: Not just related text, but text that ANSWERS
2. **Context**: Enough surrounding info to understand the answer
3. **Confidence**: Know if answer is definitive or uncertain
4. **Source**: Which document, which section

**Evidence from our failures**:
- Retrieved chunks often CONTAIN the fact but don't ANSWER the question
- Example: Query "What NOT to do?" retrieves "What TO do" (contains keywords but wrong answer)

**Hypothesis 5**: **We should retrieve chunks that ANSWER the query, not just chunks SIMILAR to the query.**

**Testable Experiment**:
- Generate synthetic questions for each chunk at indexing time
- Embed the QUESTIONS, not the chunks
- At retrieval: Match user query to synthetic questions
- Return chunks whose questions match user query
- Measure: Does this improve answer quality?

---

### Q6: Why do negation queries fail so badly?

**Deep Analysis**:
31% of failures are NEGATION_BLIND. Why?

**Root cause**:
1. **Embeddings**: "do X" and "don't do X" have similar embeddings (same words, different meaning)
2. **BM25**: Matches "do" and "X" but ignores "don't" (stopword or low weight)
3. **Query rewriting**: LLM might rewrite "what NOT to do" → "what to avoid" (loses negation)

**Evidence**:
- Query: "What should I NOT do when rate limited?"
- Retrieved: "What TO do when rate limited" (opposite answer!)
- BM25 matched "rate limited" but ignored "NOT"

**Hypothesis 6**: **Negation queries need special handling - detect negation, then FILTER OUT positive advice.**

**Testable Experiment**:
- Detect negation in query ("not", "don't", "avoid", "never")
- If negation detected:
  - Retrieve as normal
  - Post-filter: Remove chunks that give positive advice
  - Boost chunks that contain negation words
- Measure: Does this reduce NEGATION_BLIND failures?

---

### Q7: Why is code/YAML invisible to retrieval?

**Deep Analysis**:
25% of failures are EMBEDDING_BLIND, mostly for code/YAML content.

**Root cause**:
1. **Embeddings**: Trained on prose, not code. Code syntax doesn't embed well.
2. **BM25**: Code has different tokenization (camelCase, snake_case, symbols)
3. **Chunking**: Code blocks are atomic but get embedded as text

**Evidence**:
- PgBouncer config (YAML) not retrieved despite 33 mentions
- HPA config (YAML) not retrieved
- Code examples poorly matched

**Hypothesis 7**: **Code/YAML should be indexed separately with structure-aware retrieval.**

**Testable Experiment**:
- Detect code blocks in chunks (```yaml, ```python, etc.)
- Index code blocks separately with:
  - AST-based search (for code)
  - Key-value search (for YAML)
  - Exact match on config keys
- At retrieval: If query mentions config/code terms, search code index
- Measure: Does this reduce EMBEDDING_BLIND for code queries?

---

## Novel Hypotheses Summary

| Hypothesis | Addresses Failure Mode | Complexity | Expected Impact |
|------------|----------------------|------------|-----------------|
| H1: Disable semantic for negation | NEGATION_BLIND (31%) | Low | High |
| H2: Section-level chunks | FRAGMENTED (19%) | Medium | Medium |
| H3: Separate heading embeddings | WRONG_SECTION (25%) | Medium | Medium |
| H4: Adaptive BM25/semantic weights | EMBEDDING_BLIND (25%) | Low | High |
| H5: Question-based retrieval | Multiple | High | High |
| H6: Negation-aware filtering | NEGATION_BLIND (31%) | Low | High |
| H7: Structure-aware code retrieval | EMBEDDING_BLIND (25%) | High | Medium |

---

## Prioritized Experiments

### Tier 1: Low Complexity, High Impact
1. **H4: Adaptive BM25/semantic weights** - Detect technical terms, adjust weights
2. **H6: Negation-aware filtering** - Detect negation, filter/boost accordingly
3. **H1: Disable semantic for negation** - Simple flag based on query analysis

### Tier 2: Medium Complexity, Medium-High Impact
4. **H3: Separate heading embeddings** - Two embeddings per chunk
5. **H2: Section-level chunks** - Change chunking strategy

### Tier 3: High Complexity, Medium-High Impact
6. **H5: Question-based retrieval** - Generate questions at indexing
7. **H7: Structure-aware code retrieval** - Separate code index

---

## Key Insights

### What We Learned:
1. **Semantic similarity is wrong for negation** - Need special handling
2. **Code/YAML is invisible** - Need structure-aware retrieval
3. **Headings are semantic anchors** - Should be weighted higher
4. **Technical terms need exact match** - BM25 should dominate for these queries
5. **Users need answers, not similar text** - Question-based retrieval might help

### What to Question:
1. **"Embeddings solve everything"** - NO. They fail on negation, code, technical terms.
2. **"Bigger context is better"** - NO. Right context is better.
3. **"Semantic search is modern"** - NO. BM25 is essential for exact matches.
4. **"One retrieval strategy fits all"** - NO. Different query types need different strategies.

---

## Recommended Next Steps

1. **Implement H4 (Adaptive weights)** - Quick win, addresses 25% of failures
2. **Implement H6 (Negation filtering)** - Quick win, addresses 31% of failures
3. **Test H3 (Heading embeddings)** - Medium effort, addresses 25% of failures
4. **Evaluate H5 (Question-based)** - High effort but potentially transformative

---

## Anti-Patterns to Avoid

1. **"One embedding model for everything"** - Different content types need different approaches
2. **"More data solves it"** - No, better retrieval strategy solves it
3. **"Just tune hyperparameters"** - No, fundamental approach is wrong for some query types
4. **"Semantic search is always better"** - No, BM25 is better for exact matches
