# Hidden Gems: Adjacent Fields (Pre-LLM IR, Recommenders, Search)

Generated: 2026-01-25
Adjacent Field Techniques Evaluated: 5
Promising Cross-Field Gems Identified: 3

---

## Gem 1: Pseudo-Relevance Feedback (Classic IR)

**Source**: Pre-LLM Information Retrieval literature (Rocchio algorithm, 1971)

**Technique**: After initial retrieval, use the TOP retrieved documents to expand the query, then re-retrieve.

```python
# Step 1: Initial retrieval
initial_results = retrieve(user_query, k=10)

# Step 2: Extract terms from top results
top_docs = initial_results[:3]  # Top 3 assumed relevant
expansion_terms = extract_key_terms(top_docs)  # TF-IDF, YAKE, etc.

# Step 3: Expand query
expanded_query = user_query + " " + " ".join(expansion_terms)

# Step 4: Re-retrieve with expanded query
final_results = retrieve(expanded_query, k=5)
```

**Evidence from Adjacent Field**:
- Classic IR technique from 1970s-1990s
- Rocchio algorithm: proven to improve recall
- Used in search engines before neural methods
- Still used in Elasticsearch "More Like This" queries

**Applicability to RAG**:
- Works on any corpus size
- No LLM required (uses TF-IDF or YAKE for term extraction)
- Addresses: vocabulary mismatch, query incompleteness
- Can be combined with LLM query rewriting

**Implementation Complexity**: Low-Medium
- Extract terms from top docs (TF-IDF, YAKE)
- Append to query
- Re-retrieve
- Adds one extra retrieval pass (~50-100ms)

**Why It's a Gem**:
- Unconventional in LLM era: Uses retrieved docs to improve query
- Proven technique from classic IR
- Addresses: vocabulary mismatch (learns corpus vocabulary from top results)
- Simple, no LLM required

**Potential Issues**:
- If initial retrieval is completely wrong, expansion makes it worse
- Needs good initial results to work

---

## Gem 2: BM25F (Field-Weighted BM25)

**Source**: Search engine literature (Robertson & Zaragoza, 2009)

**Technique**: Instead of treating all text equally, weight different FIELDS differently in BM25 scoring. Headings get higher weight than body text.

```python
# Standard BM25: All text treated equally
score = bm25(query, chunk_text)

# BM25F: Field-weighted
score = (
    bm25(query, heading) * 3.0 +      # Headings weighted 3x
    bm25(query, first_paragraph) * 2.0 +  # First para 2x
    bm25(query, body_text) * 1.0      # Body text 1x
)
```

**Evidence from Adjacent Field**:
- Used in search engines (Elasticsearch field boosting)
- Academic papers show 10-20% improvement over standard BM25
- Solr, Elasticsearch both support field boosting

**Applicability to RAG**:
- Works on any structured content (markdown with headings)
- Addresses: "wrong section" - headings are strong signals
- Can be combined with our existing BM25

**Implementation Complexity**: Medium
- Requires parsing chunks into fields (heading, body, code, etc.)
- Modify BM25 scoring to weight fields
- Need to tune field weights

**Why It's a Gem**:
- Unconventional in RAG: Most systems treat all text equally
- Proven in search engines
- Addresses: "wrong section" - headings are semantic anchors
- We already have heading metadata from MarkdownSemanticStrategy!

**Potential Issues**:
- Requires structured content (headings, sections)
- Need to tune field weights per corpus

---

## Gem 3: Collaborative Filtering for Content (Recommender Systems)

**Source**: Recommender systems literature (item-item collaborative filtering)

**Technique**: Track which chunks are FREQUENTLY RETRIEVED TOGETHER. If chunk A is retrieved, boost chunks that are often retrieved with A.

```python
# Build co-occurrence matrix at query time
co_occurrence = {}  # {chunk_id: {other_chunk_id: count}}

# When user retrieves chunks for a query
retrieved_chunks = [A, B, C, D, E]

# Update co-occurrence
for chunk in retrieved_chunks:
    for other_chunk in retrieved_chunks:
        if chunk != other_chunk:
            co_occurrence[chunk][other_chunk] += 1

# At retrieval time
initial_results = retrieve(query, k=10)
top_chunk = initial_results[0]

# Boost chunks that co-occur with top chunk
for chunk in initial_results[1:]:
    chunk.score += co_occurrence[top_chunk][chunk.id] * 0.1
```

**Evidence from Adjacent Field**:
- Amazon's "Customers who bought X also bought Y"
- Netflix's "Because you watched X"
- Proven to improve discovery in recommender systems

**Applicability to RAG**:
- Works on any corpus size
- Learns from usage patterns (which chunks answer similar questions)
- Addresses: "fragmented facts" - if facts are spread, they'll co-occur
- Requires query logs to build co-occurrence matrix

**Implementation Complexity**: Medium-High
- Requires tracking query logs
- Building co-occurrence matrix
- Updating scores at retrieval time
- Cold start problem (no data initially)

**Why It's a Gem**:
- HIGHLY unconventional: Borrows from recommender systems
- Learns from usage patterns
- Addresses: "fragmented facts" - related chunks boost each other
- Self-improving over time

**Potential Issues**:
- Requires query logs (cold start)
- May reinforce biases (popular chunks get more popular)
- Privacy concerns if tracking user queries

---

## Evaluated But Not Applicable

### 1. PageRank-Style Authority
**Why Not**: Requires link graph. Documents don't link to each other in our corpus.

### 2. Learning to Rank (LTR)
**Why Not**: Requires large training dataset with relevance labels. We don't have this.

### 3. Faceted Search
**Why Not**: Requires pre-defined facets (categories, tags). We have doc_type but that's already covered by metadata enrichment.

### 4. Query Logs for Spelling Correction
**Why Not**: Our queries are generated, not user-typed. Spelling errors not a problem.

### 5. Diversification (MMR - Maximal Marginal Relevance)
**Why Not**: We want RELEVANT results, not diverse results. Diversification reduces relevance.

---

## Key Insights from Adjacent Fields

### What Works in Search Engines:
1. **Field weighting** - Headings matter more than body text
2. **Query expansion** - Use retrieved docs to improve query
3. **Hybrid scoring** - Combine multiple signals (we already do this)

### What Works in Recommender Systems:
1. **Co-occurrence** - Items retrieved together are related
2. **Collaborative filtering** - Learn from usage patterns
3. **Cold start solutions** - Use content-based until you have usage data

### What Doesn't Transfer Well:
1. **Link analysis** - Documents don't link to each other
2. **Click-through rate** - No user clicks in RAG
3. **Dwell time** - No user engagement signals

---

## Recommendations

**High Priority**:
1. **BM25F (Field-Weighted BM25)** - We already have heading metadata! Low complexity to implement.

**Medium Priority**:
2. **Pseudo-Relevance Feedback** - Simple, proven, addresses vocabulary mismatch

**Low Priority**:
3. **Collaborative Filtering** - Requires query logs, cold start problem, but interesting for long-term

---

## Implementation Notes

### BM25F is Easiest Win
We already have:
- Heading metadata from MarkdownSemanticStrategy
- BM25 implementation in enriched_hybrid_llm

Just need to:
1. Parse chunks into fields (heading, body, code)
2. Weight fields in BM25 scoring
3. Tune weights (start with heading=3x, body=1x)

### Pseudo-Relevance Feedback
Can be added as a post-processing step:
1. Get initial top-5 results
2. Extract YAKE keywords from top-3
3. Append to query
4. Re-retrieve
5. Compare results

### Collaborative Filtering
Requires infrastructure:
- Query log storage
- Co-occurrence matrix computation
- Real-time score boosting
- Probably not worth it for small corpus
