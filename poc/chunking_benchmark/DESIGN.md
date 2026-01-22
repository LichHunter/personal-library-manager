# POC: Chunking Strategy Benchmark

## Objective

Determine the optimal document chunking/storage strategy for the Personal Knowledge Assistant by comparing 6 different approaches across retrieval accuracy, LLM answer quality, and performance metrics.

---

## Strategies to Test

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **Fixed-Size Chunks** | Split every N tokens (512) with overlap (50) |
| 2 | **Heading-Based Sections** | Split by markdown headings (H1, H2, H3) |
| 3 | **Heading + Size Limit** | Split by headings, subdivide if > max_tokens |
| 4 | **Hierarchical (Parent-Child)** | Store at multiple levels with parent-child links |
| 5 | **Semantic Paragraphs** | Split by paragraph boundaries, merge small ones |
| 6 | **Heading + Paragraph Hybrid** | Group paragraphs under their heading |

---

## Strategy Details

### Strategy 1: Fixed-Size Chunks (Baseline)

**Algorithm:**
```
for each document:
    split into chunks of 512 tokens
    overlap 50 tokens between chunks
    store each chunk with doc_id, chunk_index
```

**Output example:**
```
chunk_id: doc1_chunk_0
content: "Python is a high-level programming language. Its design philosophy..."
metadata: {doc_id: "doc1", index: 0, start_char: 0, end_char: 2048}
```

**Parameters:**
- chunk_size: 512 tokens
- overlap: 50 tokens

---

### Strategy 2: Heading-Based Sections

**Algorithm:**
```
for each document:
    parse markdown headings (H1-H6)
    extract content between headings as sections
    store each section with heading path
```

**Output example:**
```
section_id: doc1_sec_history
heading: "## History"
heading_path: ["Python (programming language)", "History"]
content: "Python was conceived in the late 1980s by Guido van Rossum..."
metadata: {doc_id: "doc1", level: 2, parent: "doc1_root"}
```

**Edge cases to handle:**
- No headings → treat entire doc as one section
- Very long sections → no splitting (test impact)
- Deeply nested (H4+) → flatten or preserve?

---

### Strategy 3: Heading + Size Limit

**Algorithm:**
```
for each document:
    parse markdown headings
    for each section:
        if token_count(section) <= max_tokens:
            store as-is
        else:
            split into sub-chunks with overlap
            preserve heading in each sub-chunk
```

**Output example:**
```
section_id: doc1_sec_implementation_1
heading: "## Implementation"
heading_path: ["Python", "Implementation"]
content: "## Implementation\n\nCPython is the reference implementation..."
metadata: {doc_id: "doc1", level: 2, part: 1, total_parts: 3}
```

**Parameters:**
- max_tokens: 512
- overlap: 50 tokens (for split sections)

---

### Strategy 4: Hierarchical (Parent-Child)

**Algorithm:**
```
for each document:
    create root node (doc summary/intro)
    for each heading level:
        create node with content
        link to parent heading
    store all nodes with relationships
```

**Output example:**
```
nodes:
  - id: doc1_root
    level: 0
    content: "Python is a high-level programming language..."
    children: [doc1_history, doc1_design, doc1_implementation]
    
  - id: doc1_history
    level: 1
    heading: "## History"
    content: "Python was conceived..."
    parent: doc1_root
    children: [doc1_history_origins, doc1_history_growth]
    
  - id: doc1_history_origins
    level: 2
    heading: "### Origins"
    content: "In the late 1980s..."
    parent: doc1_history
    children: []
```

**Retrieval modes to test:**
- Leaf only (most specific)
- Leaf + parent context
- All levels (collapsed search)

---

### Strategy 5: Semantic Paragraphs

**Algorithm:**
```
for each document:
    split by paragraph boundaries (\n\n)
    merge consecutive small paragraphs (< min_tokens)
    split large paragraphs (> max_tokens)
    store each paragraph unit
```

**Output example:**
```
paragraph_id: doc1_para_0
content: "Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands as a successor to the ABC programming language."
metadata: {doc_id: "doc1", index: 0, start_char: 0, end_char: 156}
```

**Parameters:**
- min_tokens: 50 (merge if smaller)
- max_tokens: 256 (split if larger)

---

### Strategy 6: Heading + Paragraph Hybrid

**Algorithm:**
```
for each document:
    parse headings to get sections
    for each section:
        split into paragraphs
        store section with paragraphs as sub-units
```

**Output example:**
```
section_id: doc1_sec_history
heading: "## History"
heading_path: ["Python", "History"]
full_content: "Python was conceived... [full section text]"
paragraphs:
  - id: doc1_sec_history_p0
    content: "Python was conceived in the late 1980s..."
  - id: doc1_sec_history_p1  
    content: "Van Rossum started implementation in December 1989..."
  - id: doc1_sec_history_p2
    content: "Python 2.0 was released in 2000..."
metadata: {doc_id: "doc1", level: 2, paragraph_count: 3}
```

**Retrieval modes to test:**
- Section level (return full section)
- Paragraph level (return specific paragraph)
- Paragraph + section context (paragraph with heading prepended)

---

## Test Corpus

**Source:** Synthetic SaaS company documentation (to be generated)

**Composition:**
- ~50 documents
- Mix of document types:
  - Architecture docs (deeply nested headings)
  - API specs (structured, many small sections)
  - Decision records (ADRs) (narrative, fewer headings)
  - How-to guides (step-by-step, medium structure)
  - Meeting notes (loose structure, paragraphs)

**Why synthetic:**
- Control over structure variety
- Known ground truth for evaluation
- Can craft specific test cases

---

## Test Queries

### Query Categories

| Category | Example | Expected behavior |
|----------|---------|-------------------|
| **Simple lookup** | "What database do we use?" | Find specific fact |
| **Section retrieval** | "Explain the authentication flow" | Return coherent section |
| **Cross-section** | "How do frontend and backend communicate?" | Multiple sections needed |
| **Comparison** | "Compare REST and GraphQL endpoints" | Find both, synthesize |
| **Exhaustive** | "List all environment variables" | Find all occurrences |
| **Decision** | "Why did we choose PostgreSQL?" | Find ADR/decision doc |

### Query Set Size
- 10 queries per category = 60 total queries
- Each query has ground truth: expected document(s), section(s), and answer

---

## Metrics

### 1. Retrieval Metrics

| Metric | Description | How to measure |
|--------|-------------|----------------|
| **Chunk Recall@K** | Is the relevant chunk in top-K results? | % of queries where ground truth chunk in top-K |
| **Document Recall@K** | Is the relevant document in top-K? | % of queries where ground truth doc in top-K |
| **MRR (Mean Reciprocal Rank)** | How high is the relevant chunk ranked? | 1/rank of first relevant result, averaged |
| **Context Coverage** | Does retrieved chunk contain full answer? | Manual or LLM evaluation |

**K values to test:** 1, 3, 5, 10

### 2. LLM Answer Quality Metrics

| Metric | Description | How to measure |
|--------|-------------|----------------|
| **Answer Correctness** | Is the answer factually correct? | Compare to ground truth answer |
| **Answer Completeness** | Does answer cover all aspects? | Checklist of expected points |
| **Groundedness** | Is every claim supported by retrieved chunks? | Verify citations |
| **Coherence** | Is the answer well-structured and readable? | 1-5 scale rating |

**Evaluation method:** 
- Automated: LLM-as-judge comparing to ground truth
- Manual: Sample review for validation

### 3. Chunk Quality Metrics

| Metric | Description | How to measure |
|--------|-------------|----------------|
| **Self-containedness** | Can chunk be understood without context? | LLM rating 1-5 |
| **Semantic coherence** | Does chunk cover one topic? | LLM rating 1-5 |
| **Boundary quality** | Does chunk start/end at natural points? | Manual inspection |
| **Size distribution** | How variable are chunk sizes? | std dev of token counts |

### 4. Performance Metrics

| Metric | Description | How to measure |
|--------|-------------|----------------|
| **Index time** | Time to process all documents | Wall clock time |
| **Storage size** | Database size on disk | File size in MB |
| **Chunks per document** | Average number of chunks | Count |
| **Retrieval latency** | Time for single query | Milliseconds |
| **Embedding calls** | Number of embedding API calls | Count |

---

## Experiment Design

### Phase 1: Corpus Generation (Day 1)
1. Generate synthetic SaaS documentation
2. Create ground truth annotations
3. Validate corpus quality

### Phase 2: Implementation (Days 2-3)
1. Implement all 6 chunking strategies
2. Create common embedding/storage interface
3. Build evaluation harness

### Phase 3: Indexing Benchmark (Day 4)
1. Run each strategy on full corpus
2. Measure: index time, storage size, chunk counts
3. Analyze chunk size distributions

### Phase 4: Retrieval Benchmark (Day 5)
1. Run all queries against each strategy
2. Measure: Recall@K, MRR, retrieval latency
3. Compare retrieval accuracy

### Phase 5: LLM Quality Benchmark (Days 6-7)
1. For each strategy, generate answers using LLM
2. Evaluate: correctness, completeness, groundedness
3. Compare answer quality

### Phase 6: Analysis & Report (Day 8)
1. Aggregate all metrics
2. Create comparison visualizations
3. Make recommendation

---

## Expected Outcomes

### Hypotheses

| Strategy | Expected strength | Expected weakness |
|----------|-------------------|-------------------|
| **1. Fixed-size** | Consistent retrieval | Poor coherence, cut sentences |
| **2. Heading-based** | High coherence | Variable size, some too large |
| **3. Heading + limit** | Balanced | Complexity, split sections |
| **4. Hierarchical** | Flexible granularity | Query complexity, overhead |
| **5. Paragraphs** | Fine-grained | Loses heading context |
| **6. Heading + para** | Best of both | Most complex, storage overhead |

### Success Criteria

The winning strategy should:
1. Achieve **>90% Recall@5** for simple lookups
2. Achieve **>80% Answer Correctness** for section retrieval
3. Produce **self-contained chunks** (avg rating >4/5)
4. Index 50 docs in **<5 minutes**
5. Not exceed **2x storage** vs baseline

### Decision Matrix

| Weight | Metric | Why important |
|--------|--------|---------------|
| 30% | LLM Answer Quality | End goal is good answers |
| 25% | Retrieval Accuracy | Must find right content |
| 20% | Chunk Coherence | Affects LLM understanding |
| 15% | Performance | Must be practical |
| 10% | Simplicity | Maintenance cost |

---

## Deliverables

1. **Chunking implementations** - Python modules for each strategy
2. **Benchmark harness** - Automated test runner
3. **Synthetic corpus** - 50 docs with ground truth
4. **Results report** - Markdown with metrics, charts, recommendation
5. **Recommended strategy** - Clear winner or hybrid approach

---

## Timeline

| Day | Task |
|-----|------|
| 1 | Generate synthetic corpus + ground truth |
| 2 | Implement strategies 1-3 |
| 3 | Implement strategies 4-6 |
| 4 | Run indexing benchmarks |
| 5 | Run retrieval benchmarks |
| 6 | Run LLM quality benchmarks |
| 7 | Continue LLM benchmarks + edge cases |
| 8 | Analysis, report, recommendation |

---

## Open Questions

1. **Embedding model** - Use same as main system (all-MiniLM-L6-v2) or test multiple?
2. **LLM for evaluation** - Use local (llama3:8b) or API (GPT-4) for judging?
3. **Corpus generation** - Manual creation or LLM-generated with review?
4. **Hierarchical retrieval** - What combination of levels to test?

---

## File Structure

```
poc/chunking_benchmark/
├── DESIGN.md                 # This document
├── README.md                 # Quick start guide
├── pyproject.toml            # Dependencies
│
├── corpus/                   # Test documents
│   ├── generate_corpus.py    # Corpus generator
│   ├── documents/            # Generated docs
│   └── ground_truth.json     # Query + expected answers
│
├── strategies/               # Chunking implementations
│   ├── __init__.py
│   ├── base.py               # Common interface
│   ├── fixed_size.py         # Strategy 1
│   ├── heading_based.py      # Strategy 2
│   ├── heading_limited.py    # Strategy 3
│   ├── hierarchical.py       # Strategy 4
│   ├── paragraphs.py         # Strategy 5
│   └── heading_paragraph.py  # Strategy 6
│
├── evaluation/               # Benchmark harness
│   ├── __init__.py
│   ├── indexer.py            # Index documents
│   ├── retriever.py          # Run queries
│   ├── llm_eval.py           # LLM answer evaluation
│   └── metrics.py            # Metric calculations
│
├── results/                  # Output
│   ├── indexing_stats.csv
│   ├── retrieval_results.csv
│   ├── llm_quality.csv
│   └── report.md
│
└── scripts/
    ├── run_all.py            # Full benchmark
    ├── run_indexing.py       # Index only
    ├── run_retrieval.py      # Retrieval only
    └── run_llm_eval.py       # LLM eval only
```

---

## Next Steps

1. [ ] Review and approve this design
2. [ ] Generate synthetic corpus
3. [ ] Implement chunking strategies
4. [ ] Run benchmarks
5. [ ] Analyze and decide
