# TODO: Prepare Test Query Set

## Purpose

Create a labeled test query set that will be used to evaluate all approaches consistently.

---

## Preparation

### Prerequisites
- [ ] Access to PLM search service
- [ ] Access to PLM document corpus
- [ ] Understanding of query type categories (see `evaluation_criteria.md`)

### Reference Documents
- [ ] Read `evaluation_criteria.md` Section 4 (Query Type Categories)
- [ ] Read `evaluation_criteria.md` Section 5 (Test Data Requirements)

---

## Execution

### Step 1: Collect Queries
- [ ] Gather minimum 200 test queries
- [ ] Sources: existing benchmark queries, real user queries, synthetic generation
- [ ] Ensure coverage across all query types

### Step 2: Categorize Queries
- [ ] Assign each query to exactly one category:
  - Factoid (minimum 50)
  - Procedural (minimum 50)
  - Explanatory (minimum 50)
  - Comparison (minimum 30)
  - Troubleshooting (minimum 30)

### Step 3: Label Queries
For each query, determine and record:
- [ ] `query_type` - category from Step 2
- [ ] `relevant_chunk_ids` - chunks that contain answer information
- [ ] `optimal_granularity` - smallest context level sufficient to answer (chunk/heading/document)
- [ ] `expected_answer_keywords` - key terms that should appear in correct answer

### Step 4: Establish Oracle Performance
- [ ] For each query, test retrieval at all granularities
- [ ] Record which granularity produces answerable context
- [ ] Calculate oracle success rate (theoretical maximum)

### Step 5: Store Test Set
- [ ] Save to `benchmarks/datasets/test_queries.json` (or appropriate format)
- [ ] Include all labels
- [ ] Document format and schema

---

## Conclusion

### Deliverables
- [ ] Test query file created in `benchmarks/datasets/`
- [ ] Query count by category documented
- [ ] Oracle performance documented
- [ ] Format/schema documented

### Verification Checklist
- [ ] Minimum 200 queries total
- [ ] All 5 categories have minimum required queries
- [ ] All queries have complete labels
- [ ] Oracle results recorded

---

## Proceed to Next

Before proceeding:
1. [ ] Verify all deliverables are complete
2. [ ] Verify test set is accessible and correctly formatted
3. [ ] Document any issues or limitations of the test set

**Next:** Read and execute `02_baseline.md`
