# Draft: POC-2 Confidence Scoring and Threshold Determination

## Requirements (confirmed)

### Primary Objective
Validate whether confidence signals correlate with extraction quality, and determine the optimal threshold for routing chunks between fast and slow extraction systems.

### From Architecture Document (RAG_PIPELINE_ARCHITECTURE.md)
- **Question**: Do proposed confidence signals correlate with extraction quality? What threshold should trigger slow processing?
- **Why Critical**: Fast/slow routing depends on confidence scores. If scores don't reflect actual quality, we'll either miss bad extractions or waste resources re-processing good ones.

### Confidence Signals to Validate (from architecture)
1. **Known Term Ratio** - % of extracted terms found in term graph (< 30% = low confidence)
2. **Coverage** - % of chunk content captured by extracted terms (< 10% = low confidence)
3. **Entity Density** - Entities per 100 tokens (abnormally low or high = concern)
4. **Section Type Mismatch** - Expected vs actual extraction patterns (code block with no code entities)

### Success Criteria (from architecture)
- Strong correlation (r > 0.6) between at least one signal and quality
- Combined signal achieves >80% accuracy at classifying good vs. poor
- Identified threshold that balances quality vs. slow system load

## Existing Code Assets

### Fast Extraction System (src/plm/extraction/fast/)
- `heuristic.py`: Pattern-based extraction (CamelCase, ALL_CAPS, dot.paths, backticks, function_calls)
- `confidence.py`: Current simple confidence scoring (source_score * 0.6 + ratio_score * 0.4)

### Scoring Functions (from POC-1c)
- `scoring.py`: v3_match + many_to_many_score - validated scoring against ground truth
- Can compute precision, recall, F1, hallucination per chunk

### Test Data Available
- `poc/poc-1c-scalable-ner/artifacts/test_documents.json` - 249 docs with ground truth
- `data/vocabularies/term_index.json` - term graph for lookups
- `data/vocabularies/auto_vocab.json` - auto-generated vocabulary

## Technical Decisions

### POC Structure
- Follow existing template from `poc/POC_SPECIFICATION_TEMPLATE.md`
- Create phased execution with checkpoint artifacts
- Reuse scoring functions from POC-1c

### Test Design
- Sample 100 chunks from test corpus
- Grade each extraction with quality labels (GOOD/ACCEPTABLE/POOR)
- Use LLM-based grading for consistency, validate sample manually
- Correlate grades with each confidence signal

### Iterative Improvement Strategy
If initial results fail (<80% accuracy):
1. Phase 4a: Signal engineering - try additional signals
2. Phase 4b: Ensemble methods - combine signals differently
3. Phase 4c: Consult Oracle for architectural alternatives
4. Phase 4d: Consider different routing strategies

## Open Questions
- None remaining - requirements are clear from architecture document

## Scope Boundaries
- **INCLUDE**: Signal validation, threshold determination, routing recommendations
- **EXCLUDE**: Actually implementing the routing (that's post-POC), GLiNER (rejected in POC-1c)
