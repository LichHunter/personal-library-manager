# Benchmark Question Corpus

Structured question corpus for evaluating retrieval pipeline performance on benchmark datasets.

## Overview

This directory contains benchmark question files in JSON format. Each file represents a collection of questions designed to evaluate retrieval performance against a specific document corpus (e.g., Kubernetes documentation).

Questions follow a standardized schema with required and optional fields to support flexible evaluation scenarios.

## JSON Schema Documentation

### File Structure

Each corpus file follows this top-level structure:

```json
{
  "metadata": {
    "source": "string",
    "total": "number",
    "description": "string"
  },
  "questions": [
    { /* question objects */ }
  ]
}
```

### Metadata Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Origin dataset identifier (e.g., "kubefix", "manual", "synthetic") |
| `total` | number | Yes | Total number of questions in the corpus |
| `description` | string | No | Human-readable description of the corpus purpose and characteristics |

### Required Question Fields

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | The query text to search for. This is the actual question/search term used during retrieval evaluation. |
| `expected_answer` | string | Ground truth answer for grading retrieved chunks. Used to verify if retrieved content contains the correct information. |
| `doc_id` | string | Target document identifier without file extension. Maps to the document that should be retrieved (e.g., "tasks_configure-pod-container_configure-service-account"). |

### Optional Question Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique question identifier within the corpus (e.g., "q_001_q1", "q_024_q2"). Useful for tracking and reporting. |
| `difficulty` | string | Difficulty level classification: "easy", "medium", or "hard". Helps analyze performance by question complexity. |
| `type` | string | Question category: "factual", "comparison", "negation", or "vocabulary". Enables category-specific performance analysis. |
| `section` | string | Document section reference where the answer is located (e.g., "Configuration", "Troubleshooting"). |
| `quality_score` | float | Quality rating from 0.0 to 1.0. Indicates confidence in question-answer pair validity. |
| `source` | string | Origin dataset for this specific question (e.g., "kubefix", "manual", "synthetic"). May differ from metadata.source for mixed corpora. |
| `variant` | string | Variant identifier for question transformations (e.g., "informed_q1", "realistic_q2"). Tracks question generation method. |

## Examples

### Single Question Entry

```json
{
  "question": "What is a ServiceAccount in Kubernetes?",
  "expected_answer": "A ServiceAccount is an identity for processes that run in a Pod. It provides an identity for processes running in pods to use when contacting the API server.",
  "doc_id": "tasks_configure-pod-container_configure-service-account",
  "id": "q_024_q1",
  "difficulty": "medium",
  "type": "factual",
  "quality_score": 1.0,
  "source": "kubefix",
  "variant": "informed_q1"
}
```

### Complete File Structure

```json
{
  "metadata": {
    "source": "kubefix",
    "total": 50,
    "description": "Informed questions using technical Kubernetes terminology from kubefix dataset"
  },
  "questions": [
    {
      "question": "What is a ServiceAccount in Kubernetes?",
      "expected_answer": "A ServiceAccount is an identity for processes that run in a Pod.",
      "doc_id": "tasks_configure-pod-container_configure-service-account",
      "id": "q_001_q1",
      "difficulty": "medium",
      "type": "factual",
      "quality_score": 1.0,
      "source": "kubefix",
      "variant": "informed_q1"
    },
    {
      "question": "How do I configure a Pod to use a ServiceAccount?",
      "expected_answer": "You can specify a ServiceAccount in the Pod spec using the serviceAccountName field.",
      "doc_id": "tasks_configure-pod-container_configure-service-account",
      "id": "q_001_q2",
      "difficulty": "easy",
      "type": "factual",
      "quality_score": 0.95,
      "source": "kubefix",
      "variant": "informed_q2"
    }
  ]
}
```

## Field Mapping from Old Format

When converting from legacy question formats, use these mappings:

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `original_instruction` | `question` | Use directly for informed questions |
| `original_instruction` | `expected_answer` | Use for grading retrieved chunks |
| `doc_id` | `doc_id` | Unchanged - maps to target document |
| `realistic_q1` | `question` | Use for realistic variant questions |
| `realistic_q2` | `question` | Use for realistic variant questions |
| `quality_pass` | Filter out | Do not store; filter during conversion (keep only true values) |
| `quality_score` | `quality_score` | Preserve original score |
| `original_source` | `section` | Optional: document source path |

### Conversion Example

**Old format:**
```json
{
  "original_instruction": "What is a ServiceAccount in Kubernetes?",
  "original_source": "/content/en/docs/tasks/configure-pod-container/configure-service-account.md",
  "doc_id": "tasks_configure-pod-container_configure-service-account",
  "realistic_q1": "how do i set up a service account for my pod",
  "realistic_q2": "what's the difference between a service account and a regular user",
  "quality_score": 1.0,
  "quality_pass": true
}
```

**New format (informed variant):**
```json
{
  "question": "What is a ServiceAccount in Kubernetes?",
  "expected_answer": "What is a ServiceAccount in Kubernetes?",
  "doc_id": "tasks_configure-pod-container_configure-service-account",
  "id": "q_024_informed",
  "quality_score": 1.0,
  "source": "kubefix",
  "variant": "informed_q1"
}
```

**New format (realistic variant):**
```json
{
  "question": "how do i set up a service account for my pod",
  "expected_answer": "What is a ServiceAccount in Kubernetes?",
  "doc_id": "tasks_configure-pod-container_configure-service-account",
  "id": "q_024_realistic",
  "quality_score": 1.0,
  "source": "kubefix",
  "variant": "realistic_q1"
}
```

## Usage Guidelines

### Creating a New Corpus File

1. **Define metadata**: Specify source, total count, and description
2. **Add questions**: Include all required fields (question, expected_answer, doc_id)
3. **Enrich with optional fields**: Add difficulty, type, quality_score, variant for better analysis
4. **Validate JSON**: Ensure valid JSON syntax before use
5. **Document purpose**: Update this README if adding a new corpus type

### Evaluation Workflow

1. Load corpus file
2. For each question:
   - Use `question` as the search query
   - Retrieve chunks from document corpus
   - Compare retrieved content against `expected_answer`
   - Record hit/miss and ranking metrics
3. Aggregate results by:
   - Overall pass rate
   - Category performance (by `type`)
   - Difficulty performance (by `difficulty`)
   - Variant performance (by `variant`)

### Quality Considerations

- **quality_score**: Filter questions with score < 0.7 for strict evaluation
- **quality_pass**: Legacy field - only use questions where this was true
- **variant**: Track performance separately for informed vs. realistic questions
- **type**: Analyze failure modes by question category

## File Naming Convention

Corpus files should follow this naming pattern:

```
{dataset}_{variant}_{date}.json
```

Examples:
- `kubernetes_informed_2026-01-27.json` - Informed questions from Kubernetes dataset
- `kubernetes_realistic_2026-01-27.json` - Realistic questions from Kubernetes dataset
- `kubefix_mixed_2026-01-27.json` - Mixed questions from kubefix source

## Related Documentation

- **Retrieval Pipeline**: See `../README.md` for pipeline architecture
- **Benchmark Results**: See `../results/` for evaluation reports
- **Document Corpus**: See `../docs/` for the document collection being evaluated
