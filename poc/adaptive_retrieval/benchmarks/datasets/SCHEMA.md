# Test Query Dataset Schema

## Overview

This document defines the JSON schema for the adaptive retrieval test query set.

## File Structure

```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "2026-02-21T00:00:00Z",
    "total_queries": 210,
    "query_counts": {
      "factoid": 50,
      "procedural": 50,
      "explanatory": 50,
      "comparison": 30,
      "troubleshooting": 30
    },
    "corpus": "kubernetes-docs",
    "corpus_stats": {
      "total_chunks": 20801,
      "total_documents": 1569
    }
  },
  "queries": [
    { /* query objects */ }
  ]
}
```

## Query Object Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., "factoid_001", "proc_042") |
| `query` | string | The question text |
| `query_type` | enum | One of: `factoid`, `procedural`, `explanatory`, `comparison`, `troubleshooting` |
| `expected_answer` | string | Key answer content or summary |
| `optimal_granularity` | enum | Smallest sufficient context: `chunk`, `heading`, `document` |

### Labeling Fields (for evaluation)

| Field | Type | Description |
|-------|------|-------------|
| `relevant_chunk_ids` | string[] | Chunk IDs containing the answer |
| `relevant_heading_ids` | string[] | Heading IDs containing the answer (optional) |
| `relevant_doc_ids` | string[] | Document IDs containing the answer |
| `expected_answer_keywords` | string[] | Key terms that should appear in correct answer |

### Optional Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `difficulty` | enum | `easy`, `medium`, `hard` |
| `source` | string | Origin: `existing`, `generated`, `manual` |
| `source_file` | string | Original file if imported from existing dataset |
| `notes` | string | Additional context |

## Query Type Definitions

### Factoid
- **Definition**: Single fact lookup, answer is 1-2 sentences
- **Examples**: "What is the default port for etcd?" / "What command lists pods?"
- **Expected optimal granularity**: Usually `chunk`
- **Minimum count**: 50

### Procedural
- **Definition**: Step-by-step instructions, "how to" questions
- **Examples**: "How do I create a StatefulSet?" / "Steps to configure ingress"
- **Expected optimal granularity**: Usually `heading` (section)
- **Minimum count**: 50

### Explanatory
- **Definition**: Conceptual understanding, "why" or "how does X work"
- **Examples**: "How does Kubernetes scheduling work?" / "Explain pod lifecycle"
- **Expected optimal granularity**: Usually `heading` or `document`
- **Minimum count**: 50

### Comparison
- **Definition**: Compare two or more concepts
- **Examples**: "Deployment vs StatefulSet" / "Difference between ConfigMap and Secret"
- **Expected optimal granularity**: Usually multiple `heading`s
- **Minimum count**: 30

### Troubleshooting
- **Definition**: Diagnose or fix a problem
- **Examples**: "Why is my pod in CrashLoopBackOff?" / "How to debug OOMKilled"
- **Expected optimal granularity**: Usually `heading` with related content
- **Minimum count**: 30

## Granularity Definitions

| Granularity | Token Range | When to Use |
|-------------|-------------|-------------|
| `chunk` | ~300-600 tokens | Answer is self-contained in a single passage |
| `heading` | ~500-2000 tokens | Answer requires full section context |
| `document` | ~1000-5000 tokens | Answer requires understanding entire document |

## Example Queries

### Factoid Example
```json
{
  "id": "factoid_001",
  "query": "What is the default port for the Kubernetes API server?",
  "query_type": "factoid",
  "expected_answer": "The default port for the Kubernetes API server is 6443.",
  "optimal_granularity": "chunk",
  "relevant_chunk_ids": ["concepts_overview_components_a2b0280e6d92_3_abc123"],
  "relevant_doc_ids": ["concepts_overview_components_a2b0280e6d92"],
  "expected_answer_keywords": ["6443", "API server", "port"],
  "difficulty": "easy",
  "source": "generated"
}
```

### Procedural Example
```json
{
  "id": "proc_001",
  "query": "How do I create a Deployment in Kubernetes?",
  "query_type": "procedural",
  "expected_answer": "Create a YAML manifest with kind: Deployment, specify replicas, selector, and pod template. Apply with kubectl apply -f deployment.yaml",
  "optimal_granularity": "heading",
  "relevant_chunk_ids": ["tasks_run-application_run-stateless..._1", "tasks_run-application_run-stateless..._2"],
  "relevant_heading_ids": ["tasks_run-application_run-stateless..._h2"],
  "relevant_doc_ids": ["tasks_run-application_run-stateless-application-deployment_8d3ba70ae449"],
  "expected_answer_keywords": ["kubectl", "apply", "Deployment", "replicas", "template"],
  "difficulty": "medium",
  "source": "generated"
}
```

### Explanatory Example
```json
{
  "id": "expl_001",
  "query": "How does the Kubernetes scheduler decide which node to place a pod on?",
  "query_type": "explanatory",
  "expected_answer": "The scheduler filters nodes based on resource requirements, taints/tolerations, and affinity rules. Then scores remaining nodes based on spreading, resource balance, and other factors. Highest scoring node is selected.",
  "optimal_granularity": "heading",
  "relevant_chunk_ids": ["concepts_scheduling_kube-scheduler_..._1", "..._2", "..._3"],
  "relevant_heading_ids": ["concepts_scheduling_kube-scheduler_..._h1"],
  "relevant_doc_ids": ["concepts_scheduling_kube-scheduler_..."],
  "expected_answer_keywords": ["filter", "score", "node", "resources", "affinity"],
  "difficulty": "medium",
  "source": "generated"
}
```

### Comparison Example
```json
{
  "id": "comp_001",
  "query": "What is the difference between a Deployment and a StatefulSet?",
  "query_type": "comparison",
  "expected_answer": "Deployments manage stateless apps with interchangeable pods. StatefulSets manage stateful apps with stable network identities, persistent storage, and ordered deployment/scaling.",
  "optimal_granularity": "heading",
  "relevant_chunk_ids": ["concepts_workloads_controllers_deployment_...", "concepts_workloads_controllers_statefulset_..."],
  "relevant_heading_ids": ["..._h1", "..._h1"],
  "relevant_doc_ids": ["concepts_workloads_controllers_deployment_...", "concepts_workloads_controllers_statefulset_..."],
  "expected_answer_keywords": ["stateless", "stateful", "persistent", "identity", "ordered"],
  "difficulty": "medium",
  "source": "generated"
}
```

### Troubleshooting Example
```json
{
  "id": "trouble_001",
  "query": "Why is my pod stuck in Pending state?",
  "query_type": "troubleshooting",
  "expected_answer": "Common causes: insufficient resources (CPU/memory), no matching nodes (taints, affinity), PVC not bound, scheduler issues. Check with kubectl describe pod.",
  "optimal_granularity": "heading",
  "relevant_chunk_ids": ["tasks_debug_debug-application_debug-pods_..."],
  "relevant_heading_ids": ["tasks_debug_debug-application_debug-pods_..._h2"],
  "relevant_doc_ids": ["tasks_debug_debug-application_debug-pods_..."],
  "expected_answer_keywords": ["Pending", "resources", "scheduler", "describe", "node"],
  "difficulty": "medium",
  "source": "generated"
}
```

## Validation Rules

1. **ID uniqueness**: All `id` values must be unique
2. **Type coverage**: Must have minimum queries per type (50/50/50/30/30)
3. **Granularity labels**: All queries must have `optimal_granularity`
4. **Chunk references**: All queries must have at least one `relevant_chunk_id`
5. **Document references**: All queries must have at least one `relevant_doc_id`
6. **Keywords**: All queries should have at least 3 `expected_answer_keywords`

## Usage

```python
import json

with open("test_queries.json") as f:
    data = json.load(f)

# Access queries by type
factoid_queries = [q for q in data["queries"] if q["query_type"] == "factoid"]

# Access metadata
print(f"Total queries: {data['metadata']['total_queries']}")
print(f"Factoid count: {data['metadata']['query_counts']['factoid']}")
```
