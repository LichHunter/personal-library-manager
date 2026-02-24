# Adaptive Retrieval Test Query Dataset

## Overview

This dataset contains **229 labeled test queries** for evaluating adaptive retrieval approaches on Kubernetes documentation.

## Dataset Statistics

| Query Type | Count | Minimum Required | Status |
|------------|-------|------------------|--------|
| Factoid | 54 | 50 | OK |
| Procedural | 55 | 50 | OK |
| Explanatory | 53 | 50 | OK |
| Comparison | 33 | 30 | OK |
| Troubleshooting | 34 | 30 | OK |
| **Total** | **229** | 210 | OK |

## Corpus Statistics

- **Total chunks**: 20,801
- **Total documents**: 1,569
- **Domain**: Kubernetes official documentation

## Files

| File | Description |
|------|-------------|
| `test_queries.json` | Main test query dataset |
| `SCHEMA.md` | JSON schema documentation |
| `consolidated_existing.json` | Queries from existing POCs |
| `generated_queries.json` | Newly generated queries |

## Query Sources

### Existing Queries (65)
- `needle_questions.json` - 20 queries (Topology Manager focused)
- `needle_questions_adversarial.json` - 20 queries (adversarial patterns)
- `informed_questions.json` - 25 queries (kubefix dataset)

### Generated Queries (164)
- Factoid: 20 queries covering ports, defaults, commands
- Procedural: 50 queries covering configuration, setup, operations
- Explanatory: 40 queries covering concepts, architecture
- Comparison: 25 queries comparing K8s resources
- Troubleshooting: 30 queries for common issues

## Query Format

```json
{
  "id": "factoid_gen_001",
  "query": "What is the default port for the Kubernetes API server?",
  "query_type": "factoid",
  "expected_answer": "The default port is 6443.",
  "optimal_granularity": "chunk",
  "expected_answer_keywords": ["6443", "API server", "port"],
  "difficulty": "easy",
  "source": "generated",
  "relevant_chunk_ids": [],
  "relevant_doc_ids": []
}
```

## Query Type Definitions

| Type | Definition | Expected Granularity |
|------|------------|---------------------|
| **Factoid** | Single fact lookup, 1-2 sentence answer | chunk |
| **Procedural** | How-to, step-by-step instructions | heading |
| **Explanatory** | Conceptual, "why/how does X work" | heading/document |
| **Comparison** | Compare two+ concepts | multiple headings |
| **Troubleshooting** | Diagnose/fix problems | heading |

## Topic Coverage

Queries cover diverse Kubernetes topics:

- **Architecture**: API server, etcd, scheduler, controller-manager, kubelet
- **Workloads**: Pods, Deployments, StatefulSets, DaemonSets, Jobs, CronJobs
- **Services**: ClusterIP, NodePort, LoadBalancer, Ingress, Gateway API
- **Configuration**: ConfigMaps, Secrets, Volumes, PVCs
- **Security**: RBAC, ServiceAccounts, NetworkPolicy, Pod Security
- **Operations**: kubectl commands, scaling, rolling updates
- **Scheduling**: Affinity, taints/tolerations, resource limits

## Usage

### Load the dataset

```python
import json

with open("test_queries.json") as f:
    data = json.load(f)

queries = data["queries"]
print(f"Total queries: {data['metadata']['total_queries']}")

# Filter by type
factoid_queries = [q for q in queries if q["query_type"] == "factoid"]
```

### Evaluate against PLM

```python
# Search PLM for each query
for query in queries:
    results = plm_search(query["query"], k=10)
    
    # Check if expected doc is retrieved
    retrieved_docs = [r["doc_id"] for r in results]
    hit = any(d in retrieved_docs for d in query["relevant_doc_ids"])
```

## Next Steps

1. **Baseline measurement** - Run all queries through PLM, measure MRR@10, Hit@k
2. **Chunk labeling** - Store top retrieved chunk_ids as candidate relevance
3. **Oracle testing** - Test each granularity to find optimal for each query
4. **Approach testing** - Test adaptive retrieval approaches

## Created

- **Date**: 2026-02-21
- **Script**: `scripts/build_test_set.py`
- **POC**: `poc/adaptive_retrieval/`
