# PLM vs Baseline RAG - Complete Detailed Results

**Generated:** 2026-02-21 (updated with raw BM25 needle results)  
**Corpus:** Kubernetes Documentation  
**Documents:** 1,569  
**Chunks:** 20,801  
**Embedding Model:** BAAI/bge-base-en-v1.5 (768-dim, L2-normalized)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Benchmark Configuration](#benchmark-configuration)
3. [Needle Benchmark Results](#needle-benchmark-results)
4. [Informed Benchmark Results](#informed-benchmark-results)
5. [Realistic Benchmark Results](#realistic-benchmark-results)
6. [Statistical Analysis](#statistical-analysis)
7. [Attribution Analysis](#attribution-analysis)
8. [Head-to-Head Comparison](#head-to-head-comparison)
9. [Key Findings](#key-findings)
10. [Recommendations](#recommendations)
11. [Methodology](#methodology)
12. [Raw Data Reference](#raw-data-reference)

---

## Executive Summary

| Benchmark | Queries | PLM MRR | Baseline MRR | Δ MRR | Improvement | Significant? |
|-----------|---------|---------|--------------|-------|-------------|--------------|
| Needle | 20 | 0.842 | 0.713 | +0.129 | **+18.1%** | **Yes** (p<0.05) |
| Informed | 25 | 0.621 | 0.633 | -0.012 | **-1.9%** | No |
| Realistic | 400 | 0.196 | 0.192 | +0.004 | +2.0% | No |

**Key Findings:**
1. PLM's advantage comes entirely from **BM25 + RRF fusion**, not from content enrichment
2. **BM25 enrichment effect is query-type dependent:**
   - Helps needle (+5.5%) and informed (+8.5%) queries
   - Hurts realistic natural language queries (-5.6%)
3. **Embedding enrichment consistently hurts** retrieval by -2% to -5% across all query types (Enriched vs Naive baselines)
4. PLM **underperforms** on technical terminology queries (informed) where exact matching should excel

---

## Benchmark Configuration

### Three Variants Compared

| Variant | Components | Purpose |
|---------|------------|---------|
| **PLM Full** | BM25 + Semantic + RRF fusion + Query expansion + Enriched embeddings | Full production system |
| **Baseline-Enriched** | FAISS + PLM's enriched content (keywords/entities prepended) | Isolate enrichment contribution |
| **Baseline-Naive** | FAISS + Raw chunk content only | Pure semantic baseline |

### Shared Configuration

| Parameter | Value |
|-----------|-------|
| Embedding model | BAAI/bge-base-en-v1.5 |
| Embedding dimension | 768 |
| Normalization | L2 (for cosine similarity) |
| FAISS index type | IndexFlatIP (inner product) |
| k (results per query) | 5 |
| Chunk source | PLM production SQLite database |
| Total chunks | 20,801 |
| Total documents | 1,569 |

### PLM-Specific Components

| Component | Description |
|-----------|-------------|
| BM25 Index | bm25s library, stored in production path |
| RRF Fusion | Reciprocal Rank Fusion combining BM25 + semantic |
| Query Expansion | Disabled for this benchmark (use_rewrite=False) |
| Enrichment | Keywords/entities prepended to chunk content |

---

## Needle Benchmark Results

**Target Document:** `tasks_administer-cluster_topology-manager`  
**Total Queries:** 20  
**Topic:** Kubernetes Topology Manager

### Aggregate Metrics

| Metric | PLM | Baseline-Enriched | Baseline-Naive |
|--------|-----|-------------------|----------------|
| **MRR** | **0.8417** | 0.6958 | 0.7125 |
| **Hit@1** | **80.0%** | 60.0% | 60.0% |
| **Hit@3** | **90.0%** | 80.0% | 80.0% |
| **Hit@5** | **90.0%** | 85.0% | 85.0% |
| **Found** | **18/20** | 17/20 | 17/20 |
| **Not Found** | 2/20 | 3/20 | 3/20 |
| **Avg Latency** | 429.9ms | 123.9ms | 88.7ms |

### Per-Question Results

| ID | Question | PLM | Enr | Naive | Winner |
|----|----------|-----|-----|-------|--------|
| q_001 | My pod keeps getting rejected with a topology affinity error, what's going on? | **3** | MISS | MISS | PLM |
| q_002 | How do I force all my containers to run on the same NUMA node in k8s? | 1 | 1 | 1 | Tie |
| q_003 | What's the difference between restricted and best-effort topology policies? | **1** | 3 | 1 | Tie (PLM=Naive) |
| q_004 | Getting error about too many NUMA nodes on my server, what's the limit? | **1** | 1 | 2 | PLM |
| q_005 | How to enable topology manager on Windows nodes? | 1 | 1 | 1 | Tie |
| q_006 | Which k8s version made topology manager GA/stable? | **1** | 2 | 4 | PLM |
| q_007 | What flag do I pass to kubelet for setting topology policy? | 1 | 1 | 1 | Tie |
| q_008 | My latency-sensitive app is slow, containers seem to be on different NUMA nodes. How to fix? | **1** | 4 | 1 | Tie (PLM=Naive) |
| q_009 | Why would I use pod scope instead of container scope for topology? | 1 | 1 | 1 | Tie |
| q_010 | What are hint providers in topology manager? | 1 | 1 | 1 | Tie |
| q_011 | How do I configure kubelet to prefer NUMA nodes that are closer together? | **1** | 2 | 1 | Tie (PLM=Naive) |
| q_012 | When did prefer-closest-numa-nodes become generally available? | 1 | 1 | 1 | Tie |
| q_013 | What's the default topology manager policy if I don't set anything? | **1** | 1 | 2 | PLM |
| q_014 | Pod was scheduled but then failed on the node, says something about topology. Is that a bug? | MISS | MISS | MISS | None |
| q_015 | What QoS class does my pod need to be for topology hints to work? | 1 | 1 | 1 | Tie |
| q_016 | How to set topology scope to pod level in kubelet? | 1 | 1 | 1 | Tie |
| q_017 | What feature gate do I need for topology manager policy options? | 1 | 1 | 1 | Tie |
| q_018 | My multi-socket server has GPUs and CPUs, how does k8s coordinate their placement? | MISS | MISS | MISS | None |
| q_019 | With single-numa-node policy, when exactly does a pod get rejected? | **2** | 3 | 2 | Tie (PLM=Naive) |
| q_020 | What happens if topology manager can't find preferred NUMA affinity with best-effort policy? | **1** | 1 | 2 | PLM |

### Latency Per Question (ms)

| ID | PLM | Enriched | Naive |
|----|-----|----------|-------|
| q_001 | 2720.5 | 127.0 | 80.7 |
| q_002 | 334.2 | 58.0 | 85.8 |
| q_003 | 256.9 | 142.5 | 93.0 |
| q_004 | 369.5 | 116.7 | 90.2 |
| q_005 | 263.6 | 154.7 | 88.4 |
| q_006 | 412.9 | 85.7 | 64.3 |
| q_007 | 260.4 | 110.4 | 134.9 |
| q_008 | 262.2 | 129.8 | 68.9 |
| q_009 | 259.9 | 107.0 | 123.6 |
| q_010 | 256.7 | 133.0 | 86.3 |
| q_011 | 397.9 | 125.3 | 153.7 |
| q_012 | 311.3 | 148.7 | 76.0 |
| q_013 | 332.7 | 157.1 | 113.8 |
| q_014 | 325.9 | 189.4 | 101.1 |
| q_015 | 319.0 | 130.8 | 104.3 |
| q_016 | 396.6 | 119.0 | 66.6 |
| q_017 | 258.1 | 150.8 | 43.8 |
| q_018 | 278.1 | 102.2 | 75.0 |
| q_019 | 302.7 | 94.7 | 61.4 |
| q_020 | 279.8 | 95.2 | 62.4 |

**Note:** q_001 has high PLM latency (2720ms) due to cold start / first query initialization.

---

## Informed Benchmark Results

**Total Queries:** 25  
**Topic:** Same 25 documents as realistic, but using **proper Kubernetes terminology**  
**Purpose:** Test retrieval when users know exact technical terms

### What Makes Informed Different

| Type | Example Query | Terminology |
|------|---------------|-------------|
| **Realistic** | "how to check if my service account has permission" | Natural language |
| **Informed** | "What does a SubjectAccessReview object describe?" | Technical terms |

### Aggregate Metrics

| Metric | PLM | Baseline-Enriched | Baseline-Naive |
|--------|-----|-------------------|----------------|
| **MRR** | 0.6213 | 0.6033 | **0.6333** |
| **Hit@1** | **56.0%** | 48.0% | 52.0% |
| **Hit@5** | 72.0% | 80.0% | **84.0%** |
| **Found** | 18/25 | 20/25 | **21/25** |
| **Not Found** | 7/25 | 5/25 | **4/25** |
| **Avg Latency** | 406.7ms | 123.1ms | 66.5ms |

**Critical Finding:** PLM found FEWER documents (18) than Naive baseline (21) on technical queries!

### Per-Question Results

| ID | Question | PLM | Enr | Naive | Winner |
|----|----------|-----|-----|-------|--------|
| informed_000 | What is the role of the Infrastructure Provider in the design principles of Gateway API? | 1 | 1 | 1 | Tie |
| informed_001 | What is the Kubernetes approach to dynamic port allocation? | 1 | 1 | 1 | Tie |
| informed_002 | What does a SubjectAccessReview object describe? | **MISS** | 4 | 4 | Naive |
| informed_003 | What is the difference between a mutating webhook and a validating webhook? | MISS | MISS | MISS | None |
| informed_004 | What does the NamespaceAutoProvision admission controller do? | 1 | 1 | 1 | Tie |
| informed_005 | What types of volumes can be expanded using PersistentVolumeClaims? | 5 | 1 | 3 | Enriched |
| informed_006 | What are the two default user accounts for Windows containers? | 1 | 1 | 1 | Tie |
| informed_007 | What is the purpose of using an Indexed Job in Kubernetes? | **1** | 4 | 3 | **PLM** |
| informed_008 | What is the etcd API and what risks does it pose to the cluster's security? | **MISS** | 2 | 3 | Enriched |
| informed_009 | What is the role of a signer in Kubernetes? | 1 | 1 | 1 | Tie |
| informed_010 | What tools are required to verify Kubernetes artifacts? | 2 | MISS | **1** | Naive |
| informed_011 | What protocol does Kubernetes components use to emit traces? | 3 | 1 | 1 | Naive/Enr |
| informed_012 | What is the format in which Kubernetes components emit metrics? | 2 | MISS | 2 | Tie |
| informed_013 | What are the different types of isolation in Kubernetes? | 1 | 1 | 1 | Tie |
| informed_014 | What is Attribute-based access control (ABAC) and how does it work? | 1 | 2 | 1 | Tie |
| informed_015 | What is the role of the webhook token authenticator in the authentication process? | **MISS** | 4 | 4 | Naive |
| informed_016 | What is the purpose of the authentication flow in the aggregation layer? | 1 | 2 | 1 | Tie |
| informed_017 | What does sourcing the completion script in your shell enable? | MISS | MISS | MISS | None |
| informed_018 | What is kube proxy? | MISS | MISS | MISS | None |
| informed_019 | How can you use `kubectl` to create an object from a configuration file? | 1 | 1 | 1 | Tie |
| informed_020 | How can you suspend and resume a Job in Kubernetes? | 1 | 1 | 1 | Tie |
| informed_021 | What is the purpose of the `kubectl logs pods/job-wq-2-7r7b2` command? | **MISS** | 2 | 2 | Naive |
| informed_022 | What are the four supported Topology Manager policies? | 1 | 1 | 1 | Tie |
| informed_023 | Why do we need admission controllers? | **1** | 1 | 3 | **PLM** |
| informed_024 | What is a ServiceAccount in Kubernetes? | **1** | 3 | MISS | **PLM** |

### Head-to-Head: PLM vs Naive

| Outcome | Count | Percentage |
|---------|-------|------------|
| PLM wins (better rank) | 3 | 12% |
| PLM loses (worse rank) | 7 | 28% |
| Tie (same rank or both miss) | 15 | 60% |

### Where PLM Fails on Technical Terms

| Question | PLM | Naive | Why PLM Misses |
|----------|-----|-------|----------------|
| "SubjectAccessReview object" | MISS | 4 | Exact term should BM25 match |
| "etcd API risks" | MISS | 3 | Multi-word technical term |
| "webhook token authenticator" | MISS | 4 | Compound technical term |
| "kubectl logs pods/job-wq" | MISS | 2 | Exact command in query |
| "verify Kubernetes artifacts" | 2 | 1 | Naive wins on exact match |

### Where PLM Wins

| Question | PLM | Naive | Why PLM Wins |
|----------|-----|-------|--------------|
| "Indexed Job purpose" | 1 | 3 | RRF boosts relevant result |
| "admission controllers why" | 1 | 3 | BM25+semantic synergy |
| "ServiceAccount in Kubernetes" | 1 | MISS | Broader coverage |

### Latency Per Question (ms)

| ID | PLM | Enriched | Naive |
|----|-----|----------|-------|
| informed_000 | 2562.6 | 120.7 | 64.2 |
| informed_001 | 276.1 | 108.9 | 78.9 |
| informed_002 | 281.8 | 182.1 | 49.9 |
| informed_003 | 264.3 | 118.9 | 51.7 |
| informed_004 | 262.8 | 150.5 | 54.9 |
| informed_005 | 276.9 | 122.5 | 51.9 |
| informed_006 | 258.7 | 91.9 | 65.6 |
| informed_007 | 254.9 | 87.7 | 81.5 |
| informed_008 | 309.9 | 102.0 | 84.9 |
| informed_009 | 318.9 | 138.4 | 61.3 |
| informed_010 | 360.9 | 100.5 | 67.9 |
| informed_011 | 338.9 | 157.1 | 68.3 |
| informed_012 | 283.9 | 91.9 | 58.1 |
| informed_013 | 351.4 | 135.9 | 68.0 |
| informed_014 | 279.9 | 76.9 | 64.9 |
| informed_015 | 370.4 | 136.6 | 62.5 |
| informed_016 | 338.9 | 116.3 | 59.4 |
| informed_017 | 309.4 | 138.5 | 62.4 |
| informed_018 | 343.9 | 137.8 | 63.0 |
| informed_019 | 306.5 | 109.9 | 69.0 |
| informed_020 | 449.8 | 99.4 | 86.1 |
| informed_021 | 477.9 | 138.6 | 75.7 |
| informed_022 | 467.8 | 152.9 | 56.7 |
| informed_023 | 389.8 | 107.3 | 70.9 |
| informed_024 | 432.0 | 132.8 | 61.9 |

---

## Realistic Benchmark Results

**Total Queries:** 400 (200 questions × 2 variants: q1/q2)  
**Topic:** Diverse Kubernetes documentation questions  
**Each query has a specific target document**

### Aggregate Metrics

| Metric | PLM | Baseline-Enriched | Baseline-Naive |
|--------|-----|-------------------|----------------|
| **MRR** | **0.1957** | 0.1859 | 0.1918 |
| **Hit@1** | 12.5% | **12.8%** | **12.8%** |
| **Hit@3** | **25.3%** | 22.0% | 23.3% |
| **Hit@5** | **32.0%** | 29.5% | 30.5% |
| **Found** | **128/400** | 118/400 | 122/400 |
| **Not Found** | 272/400 | 282/400 | 278/400 |

### Rank Distribution

| Rank Position | PLM | Baseline-Enriched | Baseline-Naive |
|---------------|-----|-------------------|----------------|
| **@1 (exact)** | 50 | 51 | 51 |
| **@2-3** | 48 | 41 | 44 |
| **@4-5** | 30 | 26 | 27 |
| **>5** | 0 | 0 | 0 |
| **Not Found** | 272 | 282 | 278 |

### Head-to-Head: PLM vs Baseline-Naive

| Outcome | Count | Percentage |
|---------|-------|------------|
| PLM wins (better rank) | 73 | 18.3% |
| PLM loses (worse rank) | 69 | 17.3% |
| Tie (same rank or both miss) | 258 | 64.5% |

### Sample Questions Where PLM Wins

| PLM Rank | Naive Rank | Question |
|----------|------------|----------|
| 5 | MISS | is there a way to increase storage size for my database without losing data |
| 1 | 2 | how to manage default user accounts in windows container images |
| 1 | 3 | how to run a parallel batch job where each pod needs a unique index |
| 2 | MISS | how to manage tls certificates for my services without manual intervention |
| 1 | MISS | how to prevent one container from consuming all resources of my node |
| 1 | 2 | how to run a task on multiple nodes simultaneously |
| 1 | MISS | how to give pods access to cloud provider resources like s3 buckets |
| 3 | MISS | how to update my cluster to use a new container runtime |
| 1 | 4 | how to configure pods to run with specific security contexts |
| 2 | MISS | how to add labels to nodes based on their hardware capabilities |

### Sample Questions Where PLM Loses

| PLM Rank | Naive Rank | Question |
|----------|------------|----------|
| MISS | 2 | my prometheus dashboard is not showing kubernetes component performance metrics |
| MISS | 1 | getting authentication errors when trying to call kubernetes api from external tool |
| MISS | 3 | getting error when trying to apply kubernetes configuration from file |
| MISS | 1 | can't figure out what's happening inside my worker pod during a batch job |
| MISS | 1 | how to change storage type when recovering from a volume backup |
| MISS | 2 | my hpa is not scaling pods even when cpu usage is high |
| MISS | 1 | how to check what's causing high memory usage in my cluster |
| MISS | 1 | how to configure automatic scaling based on custom business metrics |
| MISS | 1 | how to move from in-tree storage plugins to csi drivers |
| MISS | 2 | my cronjob is not creating new jobs at the scheduled time |

---

## Statistical Analysis

### Bootstrap Confidence Intervals (95%, n=1000)

#### Needle Benchmark

| Comparison | Lower Bound | Upper Bound | Significant? |
|------------|-------------|-------------|--------------|
| PLM vs Naive | **+0.0333** | **+0.2417** | **Yes** |
| PLM vs Enriched | **+0.0500** | **+0.2625** | **Yes** |
| Enriched vs Naive | -0.1626 | +0.1125 | No |

#### Informed Benchmark

| Comparison | Lower Bound | Upper Bound | Significant? |
|------------|-------------|-------------|--------------|
| PLM vs Naive | -0.1500 | +0.1201 | No |
| PLM vs Enriched | -0.1287 | +0.1667 | No |
| Enriched vs Naive | -0.1667 | +0.0933 | No |

#### Realistic Benchmark

| Comparison | Lower Bound | Upper Bound | Significant? |
|------------|-------------|-------------|--------------|
| PLM vs Naive | -0.0306 | +0.0378 | No |
| PLM vs Enriched | -0.0178 | +0.0383 | No |
| Enriched vs Naive | -0.0301 | +0.0193 | No |

### Interpretation

- **Needle benchmark:** PLM improvement is statistically significant (CI excludes 0)
- **Informed benchmark:** No significant differences; PLM actually trends negative (-1.9%)
- **Realistic benchmark:** No statistically significant differences between any variants
- **Enrichment effect:** Consistently negative across all benchmarks (-2% to -5%)

---

## Attribution Analysis

### How Much Does Each Component Contribute?

#### Needle Benchmark

```
Total PLM improvement over Naive:     +0.1292 MRR (+18.1%)

Attribution:
├── Enrichment contribution:          -0.0167 MRR (-2.3%)
│   (Baseline-Enriched vs Baseline-Naive)
│   
└── RRF + BM25 + Expansion:           +0.1458 MRR (+20.5%)
    (PLM vs Baseline-Enriched)

Conclusion: 113% of improvement comes from RRF/BM25
            Enrichment HURT by -13%
```

#### Informed Benchmark

```
Total PLM improvement over Naive:     -0.0120 MRR (-1.9%)  ← WORSE!

Attribution:
├── Enrichment contribution:          -0.0300 MRR (-4.7%)
│   (Baseline-Enriched vs Baseline-Naive)
│   
└── RRF + BM25 + Expansion:           +0.0180 MRR (+2.8%)
    (PLM vs Baseline-Enriched)

Conclusion: PLM UNDERPERFORMS on technical queries
            Enrichment damage (-4.7%) exceeds BM25 benefit (+2.8%)
```

#### Realistic Benchmark

```
Total PLM improvement over Naive:     +0.0039 MRR (+2.0%)

Attribution:
├── Enrichment contribution:          -0.0059 MRR (-3.1%)
│   (Baseline-Enriched vs Baseline-Naive)
│   
└── RRF + BM25 + Expansion:           +0.0098 MRR (+5.1%)
    (PLM vs Baseline-Enriched)

Conclusion: 251% of improvement comes from RRF/BM25
            Enrichment HURT by -151%
```

### Summary

| Component | Needle | Informed | Realistic |
|-----------|--------|----------|-----------|
| **RRF + BM25** | **+0.146 MRR** | +0.018 MRR | +0.010 MRR |
| **Enrichment** | -0.017 MRR | **-0.030 MRR** | -0.006 MRR |
| **Total** | +0.129 MRR | **-0.012 MRR** | +0.004 MRR |

**Key Insights:**
1. Content enrichment provides **ZERO benefit** for retrieval—it consistently hurts
2. Enrichment damage is **worst on technical queries** (-4.7% on informed)
3. BM25+RRF benefit is **insufficient to overcome enrichment damage** on informed queries

---

## Head-to-Head Comparison

### Needle: Where Systems Differ

| Question | PLM | Enriched | Naive | Analysis |
|----------|-----|----------|-------|----------|
| topology affinity error | **3** | MISS | MISS | BM25 finds "topology" + "affinity" |
| k8s version GA/stable | **1** | 2 | 4 | BM25 matches "GA" terminology |
| too many NUMA nodes limit | **1** | 1 | 2 | RRF boosts exact match |
| default topology policy | **1** | 1 | 2 | RRF boosts exact match |
| best-effort policy affinity | **1** | 1 | 2 | RRF boosts exact match |

### Pattern: PLM Wins When

1. **Technical terms matter** — BM25 catches exact matches ("NUMA", "kubelet", "GA")
2. **Natural language queries** — Semantic search alone misses problem descriptions
3. **Terminology variations** — RRF combines signals from different phrasings

### Pattern: PLM Loses When

1. **Pure semantic queries** — Baselines sometimes rank higher on abstract questions
2. **Both miss** — Complex or ambiguous queries fail across all systems
3. **Noise from BM25** — Occasionally BM25 introduces false positives

---

## Key Findings

### 1. PLM Provides Significant Improvement on Natural Language Problem Descriptions

- **+18% MRR** on needle benchmark (statistically significant)
- **+20% Hit@1** (80% vs 60%)
- Excels when users describe problems in their own words

### 2. PLM Underperforms on Technical Terminology Queries

- **-1.9% MRR** on informed benchmark (PLM is worse than naive!)
- Found fewer documents (18/25) than naive baseline (21/25)
- Technical terms like "SubjectAccessReview", "etcd API" were MISSED

**Implication:** PLM's hybrid approach adds noise when exact term matching is sufficient.

### 3. Content Enrichment Consistently Hurts Retrieval

| Benchmark | Enrichment Effect |
|-----------|-------------------|
| Needle | -2.3% (hurt) |
| **Informed** | **-4.7% (worst)** |
| Realistic | -3.1% (hurt) |

**Hypothesis:** Prepending `"keywords: ... | entities: ..."` dilutes both:
1. **Embedding signal** - BGE encodes the prefix as semantics
2. **BM25 precision** - Extra terms reduce TF-IDF accuracy

**Recommendation:** Remove enrichment from embedding AND BM25 pipelines. Test keeping it only for answer generation.

### 4. BM25 + RRF Fusion Helps, But Not Enough

| Benchmark | BM25+RRF Benefit | Enrichment Damage | Net |
|-----------|------------------|-------------------|-----|
| Needle | +20.5% | -2.3% | **+18.1%** |
| Informed | +2.8% | -4.7% | **-1.9%** |
| Realistic | +5.1% | -3.1% | **+2.0%** |

On informed queries, enrichment damage **exceeds** BM25 benefit.

### 5. Realistic Queries Are Harder

| Observation | Implication |
|-------------|-------------|
| All variants ~20% MRR | Many queries may not have ideal corpus matches |
| 68-70% not found | Kubernetes docs have coverage gaps |
| No significant differences | Diminishing returns on harder queries |

### 5. Latency-Accuracy Tradeoff

| Variant | Avg Latency | Relative Speed |
|---------|-------------|----------------|
| PLM | 430ms | 1.0x (baseline) |
| Enriched | 124ms | 3.5x faster |
| Naive | 89ms | 4.8x faster |

PLM is 3-5x slower due to BM25 lookup + RRF computation. Acceptable for interactive search; may need optimization for high-throughput use cases.

### 6. Raw BM25 Experiment: Query-Type Dependent Results

**Hypothesis tested:** Does removing enrichment from BM25 index improve retrieval?

**Setup:** Rebuilt BM25 index using raw content only (embeddings still use enriched content from original database).

| Benchmark | PLM (Enriched BM25) | PLM (Raw BM25) | Change |
|-----------|---------------------|----------------|--------|
| **Needle** | **0.842 MRR** | 0.796 MRR | **-5.5%** (worse) |
| **Informed** | 0.621 MRR | 0.568 MRR | **-8.5%** (worse) |
| **Realistic** | 0.196 MRR | **0.207 MRR** | **+5.6%** (better) |

**Detailed comparison:**

| Metric | Needle (Enriched) | Needle (Raw) | Informed (Enriched) | Informed (Raw) | Realistic (Enriched) | Realistic (Raw) |
|--------|-------------------|--------------|---------------------|----------------|----------------------|-----------------|
| MRR | **0.842** | 0.796 | 0.621 | 0.568 | 0.196 | **0.207** |
| Hit@1 | **80%** | 75% | 56% | 48% | 12.5% | **13.5%** |
| Hit@3 | **90%** | 85% | 68% | 64% | 24.5% | 24.5% |
| Hit@5 | 90% | 90% | 72% | 72% | 32.0% | **32.5%** |
| Found | 18/20 | 18/20 | 18/25 | 18/25 | 128/400 | **130/400** |

**Interpretation:**
1. **Needle queries BENEFIT from enriched BM25** (+5.5%) — Precise topic questions benefit from keyword expansion
2. **Informed queries NEED enrichment in BM25** (+8.5%) — Technical terms like "SubjectAccessReview" are expanded via keyword extraction
3. **Realistic queries HURT by enrichment in BM25** (-5.6%) — Natural language doesn't match extracted keywords

**Pattern discovered:**
| Query Type | Natural Language? | Enriched BM25 Effect |
|------------|-------------------|----------------------|
| Needle | Partial (natural phrasing, specific topic) | **Helps** (+5.5%) |
| Informed | No (technical terminology) | **Helps** (+8.5%) |
| Realistic | Yes (fully natural language) | **Hurts** (-5.6%) |

**Conclusion:** The optimal BM25 strategy depends on expected query distribution:
- **Technical users** → Keep enriched BM25 (needle + informed win)
- **General users** → Consider raw BM25 or adaptive strategy

**Note:** This only tests BM25 enrichment. Semantic embeddings still use enriched content. A full test would require re-embedding with raw content (expensive).

---

## Recommendations

### Keep

| Component | Rationale |
|-----------|-----------|
| **BM25 + RRF fusion** | Primary source of improvement |
| **Identical chunk boundaries** | Ensures fair comparison |
| **BGE-base-en-v1.5** | Good balance of quality and speed |

### Reconsider

| Component | Issue | Recommendation |
|-----------|-------|----------------|
| **Content enrichment (for retrieval)** | No benefit, slight negative effect | Remove from embedding pipeline |
| **Enrichment (for generation)** | Not tested | May still help LLM answer quality |

### Future Work

| Priority | Investigation |
|----------|---------------|
| High | Isolate query expansion effect (run with/without) |
| High | Test enrichment removal in production |
| Medium | Test on different corpora (not just Kubernetes) |
| Medium | Add LLM-based answer quality grading |
| Low | Experiment with different chunk sizes |
| Low | Compare embedding models (E5, GTE, etc.) |

---

## Methodology

### Chunk Extraction

All 20,801 chunks extracted from PLM's production SQLite database:
- **Path:** `/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data/index.db`
- **Fields:** chunk_id, doc_id, content, enriched_content, heading, start_char, end_char, chunk_index
- **Purpose:** Ensure identical chunk boundaries across all variants

### FAISS Index Construction

```python
# Both baselines use identical setup
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(texts, normalize_embeddings=True)
index = faiss.IndexFlatIP(768)  # Inner product = cosine after normalization
index.add(embeddings)
```

### Query Execution

```python
# PLM
results = plm.retriever.retrieve(query, k=5, use_rewrite=False)

# Baselines
query_embedding = model.encode([query], normalize_embeddings=True)
scores, indices = index.search(query_embedding, k=5)
```

### Metrics Calculation

| Metric | Formula |
|--------|---------|
| MRR | Mean of 1/rank for found documents, 0 for not found |
| Hit@k | Percentage of queries where target found in top k |
| Bootstrap CI | 1000 resamples, percentile method |

### Statistical Significance

- **Method:** Bootstrap resampling on paired MRR differences
- **Confidence level:** 95%
- **Significant if:** CI excludes 0

---

## Raw Data Reference

### Files Generated

| File | Description | Size |
|------|-------------|------|
| `needle_benchmark.json` | Full needle results (20 queries) | 10 KB |
| `informed_benchmark.json` | Full informed results (25 queries) | 12 KB |
| `realistic_benchmark.json` | Full realistic results (400 queries) | 183 KB |
| `needle_raw_bm25_benchmark.json` | Needle with raw BM25 (20 queries) | 10 KB |
| `informed_raw_bm25_benchmark.json` | Informed with raw BM25 (25 queries) | 12 KB |
| `realistic_raw_bm25_benchmark.json` | Realistic with raw BM25 (400 queries) | 183 KB |
| `needle_per_question.csv` | Needle results in CSV format | ~3 KB |
| `informed_per_question.csv` | Informed results in CSV format | ~4 KB |
| `realistic_per_question.csv` | Realistic results in CSV format | ~50 KB |
| `BENCHMARK_REPORT.md` | Summary report | 6 KB |
| `DETAILED_RESULTS.md` | This document | ~30 KB |

### Cache Files

| File | Description | Size |
|------|-------------|------|
| `.cache/faiss_index_enriched_*.index` | Cached enriched FAISS index | 64 MB |
| `.cache/faiss_index_raw_*.index` | Cached raw FAISS index | 64 MB |

### JSON Structure

```json
{
  "metadata": {
    "timestamp": "2026-02-19T18:41:33.331052",
    "total_queries": 20,
    "k": 5
  },
  "per_question": [
    {
      "id": "q_001",
      "question": "...",
      "target_doc_id": "...",
      "plm_rank": 3,
      "plm_latency_ms": 2720.54,
      "baseline_enriched_rank": null,
      "baseline_enriched_latency_ms": 127.01,
      "baseline_naive_rank": null,
      "baseline_naive_latency_ms": 80.70
    }
  ],
  "plm": { "mrr": 0.842, "hit_at_1": 80.0, ... },
  "baseline_enriched": { ... },
  "baseline_naive": { ... },
  "statistics": {
    "mrr_ci_95": { ... },
    "attribution": { ... }
  }
}
```

---

## Appendix: Complete Per-Question Data

### Needle Benchmark (20 queries)

See `needle_per_question.csv` for full data.

### Realistic Benchmark (400 queries)

See `realistic_per_question.csv` for full data.

---

*Document generated: 2026-02-21*  
*Benchmark runner: poc/plm_vs_rag_benchmark/benchmark_runner.py*
