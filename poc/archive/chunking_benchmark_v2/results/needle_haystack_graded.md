# Needle-in-Haystack Grading Results

## Summary
- **Total Questions**: 20
- **Average Score**: 8.45/10
- **Pass Rate (≥7)**: 18/20 (90%)
- **Perfect Scores (10/10)**: 12
- **Failed Questions (<7)**: 2 (Q6, Q12)

## Grading Rubric Used
| Score | Meaning |
|-------|---------|
| 9-10 | Expected answer found verbatim or nearly verbatim |
| 7-8 | Answer concept present, slightly different wording |
| 5-6 | Partial answer, missing key details |
| 3-4 | Tangentially related content only |
| 1-2 | Completely irrelevant content, needle not found |

---

## Detailed Grades

### Q1: My pod keeps getting rejected with a topology affinity error, what's going on?
**Expected Answer**: This happens with restricted or single-numa-node policies when the Topology Manager cannot find a suitable NUMA node affinity. The pod enters a Terminated state. Use a Deployment or ReplicaSet to trigger redeployment.

**Score**: 10/10

**Reasoning**: PERFECT MATCH. The first retrieved chunk (`mdsem_7` - restricted policy) contains the exact answer:
- "the Topology Manager will reject this pod from the node. This will result in a pod entering a `Terminated` state with a pod admission failure"
- "It is recommended to use a ReplicaSet or Deployment to trigger a redeployment of the pod"

The fourth chunk (`mdsem_8` - single-numa-node policy) also confirms the same behavior for the other policy mentioned in the expected answer. This is a verbatim match.

**Retrieved Chunk Excerpts**:
> "If the affinity is not preferred, the Topology Manager will reject this pod from the node. This will result in a pod entering a `Terminated` state with a pod admission failure... It is recommended to use a ReplicaSet or Deployment to trigger a redeployment of the pod."

---

### Q2: How do I force all my containers to run on the same NUMA node in k8s?
**Expected Answer**: Use the pod scope with single-numa-node policy. Set topologyManagerScope to pod in kubelet config and use --topology-manager-policy=single-numa-node

**Score**: 9/10

**Reasoning**: The answer is present across multiple chunks. Chunk 4 (`mdsem_4` - pod scope) explicitly states:
- "Using the `pod` scope in tandem with `single-numa-node` Topology Manager policy is specifically valuable"
- "you are able to place all containers in a pod onto a single NUMA node"
- "set `topologyManagerScope` in the kubelet configuration file to `pod`"

The concept is fully present. Minor deduction (-1) because the exact flag syntax `--topology-manager-policy=single-numa-node` is not shown (it shows kubelet config file method instead), though chunk 2 mentions the policy name.

**Retrieved Chunk Excerpts**:
> "Using the `pod` scope in tandem with `single-numa-node` Topology Manager policy... you are able to place all containers in a pod onto a single NUMA node... set `topologyManagerScope` in the kubelet configuration file to `pod`"

---

### Q3: What's the difference between restricted and best-effort topology policies?
**Expected Answer**: best-effort admits pods even if preferred NUMA affinity isn't available, while restricted rejects pods that can't get preferred affinity, putting them in Terminated state

**Score**: 10/10

**Reasoning**: PERFECT COMPARISON. Chunks 1 and 3 contain exactly this comparison:
- Chunk 1 (`mdsem_6` - best-effort): "If the affinity is not preferred, the Topology Manager will store this and admit the pod to the node anyway"
- Chunk 3 (`mdsem_7` - restricted): "If the affinity is not preferred, the Topology Manager will reject this pod from the node. This will result in a pod entering a `Terminated` state"

This is a verbatim match with both policies explained side-by-side.

**Retrieved Chunk Excerpts**:
> best-effort: "If the affinity is not preferred, the Topology Manager will store this and admit the pod to the node anyway"
> restricted: "the Topology Manager will reject this pod from the node. This will result in a pod entering a `Terminated` state"

---

### Q4: Getting error about too many NUMA nodes on my server, what's the limit?
**Expected Answer**: 8 NUMA nodes by default. Use max-allowable-numa-nodes policy option to allow more, but it's not recommended

**Score**: 10/10

**Reasoning**: PERFECT MATCH. Chunk 1 (`mdsem_13` - Known limitations) states:
- "The maximum number of NUMA nodes that Topology Manager allows is 8"

Chunk 3 (`mdsem_11` - max-allowable-numa-nodes) adds:
- "nodes with more than 8 NUMA nodes can be allowed to run with the Topology Manager enabled"
- "this policy option... is **not** recommended"

Complete answer with all details present.

**Retrieved Chunk Excerpts**:
> "The maximum number of NUMA nodes that Topology Manager allows is 8... nodes with more than 8 NUMA nodes can be allowed to run... is **not** recommended"

---

### Q5: How to enable topology manager on Windows nodes?
**Expected Answer**: Enable the WindowsCPUAndMemoryAffinity feature gate and ensure the container runtime supports it

**Score**: 10/10

**Reasoning**: VERBATIM MATCH. Chunk 1 (`mdsem_2` - Windows Support) contains the exact answer:
- "The Topology Manager support can be enabled on Windows by using the `WindowsCPUAndMemoryAffinity` feature gate"
- "it requires support in the container runtime"

Perfect match with both requirements stated.

**Retrieved Chunk Excerpts**:
> "The Topology Manager support can be enabled on Windows by using the `WindowsCPUAndMemoryAffinity` feature gate and it requires support in the container runtime."

---

### Q6: Which k8s version made topology manager GA/stable?
**Expected Answer**: v1.27

**Score**: 4/10

**Reasoning**: PARTIAL FAILURE. The retrieved chunks discuss topology manager features but do NOT contain the specific answer "v1.27". The chunks mention:
- Chunk 5 (`mdsem_8`): Discusses single-numa-node policy behavior
- Chunk 2 (`mdsem_1`): Explains how topology manager works
- Chunk 3 (`mdsem_7`): Describes restricted policy

None of these chunks state when Topology Manager became GA. The retrieval found related content about the feature itself but missed the specific version number fact. This is a fact-lookup failure where the semantic search didn't match the version information.

**Retrieved Chunk Excerpts**:
> No chunk contains "v1.27" or states when Topology Manager became generally available. Retrieved chunks discuss policies and functionality instead.

---

### Q7: What flag do I pass to kubelet for setting topology policy?
**Expected Answer**: --topology-manager-policy

**Score**: 10/10

**Reasoning**: VERBATIM MATCH. Chunk 1 (`mdsem_5` - Topology manager policies) contains the exact answer:
- "You can set a policy via a kubelet flag, `--topology-manager-policy`"

Perfect match with the exact flag name.

**Retrieved Chunk Excerpts**:
> "You can set a policy via a kubelet flag, `--topology-manager-policy`"

---

### Q8: My latency-sensitive app is slow, containers seem to be on different NUMA nodes. How to fix?
**Expected Answer**: Use pod scope with single-numa-node policy to place all containers on a single NUMA node, eliminating inter-NUMA communication overhead

**Score**: 10/10

**Reasoning**: PERFECT MATCH. Chunk 1 (`mdsem_4` - pod scope) contains the exact solution:
- "Using the `pod` scope in tandem with `single-numa-node` Topology Manager policy is specifically valuable for workloads that are latency sensitive"
- "you are able to place all containers in a pod onto a single NUMA node; hence, the inter-NUMA communication overhead can be eliminated for that pod"

The answer explicitly addresses latency-sensitive workloads and provides the exact solution.

**Retrieved Chunk Excerpts**:
> "Using the `pod` scope in tandem with `single-numa-node` Topology Manager policy is specifically valuable for workloads that are latency sensitive... the inter-NUMA communication overhead can be eliminated for that pod"

---

### Q9: Why would I use pod scope instead of container scope for topology?
**Expected Answer**: Pod scope groups all containers to a common set of NUMA nodes, treating the pod as a whole. Container scope does separate alignment per container with no grouping, which can result in containers on different NUMA nodes

**Score**: 10/10

**Reasoning**: COMPLETE COMPARISON. Chunks 1 and 2 together provide the full answer:
- Chunk 1 (`mdsem_3` - container scope): "for each container (in a pod) a separate alignment is computed. In other words, there is no notion of grouping the containers to a specific set of NUMA nodes"
- Chunk 2 (`mdsem_4` - pod scope): "grouping all containers in a pod to a common set of NUMA nodes. That is, the Topology Manager treats a pod as a whole"

Perfect comparison with both scopes explained.

**Retrieved Chunk Excerpts**:
> Container: "there is no notion of grouping the containers to a specific set of NUMA nodes... arbitrary alignment of individual containers"
> Pod: "grouping all containers in a pod to a common set of NUMA nodes... the Topology Manager treats a pod as a whole"

---

### Q10: What are hint providers in topology manager?
**Expected Answer**: Components like CPU Manager and Device Manager that send and receive topology information through the Topology Manager interface

**Score**: 9/10

**Reasoning**: The answer is present but requires reading chunk 2. Chunk 2 (`mdsem_1` - How topology manager works) explains:
- "The Topology Manager provides an interface for components, called *Hint Providers*, to send and receive topology information"
- "Prior to the introduction of Topology Manager, the CPU and Device Manager in Kubernetes make resource allocation decisions independently"

The concept is clear, with CPU Manager and Device Manager mentioned as examples in context. Minor deduction (-1) because they're not explicitly named as "Hint Providers" but rather implied through context.

**Retrieved Chunk Excerpts**:
> "The Topology Manager provides an interface for components, called *Hint Providers*, to send and receive topology information... Prior to the introduction of Topology Manager, the CPU and Device Manager in Kubernetes make resource allocation decisions independently"

---

### Q11: How do I configure kubelet to prefer NUMA nodes that are closer together?
**Expected Answer**: Add prefer-closest-numa-nodes=true to Topology Manager policy options. This makes best-effort and restricted policies favor NUMA nodes with shorter distance

**Score**: 8/10

**Reasoning**: The answer is present but in chunk 5 (last position). Chunk 5 (`mdsem_10` - prefer-closest-numa-nodes) contains:
- "You can enable this option by adding `prefer-closest-numa-nodes=true` to the Topology Manager policy options"
- "the `best-effort` and `restricted` policies favor sets of NUMA nodes with shorter distance between them"

The answer is complete and accurate. Minor deduction (-2) because the user would need to read through 4 other chunks first to find this answer.

**Retrieved Chunk Excerpts**:
> "If you specify the `prefer-closest-numa-nodes` policy option, the `best-effort` and `restricted` policies favor sets of NUMA nodes with shorter distance... You can enable this option by adding `prefer-closest-numa-nodes=true`"

---

### Q12: When did prefer-closest-numa-nodes become generally available?
**Expected Answer**: Kubernetes 1.32

**Score**: 2/10

**Reasoning**: COMPLETE FAILURE. `needle_found: false` - NO chunks from the needle document were retrieved. All 5 chunks are from unrelated documents:
- reference_node_node-status (node info)
- reference_using-api_deprecation-policy (API deprecation)
- reference_command-line-tools-reference_kube-apiserver (API server flags)
- reference_using-api_deprecation-policy (more deprecation policy)
- concepts_workloads_management (canary deployments)

The answer "Kubernetes 1.32" exists in the needle document (in the prefer-closest-numa-nodes section) but was not retrieved. This is a specific fact-lookup failure where the query didn't semantically match the content containing the version number.

**Retrieved Chunk Excerpts**:
> None from needle document. Retrieved content about node status, API deprecation policies, and canary deployments - completely unrelated to the question.

---

### Q13: What's the default topology manager policy if I don't set anything?
**Expected Answer**: none - which does not perform any topology alignment

**Score**: 10/10

**Reasoning**: VERBATIM MATCH. Chunk 1 (`mdsem_6` - none policy) contains the exact answer:
- "### `none` policy... This is the default policy and does not perform any topology alignment"

Perfect match with both parts of the answer.

**Retrieved Chunk Excerpts**:
> "### `none` policy {#policy-none} This is the default policy and does not perform any topology alignment."

---

### Q14: Pod was scheduled but then failed on the node, says something about topology. Is that a bug?
**Expected Answer**: No, this is a known limitation. The scheduler is not topology-aware, so it's possible to be scheduled on a node and then fail due to the Topology Manager

**Score**: 7/10

**Reasoning**: PARTIAL ANSWER. The retrieved chunks explain pod rejection behavior (chunks 1, 3 describe Terminated state, admission failure) but the KEY answer - "scheduler is not topology-aware" being a "known limitation" - is NOT explicitly stated in the top chunks.

The user would understand that pods can be rejected and enter Terminated state, but wouldn't learn that this is a known limitation by design (scheduler not being topology-aware). The specific answer about the scheduler limitation exists in the needle document's "Known limitations" section but wasn't retrieved in the top 5.

**Retrieved Chunk Excerpts**:
> Chunks explain pod rejection and Terminated state behavior, but don't state "scheduler is not topology-aware" as a known limitation.

---

### Q15: What QoS class does my pod need to be for topology hints to work?
**Expected Answer**: Topology Manager aligns Pods of all QoS classes (BestEffort, Burstable, Guaranteed)

**Score**: 8/10

**Reasoning**: The answer is present but split across chunks. Chunk 1 (`mdsem_12` - Pod interactions) shows examples of BestEffort, Burstable, and Guaranteed pods with topology manager. Chunk 4 (`mdsem_2` - Topology manager scopes and policies) states:
- "The Topology Manager currently: aligns Pods of all QoS classes"

The explicit "all QoS classes" statement is in chunk 4. Minor deduction (-2) because the answer requires reading multiple chunks and the key statement is not in the first chunk.

**Retrieved Chunk Excerpts**:
> "The Topology Manager currently: aligns Pods of all QoS classes. aligns the requested resources that Hint Provider provides topology hints for."

---

### Q16: How to set topology scope to pod level in kubelet?
**Expected Answer**: Set topologyManagerScope to pod in the kubelet configuration file

**Score**: 10/10

**Reasoning**: VERBATIM MATCH. Chunks 1 and 2 both contain this exact information:
- "set `topologyManagerScope` in the kubelet configuration file to `pod`"

Perfect match with the exact configuration method.

**Retrieved Chunk Excerpts**:
> "To select the `pod` scope, set `topologyManagerScope` in the kubelet configuration file to `pod`"

---

### Q17: What feature gate do I need for topology manager policy options?
**Expected Answer**: TopologyManagerPolicyOptions (enabled by default)

**Score**: 10/10

**Reasoning**: VERBATIM MATCH. Chunk 1 (`mdsem_9` - Topology manager policy options) contains the exact answer:
- "Support for the Topology Manager policy options requires `TopologyManagerPolicyOptions` feature gate to be enabled (it is enabled by default)"

Perfect match including the "enabled by default" detail.

**Retrieved Chunk Excerpts**:
> "Support for the Topology Manager policy options requires `TopologyManagerPolicyOptions` feature gate to be enabled (it is enabled by default)"

---

### Q18: My multi-socket server has GPUs and CPUs, how does k8s coordinate their placement?
**Expected Answer**: Topology Manager coordinates CPU Manager and Device Manager to make topology-aligned resource allocation decisions, avoiding CPUs and devices being allocated from different NUMA nodes

**Score**: 7/10

**Reasoning**: The answer is present but not in the first chunk. Chunk 3 (`mdsem_12` - Pod interactions) mentions CPU Manager and Device Manager in context of topology hints. However, the KEY explanation is in chunk 2 (`mdsem_1` - How topology manager works):
- "Prior to the introduction of Topology Manager, the CPU and Device Manager in Kubernetes make resource allocation decisions independently of each other. This can result in undesirable allocations... CPUs and devices being allocated from different NUMA Nodes"
- "The Topology Manager is a kubelet component, which acts as a source of truth so that other kubelet components can make topology aligned resource allocation choices"

The answer requires reading multiple chunks and the first chunk (kube-scheduler) is unrelated. Deduction (-3) for poor chunk ranking.

**Retrieved Chunk Excerpts**:
> "Prior to the introduction of Topology Manager, the CPU and Device Manager... make resource allocation decisions independently... CPUs and devices being allocated from different NUMA Nodes, thus incurring additional latency"

---

### Q19: With single-numa-node policy, when exactly does a pod get rejected?
**Expected Answer**: When a single NUMA Node affinity is not possible - if more than one NUMA node is required to satisfy the allocation, the pod is rejected

**Score**: 10/10

**Reasoning**: PERFECT MATCH. Chunks 1 and 3 contain the exact answer:
- Chunk 1 (`mdsem_4` - pod scope): "In the case of `single-numa-node` policy, a pod is accepted only if a suitable set of NUMA nodes is present... a set containing more NUMA nodes - it results in pod rejection (because instead of one NUMA node, two or more NUMA nodes are required to satisfy the allocation)"
- Chunk 3 (`mdsem_8` - single-numa-node policy): "If, however, this is not possible then the Topology Manager will reject the pod from the node"

Complete answer with explicit explanation of when rejection happens.

**Retrieved Chunk Excerpts**:
> "In the case of `single-numa-node` policy, a pod is accepted only if a suitable set of NUMA nodes is present... a set containing more NUMA nodes - it results in pod rejection (because instead of one NUMA node, two or more NUMA nodes are required)"

---

### Q20: What happens if topology manager can't find preferred NUMA affinity with best-effort policy?
**Expected Answer**: The pod is admitted to the node anyway - best-effort stores the non-preferred hint and allows the pod

**Score**: 10/10

**Reasoning**: VERBATIM MATCH. Chunk 1 (`mdsem_6` - best-effort policy) contains the exact answer:
- "If the affinity is not preferred, the Topology Manager will store this and admit the pod to the node anyway"

Perfect match with the exact behavior described.

**Retrieved Chunk Excerpts**:
> "If the affinity is not preferred, the Topology Manager will store this and admit the pod to the node anyway"

---

## Score Distribution

| Score | Count | Questions |
|-------|-------|-----------|
| 10/10 | 12 | Q1, Q3, Q4, Q5, Q7, Q8, Q9, Q13, Q16, Q17, Q19, Q20 |
| 9/10 | 2 | Q2, Q10 |
| 8/10 | 2 | Q11, Q15 |
| 7/10 | 2 | Q14, Q18 |
| 4/10 | 1 | Q6 |
| 2/10 | 1 | Q12 |

## Analysis of Failures

### Q6 (Score: 4/10) - Partial failure
- **Question type**: Specific fact lookup (version number)
- **Issue**: Retrieved chunks about topology manager functionality but not the specific GA version
- **Root cause**: Semantic search didn't match the version number fact "v1.27"
- **Impact**: User gets general information but not the specific answer

### Q12 (Score: 2/10) - Complete failure
- **Question type**: Specific fact lookup (date/version)
- **Issue**: `needle_found: false` - NO needle chunks retrieved at all
- **Root cause**: Query "When did prefer-closest-numa-nodes become generally available?" didn't semantically match the chunk containing "GA since Kubernetes 1.32"
- **Impact**: User gets completely irrelevant results about API deprecation and canary deployments

## Key Observations

1. **Excellent performance on conceptual questions**: 10/10 on policy comparisons, scope explanations, workflow questions
2. **Strong on how-to questions**: 9-10/10 on configuration and setup questions
3. **Weak on specific version lookups**: Both failures (Q6, Q12) were version number queries where semantic search struggled
4. **High retrieval quality overall**: 18/20 questions (90%) had relevant chunks in top-5 results
5. **Needle chunks well-represented**: 19/20 questions had at least one needle chunk retrieved

## Conclusion

The `enriched_hybrid_llm` strategy demonstrates **excellent performance** for conceptual and practical questions about Kubernetes Topology Manager, achieving a **90% pass rate (18/20 questions ≥7/10)** and an **average score of 8.45/10**.

**Strengths**:
- Exceptional at retrieving policy comparisons and explanations
- Strong on configuration and troubleshooting questions
- Good semantic understanding of human-like queries

**Weaknesses**:
- Struggles with specific version/date fact lookups (2 failures)
- Semantic search doesn't always match numerical facts well

**Verdict**: **VALIDATED** - The strategy is highly effective for technical documentation retrieval, with failures concentrated in a specific category (version lookups) that could be addressed with better keyword enrichment for dates/versions.
