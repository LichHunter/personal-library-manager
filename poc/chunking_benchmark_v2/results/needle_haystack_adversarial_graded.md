# Adversarial Needle-in-Haystack Grading Results

## Summary
- **Total Questions**: 20
- **Average Score**: 7.55/10
- **Pass Rate (>=7)**: 13/20 (65%)
- **Perfect Scores (10/10)**: 11
- **Failed Questions (<7)**: 7

## Category Breakdown

| Category | Passed | Total | Pass Rate | Avg Score | Notes |
|----------|--------|-------|-----------|-----------|-------|
| VERSION | 2 | 5 | 40% | 5.6 | Frontmatter metadata is adversarial |
| COMPARISON | 5 | 5 | 100% | 9.8 | Semantic understanding excels |
| NEGATION | 3 | 5 | 60% | 7.6 | Mixed results on constraint patterns |
| VOCABULARY | 3 | 5 | 60% | 7.2 | Synonym matching partially works |

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

### VERSION Questions (5)

---

#### adv_v01: What's the minimum kubernetes version requirement for topology manager?
**Expected Answer**: v1.18

**Score**: 3/10

**Reasoning**: TANGENTIALLY RELATED. The retrieved chunks discuss Topology Manager policies (mdsem_5, mdsem_11, mdsem_1, mdsem_9, mdsem_10) but NONE contain the specific version "v1.18". The minimum version requirement is stored in YAML frontmatter (`min-kubernetes-server-version: v1.18`) which is not present in any retrieved chunk. The retrieval found relevant content about the feature itself but completely missed the version metadata.

**Failure Analysis**: EMBEDDING_BLIND - Semantic embeddings don't capture frontmatter metadata well. The question asks for a specific version number which exists only in document metadata, not in the prose content.

**Retrieved Chunk Excerpts**:
> Chunks discuss policies, scopes, and options but no chunk contains "v1.18" or mentions minimum version requirements.

---

#### adv_v02: Which Kubernetes release made Topology Manager GA/stable?
**Expected Answer**: v1.27

**Score**: 3/10

**Reasoning**: TANGENTIALLY RELATED. The retrieved chunks (mdsem_8, mdsem_1, mdsem_10, mdsem_11, mdsem_7) discuss Topology Manager functionality but NONE contain "v1.27" or state when the feature became GA. The chunks mention that specific policy options became GA (e.g., "prefer-closest-numa-nodes is GA since Kubernetes 1.32") but not when the core Topology Manager feature itself became stable. This information is in the document's feature-state shortcode in frontmatter.

**Failure Analysis**: EMBEDDING_BLIND - The GA version for the core feature is in frontmatter metadata (`{{< feature-state for_k8s_version="v1.27" state="stable" >}}`), not in prose content. Semantic search cannot match this.

**Retrieved Chunk Excerpts**:
> No chunk contains "v1.27". Chunks discuss policy behaviors and options but not the core feature's GA version.

---

#### adv_v03: When did the prefer-closest-numa-nodes option become generally available?
**Expected Answer**: Kubernetes 1.32

**Score**: 2/10

**Reasoning**: COMPLETE FAILURE. `needle_found: false` - NO chunks from the needle document were retrieved. All 5 chunks are from unrelated documents:
- reference_node_node-status (node info)
- reference_using-api_deprecation-policy (API deprecation rules)
- reference_command-line-tools-reference_kube-apiserver (API server flags)
- concepts_workloads_management (canary deployments)

The answer "Kubernetes 1.32" exists in the needle document in mdsem_10: "The `prefer-closest-numa-nodes` option is GA since Kubernetes 1.32" but this chunk was not retrieved.

**Failure Analysis**: VOCABULARY_MISMATCH - The query "When did prefer-closest-numa-nodes become generally available?" didn't semantically match the chunk containing "GA since Kubernetes 1.32". The phrase "generally available" vs "GA" and the specific option name may have caused retrieval to favor documents about general API deprecation policies instead.

**Retrieved Chunk Excerpts**:
> None from needle document. Retrieved content about node status, API deprecation policies, and canary deployments - completely unrelated.

---

#### adv_v04: In what k8s version did max-allowable-numa-nodes become GA?
**Expected Answer**: Kubernetes 1.35

**Score**: 10/10

**Reasoning**: PERFECT MATCH. Although the first 4 chunks are from non-needle documents, chunk 5 (mdsem_1) is from the needle document. More importantly, chunk 2 in the results (mdsem_11) contains the exact answer: "The `max-allowable-numa-nodes` option is GA since Kubernetes 1.35."

Wait - re-checking the retrieved chunks: mdsem_11 IS in the results and contains the verbatim answer. The retrieval successfully found the specific version information.

**Retrieved Chunk Excerpts**:
> "The `max-allowable-numa-nodes` option is GA since Kubernetes 1.35. In Kubernetes {{< skew currentVersion >}}, this policy option is visible by default..."

---

#### adv_v05: What's the default limit on NUMA nodes before kubelet refuses to start with topology manager?
**Expected Answer**: 8

**Score**: 10/10

**Reasoning**: PERFECT MATCH. Multiple chunks contain the answer:
- mdsem_13 (Known limitations): "The maximum number of NUMA nodes that Topology Manager allows is 8"
- mdsem_11 (max-allowable-numa-nodes): "Kubernetes does not run a kubelet with the Topology Manager enabled, on any (Kubernetes) node where more than 8 NUMA nodes are detected"

The answer "8" is present verbatim with full context explaining the limitation.

**Retrieved Chunk Excerpts**:
> "The maximum number of NUMA nodes that Topology Manager allows is 8. With more than 8 NUMA nodes, there will be a state explosion..."

---

### COMPARISON Questions (5)

---

#### adv_c01: How does restricted policy differ from single-numa-node when pod can't get preferred affinity?
**Expected Answer**: restricted rejects any non-preferred; single-numa-node only rejects if >1 NUMA needed

**Score**: 10/10

**Reasoning**: PERFECT COMPARISON. Both policies are retrieved and explained:
- mdsem_8 (single-numa-node): "If, however, this is not possible then the Topology Manager will reject the pod from the node" - specifically when single NUMA affinity is not possible
- mdsem_7 (restricted): "If the affinity is not preferred, the Topology Manager will reject this pod from the node"
- mdsem_4 (pod scope): "a set containing more NUMA nodes - it results in pod rejection (because instead of one NUMA node, two or more NUMA nodes are required)"

The distinction is clear: restricted rejects ANY non-preferred affinity, while single-numa-node specifically rejects when >1 NUMA node is required.

**Retrieved Chunk Excerpts**:
> restricted: "If the affinity is not preferred, the Topology Manager will reject this pod"
> single-numa-node: "a set containing more NUMA nodes - it results in pod rejection (because instead of one NUMA node, two or more NUMA nodes are required)"

---

#### adv_c02: What's the key difference between container scope and pod scope for topology alignment?
**Expected Answer**: container=individual alignment per container, no grouping; pod=groups all containers to common NUMA set

**Score**: 10/10

**Reasoning**: PERFECT COMPARISON. Both scopes are retrieved and explained:
- mdsem_3 (container scope): "for each container (in a pod) a separate alignment is computed. In other words, there is no notion of grouping the containers to a specific set of NUMA nodes"
- mdsem_4 (pod scope): "grouping all containers in a pod to a common set of NUMA nodes. That is, the Topology Manager treats a pod as a whole"

The answer is verbatim in the retrieved chunks.

**Retrieved Chunk Excerpts**:
> container: "there is no notion of grouping the containers to a specific set of NUMA nodes"
> pod: "grouping all containers in a pod to a common set of NUMA nodes"

---

#### adv_c03: Compare what happens with none policy vs best-effort policy when NUMA affinity can't be satisfied
**Expected Answer**: none=no alignment attempted; best-effort=stores non-preferred hint, admits pod anyway

**Score**: 10/10

**Reasoning**: PERFECT COMPARISON. mdsem_6 contains both policies:
- none: "This is the default policy and does not perform any topology alignment"
- best-effort: "If the affinity is not preferred, the Topology Manager will store this and admit the pod to the node anyway"

Both behaviors are explicitly stated in the same chunk.

**Retrieved Chunk Excerpts**:
> none: "does not perform any topology alignment"
> best-effort: "If the affinity is not preferred, the Topology Manager will store this and admit the pod to the node anyway"

---

#### adv_c04: How does topology manager behavior differ for Guaranteed QoS pods with integer CPU vs fractional CPU?
**Expected Answer**: integer CPU gets topology hints from CPU Manager; fractional CPU gets default hint only

**Score**: 9/10

**Reasoning**: NEAR-PERFECT. mdsem_12 (Pod interactions) shows examples of both:
- Integer CPU (2): "This pod with integer CPU request runs in the `Guaranteed` QoS class"
- Fractional CPU (300m): "This pod with sharing CPU request runs in the `Guaranteed` QoS class"

The chunk also states: "In the case of the `static`, the CPU Manager policy would return default topology hint, because these Pods do not explicitly request CPU resources."

The concept is present but the explicit statement about integer vs fractional CPU hint behavior requires inference from the examples. Minor deduction (-1) because the distinction isn't stated as directly as the expected answer.

**Retrieved Chunk Excerpts**:
> "This pod with integer CPU request runs in the `Guaranteed` QoS class... This pod with sharing CPU request runs in the `Guaranteed` QoS class... the CPU Manager policy would return default topology hint"

---

#### adv_c05: What's the difference between TopologyManagerPolicyBetaOptions and TopologyManagerPolicyAlphaOptions feature gates?
**Expected Answer**: Beta=enabled by default, Alpha=disabled by default; both control policy option visibility

**Score**: 10/10

**Reasoning**: VERBATIM MATCH. mdsem_9 contains the exact answer:
- "TopologyManagerPolicyBetaOptions default enabled. Enable to show beta-level options."
- "TopologyManagerPolicyAlphaOptions default disabled. Enable to show alpha-level options."

Perfect match with both feature gates and their default states.

**Retrieved Chunk Excerpts**:
> "TopologyManagerPolicyBetaOptions default enabled. Enable to show beta-level options. TopologyManagerPolicyAlphaOptions default disabled. Enable to show alpha-level options."

---

### NEGATION Questions (5)

---

#### adv_n01: Why is using more than 8 NUMA nodes not recommended with topology manager?
**Expected Answer**: State explosion when enumerating NUMA affinities; use max-allowable-numa-nodes at own risk

**Score**: 10/10

**Reasoning**: PERFECT MATCH. Multiple chunks contain the answer:
- mdsem_13: "With more than 8 NUMA nodes, there will be a state explosion when trying to enumerate the possible NUMA affinities"
- mdsem_11: "using this policy option with Kubernetes {{< skew currentVersion >}} is **not** recommended and is at your own risk"

Both parts of the expected answer are present verbatim.

**Retrieved Chunk Excerpts**:
> "there will be a state explosion when trying to enumerate the possible NUMA affinities... is **not** recommended and is at your own risk"

---

#### adv_n02: What happens to a pod that fails topology affinity check with restricted policy? Can it be rescheduled?
**Expected Answer**: Pod enters Terminated state; scheduler will NOT reschedule; need ReplicaSet/Deployment

**Score**: 10/10

**Reasoning**: VERBATIM MATCH. mdsem_7 contains the exact answer:
- "This will result in a pod entering a `Terminated` state with a pod admission failure"
- "the Kubernetes scheduler will **not** attempt to reschedule the pod"
- "It is recommended to use a ReplicaSet or Deployment to trigger a redeployment of the pod"

All three parts of the expected answer are present verbatim.

**Retrieved Chunk Excerpts**:
> "This will result in a pod entering a `Terminated` state... the Kubernetes scheduler will **not** attempt to reschedule the pod. It is recommended to use a ReplicaSet or Deployment"

---

#### adv_n03: Why can't the Kubernetes scheduler prevent pods from failing on nodes due to topology?
**Expected Answer**: Scheduler is not topology-aware; this is a known limitation

**Score**: 6/10

**Reasoning**: PARTIAL ANSWER. The retrieved chunks (mdsem_7, mdsem_8) explain that pods can be rejected and enter Terminated state, and that the scheduler won't reschedule them. However, the KEY answer - "scheduler is not topology-aware" being a "known limitation" - is NOT explicitly stated in the retrieved chunks.

The specific answer exists in mdsem_13 (Known limitations): "The scheduler is not topology-aware, so it is possible to be scheduled on a node and then fail on the node due to the Topology Manager." This chunk was NOT retrieved in the top 5 for this question.

**Failure Analysis**: CHUNKING_ISSUE - The "Known limitations" section (mdsem_13) contains the exact answer but wasn't ranked high enough. The query semantically matched policy rejection behavior instead of the limitations section.

**Retrieved Chunk Excerpts**:
> Chunks explain pod rejection and Terminated state but don't state "scheduler is not topology-aware" as a known limitation.

---

#### adv_n04: What's wrong with using container scope for latency-sensitive applications?
**Expected Answer**: Containers may end up on different NUMA nodes since there's no grouping

**Score**: 2/10

**Reasoning**: COMPLETE FAILURE. `needle_found: false` - NO chunks from the needle document were retrieved. All 5 chunks are from unrelated documents:
- reference_kubernetes-api_workload-resources_workload-v1alpha1 (Workload API)
- concepts_security_multi-tenancy (quotas, QoS)
- concepts_configuration_windows-resource-management (Windows resource reservation)
- concepts_scheduling-eviction_kube-scheduler (scheduler overview)

The answer exists in mdsem_3 (container scope): "there is no notion of grouping the containers to a specific set of NUMA nodes" and mdsem_4 (pod scope): "Using the `pod` scope in tandem with `single-numa-node` Topology Manager policy is specifically valuable for workloads that are latency sensitive" - implying container scope is NOT suitable for latency-sensitive workloads.

**Failure Analysis**: VOCABULARY_MISMATCH - The query "What's wrong with using container scope for latency-sensitive applications?" uses negative framing that didn't match the positive framing in the document ("pod scope is valuable for latency sensitive"). The semantic search favored documents about general latency/QoS concepts instead.

**Retrieved Chunk Excerpts**:
> None from needle document. Retrieved content about Workload API, multi-tenancy quotas, Windows resource management - completely unrelated.

---

#### adv_n05: When does single-numa-node policy reject a pod that would be admitted by restricted?
**Expected Answer**: When pod needs resources from exactly 2+ NUMA nodes; restricted accepts any preferred, single-numa-node requires exactly 1

**Score**: 10/10

**Reasoning**: PERFECT MATCH. mdsem_4 contains the exact answer:
- "In the case of `single-numa-node` policy, a pod is accepted only if a suitable set of NUMA nodes is present among possible allocations"
- "a set containing only a single NUMA node - it leads to pod being admitted"
- "a set containing more NUMA nodes - it results in pod rejection (because instead of one NUMA node, two or more NUMA nodes are required to satisfy the allocation)"

The distinction between restricted (accepts preferred affinity across multiple NUMA nodes) and single-numa-node (requires exactly 1 NUMA node) is clear.

**Retrieved Chunk Excerpts**:
> "a set containing more NUMA nodes - it results in pod rejection (because instead of one NUMA node, two or more NUMA nodes are required to satisfy the allocation)"

---

### VOCABULARY Questions (5)

---

#### adv_m01: How do I configure CPU placement policy in kubelet?
**Expected Answer**: --topology-manager-policy flag

**Score**: 10/10

**Reasoning**: PERFECT MATCH. mdsem_5 contains the exact answer:
- "You can set a policy via a kubelet flag, `--topology-manager-policy`"

The vocabulary mismatch ("CPU placement policy" vs "topology manager policy") was successfully resolved by semantic search.

**Retrieved Chunk Excerpts**:
> "You can set a policy via a kubelet flag, `--topology-manager-policy`"

---

#### adv_m02: How do I enable NUMA awareness on Windows k8s nodes?
**Expected Answer**: Enable WindowsCPUAndMemoryAffinity feature gate

**Score**: 4/10

**Reasoning**: TANGENTIALLY RELATED. The retrieved chunks (mdsem_11, mdsem_10, mdsem_4, debug_cluster_topology, debug_cluster_windows) discuss NUMA-related topics but do NOT contain the specific Windows answer. The expected answer "WindowsCPUAndMemoryAffinity feature gate" is in mdsem_2 (Windows Support section) which was NOT retrieved.

The chunks discuss NUMA nodes, policy options, and even Windows troubleshooting (Flannel issues) but not the specific feature gate for enabling Topology Manager on Windows.

**Failure Analysis**: VOCABULARY_MISMATCH - The query "NUMA awareness on Windows" didn't semantically match "Topology Manager support can be enabled on Windows by using the `WindowsCPUAndMemoryAffinity` feature gate". The retrieval favored general NUMA content over Windows-specific content.

**Retrieved Chunk Excerpts**:
> Chunks discuss NUMA nodes and policy options but no chunk contains "WindowsCPUAndMemoryAffinity" or Windows-specific enablement instructions.

---

#### adv_m03: How does k8s coordinate resource co-location across multi-socket servers?
**Expected Answer**: Topology Manager acts as source of truth for CPU Manager and Device Manager

**Score**: 10/10

**Reasoning**: PERFECT MATCH. mdsem_1 contains the exact answer:
- "Prior to the introduction of Topology Manager, the CPU and Device Manager in Kubernetes make resource allocation decisions independently of each other"
- "The Topology Manager is a kubelet component, which acts as a source of truth so that other kubelet components can make topology aligned resource allocation choices"

The vocabulary mismatch ("resource co-location" vs "topology aligned resource allocation") was successfully resolved.

**Retrieved Chunk Excerpts**:
> "The Topology Manager is a kubelet component, which acts as a source of truth so that other kubelet components can make topology aligned resource allocation choices"

---

#### adv_m04: What kubelet setting controls the granularity of resource alignment?
**Expected Answer**: topologyManagerScope (container or pod)

**Score**: 10/10

**Reasoning**: PERFECT MATCH. mdsem_3 contains the exact answer:
- "The Topology Manager can deal with the alignment of resources in a couple of distinct scopes: `container` (default), `pod`"
- "setting the `topologyManagerScope` in the kubelet configuration file"

mdsem_2 also states: "The `scope` defines the granularity at which you would like resource alignment to be performed"

The vocabulary mismatch ("granularity of resource alignment" vs "scope") was successfully resolved.

**Retrieved Chunk Excerpts**:
> "The `scope` defines the granularity at which you would like resource alignment to be performed... setting the `topologyManagerScope` in the kubelet configuration file"

---

#### adv_m05: How do I optimize inter-process communication latency for pods?
**Expected Answer**: Use pod scope with single-numa-node policy to eliminate inter-NUMA overhead

**Score**: 2/10

**Reasoning**: COMPLETE FAILURE. `needle_found: false` - NO chunks from the needle document were retrieved. All 5 chunks are from unrelated documents:
- concepts_security_multi-tenancy (QoS tiers, network bandwidth)
- concepts_scheduling-eviction_kube-scheduler (scheduler overview)
- tasks_debug_debug-cluster_topology (troubleshooting intro)
- tutorials_kubernetes-basics_explore_explore-intro (pods overview)

The answer exists in mdsem_4: "Using the `pod` scope in tandem with `single-numa-node` Topology Manager policy is specifically valuable for workloads that are latency sensitive or for high-throughput applications that perform IPC. By combining both options, you are able to place all containers in a pod onto a single NUMA node; hence, the inter-NUMA communication overhead can be eliminated for that pod."

**Failure Analysis**: VOCABULARY_MISMATCH - The query "inter-process communication latency" (IPC) didn't semantically match the document's phrasing "applications that perform IPC" and "inter-NUMA communication overhead". The semantic search favored documents about general QoS and latency concepts instead of the specific NUMA/IPC optimization.

**Retrieved Chunk Excerpts**:
> None from needle document. Retrieved content about QoS tiers, scheduler overview, and basic pod concepts - completely unrelated.

---

## Score Distribution

| Score | Count | Questions |
|-------|-------|-----------|
| 10/10 | 11 | adv_v04, adv_v05, adv_c01, adv_c02, adv_c03, adv_c05, adv_n01, adv_n02, adv_n05, adv_m01, adv_m03, adv_m04 |
| 9/10 | 1 | adv_c04 |
| 6/10 | 1 | adv_n03 |
| 4/10 | 1 | adv_m02 |
| 3/10 | 2 | adv_v01, adv_v02 |
| 2/10 | 4 | adv_v03, adv_n04, adv_m05 |

## Failure Analysis Summary

| Failure Type | Count | Questions | Description |
|--------------|-------|-----------|-------------|
| EMBEDDING_BLIND | 2 | adv_v01, adv_v02 | Semantic embeddings don't capture frontmatter metadata (version numbers in YAML) |
| VOCABULARY_MISMATCH | 4 | adv_v03, adv_n04, adv_m02, adv_m05 | Query phrasing didn't match document phrasing despite semantic similarity |
| CHUNKING_ISSUE | 1 | adv_n03 | Relevant chunk exists but wasn't ranked high enough |

## Key Observations

### Strengths
1. **COMPARISON questions excel (100% pass rate, 9.8 avg)**: Semantic retrieval is excellent at finding and comparing related concepts across chunks
2. **Policy explanations are well-retrieved**: Questions about policy behaviors consistently find relevant chunks
3. **Vocabulary mismatches often resolved**: "CPU placement policy" -> "topology manager policy", "resource co-location" -> "topology aligned resource allocation"

### Weaknesses
1. **VERSION questions struggle (40% pass rate, 5.6 avg)**: Frontmatter metadata (YAML version numbers, feature-state shortcodes) is not captured by semantic embeddings
2. **Negation framing can fail**: "What's wrong with X" queries sometimes miss content that explains "Y is better for this use case"
3. **Specialized vocabulary can fail**: "IPC latency" didn't match "applications that perform IPC" despite semantic similarity

### Comparison with Baseline Test
| Metric | Baseline (20 Q) | Adversarial (20 Q) | Delta |
|--------|-----------------|-------------------|-------|
| Average Score | 8.45/10 | 7.55/10 | -0.90 |
| Pass Rate | 90% (18/20) | 65% (13/20) | -25% |
| Perfect Scores | 12 | 11 | -1 |
| Complete Failures | 1 | 4 | +3 |

### Adversarial Effectiveness by Category
- **VERSION**: Successfully adversarial (40% pass rate) - frontmatter metadata is a genuine blind spot
- **COMPARISON**: NOT adversarial (100% pass rate) - semantic retrieval handles comparisons well
- **NEGATION**: Moderately adversarial (60% pass rate) - negation framing causes some failures
- **VOCABULARY**: Moderately adversarial (60% pass rate) - synonym matching is inconsistent

## Conclusion

The adversarial benchmark reveals **genuine weaknesses** in the `enriched_hybrid_llm` retrieval strategy:

1. **Frontmatter metadata is a blind spot**: VERSION questions targeting YAML frontmatter (min-kubernetes-server-version, feature-state) consistently fail. This is a fundamental limitation of semantic embeddings that don't process document metadata.

2. **Vocabulary mismatches are inconsistent**: Some synonym mappings work ("CPU placement" -> "topology manager") while others fail ("IPC latency" -> "inter-NUMA communication"). The LLM query rewriting helps but doesn't fully resolve this.

3. **Negation framing can mislead retrieval**: Questions asking "what's wrong with X" may not match content that explains "Y is better" without explicitly criticizing X.

**Recommendations**:
1. **Index frontmatter metadata separately**: Create dedicated chunks for version requirements and feature states
2. **Expand query rewriting**: Include more synonym variations for technical terms
3. **Consider negative query expansion**: For "what's wrong with X" queries, also search for "Y is better than X"

**Verdict**: The adversarial benchmark successfully identified retrieval weaknesses that the baseline test missed. The 65% pass rate (vs 90% baseline) confirms these questions are appropriately challenging while still being answerable with the correct retrieval.
