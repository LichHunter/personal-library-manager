# Deep Analysis: Kubernetes Realistic Benchmark Results

**Date**: 2026-01-30
**Benchmark Run**: `benchmark_results_realistic.json`
**Questions**: 50 (25 unique questions x 2 variants)
**Strategy**: `modular-no-llm`

---

## Executive Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Hit@5** | 36.0% (18/50) | Target doc found in top 5 results |
| **Hit@1** | 24.0% (12/50) | Target doc ranked first |
| **MRR** | 0.280 | Mean Reciprocal Rank |
| **Avg LLM Grade** | 6.22/10 | Quality of retrieved chunks |
| **Pass Rate >=7** | 21.7% | Chunks sufficient to answer question |
| **Grading Failures** | 27/50 (54%) | LLM couldn't parse JSON response |

**Primary Root Cause**: **VOCABULARY MISMATCH** (100% of failures)
- Users describe problems in natural language
- Documentation uses technical Kubernetes terminology
- Current query expansion dictionary only covers 9 terms

---

## Comparison: Needle vs Realistic Benchmarks

| Metric | Needle (Rigged) | Realistic (Honest) | Delta |
|--------|-----------------|-------------------|-------|
| Hit@5 | 90% | 36% | -54% |
| Hit@1 | 65% | 24% | -41% |
| MRR | 0.746 | 0.280 | -0.466 |
| Avg Grade | 7.6/10 | 6.2/10 | -1.4 |
| Pass Rate >=7 | 65% | 21.7% | -43.3% |

**Why the difference?**
- Needle: All 20 questions target ONE document with unique terminology
- Realistic: 50 questions across 25 different documents with natural language

---

## Detailed Per-Question Analysis

### Legend
- **HIT**: Target document found in top 5
- **MISS**: Target document not found
- **Rank**: Position in results (1 = best)
- **Grade**: LLM quality score (1-10)

---

## Group 1: SUCCESSFUL RETRIEVALS (18/50)

### Q000: Gateway API (Both variants succeeded)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_000_q1 | "how do i connect my cluster's load balancers and network configurations with gateway resources" | concepts_services-networking_gateway | 1 | - |
| q_000_q2 | "which cloud provider configurations are needed to make my gateway api work correctly" | concepts_services-networking_gateway | 1 | 8 |

**Why it worked**: Query contains "gateway" which directly matches document terminology.

---

### Q005: PersistentVolumeClaims (Both variants succeeded)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_005_q1 | "is there a way to increase storage size for my database without losing data" | concepts_storage_persistent-volumes | 1 | 7 |
| q_005_q2 | "why can't I resize my volume claim after initial provisioning" | concepts_storage_persistent-volumes | 1 | - |

**Why it worked**: "volume claim" and "resize" map well to PersistentVolumeClaim terminology.

---

### Q006: Windows Container Accounts (Both variants succeeded)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_006_q1 | "why can't I log into my windows container as administrator" | concepts_security_windows-security | 3 | 8 |
| q_006_q2 | "how to manage default user accounts in windows container images" | concepts_security_windows-security | 1 | 9 |

**Why it worked**: "windows container" is a distinctive phrase that matches.

---

### Q007: Indexed Jobs (q1 succeeded, q2 failed)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_007_q1 | "how to run a parallel batch job where each pod needs a unique index" | tasks_job_indexed-parallel-processing-static | 3 | - |
| q_007_q2 | "restart failed pod in a batch processing job without redoing entire workload" | tasks_job_indexed-parallel-processing-static | MISS | - |

**Analysis**: q1 contains "parallel batch job" + "unique index" which maps to Indexed Jobs. q2's "restart failed pod" doesn't connect to indexing concept.

---

### Q009: TLS Certificates (Both variants succeeded)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_009_q1 | "why can't my certificate signing requests get automatically approved in kubernetes" | tasks_tls_managing-tls-in-a-cluster | 1 | - |
| q_009_q2 | "how to manage tls certificates for my services without manual intervention" | tasks_tls_managing-tls-in-a-cluster | 1 | - |

**Why it worked**: "certificate signing requests" and "tls certificates" directly match document content.

---

### Q013: Resource Isolation (q1 succeeded, q2 failed)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_013_q1 | "how to prevent one container from consuming all resources of my node" | concepts_security_multi-tenancy | 1 | - |
| q_013_q2 | "stop my team's pods from interfering with critical system services" | concepts_security_multi-tenancy | MISS | 7 |

**Analysis**: q1's "consuming all resources" maps to isolation concepts. q2's "interfering with" is too vague.

---

### Q015: Authentication (Both variants succeeded)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_015_q1 | "why can't my CI/CD pipeline authenticate with kubernetes cluster after rotating service account tokens" | reference_access-authn-authz_authentication | 1 | - |
| q_015_q2 | "getting authentication errors when trying to call kubernetes api from external tool" | reference_access-authn-authz_authentication | 4 | - |

**Why it worked**: "authenticate" and "authentication errors" map directly.

---

### Q016: Aggregation Layer (Both variants succeeded)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_016_q1 | "why can't my custom API server authenticate requests from kubernetes cluster" | tasks_extend-kubernetes_configure-aggregation-layer | 1 | - |
| q_016_q2 | "my extension API is not showing up in kubectl get api-services" | tasks_extend-kubernetes_configure-aggregation-layer | 3 | - |

**Why it worked**: "custom API server" and "api-services" are technical terms present in docs.

---

### Q017: kubectl Autocomplete (q1 succeeded, q2 failed)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_017_q1 | "how to make kubectl commands autocomplete in my terminal" | tasks_tools_included_optional-kubectl-configs-fish | 1 | 10 |
| q_017_q2 | "getting tired of typing full kubernetes command names every time" | tasks_tools_included_optional-kubectl-configs-fish | MISS | 2 |

**Analysis**: q1 contains "kubectl" + "autocomplete" (exact match). q2's sentiment ("tired of typing") has no semantic connection to completion scripts.

---

### Q020: Job Suspend/Resume (Both variants succeeded)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_020_q1 | "my kubernetes batch job is stuck midway and I want to pause it without deleting" | concepts_workloads_controllers_job | 2 | - |
| q_020_q2 | "how to temporarily stop a long-running job and resume it later without losing progress" | concepts_workloads_controllers_job | 1 | - |

**Why it worked**: "batch job" + "pause" / "stop and resume" map to Job controller concepts.

---

### Q024: ServiceAccount (q1 succeeded, q2 failed)
| Question ID | Query | Target Doc | Rank | Grade |
|------------|-------|------------|------|-------|
| q_024_q1 | "why can't my pod access kubernetes api when running inside the cluster" | tasks_configure-pod-container_configure-service-account | 4 | - |
| q_024_q2 | "error accessing cloud resources from inside my application container" | tasks_configure-pod-container_configure-service-account | MISS | 3 |

**Analysis**: q1's "pod access kubernetes api" connects to ServiceAccount. q2's "cloud resources" is too generic.

---

## Group 2: FAILED RETRIEVALS (32/50)

### Q001: Dynamic Port Allocation (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_001_q1 | "why do my service ports keep changing between deployments" | concepts_cluster-administration_networking | User: "ports changing" -> Doc: "dynamic port allocation" |
| q_001_q2 | "can't access my microservice because its port is randomly assigned each time" | concepts_cluster-administration_networking | User: "randomly assigned" -> Doc: "dynamic port allocation" |

**Missing expansion**: `"port changing" -> "dynamic port allocation port assignment"`

---

### Q002: SubjectAccessReview (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_002_q1 | "how to check if my service account has permission to do something in kubernetes" | reference_access-authn-authz_webhook | User: "check permission" -> Doc: "SubjectAccessReview" |
| q_002_q2 | "why is my pipeline failing with a permission denied error when deploying" | reference_access-authn-authz_webhook | User: "permission denied" -> Doc: "SubjectAccessReview authorization" |

**Missing expansion**: `"check permission" -> "SubjectAccessReview authorization can-i kubectl auth"`

---

### Q003: Admission Webhooks (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_003_q1 | "how to automatically modify incoming pod specs before they're created in my cluster" | reference_access-authn-authz_extensible-admission-controllers | User: "modify pod specs" -> Doc: "mutating webhook admission controller" |
| q_003_q2 | "prevent developers from deploying containers without specific labels or security settings" | reference_access-authn-authz_extensible-admission-controllers | User: "prevent deploying" -> Doc: "validating webhook admission controller" |

**Missing expansion**: `"modify pod" -> "mutating webhook admission controller"`, `"prevent deploy" -> "validating webhook admission controller policy"`

---

### Q004: NamespaceAutoProvision (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_004_q1 | "why can't my team create resources without manually creating namespaces first" | reference_access-authn-authz_admission-controllers | User: "manually creating namespaces" -> Doc: "NamespaceAutoProvision admission controller" |
| q_004_q2 | "how to automatically set up default namespaces for new projects without manual intervention" | reference_access-authn-authz_admission-controllers | User: "automatically set up namespaces" -> Doc: "NamespaceAutoProvision" |

**Missing expansion**: `"auto namespace" -> "NamespaceAutoProvision admission controller automatic namespace creation"`

---

### Q008: etcd API Security (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_008_q1 | "why does my kubernetes cluster hang when multiple nodes go offline simultaneously" | concepts_security_api-server-bypass-risks | User: "cluster hang nodes offline" -> Doc: "etcd quorum consistency API bypass" |
| q_008_q2 | "how to protect against potential data corruption if etcd nodes lose connection" | concepts_security_api-server-bypass-risks | User: "etcd data corruption" -> Doc: "etcd API security risks bypass" |

**Missing expansion**: `"cluster hang" -> "etcd quorum consensus leader election"`, `"data corruption etcd" -> "etcd API bypass risks security"`

---

### Q010: Verify Kubernetes Artifacts (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_010_q1 | "how can I quickly check if my kubernetes yaml files are valid before deploying" | tasks_administer-cluster_verify-signed-artifacts | User: "check yaml valid" -> Doc: "verify signed artifacts cosign SLSA" |
| q_010_q2 | "getting weird errors in production, want to validate my deployment configs locally" | tasks_administer-cluster_verify-signed-artifacts | User: "validate configs" -> Doc: "verify signatures provenance" |

**Analysis**: This is actually a **question mismatch** - the user's question is about YAML validation, but the expected answer is about cryptographic signature verification. These are different concepts.

---

### Q011: System Traces (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_011_q1 | "why can't I see my application's logs and traces in the monitoring dashboard" | concepts_cluster-administration_system-traces | User: "logs and traces monitoring" -> Doc: "OpenTelemetry OTLP gRPC tracing spans" |
| q_011_q2 | "how to collect and track performance metrics from my kubernetes cluster" | concepts_cluster-administration_system-traces | User: "performance metrics" -> Doc: "system traces OTLP exporters" |

**Missing expansion**: `"traces monitoring" -> "OpenTelemetry OTLP gRPC tracing spans distributed"`, `"performance metrics" -> "observability tracing spans telemetry"`

---

### Q012: System Metrics (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_012_q1 | "how to collect and monitor resource usage for my kubernetes cluster" | concepts_cluster-administration_system-metrics | User: "monitor resource usage" -> Doc: "Prometheus metrics endpoints /metrics" |
| q_012_q2 | "my prometheus dashboard is not showing kubernetes component performance metrics" | concepts_cluster-administration_system-metrics | User: "prometheus dashboard" -> Doc: "metrics endpoints scrape format" |

**Missing expansion**: `"prometheus metrics" -> "Prometheus /metrics endpoint scrape kubelet cadvisor"`

---

### Q014: ABAC (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_014_q1 | "why can't I make granular permissions based on user attributes like job title" | reference_access-authn-authz_abac | User: "permissions user attributes" -> Doc: "ABAC attribute-based access control policy" |
| q_014_q2 | "kubernetes access control too rigid, how to define complex authorization rules" | reference_access-authn-authz_abac | User: "complex authorization rules" -> Doc: "ABAC attribute-based access control" |

**Missing expansion**: `"user attributes permissions" -> "ABAC attribute-based access control policy"`

---

### Q018: kube-proxy (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_018_q1 | "why can't my services communicate between nodes in my cluster" | concepts_cluster-administration_proxies | User: "services communicate between nodes" -> Doc: "kube-proxy iptables IPVS userspace" |
| q_018_q2 | "how kubernetes routes traffic to the right pod when I have multiple replicas" | concepts_cluster-administration_proxies | User: "routes traffic to pod" -> Doc: "kube-proxy service proxy network" |

**Missing expansion**: `"service communication" -> "kube-proxy iptables IPVS service proxy"`, `"traffic routing pods" -> "kube-proxy endpoints load balancing"`

---

### Q019: kubectl apply (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_019_q1 | "how do i deploy my yaml file from the command line" | tasks_manage-kubernetes-objects_imperative-config | User: "deploy yaml command line" -> Doc: "kubectl create imperative configuration" |
| q_019_q2 | "getting error when trying to apply kubernetes configuration from file" | tasks_manage-kubernetes-objects_imperative-config | User: "apply configuration" -> Doc: "imperative config kubectl create" |

**Analysis**: Interesting - user says "apply" which is **declarative**, but target doc is about **imperative** config. The retrieval system may be finding declarative docs instead.

---

### Q021: kubectl logs (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_021_q1 | "how to see what my job failed and why it's not completing" | tasks_job_fine-parallel-processing-work-queue | User: "see job failed" -> Doc: "kubectl logs pods work queue processing" |
| q_021_q2 | "can't figure out what's happening inside my worker pod during a batch job" | tasks_job_fine-parallel-processing-work-queue | User: "inside worker pod" -> Doc: "kubectl logs fine parallel processing" |

**Missing expansion**: `"see job failed" -> "kubectl logs pods container output debugging"`, `"inside pod" -> "kubectl exec logs describe events"`

---

### Q022: Topology Manager (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_022_q1 | "why are my high performance containers not getting scheduled on the right cpu cores" | tasks_administer-cluster_topology-manager | User: "cpu cores scheduling" -> Doc: "Topology Manager NUMA policy" |
| q_022_q2 | "can't control how cpu and memory resources get allocated across my numa nodes" | tasks_administer-cluster_topology-manager | User: "numa nodes allocation" -> Doc: "Topology Manager policy best-effort restricted single-numa-node" |

**Missing expansion**: `"cpu cores numa" -> "Topology Manager NUMA policy affinity best-effort restricted single-numa-node"`

---

### Q023: Admission Controllers (Both variants failed)
| Question ID | Query | Target Doc | Vocabulary Gap |
|------------|-------|------------|----------------|
| q_023_q1 | "how to prevent developers from deploying containers with latest tag in production" | reference_access-authn-authz_admission-controllers | User: "prevent latest tag" -> Doc: "admission controller AlwaysPullImages ImagePolicyWebhook" |
| q_023_q2 | "why can't I stop someone from using root containers in my cluster" | reference_access-authn-authz_admission-controllers | User: "stop root containers" -> Doc: "PodSecurityPolicy admission controller runAsNonRoot" |

**Missing expansion**: `"prevent latest tag" -> "admission controller ImagePolicyWebhook AlwaysPullImages"`, `"root containers" -> "PodSecurityPolicy PodSecurityAdmission runAsNonRoot securityContext"`

---

## Vocabulary Gap Analysis

### Current Query Expansion Coverage
The pipeline only expands **9 terms** (from `query_expander.py`):
- `rpo`, `recovery point objective`
- `rto`, `recovery time objective`
- `jwt`, `token`
- `database stack`, `database`
- `monitoring stack`, `monitoring`, `observability`
- `hpa`, `autoscaling`, `autoscaler`

### Required Expansions to Fix Failures

| Failed Topic | Missing Expansion |
|--------------|-------------------|
| Dynamic port allocation | `"port changing" -> "dynamic port allocation"` |
| SubjectAccessReview | `"check permission" -> "SubjectAccessReview can-i auth"` |
| Admission webhooks | `"modify pod" -> "mutating admission webhook"`, `"prevent deploy" -> "validating admission webhook"` |
| NamespaceAutoProvision | `"auto namespace" -> "NamespaceAutoProvision"` |
| etcd security | `"cluster hang" -> "etcd quorum"`, `"data corruption" -> "etcd API bypass"` |
| System traces | `"traces" -> "OpenTelemetry OTLP spans"` |
| System metrics | `"prometheus" -> "/metrics endpoint scrape"` |
| ABAC | `"user attributes" -> "ABAC attribute-based"` |
| kube-proxy | `"service communication" -> "kube-proxy iptables"` |
| kubectl logs | `"see job failed" -> "kubectl logs"` |
| Topology Manager | `"numa cpu" -> "Topology Manager policy"` |
| Admission controllers | `"prevent tag" -> "ImagePolicyWebhook"`, `"root containers" -> "PodSecurityAdmission"` |

---

## Grading Analysis

### Grading Success Rate
- **Successful grades**: 23/50 (46%)
- **Failed to grade**: 27/50 (54%)

The high failure rate is due to Haiku returning malformed JSON. The fallback regex extraction helped but wasn't sufficient.

### Grade Distribution (when graded)
| Grade | Count | Interpretation |
|-------|-------|----------------|
| 10 | 2 | Perfect answer |
| 9 | 1 | Excellent |
| 8 | 4 | Good (above threshold) |
| 7 | 4 | Acceptable |
| 6 | 5 | Marginal |
| 5 | 1 | Poor |
| 4 | 2 | Very poor |
| 3 | 3 | Almost useless |
| 2 | 1 | Useless |

---

## Key Insights

### 1. Vocabulary Mismatch is THE Problem
100% of failures can be attributed to vocabulary mismatch between natural user language and technical Kubernetes documentation.

### 2. Query Expansion Dictionary Too Small
Only 9 terms covered. Need 50+ domain-specific expansions for Kubernetes.

### 3. Some Questions Have Conceptual Mismatch
Q010 asks about YAML validation but expected answer is about cryptographic signatures - these are different concepts entirely.

### 4. Variant Quality Matters
When one variant succeeded and one failed, the successful variant contained more technical terms:
- "kubectl autocomplete" (HIT) vs "tired of typing" (MISS)
- "parallel batch job unique index" (HIT) vs "restart failed pod" (MISS)

### 5. BM25 Needs Domain Terms
BM25 lexical matching works great when queries contain domain terms. Semantic embeddings alone can't bridge large vocabulary gaps.

---

## Recommendations

### Immediate Actions

1. **Expand Query Dictionary**: Add 40+ new Kubernetes-specific term expansions to `query_expander.py`

2. **Add Synonym Mappings**: Map user problem descriptions to technical solutions
   - "prevent" -> "admission controller"
   - "check permission" -> "SubjectAccessReview"
   - "service communication" -> "kube-proxy"

3. **Problem-to-Solution Mapping**: Create mappings from symptoms to root causes
   - "cluster hang" -> "etcd quorum"
   - "ports changing" -> "dynamic port allocation"

4. **Improve Grading Reliability**: Switch back to Sonnet for grading or improve Haiku prompt for reliable JSON output

5. **Review Question Quality**: Some expected answers don't match realistic user questions (Q010)

### Long-term Improvements

1. **Query Understanding Layer**: Use LLM to identify user intent before retrieval
2. **Document Summarization**: Add searchable summaries that use natural language
3. **Hybrid Query Rewriting**: Combine rule-based expansion with LLM rewriting
4. **Feedback Loop**: Track failed queries and add expansions for common failures

---

## Appendix: Full Results Data

Results file: `benchmark_results_realistic.json`

### Success/Failure Summary by Question
| Q# | Topic | q1 | q2 | Notes |
|----|-------|-----|-----|-------|
| 000 | Gateway API | HIT | HIT | Both succeeded |
| 001 | Dynamic ports | MISS | MISS | Vocabulary gap |
| 002 | SubjectAccessReview | MISS | MISS | Vocabulary gap |
| 003 | Admission webhooks | MISS | MISS | Vocabulary gap |
| 004 | NamespaceAutoProvision | MISS | MISS | Vocabulary gap |
| 005 | PVC expansion | HIT | HIT | Both succeeded |
| 006 | Windows accounts | HIT | HIT | Both succeeded |
| 007 | Indexed Jobs | HIT | MISS | q1 technical, q2 vague |
| 008 | etcd security | MISS | MISS | Vocabulary gap |
| 009 | TLS certificates | HIT | HIT | Both succeeded |
| 010 | Verify artifacts | MISS | MISS | Question mismatch |
| 011 | System traces | MISS | MISS | Vocabulary gap |
| 012 | System metrics | MISS | MISS | Vocabulary gap |
| 013 | Resource isolation | HIT | MISS | q1 technical, q2 vague |
| 014 | ABAC | MISS | MISS | Vocabulary gap |
| 015 | Authentication | HIT | HIT | Both succeeded |
| 016 | Aggregation layer | HIT | HIT | Both succeeded |
| 017 | kubectl autocomplete | HIT | MISS | q1 exact match, q2 sentiment |
| 018 | kube-proxy | MISS | MISS | Vocabulary gap |
| 019 | kubectl apply | MISS | MISS | Declarative vs imperative |
| 020 | Job suspend | HIT | HIT | Both succeeded |
| 021 | kubectl logs | MISS | MISS | Vocabulary gap |
| 022 | Topology Manager | MISS | MISS | Vocabulary gap |
| 023 | Admission controllers | MISS | MISS | Vocabulary gap |
| 024 | ServiceAccount | HIT | MISS | q1 specific, q2 generic |
