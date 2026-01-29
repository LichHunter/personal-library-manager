# Realistic Questions Benchmark: Failure Analysis

**Date**: 2026-01-27
**Benchmark**: `results/realistic_2026-01-27_110955/`
**Hit@5**: 40.75% (163/400 queries successful)
**Failure Rate**: 59.25% (237 failures)

---

## Executive Summary

The realistic questions benchmark revealed that **vocabulary mismatch** is the dominant retrieval challenge, but this umbrella term masks distinct failure modes with different solutions. This document breaks down the 237 failures into actionable categories.

---

## Failure Taxonomy

### 1. CONCEPT-TERM MISMATCH (~40% of failures)

**Pattern**: User describes a *concept* or *problem*, but the correct doc uses a specific *technical term* they don't know.

| Question | Expected Doc | Retrieved | Vocabulary Gap |
|----------|--------------|-----------|----------------|
| "how to automatically modify incoming pod specs before they're created" | `extensible-admission-controllers` | `admission-webhooks-good-practices` | User doesn't know "mutating webhook" |
| "why can't my team create resources without manually creating namespaces first" | `admission-controllers` | `namespaces`, `manage-resources` | User doesn't know "NamespaceAutoProvision" |
| "prevent developers from deploying containers with latest tag" | `admission-controllers` | `containers_images` | User doesn't know "ImagePolicyWebhook" |
| "how to check if my service account has permission" | `webhook` (SubjectAccessReview) | `service-accounts`, `rbac` | User doesn't know "SubjectAccessReview" |
| "restart failed pod in batch job without redoing entire workload" | `indexed-parallel-processing-static` | `job`, `pod-failure-policy` | User doesn't know "Indexed Job" |
| "why do some of my pods have weird leader election behavior" | `leases` | `coordinated-leader-election` | User doesn't know "Lease object" |
| "how to keep my application running if an entire data center goes down" | `multiple-zones` | `disruptions`, `self-healing` | User doesn't know "multi-zone deployment" |

**Root Cause**: Documentation uses K8s-specific terminology; users describe problems in natural language.

**Potential Solutions**:
- Query expansion with domain-specific synonym dictionary
- Document enrichment with "problem pattern" metadata
- Fine-tuned embeddings on (user-query, doc-term) pairs

---

### 2. WRONG ABSTRACTION LEVEL (~25% of failures)

**Pattern**: Question targets a specific implementation detail, but retrieval returns the general concept doc (or vice versa).

| Question | Expected Doc | Retrieved | Abstraction Issue |
|----------|--------------|-----------|-------------------|
| "why do my service ports keep changing between deployments" | `cluster-administration_networking` (dynamic port allocation) | `services-networking_service` (general Services) | Too general |
| "how to see what my job failed and why it's not completing" | `fine-parallel-processing-work-queue` (kubectl logs example) | `concepts_workloads_controllers_job` (Job concepts) | Too general |
| "why are my pods stuck in pending state" | `kube-scheduler` | `assign-pod-node`, `node-autoscaling` | Concept vs implementation |
| "how does kubernetes route traffic to the right pod" | `proxies` (kube-proxy) | `create-external-load-balancer`, `virtual-ips` | Implementation vs concept |
| "how to configure external authentication for kubernetes api server" | `authentication` | `access-api-from-pod`, `mixed-version-proxy` | Task-level vs reference-level |
| "why are my pods constantly running out of cpu and getting evicted" | `cluster-large` (VPA mention) | `pid-limiting`, `pod-overhead` | Best practices vs specific features |

**Root Cause**: Same topic exists at multiple abstraction levels in K8s docs (concepts → tasks → reference). Query doesn't signal which level is needed.

**Potential Solutions**:
- Hierarchical chunking that preserves concept→task→reference relationships
- Multi-level retrieval (retrieve from multiple abstraction levels, let LLM choose)
- Query classification to predict desired abstraction level

---

### 3. SYMPTOM-SOLUTION MISMATCH (~15% of failures)

**Pattern**: User describes a *symptom/error*, but the solution is in a doc about a different *feature*.

| Question | Expected Doc | Retrieved | Mental Model Gap |
|----------|--------------|-----------|------------------|
| "why does my kubernetes cluster hang when multiple nodes go offline" | `api-server-bypass-risks` (etcd quorum) | `configure-upgrade-etcd`, `topology-spread-constraints` | Symptom: "hang" → Solution: etcd security |
| "why are my pods not getting marked unhealthy even when app stops responding" | `volume-health-monitoring` | `configure-liveness-readiness-startup-probes` | User wants app health, doc is volume health |
| "how can kubernetes automatically detect and replace failing containers" | `volume-health-monitoring` | `observability`, `monitor-node-health` | Same - app health vs storage health |
| "getting weird errors in production, want to validate deployment configs locally" | `verify-signed-artifacts` | `admission-webhooks-good-practices` | User wants YAML validation, doc is signature verification |
| "my security team is complaining about pod privilege escalation risks" | `pod-security-policy` | `rbac-good-practices`, `pod-security-standards` | Security concern → deprecated feature |
| "docker and kubernetes using different resource management settings causing pod failures" | `configure-cgroup-driver` | `node-autoscaling`, `scheduling-eviction` | Symptom: failures → Solution: cgroup mismatch |

**Root Cause**: User's mental model of the problem doesn't match K8s architecture boundaries. The causal chain from symptom to solution requires domain expertise.

**Potential Solutions**:
- Symptom→Solution knowledge graph
- Document enrichment with "common symptoms this doc solves"
- Chain-of-thought retrieval (retrieve symptoms first, then solutions)

---

### 4. OVERLAPPING TOPICS (~10% of failures)

**Pattern**: Multiple docs cover similar ground; retrieval picks a "close but wrong" match.

| Question | Expected Doc | Retrieved | Overlap Issue |
|----------|--------------|-----------|---------------|
| "how to prevent containers from running as root" | `pod-security-policy` | `linux-kernel-security-constraints`, `rbac-good-practices` | Multiple security docs |
| "prevent one team's greedy containers from consuming resources" | `memory-default-namespace` | `multi-tenancy`, `quality-service-pod` | Resource limits vs quotas vs tenancy |
| "how to give my frontend team read-only access to only their pods" | `rbac` | `service-accounts`, `rbac` glossary | RBAC info split across docs |
| "kubernetes access control too rigid, how to define complex authorization rules" | `abac` | `rbac`, `service-accounts` | RBAC more popular than ABAC |
| "why can't my teammate access the production namespace" | `rbac` | `service-accounts`, `rbac` glossary, `multi-tenancy` | Access control spread across docs |
| "how do I make sure my sensitive data isn't stored as plain text" | `encrypt-data` | `secrets-good-practices`, `kms-provider` | Encryption info in multiple places |

**Root Cause**: K8s documentation has intentional overlap (concepts, tasks, reference for same feature). Ground truth targets one specific doc, but query legitimately matches multiple.

**Potential Solutions**:
- Document deduplication or consolidation during indexing
- Multi-document retrieval with re-ranking
- Accept multiple valid answers in evaluation (relaxed ground truth)

---

### 5. GROUND TRUTH QUESTIONABLE (~10% of failures)

**Pattern**: The expected document seems like a poor ground truth for the realistic question generated.

| Question | Expected Doc | Retrieved | Issue |
|----------|--------------|-----------|-------|
| "how can I quickly check if my kubernetes yaml files are valid" | `verify-signed-artifacts` | `update-daemon-set`, `manifest` glossary | User wants YAML validation, not signature verification |
| "what's the fastest way to clean up my test environment after running a demo" | `extended-resource` | `use-cascading-deletion`, `finalizers` | Cleanup is tangential to extended-resource doc |
| "how to remove all kubernetes resources i created" | `extended-resource` | `workloads_management`, `namespaces` | Same - cleanup is a side note |
| "how do my microservices communicate and discover each other" | `environment-variable-expose-pod-information` | `services-networking`, `service` | Services is the real answer |
| "pull specific branch or commit from github into my kubernetes pod" | `volumes` (gitRepo) | `contribute_generate-ref-docs`, `open-a-pr` | gitRepo is deprecated, docs contribution matched |

**Root Cause**: The original kubefix question targeted a specific detail in the doc; our LLM transformation made it too general, breaking the ground truth mapping.

**Potential Solutions**:
- Filter/fix dataset: remove questions where transformation broke ground truth
- Use human validation on a subset
- Accept these as "not retrieval failures" in evaluation

---

## Summary Statistics

| Category | Est. % of Failures | Difficulty to Fix | Approach |
|----------|-------------------|-------------------|----------|
| Concept-Term Mismatch | ~40% | Medium | Query expansion, doc enrichment |
| Wrong Abstraction Level | ~25% | Hard | Hierarchical retrieval |
| Symptom-Solution Mismatch | ~15% | Very Hard | Knowledge graphs, chain-of-thought |
| Overlapping Topics | ~10% | Medium | Multi-doc retrieval, relaxed eval |
| Ground Truth Issues | ~10% | Easy | Dataset cleanup |

---

## Concrete Examples by Category

### Concept-Term Mismatch Examples

```
Q: "how to automatically modify incoming pod specs before they're created in my cluster"
Expected: reference_access-authn-authz_extensible-admission-controllers
Retrieved: concepts_cluster-administration_admission-webhooks-good-practices

Gap: User doesn't know the term "mutating admission webhook"
The doc title literally says "extensible-admission-controllers" but user says "modify pod specs"
```

```
Q: "why can't my team create resources without manually creating namespaces first"  
Expected: reference_access-authn-authz_admission-controllers
Retrieved: tasks_administer-cluster_namespaces

Gap: User doesn't know "NamespaceAutoProvision admission controller"
Retrieved docs are about namespaces in general, not the auto-provisioning feature
```

```
Q: "restart failed pod in a batch processing job without redoing entire workload"
Expected: tasks_job_indexed-parallel-processing-static
Retrieved: concepts_workloads_controllers_job, tasks_job_pod-failure-policy

Gap: User doesn't know "Indexed Job" - the specific K8s feature for this use case
```

### Wrong Abstraction Level Examples

```
Q: "why do my service ports keep changing between deployments"
Expected: concepts_cluster-administration_networking (dynamic port allocation)
Retrieved: concepts_services-networking_service (general Services concept)

Issue: Both docs discuss ports, but at different levels of specificity
```

```
Q: "how to see what my job failed and why it's not completing"
Expected: tasks_job_fine-parallel-processing-work-queue (has kubectl logs example)
Retrieved: concepts_workloads_controllers_job (general Job concept)

Issue: User wants debugging steps (task-level), got concept overview
```

### Symptom-Solution Mismatch Examples

```
Q: "why does my kubernetes cluster hang when multiple nodes go offline simultaneously"
Expected: concepts_security_api-server-bypass-risks (etcd quorum risks)
Retrieved: tasks_administer-cluster_configure-upgrade-etcd, topology-spread-constraints

Issue: Symptom is "cluster hang", but cause is etcd quorum loss - not obvious connection
```

```
Q: "why are my pods not getting marked unhealthy even when my app stops responding"
Expected: concepts_storage_volume-health-monitoring
Retrieved: tasks_configure-pod-container_configure-liveness-readiness-startup-probes

Issue: User thinks "app health" but expected doc is about "volume health" - confusing ground truth
```

---

## Research Directions

### High-Impact (Address ~40% of failures)

1. **Domain-Specific Query Expansion**
   - Build K8s terminology dictionary mapping user language → technical terms
   - Example mappings:
     - "modify pod specs before creation" → "mutating admission webhook"
     - "check permissions" → "SubjectAccessReview"
     - "auto-create namespaces" → "NamespaceAutoProvision"
   - Could be rule-based or learned from query logs

2. **Document Enrichment with Problem Patterns**
   - Add "common user questions" or "when to use this" metadata to chunks
   - Extract from: Stack Overflow, GitHub issues, Reddit r/kubernetes
   - Embed these patterns alongside document content

### Medium-Impact (Address ~25% of failures)

3. **Hierarchical/Multi-Level Retrieval**
   - Index documents at multiple granularities
   - Retrieve from concept, task, AND reference levels
   - Let LLM or re-ranker select appropriate level

4. **Query Intent Classification**
   - Classify queries as: conceptual, how-to, troubleshooting, reference
   - Route to appropriate document type

### Lower-Impact but Valuable

5. **Relaxed Evaluation**
   - Accept multiple valid documents as correct
   - Build equivalence classes of documents covering same topic

6. **Dataset Quality Improvement**
   - Human review of ground truth mappings
   - Filter questions where transformation broke the mapping

---

## Appendix: Sample Failure Pairs

### Failures Where Retrieved Doc Was Actually Reasonable

Sometimes the "failure" is debatable - retrieved docs are relevant, just not the exact expected doc:

| Question | Expected | Retrieved | Assessment |
|----------|----------|-----------|------------|
| "how to give a service account permission to list and create deployments" | `rbac` | `rbac` (hit!), `rbac` glossary, `service-accounts` | Actually passed |
| "why can't my pod access cloud storage" | `service-accounts` | `securing-a-cluster`, `service-accounts` (rank 2) | Close miss |
| "how to inject database password into container without putting secrets in git" | `distribute-credentials-secure` | `secrets-good-practices` | Reasonable alternative |

### Clear Failures (Retrieved Completely Wrong)

| Question | Expected | Retrieved | Assessment |
|----------|----------|-----------|------------|
| "pull specific branch or commit from github into my kubernetes pod" | `volumes` (gitRepo) | `contribute_generate-ref-docs`, `open-a-pr` | Matched "github" to docs contribution |
| "how to keep my application running if an entire data center goes down" | `multiple-zones` | `create-cluster-kubeadm`, `disruptions` | Missed "zones" concept entirely |
| "docker and kubernetes using different resource management settings" | `configure-cgroup-driver` | `node-autoscaling`, `scheduling-eviction` | Missed "cgroup" entirely |
