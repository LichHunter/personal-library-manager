# Realistic Questions Benchmark Report

Generated: 2026-01-27 11:19:00

Results folder: `results/realistic_2026-01-27_110955`

---

## Summary

- **Corpus size**: 1569 documents
- **Chunk count**: 7269 chunks
- **Total queries**: 400
- **Strategy**: enriched_hybrid_llm
- **Chunking**: MarkdownSemanticStrategy
- **Hit@1**: 20.2%
- **Hit@5**: 40.8%
- **MRR**: 0.274

---

## Q1 vs Q2 Performance

- **Q1 (realistic variant 1) Hit@5**: 41.5%
- **Q2 (realistic variant 2) Hit@5**: 40.0%

---

## Failure Analysis

| Category | Count | % |
|----------|-------|---|
| VOCABULARY_MISMATCH | 237 | 100.0% |
| RANKING_ERROR | 0 | 0.0% |
| CHUNKING_ISSUE | 0 | 0.0% |
| EMBEDDING_BLIND | 0 | 0.0% |

---

## Worst Failures

### 1. Question: "how do i connect my cluster's load balancers and network configurations with gateway resources"

- **Expected**: `concepts_services-networking_gateway`
- **Retrieved**: `reference_labels-annotations-taints__index`, `reference_labels-annotations-taints__index`, `reference_labels-annotations-taints__index`
- **Rank**: Not found

### 2. Question: "why do my service ports keep changing between deployments"

- **Expected**: `concepts_cluster-administration_networking`
- **Retrieved**: `concepts_services-networking_service`, `concepts_overview__index`, `tutorials_kubernetes-basics__index`
- **Rank**: Not found

### 3. Question: "can't access my microservice because its port is randomly assigned each time"

- **Expected**: `concepts_cluster-administration_networking`
- **Retrieved**: `tasks_access-application-cluster_connecting-frontend-backend`, `concepts_services-networking_service`, `concepts_services-networking_service`
- **Rank**: Not found

### 4. Question: "how to check if my service account has permission to do something in kubernetes"

- **Expected**: `reference_access-authn-authz_webhook`
- **Retrieved**: `concepts_security_service-accounts`, `reference_glossary_rbac`, `concepts_security_rbac-good-practices`
- **Rank**: Not found

### 5. Question: "why is my pipeline failing with a permission denied error when deploying"

- **Expected**: `reference_access-authn-authz_webhook`
- **Retrieved**: `reference_access-authn-authz_authorization`, `concepts_security_controlling-access`, `reference_access-authn-authz_authorization`
- **Rank**: Not found

### 6. Question: "how to automatically modify incoming pod specs before they're created in my cluster"

- **Expected**: `reference_access-authn-authz_extensible-admission-controllers`
- **Retrieved**: `concepts_cluster-administration_admission-webhooks-good-practices`, `concepts_cluster-administration_admission-webhooks-good-practices`, `concepts_cluster-administration_admission-webhooks-good-practices`
- **Rank**: Not found

### 7. Question: "prevent developers from deploying containers without specific labels or security settings"

- **Expected**: `reference_access-authn-authz_extensible-admission-controllers`
- **Retrieved**: `concepts_security_pod-security-admission`, `reference_labels-annotations-taints__index`, `concepts_security_cloud-native-security`
- **Rank**: Not found

### 8. Question: "why can't my team create resources without manually creating namespaces first"

- **Expected**: `reference_access-authn-authz_admission-controllers`
- **Retrieved**: `tasks_administer-cluster_namespaces`, `tasks_administer-cluster_manage-resources_memory-default-namespace`, `concepts_overview_working-with-objects_namespaces`
- **Rank**: Not found

### 9. Question: "how to automatically set up default namespaces for new projects without manual intervention"

- **Expected**: `reference_access-authn-authz_admission-controllers`
- **Retrieved**: `tutorials_cluster-management_namespaces-walkthrough`, `tasks_administer-cluster_namespaces`, `tasks_administer-cluster_manage-resources_cpu-default-namespace`
- **Rank**: Not found

### 10. Question: "restart failed pod in a batch processing job without redoing entire workload"

- **Expected**: `tasks_job_indexed-parallel-processing-static`
- **Retrieved**: `concepts_workloads_controllers_job`, `tasks_job_pod-failure-policy`, `tasks_job_pod-failure-policy`
- **Rank**: Not found

