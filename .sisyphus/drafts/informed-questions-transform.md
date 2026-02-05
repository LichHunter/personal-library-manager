# Informed User Questions Transform

**Purpose**: Transform realistic questions (natural language, problem-oriented) into informed questions (technical terminology, solution-oriented) for benchmark comparison.

**Target File**: `poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json`

---

## Transformation Principle

| Realistic User | Informed User |
|----------------|---------------|
| Describes symptoms/problems | Uses Kubernetes terminology |
| "why can't I..." | "how does X work" |
| Natural language | Technical vocabulary |
| Problem-oriented | Solution-oriented |

---

## JSON Structure

```json
{
  "metadata": {
    "source": "kubefix",
    "model": "transformed from realistic_questions.json",
    "prompt_version": "informed_user_v1",
    "description": "Questions transformed to use proper Kubernetes terminology - simulating users who know what they're looking for",
    "total": 25,
    "variants_per_question": 2
  },
  "questions": [
    // ... questions array below
  ]
}
```

---

## Transformed Questions (25 questions x 2 variants = 50 total)

### Q000: Gateway API
- **Doc ID**: `concepts_services-networking_gateway`
- **Original**: "What is the role of the Infrastructure Provider in the design principles of Gateway API?"
- **Realistic q1**: "how do i connect my cluster's load balancers and network configurations with gateway resources"
- **Realistic q2**: "which cloud provider configurations are needed to make my gateway api work correctly"
- **Informed q1**: "what is the Infrastructure Provider role in Gateway API design"
- **Informed q2**: "how does Gateway API handle infrastructure provider integration for load balancers"

### Q001: Dynamic Port Allocation
- **Doc ID**: `concepts_cluster-administration_networking`
- **Original**: "What is the Kubernetes approach to dynamic port allocation?"
- **Realistic q1**: "why do my service ports keep changing between deployments"
- **Realistic q2**: "can't access my microservice because its port is randomly assigned each time"
- **Informed q1**: "how does kubernetes handle dynamic port allocation for services"
- **Informed q2**: "what is the kubernetes networking approach to service port assignment"

### Q002: SubjectAccessReview
- **Doc ID**: `reference_access-authn-authz_webhook`
- **Original**: "What does a SubjectAccessReview object describe?"
- **Realistic q1**: "how to check if my service account has permission to do something in kubernetes"
- **Realistic q2**: "why is my pipeline failing with a permission denied error when deploying"
- **Informed q1**: "how to use SubjectAccessReview to check permissions in kubernetes"
- **Informed q2**: "what does SubjectAccessReview describe for authorization checks"

### Q003: Admission Webhooks
- **Doc ID**: `reference_access-authn-authz_extensible-admission-controllers`
- **Original**: "What is the difference between a mutating webhook and a validating webhook?"
- **Realistic q1**: "how to automatically modify incoming pod specs before they're created in my cluster"
- **Realistic q2**: "prevent developers from deploying containers without specific labels or security settings"
- **Informed q1**: "what is the difference between mutating and validating admission webhooks"
- **Informed q2**: "how do mutating webhooks modify pod specs before creation"

### Q004: NamespaceAutoProvision
- **Doc ID**: `reference_access-authn-authz_admission-controllers`
- **Original**: "What does the NamespaceAutoProvision admission controller do?"
- **Realistic q1**: "why can't my team create resources without manually creating namespaces first"
- **Realistic q2**: "how to automatically set up default namespaces for new projects without manual intervention"
- **Informed q1**: "what does NamespaceAutoProvision admission controller do"
- **Informed q2**: "how to enable automatic namespace creation with admission controllers"

### Q005: PersistentVolumeClaim Expansion
- **Doc ID**: `concepts_storage_persistent-volumes`
- **Original**: "What types of volumes can be expanded using PersistentVolumeClaims?"
- **Realistic q1**: "is there a way to increase storage size for my database without losing data"
- **Realistic q2**: "why can't I resize my volume claim after initial provisioning"
- **Informed q1**: "which volume types support PersistentVolumeClaim expansion"
- **Informed q2**: "how to expand storage using PersistentVolumeClaims in kubernetes"

### Q006: Windows Container Accounts
- **Doc ID**: `concepts_security_windows-security`
- **Original**: "What are the two default user accounts for Windows containers?"
- **Realistic q1**: "why can't I log into my windows container as administrator"
- **Realistic q2**: "how to manage default user accounts in windows container images"
- **Informed q1**: "what are the default user accounts for Windows containers in kubernetes"
- **Informed q2**: "ContainerUser vs ContainerAdministrator in windows containers"

### Q007: Indexed Jobs
- **Doc ID**: `tasks_job_indexed-parallel-processing-static`
- **Original**: "What is the purpose of using an Indexed Job in Kubernetes?"
- **Realistic q1**: "how to run a parallel batch job where each pod needs a unique index"
- **Realistic q2**: "restart failed pod in a batch processing job without redoing entire workload"
- **Informed q1**: "what is the purpose of Indexed Jobs in kubernetes"
- **Informed q2**: "how do Indexed Jobs assign completion indexes to pods"

### Q008: etcd API Security Risks
- **Doc ID**: `concepts_security_api-server-bypass-risks`
- **Original**: "What is the etcd API and what risks does it pose to the cluster's security?"
- **Realistic q1**: "why does my kubernetes cluster hang when multiple nodes go offline simultaneously"
- **Realistic q2**: "how to protect against potential data corruption if etcd nodes lose connection"
- **Informed q1**: "what are the security risks of direct etcd API access bypassing kube-apiserver"
- **Informed q2**: "how does etcd API bypass affect cluster security"

### Q009: Certificate Signers
- **Doc ID**: `tasks_tls_managing-tls-in-a-cluster`
- **Original**: "What is the role of a signer in Kubernetes?"
- **Realistic q1**: "why can't my certificate signing requests get automatically approved in kubernetes"
- **Realistic q2**: "how to manage tls certificates for my services without manual intervention"
- **Informed q1**: "what is the role of a certificate signer in kubernetes TLS management"
- **Informed q2**: "how do kubernetes certificate signers approve CSRs"

### Q010: Verify Signed Artifacts
- **Doc ID**: `tasks_administer-cluster_verify-signed-artifacts`
- **Original**: "What tools are required to verify Kubernetes artifacts?"
- **Realistic q1**: "how can I quickly check if my kubernetes yaml files are valid before deploying"
- **Realistic q2**: "getting weird errors in production, want to validate my deployment configs locally"
- **Informed q1**: "what tools are needed to verify signed kubernetes artifacts with cosign"
- **Informed q2**: "how to verify kubernetes release artifact signatures and provenance"

### Q011: System Traces
- **Doc ID**: `concepts_cluster-administration_system-traces`
- **Original**: "What protocol does Kubernetes components use to emit traces?"
- **Realistic q1**: "why can't I see my application's logs and traces in the monitoring dashboard"
- **Realistic q2**: "how to collect and track performance metrics from my kubernetes cluster"
- **Informed q1**: "what protocol does kubernetes use to emit traces OTLP or gRPC"
- **Informed q2**: "how to configure OpenTelemetry tracing for kubernetes components"

### Q012: System Metrics
- **Doc ID**: `concepts_cluster-administration_system-metrics`
- **Original**: "What is the format in which Kubernetes components emit metrics?"
- **Realistic q1**: "how to collect and monitor resource usage for my kubernetes cluster"
- **Realistic q2**: "my prometheus dashboard is not showing kubernetes component performance metrics"
- **Informed q1**: "what format do kubernetes components use to emit metrics prometheus or openmetrics"
- **Informed q2**: "how to scrape metrics from kubernetes component /metrics endpoints"

### Q013: Multi-tenancy Isolation
- **Doc ID**: `concepts_security_multi-tenancy`
- **Original**: "What are the different types of isolation in Kubernetes?"
- **Realistic q1**: "how to prevent one container from consuming all resources of my node"
- **Realistic q2**: "stop my team's pods from interfering with critical system services"
- **Informed q1**: "what types of isolation does kubernetes support for multi-tenancy"
- **Informed q2**: "how does kubernetes implement namespace and node isolation for multi-tenant clusters"

### Q014: ABAC Authorization
- **Doc ID**: `reference_access-authn-authz_abac`
- **Original**: "What is Attribute-based access control (ABAC) and how does it work?"
- **Realistic q1**: "why can't I make granular permissions based on user attributes like job title"
- **Realistic q2**: "kubernetes access control too rigid, how to define complex authorization rules"
- **Informed q1**: "how does ABAC attribute-based access control work in kubernetes"
- **Informed q2**: "how to configure ABAC policy file for kubernetes authorization"

### Q015: Webhook Token Authentication
- **Doc ID**: `reference_access-authn-authz_authentication`
- **Original**: "What is the role of the webhook token authenticator in the authentication process?"
- **Realistic q1**: "why can't my CI/CD pipeline authenticate with kubernetes cluster after rotating service account tokens"
- **Realistic q2**: "getting authentication errors when trying to call kubernetes api from external tool"
- **Informed q1**: "how does webhook token authenticator validate bearer tokens in kubernetes"
- **Informed q2**: "what is the role of webhook token authentication in kubernetes API server"

### Q016: API Aggregation Layer
- **Doc ID**: `tasks_extend-kubernetes_configure-aggregation-layer`
- **Original**: "What is the purpose of the authentication flow in the aggregation layer?"
- **Realistic q1**: "why can't my custom API server authenticate requests from kubernetes cluster"
- **Realistic q2**: "my extension API is not showing up in kubectl get api-services"
- **Informed q1**: "how does authentication work in kubernetes API aggregation layer"
- **Informed q2**: "what is the purpose of front-proxy authentication in aggregated API servers"

### Q017: kubectl Completion
- **Doc ID**: `tasks_tools_included_optional-kubectl-configs-fish`
- **Original**: "What does sourcing the completion script in your shell enable?"
- **Realistic q1**: "how to make kubectl commands autocomplete in my terminal"
- **Realistic q2**: "getting tired of typing full kubernetes command names every time"
- **Informed q1**: "how to enable kubectl shell completion for bash zsh or fish"
- **Informed q2**: "what does sourcing kubectl completion script enable in shell"

### Q018: kube-proxy
- **Doc ID**: `concepts_cluster-administration_proxies`
- **Original**: "What is kube proxy?"
- **Realistic q1**: "why can't my services communicate between nodes in my cluster"
- **Realistic q2**: "how kubernetes routes traffic to the right pod when I have multiple replicas"
- **Informed q1**: "what is kube-proxy and how does it implement service networking"
- **Informed q2**: "how does kube-proxy handle iptables or IPVS for service load balancing"

### Q019: Imperative Configuration
- **Doc ID**: `tasks_manage-kubernetes-objects_imperative-config`
- **Original**: "How can you use `kubectl` to create an object from a configuration file?"
- **Realistic q1**: "how do i deploy my yaml file from the command line"
- **Realistic q2**: "getting error when trying to apply kubernetes configuration from file"
- **Informed q1**: "how to use kubectl create with imperative object configuration"
- **Informed q2**: "what is the difference between kubectl create and kubectl apply for config files"

### Q020: Job Suspend/Resume
- **Doc ID**: `concepts_workloads_controllers_job`
- **Original**: "How can you suspend and resume a Job in Kubernetes?"
- **Realistic q1**: "my kubernetes batch job is stuck midway and I want to pause it without deleting"
- **Realistic q2**: "how to temporarily stop a long-running job and resume it later without losing progress"
- **Informed q1**: "how to suspend and resume a Job in kubernetes using spec.suspend"
- **Informed q2**: "what happens to pods when a kubernetes Job is suspended"

### Q021: kubectl logs for Jobs
- **Doc ID**: `tasks_job_fine-parallel-processing-work-queue`
- **Original**: "What is the purpose of the `kubectl logs pods/job-wq-2-7r7b2` command?"
- **Realistic q1**: "how to see what my job failed and why it's not completing"
- **Realistic q2**: "can't figure out what's happening inside my worker pod during a batch job"
- **Informed q1**: "how to use kubectl logs to debug Job pod failures"
- **Informed q2**: "how to view container logs from kubernetes Job worker pods"

### Q022: Topology Manager
- **Doc ID**: `tasks_administer-cluster_topology-manager`
- **Original**: "What are the four supported Topology Manager policies?"
- **Realistic q1**: "why are my high performance containers not getting scheduled on the right cpu cores"
- **Realistic q2**: "can't control how cpu and memory resources get allocated across my numa nodes"
- **Informed q1**: "what are the Topology Manager policies none best-effort restricted single-numa-node"
- **Informed q2**: "how does Topology Manager coordinate NUMA-aware resource allocation"

### Q023: Admission Controllers
- **Doc ID**: `reference_access-authn-authz_admission-controllers`
- **Original**: "Why do we need admission controllers?"
- **Realistic q1**: "how to prevent developers from deploying containers with latest tag in production"
- **Realistic q2**: "why can't I stop someone from using root containers in my cluster"
- **Informed q1**: "what admission controllers are recommended for production kubernetes clusters"
- **Informed q2**: "how do admission controllers enforce security policies on pod creation"

### Q024: ServiceAccount Configuration
- **Doc ID**: `tasks_configure-pod-container_configure-service-account`
- **Original**: "What is a ServiceAccount in Kubernetes?"
- **Realistic q1**: "why can't my pod access kubernetes api when running inside the cluster"
- **Realistic q2**: "error accessing cloud resources from inside my application container"
- **Informed q1**: "how to configure a ServiceAccount for pod API access in kubernetes"
- **Informed q2**: "what is the purpose of ServiceAccount tokens for pod authentication"

---

## JSON Output (Copy-Paste Ready)

```json
{
  "metadata": {
    "source": "kubefix",
    "model": "transformed from realistic_questions.json",
    "prompt_version": "informed_user_v1",
    "description": "Questions transformed to use proper Kubernetes terminology - simulating users who know what they're looking for",
    "total": 25,
    "variants_per_question": 2
  },
  "questions": [
    {
      "original_instruction": "What is the role of the Infrastructure Provider in the design principles of Gateway API?",
      "doc_id": "concepts_services-networking_gateway",
      "realistic_q1": "how do i connect my cluster's load balancers and network configurations with gateway resources",
      "realistic_q2": "which cloud provider configurations are needed to make my gateway api work correctly",
      "informed_q1": "what is the Infrastructure Provider role in Gateway API design",
      "informed_q2": "how does Gateway API handle infrastructure provider integration for load balancers"
    },
    {
      "original_instruction": "What is the Kubernetes approach to dynamic port allocation?",
      "doc_id": "concepts_cluster-administration_networking",
      "realistic_q1": "why do my service ports keep changing between deployments",
      "realistic_q2": "can't access my microservice because its port is randomly assigned each time",
      "informed_q1": "how does kubernetes handle dynamic port allocation for services",
      "informed_q2": "what is the kubernetes networking approach to service port assignment"
    },
    {
      "original_instruction": "What does a SubjectAccessReview object describe?",
      "doc_id": "reference_access-authn-authz_webhook",
      "realistic_q1": "how to check if my service account has permission to do something in kubernetes",
      "realistic_q2": "why is my pipeline failing with a permission denied error when deploying",
      "informed_q1": "how to use SubjectAccessReview to check permissions in kubernetes",
      "informed_q2": "what does SubjectAccessReview describe for authorization checks"
    },
    {
      "original_instruction": "What is the difference between a mutating webhook and a validating webhook?",
      "doc_id": "reference_access-authn-authz_extensible-admission-controllers",
      "realistic_q1": "how to automatically modify incoming pod specs before they're created in my cluster",
      "realistic_q2": "prevent developers from deploying containers without specific labels or security settings",
      "informed_q1": "what is the difference between mutating and validating admission webhooks",
      "informed_q2": "how do mutating webhooks modify pod specs before creation"
    },
    {
      "original_instruction": "What does the NamespaceAutoProvision admission controller do?",
      "doc_id": "reference_access-authn-authz_admission-controllers",
      "realistic_q1": "why can't my team create resources without manually creating namespaces first",
      "realistic_q2": "how to automatically set up default namespaces for new projects without manual intervention",
      "informed_q1": "what does NamespaceAutoProvision admission controller do",
      "informed_q2": "how to enable automatic namespace creation with admission controllers"
    },
    {
      "original_instruction": "What types of volumes can be expanded using PersistentVolumeClaims?",
      "doc_id": "concepts_storage_persistent-volumes",
      "realistic_q1": "is there a way to increase storage size for my database without losing data",
      "realistic_q2": "why can't I resize my volume claim after initial provisioning",
      "informed_q1": "which volume types support PersistentVolumeClaim expansion",
      "informed_q2": "how to expand storage using PersistentVolumeClaims in kubernetes"
    },
    {
      "original_instruction": "What are the two default user accounts for Windows containers?",
      "doc_id": "concepts_security_windows-security",
      "realistic_q1": "why can't I log into my windows container as administrator",
      "realistic_q2": "how to manage default user accounts in windows container images",
      "informed_q1": "what are the default user accounts for Windows containers in kubernetes",
      "informed_q2": "ContainerUser vs ContainerAdministrator in windows containers"
    },
    {
      "original_instruction": "What is the purpose of using an Indexed Job in Kubernetes?",
      "doc_id": "tasks_job_indexed-parallel-processing-static",
      "realistic_q1": "how to run a parallel batch job where each pod needs a unique index",
      "realistic_q2": "restart failed pod in a batch processing job without redoing entire workload",
      "informed_q1": "what is the purpose of Indexed Jobs in kubernetes",
      "informed_q2": "how do Indexed Jobs assign completion indexes to pods"
    },
    {
      "original_instruction": "What is the etcd API and what risks does it pose to the cluster's security?",
      "doc_id": "concepts_security_api-server-bypass-risks",
      "realistic_q1": "why does my kubernetes cluster hang when multiple nodes go offline simultaneously",
      "realistic_q2": "how to protect against potential data corruption if etcd nodes lose connection",
      "informed_q1": "what are the security risks of direct etcd API access bypassing kube-apiserver",
      "informed_q2": "how does etcd API bypass affect cluster security"
    },
    {
      "original_instruction": "What is the role of a signer in Kubernetes?",
      "doc_id": "tasks_tls_managing-tls-in-a-cluster",
      "realistic_q1": "why can't my certificate signing requests get automatically approved in kubernetes",
      "realistic_q2": "how to manage tls certificates for my services without manual intervention",
      "informed_q1": "what is the role of a certificate signer in kubernetes TLS management",
      "informed_q2": "how do kubernetes certificate signers approve CSRs"
    },
    {
      "original_instruction": "What tools are required to verify Kubernetes artifacts?",
      "doc_id": "tasks_administer-cluster_verify-signed-artifacts",
      "realistic_q1": "how can I quickly check if my kubernetes yaml files are valid before deploying",
      "realistic_q2": "getting weird errors in production, want to validate my deployment configs locally",
      "informed_q1": "what tools are needed to verify signed kubernetes artifacts with cosign",
      "informed_q2": "how to verify kubernetes release artifact signatures and provenance"
    },
    {
      "original_instruction": "What protocol does Kubernetes components use to emit traces?",
      "doc_id": "concepts_cluster-administration_system-traces",
      "realistic_q1": "why can't I see my application's logs and traces in the monitoring dashboard",
      "realistic_q2": "how to collect and track performance metrics from my kubernetes cluster",
      "informed_q1": "what protocol does kubernetes use to emit traces OTLP or gRPC",
      "informed_q2": "how to configure OpenTelemetry tracing for kubernetes components"
    },
    {
      "original_instruction": "What is the format in which Kubernetes components emit metrics?",
      "doc_id": "concepts_cluster-administration_system-metrics",
      "realistic_q1": "how to collect and monitor resource usage for my kubernetes cluster",
      "realistic_q2": "my prometheus dashboard is not showing kubernetes component performance metrics",
      "informed_q1": "what format do kubernetes components use to emit metrics prometheus or openmetrics",
      "informed_q2": "how to scrape metrics from kubernetes component /metrics endpoints"
    },
    {
      "original_instruction": "What are the different types of isolation in Kubernetes?",
      "doc_id": "concepts_security_multi-tenancy",
      "realistic_q1": "how to prevent one container from consuming all resources of my node",
      "realistic_q2": "stop my team's pods from interfering with critical system services",
      "informed_q1": "what types of isolation does kubernetes support for multi-tenancy",
      "informed_q2": "how does kubernetes implement namespace and node isolation for multi-tenant clusters"
    },
    {
      "original_instruction": "What is Attribute-based access control (ABAC) and how does it work?",
      "doc_id": "reference_access-authn-authz_abac",
      "realistic_q1": "why can't I make granular permissions based on user attributes like job title",
      "realistic_q2": "kubernetes access control too rigid, how to define complex authorization rules",
      "informed_q1": "how does ABAC attribute-based access control work in kubernetes",
      "informed_q2": "how to configure ABAC policy file for kubernetes authorization"
    },
    {
      "original_instruction": "What is the role of the webhook token authenticator in the authentication process?",
      "doc_id": "reference_access-authn-authz_authentication",
      "realistic_q1": "why can't my CI/CD pipeline authenticate with kubernetes cluster after rotating service account tokens",
      "realistic_q2": "getting authentication errors when trying to call kubernetes api from external tool",
      "informed_q1": "how does webhook token authenticator validate bearer tokens in kubernetes",
      "informed_q2": "what is the role of webhook token authentication in kubernetes API server"
    },
    {
      "original_instruction": "What is the purpose of the authentication flow in the aggregation layer?",
      "doc_id": "tasks_extend-kubernetes_configure-aggregation-layer",
      "realistic_q1": "why can't my custom API server authenticate requests from kubernetes cluster",
      "realistic_q2": "my extension API is not showing up in kubectl get api-services",
      "informed_q1": "how does authentication work in kubernetes API aggregation layer",
      "informed_q2": "what is the purpose of front-proxy authentication in aggregated API servers"
    },
    {
      "original_instruction": "What does sourcing the completion script in your shell enable?",
      "doc_id": "tasks_tools_included_optional-kubectl-configs-fish",
      "realistic_q1": "how to make kubectl commands autocomplete in my terminal",
      "realistic_q2": "getting tired of typing full kubernetes command names every time",
      "informed_q1": "how to enable kubectl shell completion for bash zsh or fish",
      "informed_q2": "what does sourcing kubectl completion script enable in shell"
    },
    {
      "original_instruction": "What is kube proxy?",
      "doc_id": "concepts_cluster-administration_proxies",
      "realistic_q1": "why can't my services communicate between nodes in my cluster",
      "realistic_q2": "how kubernetes routes traffic to the right pod when I have multiple replicas",
      "informed_q1": "what is kube-proxy and how does it implement service networking",
      "informed_q2": "how does kube-proxy handle iptables or IPVS for service load balancing"
    },
    {
      "original_instruction": "How can you use `kubectl` to create an object from a configuration file?",
      "doc_id": "tasks_manage-kubernetes-objects_imperative-config",
      "realistic_q1": "how do i deploy my yaml file from the command line",
      "realistic_q2": "getting error when trying to apply kubernetes configuration from file",
      "informed_q1": "how to use kubectl create with imperative object configuration",
      "informed_q2": "what is the difference between kubectl create and kubectl apply for config files"
    },
    {
      "original_instruction": "How can you suspend and resume a Job in Kubernetes?",
      "doc_id": "concepts_workloads_controllers_job",
      "realistic_q1": "my kubernetes batch job is stuck midway and I want to pause it without deleting",
      "realistic_q2": "how to temporarily stop a long-running job and resume it later without losing progress",
      "informed_q1": "how to suspend and resume a Job in kubernetes using spec.suspend",
      "informed_q2": "what happens to pods when a kubernetes Job is suspended"
    },
    {
      "original_instruction": "What is the purpose of the `kubectl logs pods/job-wq-2-7r7b2` command?",
      "doc_id": "tasks_job_fine-parallel-processing-work-queue",
      "realistic_q1": "how to see what my job failed and why it's not completing",
      "realistic_q2": "can't figure out what's happening inside my worker pod during a batch job",
      "informed_q1": "how to use kubectl logs to debug Job pod failures",
      "informed_q2": "how to view container logs from kubernetes Job worker pods"
    },
    {
      "original_instruction": "What are the four supported Topology Manager policies?",
      "doc_id": "tasks_administer-cluster_topology-manager",
      "realistic_q1": "why are my high performance containers not getting scheduled on the right cpu cores",
      "realistic_q2": "can't control how cpu and memory resources get allocated across my numa nodes",
      "informed_q1": "what are the Topology Manager policies none best-effort restricted single-numa-node",
      "informed_q2": "how does Topology Manager coordinate NUMA-aware resource allocation"
    },
    {
      "original_instruction": "Why do we need admission controllers?",
      "doc_id": "reference_access-authn-authz_admission-controllers",
      "realistic_q1": "how to prevent developers from deploying containers with latest tag in production",
      "realistic_q2": "why can't I stop someone from using root containers in my cluster",
      "informed_q1": "what admission controllers are recommended for production kubernetes clusters",
      "informed_q2": "how do admission controllers enforce security policies on pod creation"
    },
    {
      "original_instruction": "What is a ServiceAccount in Kubernetes?",
      "doc_id": "tasks_configure-pod-container_configure-service-account",
      "realistic_q1": "why can't my pod access kubernetes api when running inside the cluster",
      "realistic_q2": "error accessing cloud resources from inside my application container",
      "informed_q1": "how to configure a ServiceAccount for pod API access in kubernetes",
      "informed_q2": "what is the purpose of ServiceAccount tokens for pod authentication"
    }
  ]
}
```

---

## To Create the JSON File

Copy the JSON block above and save to:
```
poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json
```

Or run:
```bash
cat > poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json << 'EOF'
<paste JSON here>
EOF
```

---

## Expected Benchmark Improvement

| Benchmark Type | Expected Hit@5 | Rationale |
|----------------|----------------|-----------|
| Needle (rigged) | 90% | All queries target same doc |
| Realistic (natural language) | 36% | Vocabulary mismatch |
| **Informed (technical terms)** | **70-85%** | Direct terminology match |

The informed questions should show significantly better retrieval because they use the exact technical terms found in the documentation.
