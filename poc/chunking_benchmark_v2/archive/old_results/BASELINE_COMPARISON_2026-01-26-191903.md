# Baseline Questions - Strategy Comparison

**Date**: 2026-01-26T19:19:03.868628
**Strategies**: adaptive_hybrid, synthetic_variants, enriched_hybrid_llm
**Questions**: 10

## Question base_001: How many requests per minute are allowed in the CloudFlow API?...

**Expected Answer**: 100 requests per minute per authenticated user, 20 requests per minute for unauthenticated requests, with a burst allowance of 150 requests in a 10-second window

### adaptive_hybrid
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 1.000]
> ## Rate Limiting  To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.  **Default Limits:** - 100 requests per minute per authenticated user - 20 requests pe...

### synthetic_variants
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 1.000]
> ## Rate Limiting  To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.  **Default Limits:** - 100 requests per minute per authenticated user - 20 requests pe...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 1.000]
> ## Rate Limiting  To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.  **Default Limits:** - 100 requests per minute per authenticated user - 20 requests pe...

---

## Question base_002: What happens when I exceed the CloudFlow API rate limit?...

**Expected Answer**: When you exceed the rate limit, you'll receive a `429 Too Many Requests` response with an error message indicating how long to wait before retrying

### adaptive_hybrid
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 1.000]
> ## Rate Limiting  To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.  **Default Limits:** - 100 requests per minute per authenticated user - 20 requests pe...

### synthetic_variants
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 1.000]
> ### Error Codes  CloudFlow returns specific error codes to help you identify and resolve issues:  - `invalid_parameter`: One or more request parameters are invalid - `missing_required_field`: Required...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 1.000]
> ### Error Codes  CloudFlow returns specific error codes to help you identify and resolve issues:  - `invalid_parameter`: One or more request parameters are invalid - `missing_required_field`: Required...

---

## Question base_003: How do I handle rate limit errors in my code?...

**Expected Answer**: Implement exponential backoff, monitor `X-RateLimit-Remaining` header values, cache responses when appropriate, and consider upgrading to Enterprise tier for higher limits

### adaptive_hybrid
**Top Chunk**: [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 1.000]
> #### Handling Rate Limits in Code  **Python example with retry logic:** ```python import time import requests  def cloudflow_api_call_with_retry(url, headers, max_retries=3):     for attempt in range(...

### synthetic_variants
**Top Chunk**: [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 1.000]
> #### Handling Rate Limits in Code  **Python example with retry logic:** ```python import time import requests  def cloudflow_api_call_with_retry(url, headers, max_retries=3):     for attempt in range(...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_5, score: 1.000]
> # Configure error handling  cloudflow workflows update wf_9k2n4m8p1q \   --step data_validation \   --on-error continue \   --error-threshold 5%  # Fail if > 5% of records invalid ```  **2. External A...

---

## Question base_004: What are the CloudFlow API authentication methods?...

**Expected Answer**: CloudFlow supports three authentication methods: API Keys, OAuth 2.0, and JWT Tokens

### adaptive_hybrid
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_1, score: 1.000]
> ## Authentication  CloudFlow supports three authentication methods to suit different use cases and security requirements.  ### API Keys  API keys provide simple authentication for server-to-server com...

### synthetic_variants
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_1, score: 1.000]
> ## Authentication  CloudFlow supports three authentication methods to suit different use cases and security requirements.  ### API Keys  API keys provide simple authentication for server-to-server com...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_1, score: 1.000]
> ## Authentication  CloudFlow supports three authentication methods to suit different use cases and security requirements.  ### API Keys  API keys provide simple authentication for server-to-server com...

---

## Question base_005: How long do JWT tokens last in CloudFlow?...

**Expected Answer**: All tokens expire after 3600 seconds (1 hour). Implement token refresh logic in your application.

### adaptive_hybrid
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_3, score: 1.000]
> ### JWT Tokens  For advanced use cases, CloudFlow supports JSON Web Tokens (JWT) with RS256 signing algorithm. JWTs must include the following claims:  - `iss` (issuer): Your application identifier - ...

### synthetic_variants
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_3, score: 1.000]
> ### JWT Tokens  For advanced use cases, CloudFlow supports JSON Web Tokens (JWT) with RS256 signing algorithm. JWTs must include the following claims:  - `iss` (issuer): Your application identifier - ...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: api_reference, chunk_id: api_reference_mdsem_3, score: 1.000]
> ### JWT Tokens  For advanced use cases, CloudFlow supports JSON Web Tokens (JWT) with RS256 signing algorithm. JWTs must include the following claims:  - `iss` (issuer): Your application identifier - ...

---

## Question base_006: What should I do if my workflow exceeds the maximum execution timeout?...

**Expected Answer**: Workflows have a default timeout of 3600 seconds (60 minutes). You can increase the timeout, optimize workflow steps, or split the workflow into smaller workflows

### adaptive_hybrid
**Top Chunk**: [doc_id: user_guide, chunk_id: user_guide_mdsem_14, score: 1.000]
> ### Steps Per Workflow  - **Maximum**: 50 steps per workflow - **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via ...

### synthetic_variants
**Top Chunk**: [doc_id: user_guide, chunk_id: user_guide_mdsem_14, score: 1.000]
> ### Steps Per Workflow  - **Maximum**: 50 steps per workflow - **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via ...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: user_guide, chunk_id: user_guide_mdsem_14, score: 1.000]
> ### Steps Per Workflow  - **Maximum**: 50 steps per workflow - **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via ...

---

## Question base_007: How do I restart a failed workflow execution?...

**Expected Answer**: View failed executions in the Dead Letter Queue, inspect the execution context and error details, and use the 'Retry' button to reprocess with the same input data

### adaptive_hybrid
**Top Chunk**: [doc_id: user_guide, chunk_id: user_guide_mdsem_13, score: 1.000]
> ### Fallback Actions  Execute alternative actions when the primary action fails:  ```yaml - id: primary_payment   action: http_request   config:     url: "https://primary-payment-gateway.com/charge"  ...

### synthetic_variants
**Top Chunk**: [doc_id: user_guide, chunk_id: user_guide_mdsem_13, score: 1.000]
> ### Fallback Actions  Execute alternative actions when the primary action fails:  ```yaml - id: primary_payment   action: http_request   config:     url: "https://primary-payment-gateway.com/charge"  ...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: user_guide, chunk_id: user_guide_mdsem_12, score: 1.000]
> ## Error Handling  Robust error handling ensures your workflows are resilient and reliable.  ### Retry Policies  Configure automatic retries for failed actions:  ```yaml - id: api_call   action: http_...

---

## Question base_008: What are the recommended steps for troubleshooting a workflow failure?...

**Expected Answer**: Verify service health, check API connectivity, review recent deployments, inspect platform metrics, check logs, and follow the escalation procedure based on the severity level

### adaptive_hybrid
**Top Chunk**: [doc_id: user_guide, chunk_id: user_guide_mdsem_14, score: 1.000]
> ### Steps Per Workflow  - **Maximum**: 50 steps per workflow - **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via ...

### synthetic_variants
**Top Chunk**: [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_1, score: 1.000]
> ## Overview  This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analy...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_12, score: 1.000]
> ### Getting Help  If this troubleshooting guide doesn't resolve your issue:  1. Search the knowledge base: `cloudflow kb search "your issue"` 2. Check community forum for similar issues 3. Contact sup...

---

## Question base_009: How do I set up database connection pooling in CloudFlow?...

**Expected Answer**: Use PgBouncer for connection pooling, configure CloudFlow to use PgBouncer, set connection pool modes, and add read replicas to optimize database performance

### adaptive_hybrid
**Top Chunk**: [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_3, score: 1.000]
> # Check database slow query log  kubectl logs -n cloudflow deploy/cloudflow-db-primary | \   grep "slow query" | \   tail -n 50  # Analyze query patterns  cloudflow db analyze-queries --min-duration 5...

### synthetic_variants
**Top Chunk**: [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_3, score: 1.000]
> # Check database slow query log  kubectl logs -n cloudflow deploy/cloudflow-db-primary | \   grep "slow query" | \   tail -n 50  # Analyze query patterns  cloudflow db analyze-queries --min-duration 5...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_2, score: 1.000]
> ### Architecture Diagram (Conceptual)  ``` ┌─────────────────────────────────────────────────────────────────┐ │                        Load Balancer (ALB)                       │ │                   ...

---

## Question base_010: What Kubernetes resources are needed to deploy CloudFlow?...

**Expected Answer**: Deploy an EKS cluster with managed node groups, configure storage with EBS CSI driver, set up a namespace with resource quotas, and use Helm charts for deployment

### adaptive_hybrid
**Top Chunk**: [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_3, score: 1.000]
> ### Environment Configuration  CloudFlow requires the following environment variables:  | Variable | Description | Example | Required | |----------|-------------|---------|----------| | `DATABASE_URL`...

### synthetic_variants
**Top Chunk**: [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_1, score: 1.000]
> ## Overview  CloudFlow is a cloud-native workflow orchestration platform designed for high-availability production environments. This guide provides comprehensive instructions for deploying and operat...

### enriched_hybrid_llm
**Top Chunk**: [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_1, score: 1.000]
> ## Overview  CloudFlow is a cloud-native workflow orchestration platform designed for high-availability production environments. This guide provides comprehensive instructions for deploying and operat...

---
