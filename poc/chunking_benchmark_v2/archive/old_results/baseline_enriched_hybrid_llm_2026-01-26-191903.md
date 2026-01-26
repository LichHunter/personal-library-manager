# Gem Strategy Test Results

**Strategy**: enriched_hybrid_llm
**Date**: 2026-01-26T19:19:03.868130
**Queries Tested**: 10

## Query: base_001
**Type**: factual

**Query**: How many requests per minute are allowed in the CloudFlow API?

**Expected Answer**: 100 requests per minute per authenticated user, 20 requests per minute for unauthenticated requests, with a burst allowance of 150 requests in a 10-second window

**Retrieved Chunks**:
1. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 1.000]
   > ## Rate Limiting
   > 
   > To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.
   > 
   > **Default Limits:**
   > - 100 requests per minute per authenticated user
   > - 20 requests per minute for unauthenticated requests
   > - Burst allowance: 150 requests in a 10-second window
   > 
   > ### Rate Limit Headers
   > 
   > Every API response includes rate limit information:
   > 
   > ```
   > X-RateLimit-Limit: 100
   > X-RateLimit-Remaining: 87
   > X-RateLimit-Reset: 1640995200
   > ```
   > 
   > When you exceed the rate limit, you'll rec...

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_6, score: 0.500]
   > #### Rate Limit Tiers
   > 
   > CloudFlow enforces the following rate limits per workspace:
   > 
   > | Tier | Requests/Minute | Requests/Hour | Concurrent Workflows |
   > |------|-----------------|---------------|----------------------|
   > | Free | 60 | 1,000 | 5 |
   > | Standard | 1,000 | 50,000 | 50 |
   > | Premium | 5,000 | 250,000 | 200 |
   > | Enterprise | Custom | Custom | Unlimited |
   > 
   > #### Checking Rate Limit Status
   > 
   > ```bash
   > 
   > # Check current rate limit status
   > 
   > curl -I https://api.cloudflow.io/api/v1/workflows \
   >   -H "Author...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_18, score: 0.333]
   > ### Throughput Capacity
   > 
   > **API Gateway**:
   > - Sustained: 10,000 requests per second
   > - Peak: 25,000 requests per second (5-minute burst)
   > - Rate limiting: 1,000 requests per minute per API key
   > 
   > **Workflow Engine**:
   > - Concurrent executions: 8,000 workflows (across 16 pods)
   > - Execution start rate: 500 per second
   > - Completion rate: 450 per second (average 2-second execution time)
   > 
   > **Database**:
   > - Read throughput: 50,000 queries per second (across replicas)
   > - Write throughput: 15,000 transactions per se...

4. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.250]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

5. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 0.200]
   > #### Handling Rate Limits in Code
   > 
   > **Python example with retry logic:**
   > ```python
   > import time
   > import requests
   > 
   > def cloudflow_api_call_with_retry(url, headers, max_retries=3):
   >     for attempt in range(max_retries):
   >         response = requests.get(url, headers=headers)
   >         
   >         if response.status_code == 429:
   >             retry_after = int(response.headers.get('Retry-After', 60))
   >             print(f"Rate limited. Waiting {retry_after} seconds...")
   >             time.sleep(retry_after)
   >        ...

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_002
**Type**: factual

**Query**: What happens when I exceed the CloudFlow API rate limit?

**Expected Answer**: When you exceed the rate limit, you'll receive a `429 Too Many Requests` response with an error message indicating how long to wait before retrying

**Retrieved Chunks**:
1. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 1.000]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_5, score: 0.500]
   > # Configure error handling
   > 
   > cloudflow workflows update wf_9k2n4m8p1q \
   >   --step data_validation \
   >   --on-error continue \
   >   --error-threshold 5%  # Fail if > 5% of records invalid
   > ```
   > 
   > **2. External API Failures**
   > ```
   > ExternalAPIError: API request to https://partner-api.example.com failed with status 502
   > ```
   > 
   > **Resolution:**
   > ```bash
   > 
   > # Add circuit breaker
   > 
   > cloudflow workflows update wf_9k2n4m8p1q \
   >   --step external_api_call \
   >   --circuit-breaker-enabled true \
   >   --circuit-breaker-threshold 5 \
   > ...

3. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 0.333]
   > ## Rate Limiting
   > 
   > To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.
   > 
   > **Default Limits:**
   > - 100 requests per minute per authenticated user
   > - 20 requests per minute for unauthenticated requests
   > - Burst allowance: 150 requests in a 10-second window
   > 
   > ### Rate Limit Headers
   > 
   > Every API response includes rate limit information:
   > 
   > ```
   > X-RateLimit-Limit: 100
   > X-RateLimit-Remaining: 87
   > X-RateLimit-Reset: 1640995200
   > ```
   > 
   > When you exceed the rate limit, you'll rec...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 0.250]
   > #### Handling Rate Limits in Code
   > 
   > **Python example with retry logic:**
   > ```python
   > import time
   > import requests
   > 
   > def cloudflow_api_call_with_retry(url, headers, max_retries=3):
   >     for attempt in range(max_retries):
   >         response = requests.get(url, headers=headers)
   >         
   >         if response.status_code == 429:
   >             retry_after = int(response.headers.get('Retry-After', 60))
   >             print(f"Rate limited. Waiting {retry_after} seconds...")
   >             time.sleep(retry_after)
   >        ...

5. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_2, score: 0.200]
   > #### RBAC Policy Violations
   > 
   > CloudFlow uses role-based access control (RBAC) with the following hierarchy:
   > - `viewer` - Read-only access
   > - `developer` - Create and modify workflows (non-production)
   > - `operator` - Execute workflows, view logs
   > - `admin` - Full access to workspace
   > - `platform-admin` - Cross-workspace administration
   > 
   > **Verify resource permissions:**
   > ```bash
   > 
   > # Check effective permissions
   > 
   > cloudflow rbac check \
   >   --user john.doe@company.com \
   >   --resource workflow:prod-pipeline \
   >   ...

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_003
**Type**: procedural

**Query**: How do I handle rate limit errors in my code?

**Expected Answer**: Implement exponential backoff, monitor `X-RateLimit-Remaining` header values, cache responses when appropriate, and consider upgrading to Enterprise tier for higher limits

**Retrieved Chunks**:
1. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_5, score: 1.000]
   > # Configure error handling
   > 
   > cloudflow workflows update wf_9k2n4m8p1q \
   >   --step data_validation \
   >   --on-error continue \
   >   --error-threshold 5%  # Fail if > 5% of records invalid
   > ```
   > 
   > **2. External API Failures**
   > ```
   > ExternalAPIError: API request to https://partner-api.example.com failed with status 502
   > ```
   > 
   > **Resolution:**
   > ```bash
   > 
   > # Add circuit breaker
   > 
   > cloudflow workflows update wf_9k2n4m8p1q \
   >   --step external_api_call \
   >   --circuit-breaker-enabled true \
   >   --circuit-breaker-threshold 5 \
   > ...

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.500]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

3. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 0.333]
   > #### Handling Rate Limits in Code
   > 
   > **Python example with retry logic:**
   > ```python
   > import time
   > import requests
   > 
   > def cloudflow_api_call_with_retry(url, headers, max_retries=3):
   >     for attempt in range(max_retries):
   >         response = requests.get(url, headers=headers)
   >         
   >         if response.status_code == 429:
   >             retry_after = int(response.headers.get('Retry-After', 60))
   >             print(f"Rate limited. Waiting {retry_after} seconds...")
   >             time.sleep(retry_after)
   >        ...

4. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 0.250]
   > ## Rate Limiting
   > 
   > To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.
   > 
   > **Default Limits:**
   > - 100 requests per minute per authenticated user
   > - 20 requests per minute for unauthenticated requests
   > - Burst allowance: 150 requests in a 10-second window
   > 
   > ### Rate Limit Headers
   > 
   > Every API response includes rate limit information:
   > 
   > ```
   > X-RateLimit-Limit: 100
   > X-RateLimit-Remaining: 87
   > X-RateLimit-Reset: 1640995200
   > ```
   > 
   > When you exceed the rate limit, you'll rec...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_12, score: 0.200]
   > ## Error Handling
   > 
   > Robust error handling ensures your workflows are resilient and reliable.
   > 
   > ### Retry Policies
   > 
   > Configure automatic retries for failed actions:
   > 
   > ```yaml
   > - id: api_call
   >   action: http_request
   >   config:
   >     url: "https://api.example.com/data"
   >   retry:
   >     max_attempts: 3
   >     backoff_type: "exponential"  # or "fixed", "linear"
   >     initial_interval: 1000       # milliseconds
   >     max_interval: 30000
   >     multiplier: 2.0
   >     retry_on:
   >       - timeout
   >       - network_error
   >       - statu...

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_004
**Type**: factual

**Query**: What are the CloudFlow API authentication methods?

**Expected Answer**: CloudFlow supports three authentication methods: API Keys, OAuth 2.0, and JWT Tokens

**Retrieved Chunks**:
1. [doc_id: api_reference, chunk_id: api_reference_mdsem_1, score: 1.000]
   > ## Authentication
   > 
   > CloudFlow supports three authentication methods to suit different use cases and security requirements.
   > 
   > ### API Keys
   > 
   > API keys provide simple authentication for server-to-server communication. Include your API key in the request header:
   > 
   > ```bash
   > curl -H "X-API-Key: cf_live_a1b2c3d4e5f6g7h8i9j0" \
   >   https://api.cloudflow.io/v2/workflows
   > ```
   > 
   > **Security Notes:**
   > - Never expose API keys in client-side code
   > - Rotate keys every 90 days
   > - Use separate keys for development and produc...

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_2, score: 0.500]
   > ### OAuth 2.0
   > 
   > OAuth 2.0 is recommended for applications that access CloudFlow on behalf of users. We support the Authorization Code flow with PKCE.
   > 
   > **Authorization Endpoint:** `https://auth.cloudflow.io/oauth/authorize`
   > 
   > **Token Endpoint:** `https://auth.cloudflow.io/oauth/token`
   > 
   > **Supported Scopes:**
   > - `workflows:read` - Read workflow configurations
   > - `workflows:write` - Create and modify workflows
   > - `pipelines:read` - Read pipeline data
   > - `pipelines:write` - Create and manage pipelines
   > - `a...

3. [doc_id: api_reference, chunk_id: api_reference_mdsem_0, score: 0.333]
   > # CloudFlow API Reference
   > 
   > Version 2.1.0 | Last Updated: January 2026
   > 
   > ## Overview
   > 
   > The CloudFlow API is a RESTful service that enables developers to programmatically manage cloud workflows, data pipelines, and automation tasks. This documentation provides comprehensive details on authentication, endpoints, request/response formats, error handling, and best practices.
   > 
   > **Base URL:** `https://api.cloudflow.io/v2`
   > 
   > **API Status:** https://status.cloudflow.io

4. [doc_id: user_guide, chunk_id: user_guide_mdsem_6, score: 0.250]
   > ## Available Actions
   > 
   > CloudFlow provides a comprehensive library of actions to build powerful automations.
   > 
   > ### HTTP Requests
   > 
   > Make HTTP requests to any API endpoint:
   > 
   > **Configuration:**
   > - **Method**: GET, POST, PUT, PATCH, DELETE, HEAD
   > - **URL**: Full endpoint URL (supports variable interpolation)
   > - **Headers**: Custom headers as key-value pairs
   > - **Query Parameters**: URL parameters
   > - **Body**: JSON, form data, or raw text
   > - **Authentication**: Basic Auth, Bearer Token, API Key, OAuth 2.0
   > 
   > **E...

5. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_1, score: 0.200]
   > ## Overview
   > 
   > This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analysis, resolution steps, and preventive measures.
   > 
   > ### Quick Diagnostic Checklist
   > 
   > Before diving into specific issues, perform these initial checks:
   > 
   > - Verify service health: `cloudflow status --all`
   > - Check API connectivity: `curl -I https://api.cloudflow.io/health`
   > - Review recent deployments: `kube...

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_005
**Type**: factual

**Query**: How long do JWT tokens last in CloudFlow?

**Expected Answer**: All tokens expire after 3600 seconds (1 hour). Implement token refresh logic in your application.

**Retrieved Chunks**:
1. [doc_id: api_reference, chunk_id: api_reference_mdsem_3, score: 1.000]
   > ### JWT Tokens
   > 
   > For advanced use cases, CloudFlow supports JSON Web Tokens (JWT) with RS256 signing algorithm. JWTs must include the following claims:
   > 
   > - `iss` (issuer): Your application identifier
   > - `sub` (subject): User or service account ID
   > - `aud` (audience): `https://api.cloudflow.io`
   > - `exp` (expiration): Unix timestamp (max 3600 seconds from `iat`)
   > - `iat` (issued at): Unix timestamp
   > - `scope`: Space-separated list of requested scopes
   > 
   > Example JWT header:
   > 
   > ```python
   > import jwt
   > import time...

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_1, score: 0.500]
   > ## Overview
   > 
   > This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analysis, resolution steps, and preventive measures.
   > 
   > ### Quick Diagnostic Checklist
   > 
   > Before diving into specific issues, perform these initial checks:
   > 
   > - Verify service health: `cloudflow status --all`
   > - Check API connectivity: `curl -I https://api.cloudflow.io/health`
   > - Review recent deployments: `kube...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_15, score: 0.333]
   > ### Authentication & Authorization
   > 
   > **JWT Token Validation**:
   > - Algorithm: RS256 (asymmetric signing)
   > - Key rotation: Every 30 days with 7-day overlap period
   > - Public key distribution: JWKS endpoint cached in Redis
   > - Validation: Signature, expiry, issuer, audience claims
   > - Token revocation: Blacklist in Redis for compromised tokens
   > 
   > **Permission Model**:
   > ```
   > User → Roles → Permissions
   >      ↘       ↗
   >       Tenants (Multi-tenancy isolation)
   > ```
   > 
   > Example permissions:
   > - `workflow:read` - View workfl...

4. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_3, score: 0.250]
   > ## Microservices Breakdown
   > 
   > 
   > 
   > ### API Gateway
   > 
   > **Purpose**: Single entry point for all client requests, providing authentication, rate limiting, request routing, and protocol translation.
   > 
   > **Technology**: Node.js with Express.js framework  
   > **Replicas**: 12 pods (production), auto-scaling 8-20 based on CPU  
   > **Resource Allocation**: 2 vCPU, 4GB RAM per pod
   > 
   > **Key Responsibilities**:
   > - JWT token validation (delegated to Auth Service for initial validation)
   > - Rate limiting: 1000 requests per minut...

5. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_4, score: 0.200]
   > ### Auth Service
   > 
   > **Purpose**: Centralized authentication and authorization service handling user identity, token generation, and permission validation.
   > 
   > **Technology**: Go with gRPC for internal communication, REST for external  
   > **Replicas**: 8 pods (production), auto-scaling 6-12  
   > **Resource Allocation**: 1 vCPU, 2GB RAM per pod
   > 
   > **Key Responsibilities**:
   > - User authentication via multiple providers (OAuth2, SAML, local credentials)
   > - JWT token generation and validation (RS256 algorithm)
   > - R...

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_006
**Type**: procedural

**Query**: What should I do if my workflow exceeds the maximum execution timeout?

**Expected Answer**: Workflows have a default timeout of 3600 seconds (60 minutes). You can increase the timeout, optimize workflow steps, or split the workflow into smaller workflows

**Retrieved Chunks**:
1. [doc_id: user_guide, chunk_id: user_guide_mdsem_14, score: 1.000]
   > ### Steps Per Workflow
   > 
   > - **Maximum**: 50 steps per workflow
   > - **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via webhooks.
   > 
   > ### Execution Timeout
   > 
   > - **Default**: 3600 seconds (60 minutes)
   > - **Behavior**: Workflows exceeding this timeout are automatically terminated
   > - **Custom Timeouts**: Enterprise plans can request custom timeout limits
   > 
   > **Setting Step-Level Timeouts:**
   > ```yaml
   > - id: long_running_task
   >   actio...

2. [doc_id: user_guide, chunk_id: user_guide_mdsem_12, score: 0.500]
   > ## Error Handling
   > 
   > Robust error handling ensures your workflows are resilient and reliable.
   > 
   > ### Retry Policies
   > 
   > Configure automatic retries for failed actions:
   > 
   > ```yaml
   > - id: api_call
   >   action: http_request
   >   config:
   >     url: "https://api.example.com/data"
   >   retry:
   >     max_attempts: 3
   >     backoff_type: "exponential"  # or "fixed", "linear"
   >     initial_interval: 1000       # milliseconds
   >     max_interval: 30000
   >     multiplier: 2.0
   >     retry_on:
   >       - timeout
   >       - network_error
   >       - statu...

3. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_4, score: 0.333]
   > # Create sub-workflows
   > 
   > cloudflow workflows create data-pipeline-part1 \
   >   --steps "data_ingestion,data_validation" \
   >   --timeout 1800
   > 
   > cloudflow workflows create data-pipeline-part2 \
   >   --steps "data_transformation,data_export" \
   >   --timeout 3600 \
   >   --trigger workflow_completed \
   >   --trigger-workflow data-pipeline-part1
   > ```
   > 
   > ### Retry Logic and Exponential Backoff
   > 
   > CloudFlow implements automatic retry with exponential backoff for transient failures:
   > - Max retries: 3
   > - Initial delay: 1 second
   > -...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_2, score: 0.250]
   > #### RBAC Policy Violations
   > 
   > CloudFlow uses role-based access control (RBAC) with the following hierarchy:
   > - `viewer` - Read-only access
   > - `developer` - Create and modify workflows (non-production)
   > - `operator` - Execute workflows, view logs
   > - `admin` - Full access to workspace
   > - `platform-admin` - Cross-workspace administration
   > 
   > **Verify resource permissions:**
   > ```bash
   > 
   > # Check effective permissions
   > 
   > cloudflow rbac check \
   >   --user john.doe@company.com \
   >   --resource workflow:prod-pipeline \
   >   ...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_16, score: 0.200]
   > ### Data Limits
   > 
   > - **Maximum request/response size**: 10MB per action
   > - **Maximum execution payload**: 50MB total
   > - **Variable value size**: 1MB per variable
   > 
   > ### Enterprise Plan Limits
   > 
   > Enterprise customers can request increased limits:
   > - Up to 100 steps per workflow
   > - Up to 10,000 executions per day
   > - Up to 7200 second timeout (2 hours)
   > - Priority execution queue
   > - Dedicated capacity allocation
   > 
   > Contact sales@cloudflow.io for Enterprise pricing and custom limits.
   > 
   > ## Best Practices
   > 
   > Follow the...

**Baseline Score**: 9/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_007
**Type**: procedural

**Query**: How do I restart a failed workflow execution?

**Expected Answer**: View failed executions in the Dead Letter Queue, inspect the execution context and error details, and use the 'Retry' button to reprocess with the same input data

**Retrieved Chunks**:
1. [doc_id: user_guide, chunk_id: user_guide_mdsem_12, score: 1.000]
   > ## Error Handling
   > 
   > Robust error handling ensures your workflows are resilient and reliable.
   > 
   > ### Retry Policies
   > 
   > Configure automatic retries for failed actions:
   > 
   > ```yaml
   > - id: api_call
   >   action: http_request
   >   config:
   >     url: "https://api.example.com/data"
   >   retry:
   >     max_attempts: 3
   >     backoff_type: "exponential"  # or "fixed", "linear"
   >     initial_interval: 1000       # milliseconds
   >     max_interval: 30000
   >     multiplier: 2.0
   >     retry_on:
   >       - timeout
   >       - network_error
   >       - statu...

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_4, score: 0.500]
   > # Create sub-workflows
   > 
   > cloudflow workflows create data-pipeline-part1 \
   >   --steps "data_ingestion,data_validation" \
   >   --timeout 1800
   > 
   > cloudflow workflows create data-pipeline-part2 \
   >   --steps "data_transformation,data_export" \
   >   --timeout 3600 \
   >   --trigger workflow_completed \
   >   --trigger-workflow data-pipeline-part1
   > ```
   > 
   > ### Retry Logic and Exponential Backoff
   > 
   > CloudFlow implements automatic retry with exponential backoff for transient failures:
   > - Max retries: 3
   > - Initial delay: 1 second
   > -...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_6, score: 0.333]
   > ### Scheduler Service
   > 
   > **Purpose**: Time-based workflow triggering system supporting cron-like schedules and one-time delayed executions.
   > 
   > **Technology**: Go with distributed locking via Redis  
   > **Replicas**: 4 pods (production), active-passive with leader election  
   > **Resource Allocation**: 2 vCPU, 4GB RAM per pod
   > 
   > **Key Responsibilities**:
   > - Parse and validate cron expressions (extended format supporting seconds)
   > - Maintain schedule registry in PostgreSQL
   > - Distributed scheduling with leader e...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_5, score: 0.250]
   > # Configure error handling
   > 
   > cloudflow workflows update wf_9k2n4m8p1q \
   >   --step data_validation \
   >   --on-error continue \
   >   --error-threshold 5%  # Fail if > 5% of records invalid
   > ```
   > 
   > **2. External API Failures**
   > ```
   > ExternalAPIError: API request to https://partner-api.example.com failed with status 502
   > ```
   > 
   > **Resolution:**
   > ```bash
   > 
   > # Add circuit breaker
   > 
   > cloudflow workflows update wf_9k2n4m8p1q \
   >   --step external_api_call \
   >   --circuit-breaker-enabled true \
   >   --circuit-breaker-threshold 5 \
   > ...

5. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.200]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_008
**Type**: procedural

**Query**: What are the recommended steps for troubleshooting a workflow failure?

**Expected Answer**: Verify service health, check API connectivity, review recent deployments, inspect platform metrics, check logs, and follow the escalation procedure based on the severity level

**Retrieved Chunks**:
1. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_12, score: 1.000]
   > ### Getting Help
   > 
   > If this troubleshooting guide doesn't resolve your issue:
   > 
   > 1. Search the knowledge base: `cloudflow kb search "your issue"`
   > 2. Check community forum for similar issues
   > 3. Contact support with detailed logs and reproduction steps
   > 4. For urgent issues, use emergency escalation procedures
   > 
   > **Remember:** Always capture logs, metrics, and reproduction steps before escalating!
   > 
   > ---
   > 
   > *Last updated: January 24, 2026*  
   > *Document version: 3.2.1*  
   > *Feedback: docs-feedback@cloudflow.io*

2. [doc_id: user_guide, chunk_id: user_guide_mdsem_12, score: 0.500]
   > ## Error Handling
   > 
   > Robust error handling ensures your workflows are resilient and reliable.
   > 
   > ### Retry Policies
   > 
   > Configure automatic retries for failed actions:
   > 
   > ```yaml
   > - id: api_call
   >   action: http_request
   >   config:
   >     url: "https://api.example.com/data"
   >   retry:
   >     max_attempts: 3
   >     backoff_type: "exponential"  # or "fixed", "linear"
   >     initial_interval: 1000       # milliseconds
   >     max_interval: 30000
   >     multiplier: 2.0
   >     retry_on:
   >       - timeout
   >       - network_error
   >       - statu...

3. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_1, score: 0.333]
   > ## Overview
   > 
   > This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analysis, resolution steps, and preventive measures.
   > 
   > ### Quick Diagnostic Checklist
   > 
   > Before diving into specific issues, perform these initial checks:
   > 
   > - Verify service health: `cloudflow status --all`
   > - Check API connectivity: `curl -I https://api.cloudflow.io/health`
   > - Review recent deployments: `kube...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_5, score: 0.250]
   > # Configure error handling
   > 
   > cloudflow workflows update wf_9k2n4m8p1q \
   >   --step data_validation \
   >   --on-error continue \
   >   --error-threshold 5%  # Fail if > 5% of records invalid
   > ```
   > 
   > **2. External API Failures**
   > ```
   > ExternalAPIError: API request to https://partner-api.example.com failed with status 502
   > ```
   > 
   > **Resolution:**
   > ```bash
   > 
   > # Add circuit breaker
   > 
   > cloudflow workflows update wf_9k2n4m8p1q \
   >   --step external_api_call \
   >   --circuit-breaker-enabled true \
   >   --circuit-breaker-threshold 5 \
   > ...

5. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_0, score: 0.200]
   > # CloudFlow Platform Troubleshooting Guide
   > 
   > **Version:** 3.2.1  
   > **Last Updated:** January 2026  
   > **Audience:** Platform Engineers, DevOps, Support Teams
   > 
   > ## Table of Contents
   > 
   > 1. [Overview](#overview)
   > 2. [Authentication & Authorization Issues](#authentication--authorization-issues)
   > 3. [Performance Problems](#performance-problems)
   > 4. [Database Connection Issues](#database-connection-issues)
   > 5. [Workflow Execution Failures](#workflow-execution-failures)
   > 6. [Rate Limiting & Throttling](#rate-limit...

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_009
**Type**: procedural

**Query**: How do I set up database connection pooling in CloudFlow?

**Expected Answer**: Use PgBouncer for connection pooling, configure CloudFlow to use PgBouncer, set connection pool modes, and add read replicas to optimize database performance

**Retrieved Chunks**:
1. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_2, score: 1.000]
   > ### Architecture Diagram (Conceptual)
   > 
   > ```
   > ┌─────────────────────────────────────────────────────────────────┐
   > │                        Load Balancer (ALB)                       │
   > │                     (TLS Termination - 443)                      │
   > └────────────────────────────┬────────────────────────────────────┘
   >                              │
   >                              ▼
   > ┌─────────────────────────────────────────────────────────────────┐
   > │                        API Gateway Layer           ...

2. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_3, score: 0.500]
   > ### Environment Configuration
   > 
   > CloudFlow requires the following environment variables:
   > 
   > | Variable | Description | Example | Required |
   > |----------|-------------|---------|----------|
   > | `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/cloudflow` | Yes |
   > | `REDIS_URL` | Redis connection string | `redis://redis-master.cloudflow-prod.svc.cluster.local:6379` | Yes |
   > | `JWT_SECRET` | Secret key for JWT token signing | `<generated-secret-256-bit>` | Yes |
   > | `LOG_LEVEL`...

3. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_4, score: 0.333]
   > # postgres-values.yaml
   > 
   > global:
   >   postgresql:
   >     auth:
   >       username: cloudflow
   >       database: cloudflow
   >       existingSecret: postgres-credentials
   > 
   > image:
   >   tag: "14.10.0"
   > 
   > primary:
   >   resources:
   >     limits:
   >       cpu: 4000m
   >       memory: 8Gi
   >     requests:
   >       cpu: 2000m
   >       memory: 4Gi
   >   
   >   persistence:
   >     enabled: true
   >     size: 100Gi
   >     storageClass: gp3
   >   
   >   extendedConfiguration: |
   >     max_connections = 100
   >     shared_buffers = 2GB
   >     effective_cache_size = 6GB
   >     maintenance_wor...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_3, score: 0.250]
   > # Check database slow query log
   > 
   > kubectl logs -n cloudflow deploy/cloudflow-db-primary | \
   >   grep "slow query" | \
   >   tail -n 50
   > 
   > # Analyze query patterns
   > 
   > cloudflow db analyze-queries --min-duration 5000 --limit 20
   > ```
   > 
   > **2. Review Query Execution Plans**
   > 
   > ```sql
   > -- Connect to CloudFlow database
   > cloudflow db connect --readonly
   > 
   > -- Explain slow query
   > EXPLAIN ANALYZE
   > SELECT w.*, e.status, e.error_message
   > FROM workflows w
   > LEFT JOIN executions e ON w.id = e.workflow_id
   > WHERE w.workspace_id = 'ws_abc...

5. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_6, score: 0.200]
   > ### Grafana Dashboards
   > 
   > Access Grafana to view CloudFlow dashboards:
   > 
   > ```bash
   > kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
   > ```
   > 
   > Import the CloudFlow dashboard (ID: 15847) or use the provided JSON template. Key dashboard panels include:
   > 
   > 1. **API Performance**: Request rate, P95/P99 latency, error rate
   > 2. **Resource Usage**: CPU, memory, disk I/O per pod
   > 3. **Database Health**: Connection pool utilization, query performance
   > 4. **Worker Status**: Queue depth, processing rate, ...

**Baseline Score**: 8/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_010
**Type**: procedural

**Query**: What Kubernetes resources are needed to deploy CloudFlow?

**Expected Answer**: Deploy an EKS cluster with managed node groups, configure storage with EBS CSI driver, set up a namespace with resource quotas, and use Helm charts for deployment

**Retrieved Chunks**:
1. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_1, score: 1.000]
   > ## Overview
   > 
   > CloudFlow is a cloud-native workflow orchestration platform designed for high-availability production environments. This guide provides comprehensive instructions for deploying and operating CloudFlow on Amazon EKS (Elastic Kubernetes Service).
   > 
   > ### Architecture Summary
   > 
   > CloudFlow consists of the following components:
   > 
   > - **API Server**: REST API for workflow management (Node.js/Express)
   > - **Worker Service**: Background job processor (Node.js)
   > - **Scheduler**: Cron-based task schedul...

2. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_2, score: 0.500]
   > ### Required Tools
   > 
   > - `kubectl` (v1.28 or later)
   > - `helm` (v3.12 or later)
   > - `aws-cli` (v2.13 or later)
   > - `eksctl` (v0.165 or later)
   > - `terraform` (v1.6 or later) - for infrastructure provisioning
   > 
   > ### Access Requirements
   > 
   > - AWS account with appropriate IAM permissions
   > - EKS cluster admin access
   > - Container registry access (ECR)
   > - Domain name and SSL certificates
   > - Secrets management access (AWS Secrets Manager or Vault)
   > 
   > ### Network Requirements
   > 
   > - VPC with at least 3 public and 3 private subne...

3. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_3, score: 0.333]
   > ### Environment Configuration
   > 
   > CloudFlow requires the following environment variables:
   > 
   > | Variable | Description | Example | Required |
   > |----------|-------------|---------|----------|
   > | `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/cloudflow` | Yes |
   > | `REDIS_URL` | Redis connection string | `redis://redis-master.cloudflow-prod.svc.cluster.local:6379` | Yes |
   > | `JWT_SECRET` | Secret key for JWT token signing | `<generated-secret-256-bit>` | Yes |
   > | `LOG_LEVEL`...

4. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_4, score: 0.250]
   > # postgres-values.yaml
   > 
   > global:
   >   postgresql:
   >     auth:
   >       username: cloudflow
   >       database: cloudflow
   >       existingSecret: postgres-credentials
   > 
   > image:
   >   tag: "14.10.0"
   > 
   > primary:
   >   resources:
   >     limits:
   >       cpu: 4000m
   >       memory: 8Gi
   >     requests:
   >       cpu: 2000m
   >       memory: 4Gi
   >   
   >   persistence:
   >     enabled: true
   >     size: 100Gi
   >     storageClass: gp3
   >   
   >   extendedConfiguration: |
   >     max_connections = 100
   >     shared_buffers = 2GB
   >     effective_cache_size = 6GB
   >     maintenance_wor...

5. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_9, score: 0.200]
   > #### Pods in CrashLoopBackOff
   > 
   > **Symptoms**: Pods continuously restart
   > **Diagnosis**:
   > ```bash
   > kubectl logs -n cloudflow-prod <pod-name> --previous
   > kubectl describe pod -n cloudflow-prod <pod-name>
   > ```
   > 
   > **Common Causes**:
   > - Database connection failure
   > - Invalid environment variables
   > - Insufficient resources
   > 
   > #### High Memory Usage
   > 
   > **Symptoms**: Pods being OOMKilled
   > **Diagnosis**:
   > ```bash
   > kubectl top pods -n cloudflow-prod
   > ```
   > 
   > **Resolution**:
   > - Increase memory limits in deployment
   > - Check for me...

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---
