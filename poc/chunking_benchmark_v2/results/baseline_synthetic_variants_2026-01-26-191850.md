# Gem Strategy Test Results

**Strategy**: synthetic_variants
**Date**: 2026-01-26T19:18:50.238936
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

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_0, score: 0.500]
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

3. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.333]
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

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_1, score: 0.250]
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

5. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_6, score: 0.200]
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

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 0.500]
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

3. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_6, score: 0.333]
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

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_003
**Type**: procedural

**Query**: How do I handle rate limit errors in my code?

**Expected Answer**: Implement exponential backoff, monitor `X-RateLimit-Remaining` header values, cache responses when appropriate, and consider upgrading to Enterprise tier for higher limits

**Retrieved Chunks**:
1. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 1.000]
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

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 0.500]
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

3. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_6, score: 0.333]
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

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_0, score: 0.500]
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

4. [doc_id: api_reference, chunk_id: api_reference_mdsem_2, score: 0.250]
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

5. [doc_id: api_reference, chunk_id: api_reference_mdsem_3, score: 0.200]
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

3. [doc_id: api_reference, chunk_id: api_reference_mdsem_1, score: 0.333]
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

5. [doc_id: api_reference, chunk_id: api_reference_mdsem_0, score: 0.200]
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

3. [doc_id: user_guide, chunk_id: user_guide_mdsem_16, score: 0.333]
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

**Baseline Score**: 9/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_007
**Type**: procedural

**Query**: How do I restart a failed workflow execution?

**Expected Answer**: View failed executions in the Dead Letter Queue, inspect the execution context and error details, and use the 'Retry' button to reprocess with the same input data

**Retrieved Chunks**:
1. [doc_id: user_guide, chunk_id: user_guide_mdsem_13, score: 1.000]
   > ### Fallback Actions
   > 
   > Execute alternative actions when the primary action fails:
   > 
   > ```yaml
   > - id: primary_payment
   >   action: http_request
   >   config:
   >     url: "https://primary-payment-gateway.com/charge"
   >     method: POST
   >     body:
   >       amount: "{{amount}}"
   >   on_error:
   >     - id: fallback_payment
   >       action: http_request
   >       config:
   >         url: "https://backup-payment-gateway.com/charge"
   >         method: POST
   >         body:
   >           amount: "{{amount}}"
   >     - id: notify_admin
   >       action: email
   >  ...

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

3. [doc_id: user_guide, chunk_id: user_guide_mdsem_17, score: 0.333]
   > ### 8. Test Thoroughly
   > 
   > Before activating a workflow:
   > 1. Use test mode with sample data
   > 2. Verify all actions execute correctly
   > 3. Test error handling paths
   > 4. Review execution logs
   > 5. Start with a limited scope (e.g., test channel, small dataset)
   > 
   > ### 9. Document Your Workflows
   > 
   > Add descriptions to workflows and steps:
   > 
   > ```yaml
   > workflow:
   >   name: "Daily Sales Report"
   >   description: |
   >     Generates a daily sales report and distributes it to the sales team.
   >     Runs at 8:00 AM EST Monday-Friday.
   >  ...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_11, score: 0.250]
   > #### Step 2: Gather Information (5-15 minutes)
   > 
   > Create incident document with:
   > - Incident timestamp and duration
   > - Affected services and endpoints
   > - Error rates and user impact
   > - Recent changes or deployments
   > - Relevant log excerpts
   > - Correlation IDs for failed requests
   > 
   > ```bash
   > 
   > # Generate incident report
   > 
   > cloudflow debug incident-report \
   >   --start "2026-01-24T10:30:00Z" \
   >   --end "2026-01-24T11:00:00Z" \
   >   --output incident-report.md
   > 
   > # Capture system snapshot
   > 
   > cloudflow debug snapshot --outp...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_3, score: 0.200]
   > ## Workflow Creation
   > 
   > CloudFlow offers multiple ways to create and manage workflows, giving you the flexibility to work in the way that suits you best.
   > 
   > ### Visual Editor
   > 
   > The Visual Editor is CloudFlow's drag-and-drop interface for building workflows without code. It's perfect for users who prefer a graphical approach:
   > 
   > **Key Features:**
   > - Drag-and-drop action blocks from the sidebar
   > - Visual connections between steps show your workflow logic
   > - Inline configuration for each action
   > - Real-time v...

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: base_008
**Type**: procedural

**Query**: What are the recommended steps for troubleshooting a workflow failure?

**Expected Answer**: Verify service health, check API connectivity, review recent deployments, inspect platform metrics, check logs, and follow the escalation procedure based on the severity level

**Retrieved Chunks**:
1. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_1, score: 1.000]
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

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_12, score: 0.500]
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

3. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_11, score: 0.333]
   > #### Step 2: Gather Information (5-15 minutes)
   > 
   > Create incident document with:
   > - Incident timestamp and duration
   > - Affected services and endpoints
   > - Error rates and user impact
   > - Recent changes or deployments
   > - Relevant log excerpts
   > - Correlation IDs for failed requests
   > 
   > ```bash
   > 
   > # Generate incident report
   > 
   > cloudflow debug incident-report \
   >   --start "2026-01-24T10:30:00Z" \
   >   --end "2026-01-24T11:00:00Z" \
   >   --output incident-report.md
   > 
   > # Capture system snapshot
   > 
   > cloudflow debug snapshot --outp...

4. [doc_id: user_guide, chunk_id: user_guide_mdsem_17, score: 0.250]
   > ### 8. Test Thoroughly
   > 
   > Before activating a workflow:
   > 1. Use test mode with sample data
   > 2. Verify all actions execute correctly
   > 3. Test error handling paths
   > 4. Review execution logs
   > 5. Start with a limited scope (e.g., test channel, small dataset)
   > 
   > ### 9. Document Your Workflows
   > 
   > Add descriptions to workflows and steps:
   > 
   > ```yaml
   > workflow:
   >   name: "Daily Sales Report"
   >   description: |
   >     Generates a daily sales report and distributes it to the sales team.
   >     Runs at 8:00 AM EST Monday-Friday.
   >  ...

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
1. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_3, score: 1.000]
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

2. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_1, score: 0.500]
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

4. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_3, score: 0.250]
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

5. [doc_id: api_reference, chunk_id: api_reference_mdsem_0, score: 0.200]
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

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_0, score: 0.333]
   > # CloudFlow Platform - System Architecture Overview
   > 
   > **Document Version:** 2.3.1  
   > **Last Updated:** January 15, 2026  
   > **Owner:** Platform Architecture Team  
   > **Status:** Production
   > 
   > ## Executive Summary
   > 
   > CloudFlow is a distributed, cloud-native workflow automation platform designed to orchestrate complex business processes at scale. The platform processes over 2.5 million workflow executions daily with an average P99 latency of 180ms for API operations and 4.2 seconds for workflow execution. T...

4. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_0, score: 0.250]
   > # CloudFlow Platform - Deployment and Operations Guide
   > 
   > **Version:** 2.4.0  
   > **Last Updated:** January 2026  
   > **Target Environment:** Production (AWS EKS)
   > 
   > ## Table of Contents
   > 
   > 1. [Overview](#overview)
   > 2. [Prerequisites](#prerequisites)
   > 3. [Infrastructure Setup](#infrastructure-setup)
   > 4. [Kubernetes Deployment](#kubernetes-deployment)
   > 5. [Database Configuration](#database-configuration)
   > 6. [Monitoring and Observability](#monitoring-and-observability)
   > 7. [Backup and Disaster Recovery](#backup-an...

5. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_3, score: 0.200]
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

**Baseline Score**: 10/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---
