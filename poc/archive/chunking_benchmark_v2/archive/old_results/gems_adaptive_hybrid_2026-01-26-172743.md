# Gem Strategy Test Results

**Strategy**: adaptive_hybrid
**Date**: 2026-01-26T17:27:43.211377
**Queries Tested**: 1

## Query: neg_001
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: What should I NOT do when I'm rate limited?

**Expected Answer**: Don't keep hammering the API. Instead: check Retry-After header, implement exponential backoff, monitor X-RateLimit-Remaining, cache responses, consider upgrading tier.

**Retrieved Chunks**:
1. [doc_id: api_reference, chunk_id: api_reference_fix_1, score: 1.000]
   > ## Rate Limiting To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints. **Default Limits:** - 100 requests per minute per authenticated user - 20 requests per minute for unauthenticated requests - Burst allowance: 150 requests in a 10-second window ### Rate Limit Headers Every API response includes rate limit information: ``` X-RateLimit-Limit: 100 X-RateLimit-Remaining: 87 X-RateLimit-Reset: 1640995200 ``` When you exceed the rate limit, you'll receive a...

2. [doc_id: api_reference, chunk_id: api_reference_fix_4, score: 0.500]
   > ### Error Response Format ```json { "error": { "code": "invalid_parameter", "message": "The 'limit' parameter must be between 1 and 100", "field": "limit", "request_id": "req_8k3m9x2p" } } ``` ### HTTP Status Codes | Status Code | Description | Common Causes | |------------|-------------|---------------| | 400 | Bad Request | Invalid parameters, malformed JSON, validation errors | | 401 | Unauthorized | Missing or invalid authentication credentials | | 403 | Forbidden | Insufficient permissions ...

3. [doc_id: user_guide, chunk_id: user_guide_fix_5, score: 0.333]
   > Save and activate **Testing Schedules:** Use the built-in schedule calculator to preview upcoming executions: ``` Next 5 executions: 1. 2026-01-24 14:00:00 UTC 2. 2026-01-25 14:00:00 UTC 3. 2026-01-26 14:00:00 UTC 4. 2026-01-27 14:00:00 UTC 5. 2026-01-28 14:00:00 UTC ``` ## Error Handling Robust error handling ensures your workflows are resilient and reliable. ### Retry Policies Configure automatic retries for failed actions: ```yaml - id: api_call action: http_request config: url: "https://api....

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_fix_7, score: 0.250]
   > Waiting {retry_after} seconds...") time.sleep(retry_after) continue remaining = int(response.headers.get('X-RateLimit-Remaining', 0)) if remaining < 10: print(f"Warning: Only {remaining} requests remaining") return response raise Exception("Max retries exceeded due to rate limiting") ``` **Bash script with rate limit checking:** ```bash #!/bin/bash check_rate_limit() { local remaining=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \ -H "Authorization: Bearer $CF_ACCESS_TOKEN" | \ grep -i...

5. [doc_id: user_guide, chunk_id: user_guide_fix_6, score: 0.200]
   > Filter by error type, date range, or execution ID 3. Click an execution to view full details 4. Click **"Retry"** to reprocess with the same input data ### Error Notifications Get notified when workflows fail: ```yaml workflow: notifications: on_failure: - type: email to: "ops-team@company.com" - type: slack channel: "#alerts" message: "Workflow {{workflow.name}} failed: {{error.message}}" on_success_after_retry: - type: slack channel: "#monitoring" message: "Workflow recovered after {{error.att...

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---
