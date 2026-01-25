# Manual Testing Report

Generated: 2026-01-25 17:50:58
Strategy: enriched_hybrid_llm
Questions: 10

## Instructions

For each question below, manually grade the retrieved chunks:

1. Read the **QUESTION**
2. Read the **EXPECTED ANSWER**
3. Review the **RETRIEVED CHUNKS**
4. Assign a score 1-10 using the rubric below

### Grading Rubric


GRADING RUBRIC (1-10 Scale)

10: Perfect
    - All requested information retrieved
    - Directly answers the question
    - No irrelevant content

9: Excellent
    - Complete answer provided
    - Minor irrelevant content present
    - Core facts are clear and accurate

8: Very Good
    - Answer is present but buried in some noise
    - Requires some parsing to extract the answer
    - Supporting details may be incomplete

7: Good
    - Core answer is present
    - Missing some supporting details
    - Requires inference to fully answer the question

6: Adequate
    - Partial answer provided
    - Enough information to be useful
    - Significant gaps in coverage

5: Borderline
    - Some relevant information present
    - Misses key point or central fact
    - Requires substantial additional context

4: Poor
    - Tangentially related content only
    - Does not directly address the question
    - Requires significant interpretation

3: Very Poor
    - Mostly irrelevant content
    - Hint of the topic present
    - Unlikely to help answer the question

2: Bad
    - Almost entirely irrelevant
    - Minimal connection to the question
    - Misleading or confusing

1: Failed
    - No relevant content retrieved
    - Completely off-topic
    - Useless for answering the question


## Questions

### Question 1: How many requests per minute are allowed in the CloudFlow API?

**Expected Answer**: 100 requests per minute per authenticated user, 20 requests per minute for unauthenticated requests, with a burst allowance of 150 requests in a 10-second window

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[api_reference]**

   ## Rate Limiting

To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.

**Default Limits:**
- 100 requests per minute per authenticated user
- 20 requests per minute for unauthenticated requests
- Burst allowance: 150 requests in a 10-second window

### Rate Limit Headers

Every API response includes rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1640995200
```

When you exceed the rate limit, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please retry after 42 seconds.",
    "retry_after": 42
  }
}
```

**Best Practices:**
- Monitor `X-RateLimit-Remaining` header values
- Implement exponential backoff when receiving 429 responses
- Cache responses when appropriate to reduce API calls
- Consider upgrading to Enterprise tier for higher limits (1000 req/min)

## Pagination

List endpoints return paginated results to optimize performance. Use `limit` and `offset` parameters to navigate through result sets.

**Parameters:**
- `limit`: Number of items per page (default: 20, max: 100)
- `offset`: Number of items to skip (default: 0)

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://api.cloudflow.io/v2/workflows?limit=50&offset=100"
```

**Response Structure:**

```json
{
  "data": [...],
  "pagination": {
    "total": 347,
    "limit": 50,
    "offset": 100,
    "has_more": true
  }
}
```

**Python Example:**

```python
import requests

def fetch_all_workflows(api_key):
    base_url = "https://api.cloudflow.io/v2/workflows"
    headers = {"X-API-Key": api_key}
    all_workflows = []
    offset = 0
    limit = 100
    
    while True:
        response = requests.get(
            base_url,
            headers=headers,
            params={"limit": limit, "offset": offset}
        )
        data = response.json()
        all_workflows.extend(data['data'])
        
        if not data['pagination']['has_more']:
            break
        
        offset += limit
    
    return all_workflows
```

## Endpoints



### Workflows



#### List Workflows

Retrieve a paginated list of all workflows in your account.

**Endpoint:** `GET /workflows`

**Query Parameters:**
- `status` (optional): Filter by status (`active`, `paused`, `archived`)
- `created_after` (optional): ISO 8601 timestamp
- `limit` (optional): Items per page (max 100)
- `offset` (optional): Pagination offset

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://api.cloudflow.io/v2/workflows?status=active&limit=25"
```

**Example Response (200 OK):**

```json
{
  "data": [
    {
      "id": "wf_8x7k2m9p",
      "name": "Data Processing Pipeline",
      "description": "Processes customer data every hour",
      "status": "active",
      "trigger": {
        "type": "schedule",
        "cron": "0 * * * *"
      },
      "steps": 12,
      "created_at": "2026-01-15T10:30:00Z",
      "updated_at": "2026-01-23T14:22:00Z",
      "last_run": "2026-01-24T09:00:00Z"
    }
  ],
  "pagination": {
    "total": 1,
    "limit": 25,
    "offset": 0,
    "has_more": false
  }
}
```

#### Create Workflow

Create a new workflow with specified configuration.

**Endpoint:** `POST /workflows`

**Request Body:**

```json
{
  "name": "Email Campaign Automation",
  "description": "Sends personalized emails based on user behavior",
  "trigger": {
    "type": "webhook",
    "url": "https://your-app.com/webhook"
  },
  "steps": [
    {
      "type": "fetch_data",
      "source": "users_table",
      "filters": {"active": true}
    },
    {
      "type": "transform",
      "script": "data.map(user => ({ ...user, segment: calculateSegment(user) }))"
    },
    {
      "type": "send_email",
      "template_id": "tpl_welcome_v2"
    }
  ]
}
```

**Example Request:**

```bash
curl -X POST https://api.cloudflow.io/v2/workflows \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Email Campaign Automation",
    "trigger": {"type": "webhook"},
    "steps": [{"type": "fetch_data", "source": "users_table"}]
  }'
```

**Example Response (201 Created):**

```json
{
  "id": "wf_9k3m7n2q",
  "name": "Email Campaign Automation",
  "status": "active",
  "created_at": "2026-01-24T10:15:00Z",
  "webhook_url": "https://api.cloudflow.io/v2/webhooks/wf_9k3m7n2q/trigger"
}
```

#### Get Workflow Details

Retrieve detailed information about a specific workflow.

**Endpoint:** `GET /workflows/{workflow_id}`

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p
```

**Example Response (200 OK):**

```json
{
  "id": "wf_8x7k2m9p",
  "name": "Data Processing Pipeline",
  "description": "Processes customer data every hour",
  "status": "active",
  "trigger": {
    "type": "schedule",
    "cron": "0 * * * *",
    "timezone": "UTC"
  },
  "steps": [
    {
      "id": "step_1",
      "type": "fetch_data",
      "source": "customer_db",
      "query": "SELECT * FROM customers WHERE updated_at > :last_run"
    },
    {
      "id": "step_2",
      "type": "transform",
      "script_id": "scr_transform_v3"
    }
  ],
  "metrics": {
    "total_runs": 1247,
    "success_rate": 98.7,
    "avg_duration_ms": 3420,
    "last_error": null
  },
  "created_at": "2026-01-15T10:30:00Z",
  "updated_at": "2026-01-23T14:22:00Z"
}
```

#### Update Workflow

Modify an existing workflow's configuration.

**Endpoint:** `PATCH /workflows/{workflow_id}`

**Example Request:**

```bash
curl -X PATCH https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"status": "paused"}'
```

**Example Response (200 OK):**

```json
{
  "id": "wf_8x7k2m9p",
  "status": "paused",
  "updated_at": "2026-01-24T10:30:00Z"
}
```

#### Delete Workflow

Permanently delete a workflow.

**Endpoint:** `DELETE /workflows/{workflow_id}`

**Example Request:**

```bash
curl -X DELETE https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p \
  -H "X-API-Key: YOUR_API_KEY"
```

**Example Response (204 No Content)**

2. **[troubleshooting_guide]**

   # Configure error handling

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_validation \
  --on-error continue \
  --error-threshold 5%  # Fail if > 5% of records invalid
```

**2. External API Failures**
```
ExternalAPIError: API request to https://partner-api.example.com failed with status 502
```

**Resolution:**
```bash

# Add circuit breaker

cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --circuit-breaker-enabled true \
  --circuit-breaker-threshold 5 \
  --circuit-breaker-timeout 30000

# Configure fallback behavior

cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --fallback-action use_cached_data \
  --cache-ttl 3600
```

---

## Rate Limiting & Throttling



### 429 Too Many Requests



#### Error Response

```json
{
  "error": "rate_limit_exceeded",
  "message": "API rate limit exceeded. Retry after 45 seconds.",
  "status": 429,
  "headers": {
    "X-RateLimit-Limit": "1000",
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1706112345",
    "Retry-After": "45"
  }
}
```

3. **[troubleshooting_guide]**

   #### Rate Limit Tiers

CloudFlow enforces the following rate limits per workspace:

\| Tier \| Requests/Minute \| Requests/Hour \| Concurrent Workflows \|
\|------\|-----------------\|---------------\|----------------------\|
\| Free \| 60 \| 1,000 \| 5 \|
\| Standard \| 1,000 \| 50,000 \| 50 \|
\| Premium \| 5,000 \| 250,000 \| 200 \|
\| Enterprise \| Custom \| Custom \| Unlimited \|

#### Checking Rate Limit Status

```bash

# Check current rate limit status

curl -I https://api.cloudflow.io/api/v1/workflows \
  -H "Authorization: Bearer $CF_ACCESS_TOKEN"

# Extract rate limit headers

curl -s -I https://api.cloudflow.io/api/v1/workflows \
  -H "Authorization: Bearer $CF_ACCESS_TOKEN" \| \
  grep -i "x-ratelimit"

# Output:



# X-RateLimit-Limit: 1000



# X-RateLimit-Remaining: 247



# X-RateLimit-Reset: 1706112400



# Monitor rate limit usage

cloudflow metrics query \
  --metric rate_limit_remaining \
  --workspace ws_abc123 \
  --last 1h \
  --interval 1m
```

4. **[troubleshooting_guide]**

   #### Handling Rate Limits in Code

**Python example with retry logic:**
```python
import time
import requests

def cloudflow_api_call_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        if remaining < 10:
            print(f"Warning: Only {remaining} requests remaining")
            
        return response
    
    raise Exception("Max retries exceeded due to rate limiting")
```

**Bash script with rate limit checking:**
```bash
#!/bin/bash

check_rate_limit() {
    local remaining=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \
        -H "Authorization: Bearer $CF_ACCESS_TOKEN" \| \
        grep -i "x-ratelimit-remaining" \| \
        awk '{print $2}' \| tr -d '\r')
    
    if [ "$remaining" -lt 10 ]; then
        echo "Warning: Only $remaining requests remaining"
        local reset=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \
            -H "Authorization: Bearer $CF_ACCESS_TOKEN" \| \
            grep -i "x-ratelimit-reset" \| \
            awk '{print $2}' \| tr -d '\r')
        local wait_time=$((reset - $(date +%s)))
        echo "Rate limit resets in $wait_time seconds"
        sleep $wait_time
    fi
}

# Use before API calls

check_rate_limit
cloudflow workflows execute wf_9k2n4m8p1q
```

#### Optimization Strategies

**1. Implement request batching:**
```bash

# Batch multiple workflow executions

cloudflow workflows execute-batch \
  --workflow-ids "wf_id1,wf_id2,wf_id3,wf_id4,wf_id5" \
  --batch-size 5

# This counts as 1 API request instead of 5

```

**2. Use webhooks instead of polling:**
```bash

# Configure webhook for workflow completion

cloudflow webhooks create \
  --event workflow.completed \
  --url https://your-service.com/webhooks/cloudflow \
  --secret $WEBHOOK_SECRET

# Verify webhook

cloudflow webhooks test webhook_abc123
```

**3. Leverage caching:**
```bash

# Enable client-side caching

export CLOUDFLOW_CACHE_ENABLED=true
export CLOUDFLOW_CACHE_TTL=300

# Cache workflow metadata

cloudflow workflows list --use-cache --cache-ttl 600
```

---

## Log Analysis & Debugging



### Accessing CloudFlow Logs



#### Kubernetes Deployments

```bash

# List all CloudFlow pods

kubectl get pods -n cloudflow

# Tail logs from API server

kubectl logs -f -n cloudflow deployment/cloudflow-api --tail=100

# Get logs from specific pod

kubectl logs -n cloudflow cloudflow-api-7d4f6b8c9d-x7k2m

# Get logs from previous crashed pod

kubectl logs -n cloudflow cloudflow-api-7d4f6b8c9d-x7k2m --previous

# Get logs from all pods in deployment

kubectl logs -n cloudflow deployment/cloudflow-api --all-containers=true

# Stream logs from multiple pods

kubectl logs -n cloudflow -l app=cloudflow-api -f --max-log-requests=10
```

#### Log Levels

CloudFlow supports the following log levels:
- `TRACE` - Very detailed debugging information
- `DEBUG` - Detailed debugging information
- `INFO` - Informational messages (default)
- `WARN` - Warning messages
- `ERROR` - Error messages
- `FATAL` - Fatal errors causing shutdown

**Changing log levels:**
```bash

# Set global log level

cloudflow config set logging.level DEBUG

# Set log level for specific component

cloudflow config set logging.components.database DEBUG
cloudflow config set logging.components.auth INFO
cloudflow config set logging.components.workflows TRACE

# Temporary log level increase (resets after 1 hour)

cloudflow debug set-log-level DEBUG --duration 3600

# View current log configuration

cloudflow config get logging --format json
```

### Grep Patterns for Common Issues



#### Authentication Failures

```bash

# Find all authentication errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "authentication\\|401\\|unauthorized" \| \
  tail -n 50

# Find JWT validation failures

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "JWT\|token.*invalid\|signature.*failed"

# Find clock skew issues

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "clock skew\\|time.*sync\\|nbf\\|exp"
```

#### Database Errors

```bash

# Find database connection errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "connection.*pool\|could not connect\|database.*timeout"

# Find slow queries

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep "slow query" \| \
  awk '{print $NF}' \| \
  sort -n \| \
  tail -n 20

# Find deadlock errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "deadlock detected"
```

#### Workflow Execution Errors

```bash

# Find workflow timeout errors

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep -E "timeout\|exceeded.*3600"

# Find workflow retry attempts

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep -E "retry attempt [0-9]\|retrying in"

# Find workflow failures by ID

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep "exec_7h3j6k9m2n"
```

#### Rate Limiting

```bash

# Find rate limit events

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "429\|rate.*limit\|throttle"

# Count rate limit errors by hour

kubectl logs -n cloudflow deployment/cloudflow-api --since=24h \| \
  grep "rate_limit_exceeded" \| \
  awk '{print $1}' \| \
  cut -d'T' -f1-2 \| \
  sort \| uniq -c
```

### Advanced Log Analysis



#### Using jq for JSON Logs

```bash

# Parse JSON logs and filter by level

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  jq 'select(.level == "ERROR")'

# Extract specific fields

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  jq '{timestamp: .timestamp, level: .level, message: .message, execution_id: .context.execution_id}'

# Filter by workflow ID

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  jq 'select(.workflow_id == "wf_9k2n4m8p1q")'

# Count errors by type

kubectl logs -n cloudflow deployment/cloudflow-api --since=1h \| \
  jq -r 'select(.level == "ERROR") \| .error_type' \| \
  sort \| uniq -c \| sort -rn
```

#### Correlation IDs

CloudFlow uses correlation IDs to trace requests across services.

```bash

# Extract correlation ID from error

CORRELATION_ID="corr_8h4j9k2m5n"

# Trace request across all services

for pod in $(kubectl get pods -n cloudflow -l tier=backend -o name); do
  echo "=== $pod ==="
  kubectl logs -n cloudflow $pod \| grep $CORRELATION_ID
done

# Export full trace to file

cloudflow debug trace $CORRELATION_ID --output trace-$CORRELATION_ID.json
```

### Debugging Commands



#### Enable Debug Mode for Workflow Execution

```bash

# Execute workflow with debug logging

cloudflow workflows execute wf_9k2n4m8p1q \
  --debug \
  --log-level TRACE \
  --output-logs /tmp/workflow-debug.log

# Enable step-by-step execution

cloudflow workflows execute wf_9k2n4m8p1q \
  --step-mode interactive \
  --breakpoint-on-error

# Capture full execution context

cloudflow workflows execute wf_9k2n4m8p1q \
  --capture-context \
  --context-output /tmp/execution-context.json
```

#### Database Query Debugging

```bash

# Enable query logging

cloudflow db config set log_statement all
cloudflow db config set log_duration on
cloudflow db config set log_min_duration_statement 1000  # Log queries > 1s

# Capture query plan for slow endpoint

cloudflow debug capture-queries \
  --endpoint "/api/v1/workflows/list" \
  --duration 60 \
  --output query-analysis.txt

# Analyze query performance

cloudflow db analyze-performance --last 1h
```

#### Network Debugging

```bash

# Test connectivity from CloudFlow pod

kubectl run -n cloudflow netdebug --rm -i --tty \
  --image=nicolaka/netshoot -- /bin/bash

# Inside pod:



# Check DNS resolution

nslookup api.cloudflow.io

# Check connectivity to database

nc -zv cloudflow-db.internal.company.com 5432

# Trace route

traceroute api.cloudflow.io

# Capture packets

tcpdump -i any -w /tmp/capture.pcap port 5432
```

---

## Escalation Procedures



### Severity Levels

CloudFlow incidents are classified into four severity levels:

5. **[api_reference]**

   ### Error Codes

CloudFlow returns specific error codes to help you identify and resolve issues:

- `invalid_parameter`: One or more request parameters are invalid
- `missing_required_field`: Required field is missing from request body
- `authentication_failed`: Invalid API key or token
- `insufficient_permissions`: User lacks required scope or permission
- `resource_not_found`: Requested resource does not exist
- `rate_limit_exceeded`: Too many requests, see rate limiting section
- `workflow_execution_failed`: Workflow execution encountered an error
- `invalid_json`: Request body contains malformed JSON
- `duplicate_resource`: Resource with same identifier already exists
- `quota_exceeded`: Account quota limit reached

### Error Handling Best Practices

**Python Example:**

```python
import requests
import time

def make_api_request(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(response.headers.get('X-RateLimit-Reset', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            elif response.status_code >= 500:
                # Server error - retry with exponential backoff
                wait_time = 2 ** attempt
                print(f"Server error. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            else:
                # Client error - don't retry
                error_data = response.json()
                raise Exception(f"API Error: {error_data['error']['message']}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception("Max retries exceeded")
```

## Webhooks

CloudFlow can send webhook notifications when specific events occur in your workflows.

**Supported Events:**
- `workflow.started`
- `workflow.completed`
- `workflow.failed`
- `pipeline.completed`
- `pipeline.failed`

**Webhook Payload Example:**

```json
{
  "event": "workflow.completed",
  "timestamp": "2026-01-24T10:45:00Z",
  "data": {
    "workflow_id": "wf_8x7k2m9p",
    "execution_id": "exec_9k3m7n2q",
    "status": "completed",
    "duration_ms": 3420,
    "records_processed": 1247
  }
}
```

Configure webhooks in your account settings or via the API:

```bash
curl -X POST https://api.cloudflow.io/v2/webhooks \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/cloudflow-webhook",
    "events": ["workflow.completed", "workflow.failed"],
    "secret": "whsec_your_webhook_secret"
  }'
```

## Support

For additional help and resources:

- **Documentation:** https://docs.cloudflow.io
- **API Status:** https://status.cloudflow.io
- **Support Email:** support@cloudflow.io
- **Community Forum:** https://community.cloudflow.io

Enterprise customers have access to 24/7 priority support via phone and dedicated Slack channels.

---

### Question 2: What happens when I exceed the CloudFlow API rate limit?

**Expected Answer**: When you exceed the rate limit, you'll receive a `429 Too Many Requests` response with an error message indicating how long to wait before retrying

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[api_reference]**

   ## Rate Limiting

To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.

**Default Limits:**
- 100 requests per minute per authenticated user
- 20 requests per minute for unauthenticated requests
- Burst allowance: 150 requests in a 10-second window

### Rate Limit Headers

Every API response includes rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1640995200
```

When you exceed the rate limit, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please retry after 42 seconds.",
    "retry_after": 42
  }
}
```

**Best Practices:**
- Monitor `X-RateLimit-Remaining` header values
- Implement exponential backoff when receiving 429 responses
- Cache responses when appropriate to reduce API calls
- Consider upgrading to Enterprise tier for higher limits (1000 req/min)

## Pagination

List endpoints return paginated results to optimize performance. Use `limit` and `offset` parameters to navigate through result sets.

**Parameters:**
- `limit`: Number of items per page (default: 20, max: 100)
- `offset`: Number of items to skip (default: 0)

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://api.cloudflow.io/v2/workflows?limit=50&offset=100"
```

**Response Structure:**

```json
{
  "data": [...],
  "pagination": {
    "total": 347,
    "limit": 50,
    "offset": 100,
    "has_more": true
  }
}
```

**Python Example:**

```python
import requests

def fetch_all_workflows(api_key):
    base_url = "https://api.cloudflow.io/v2/workflows"
    headers = {"X-API-Key": api_key}
    all_workflows = []
    offset = 0
    limit = 100
    
    while True:
        response = requests.get(
            base_url,
            headers=headers,
            params={"limit": limit, "offset": offset}
        )
        data = response.json()
        all_workflows.extend(data['data'])
        
        if not data['pagination']['has_more']:
            break
        
        offset += limit
    
    return all_workflows
```

## Endpoints



### Workflows



#### List Workflows

Retrieve a paginated list of all workflows in your account.

**Endpoint:** `GET /workflows`

**Query Parameters:**
- `status` (optional): Filter by status (`active`, `paused`, `archived`)
- `created_after` (optional): ISO 8601 timestamp
- `limit` (optional): Items per page (max 100)
- `offset` (optional): Pagination offset

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://api.cloudflow.io/v2/workflows?status=active&limit=25"
```

**Example Response (200 OK):**

```json
{
  "data": [
    {
      "id": "wf_8x7k2m9p",
      "name": "Data Processing Pipeline",
      "description": "Processes customer data every hour",
      "status": "active",
      "trigger": {
        "type": "schedule",
        "cron": "0 * * * *"
      },
      "steps": 12,
      "created_at": "2026-01-15T10:30:00Z",
      "updated_at": "2026-01-23T14:22:00Z",
      "last_run": "2026-01-24T09:00:00Z"
    }
  ],
  "pagination": {
    "total": 1,
    "limit": 25,
    "offset": 0,
    "has_more": false
  }
}
```

#### Create Workflow

Create a new workflow with specified configuration.

**Endpoint:** `POST /workflows`

**Request Body:**

```json
{
  "name": "Email Campaign Automation",
  "description": "Sends personalized emails based on user behavior",
  "trigger": {
    "type": "webhook",
    "url": "https://your-app.com/webhook"
  },
  "steps": [
    {
      "type": "fetch_data",
      "source": "users_table",
      "filters": {"active": true}
    },
    {
      "type": "transform",
      "script": "data.map(user => ({ ...user, segment: calculateSegment(user) }))"
    },
    {
      "type": "send_email",
      "template_id": "tpl_welcome_v2"
    }
  ]
}
```

**Example Request:**

```bash
curl -X POST https://api.cloudflow.io/v2/workflows \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Email Campaign Automation",
    "trigger": {"type": "webhook"},
    "steps": [{"type": "fetch_data", "source": "users_table"}]
  }'
```

**Example Response (201 Created):**

```json
{
  "id": "wf_9k3m7n2q",
  "name": "Email Campaign Automation",
  "status": "active",
  "created_at": "2026-01-24T10:15:00Z",
  "webhook_url": "https://api.cloudflow.io/v2/webhooks/wf_9k3m7n2q/trigger"
}
```

#### Get Workflow Details

Retrieve detailed information about a specific workflow.

**Endpoint:** `GET /workflows/{workflow_id}`

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p
```

**Example Response (200 OK):**

```json
{
  "id": "wf_8x7k2m9p",
  "name": "Data Processing Pipeline",
  "description": "Processes customer data every hour",
  "status": "active",
  "trigger": {
    "type": "schedule",
    "cron": "0 * * * *",
    "timezone": "UTC"
  },
  "steps": [
    {
      "id": "step_1",
      "type": "fetch_data",
      "source": "customer_db",
      "query": "SELECT * FROM customers WHERE updated_at > :last_run"
    },
    {
      "id": "step_2",
      "type": "transform",
      "script_id": "scr_transform_v3"
    }
  ],
  "metrics": {
    "total_runs": 1247,
    "success_rate": 98.7,
    "avg_duration_ms": 3420,
    "last_error": null
  },
  "created_at": "2026-01-15T10:30:00Z",
  "updated_at": "2026-01-23T14:22:00Z"
}
```

#### Update Workflow

Modify an existing workflow's configuration.

**Endpoint:** `PATCH /workflows/{workflow_id}`

**Example Request:**

```bash
curl -X PATCH https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"status": "paused"}'
```

**Example Response (200 OK):**

```json
{
  "id": "wf_8x7k2m9p",
  "status": "paused",
  "updated_at": "2026-01-24T10:30:00Z"
}
```

#### Delete Workflow

Permanently delete a workflow.

**Endpoint:** `DELETE /workflows/{workflow_id}`

**Example Request:**

```bash
curl -X DELETE https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p \
  -H "X-API-Key: YOUR_API_KEY"
```

**Example Response (204 No Content)**

2. **[troubleshooting_guide]**

   # Configure error handling

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_validation \
  --on-error continue \
  --error-threshold 5%  # Fail if > 5% of records invalid
```

**2. External API Failures**
```
ExternalAPIError: API request to https://partner-api.example.com failed with status 502
```

**Resolution:**
```bash

# Add circuit breaker

cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --circuit-breaker-enabled true \
  --circuit-breaker-threshold 5 \
  --circuit-breaker-timeout 30000

# Configure fallback behavior

cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --fallback-action use_cached_data \
  --cache-ttl 3600
```

---

## Rate Limiting & Throttling



### 429 Too Many Requests



#### Error Response

```json
{
  "error": "rate_limit_exceeded",
  "message": "API rate limit exceeded. Retry after 45 seconds.",
  "status": 429,
  "headers": {
    "X-RateLimit-Limit": "1000",
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1706112345",
    "Retry-After": "45"
  }
}
```

3. **[api_reference]**

   ### Error Codes

CloudFlow returns specific error codes to help you identify and resolve issues:

- `invalid_parameter`: One or more request parameters are invalid
- `missing_required_field`: Required field is missing from request body
- `authentication_failed`: Invalid API key or token
- `insufficient_permissions`: User lacks required scope or permission
- `resource_not_found`: Requested resource does not exist
- `rate_limit_exceeded`: Too many requests, see rate limiting section
- `workflow_execution_failed`: Workflow execution encountered an error
- `invalid_json`: Request body contains malformed JSON
- `duplicate_resource`: Resource with same identifier already exists
- `quota_exceeded`: Account quota limit reached

### Error Handling Best Practices

**Python Example:**

```python
import requests
import time

def make_api_request(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(response.headers.get('X-RateLimit-Reset', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            elif response.status_code >= 500:
                # Server error - retry with exponential backoff
                wait_time = 2 ** attempt
                print(f"Server error. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            else:
                # Client error - don't retry
                error_data = response.json()
                raise Exception(f"API Error: {error_data['error']['message']}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception("Max retries exceeded")
```

## Webhooks

CloudFlow can send webhook notifications when specific events occur in your workflows.

**Supported Events:**
- `workflow.started`
- `workflow.completed`
- `workflow.failed`
- `pipeline.completed`
- `pipeline.failed`

**Webhook Payload Example:**

```json
{
  "event": "workflow.completed",
  "timestamp": "2026-01-24T10:45:00Z",
  "data": {
    "workflow_id": "wf_8x7k2m9p",
    "execution_id": "exec_9k3m7n2q",
    "status": "completed",
    "duration_ms": 3420,
    "records_processed": 1247
  }
}
```

Configure webhooks in your account settings or via the API:

```bash
curl -X POST https://api.cloudflow.io/v2/webhooks \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/cloudflow-webhook",
    "events": ["workflow.completed", "workflow.failed"],
    "secret": "whsec_your_webhook_secret"
  }'
```

## Support

For additional help and resources:

- **Documentation:** https://docs.cloudflow.io
- **API Status:** https://status.cloudflow.io
- **Support Email:** support@cloudflow.io
- **Community Forum:** https://community.cloudflow.io

Enterprise customers have access to 24/7 priority support via phone and dedicated Slack channels.

4. **[troubleshooting_guide]**

   #### Handling Rate Limits in Code

**Python example with retry logic:**
```python
import time
import requests

def cloudflow_api_call_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        if remaining < 10:
            print(f"Warning: Only {remaining} requests remaining")
            
        return response
    
    raise Exception("Max retries exceeded due to rate limiting")
```

**Bash script with rate limit checking:**
```bash
#!/bin/bash

check_rate_limit() {
    local remaining=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \
        -H "Authorization: Bearer $CF_ACCESS_TOKEN" \| \
        grep -i "x-ratelimit-remaining" \| \
        awk '{print $2}' \| tr -d '\r')
    
    if [ "$remaining" -lt 10 ]; then
        echo "Warning: Only $remaining requests remaining"
        local reset=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \
            -H "Authorization: Bearer $CF_ACCESS_TOKEN" \| \
            grep -i "x-ratelimit-reset" \| \
            awk '{print $2}' \| tr -d '\r')
        local wait_time=$((reset - $(date +%s)))
        echo "Rate limit resets in $wait_time seconds"
        sleep $wait_time
    fi
}

# Use before API calls

check_rate_limit
cloudflow workflows execute wf_9k2n4m8p1q
```

#### Optimization Strategies

**1. Implement request batching:**
```bash

# Batch multiple workflow executions

cloudflow workflows execute-batch \
  --workflow-ids "wf_id1,wf_id2,wf_id3,wf_id4,wf_id5" \
  --batch-size 5

# This counts as 1 API request instead of 5

```

**2. Use webhooks instead of polling:**
```bash

# Configure webhook for workflow completion

cloudflow webhooks create \
  --event workflow.completed \
  --url https://your-service.com/webhooks/cloudflow \
  --secret $WEBHOOK_SECRET

# Verify webhook

cloudflow webhooks test webhook_abc123
```

**3. Leverage caching:**
```bash

# Enable client-side caching

export CLOUDFLOW_CACHE_ENABLED=true
export CLOUDFLOW_CACHE_TTL=300

# Cache workflow metadata

cloudflow workflows list --use-cache --cache-ttl 600
```

---

## Log Analysis & Debugging



### Accessing CloudFlow Logs



#### Kubernetes Deployments

```bash

# List all CloudFlow pods

kubectl get pods -n cloudflow

# Tail logs from API server

kubectl logs -f -n cloudflow deployment/cloudflow-api --tail=100

# Get logs from specific pod

kubectl logs -n cloudflow cloudflow-api-7d4f6b8c9d-x7k2m

# Get logs from previous crashed pod

kubectl logs -n cloudflow cloudflow-api-7d4f6b8c9d-x7k2m --previous

# Get logs from all pods in deployment

kubectl logs -n cloudflow deployment/cloudflow-api --all-containers=true

# Stream logs from multiple pods

kubectl logs -n cloudflow -l app=cloudflow-api -f --max-log-requests=10
```

#### Log Levels

CloudFlow supports the following log levels:
- `TRACE` - Very detailed debugging information
- `DEBUG` - Detailed debugging information
- `INFO` - Informational messages (default)
- `WARN` - Warning messages
- `ERROR` - Error messages
- `FATAL` - Fatal errors causing shutdown

**Changing log levels:**
```bash

# Set global log level

cloudflow config set logging.level DEBUG

# Set log level for specific component

cloudflow config set logging.components.database DEBUG
cloudflow config set logging.components.auth INFO
cloudflow config set logging.components.workflows TRACE

# Temporary log level increase (resets after 1 hour)

cloudflow debug set-log-level DEBUG --duration 3600

# View current log configuration

cloudflow config get logging --format json
```

### Grep Patterns for Common Issues



#### Authentication Failures

```bash

# Find all authentication errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "authentication\\|401\\|unauthorized" \| \
  tail -n 50

# Find JWT validation failures

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "JWT\|token.*invalid\|signature.*failed"

# Find clock skew issues

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "clock skew\\|time.*sync\\|nbf\\|exp"
```

#### Database Errors

```bash

# Find database connection errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "connection.*pool\|could not connect\|database.*timeout"

# Find slow queries

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep "slow query" \| \
  awk '{print $NF}' \| \
  sort -n \| \
  tail -n 20

# Find deadlock errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "deadlock detected"
```

#### Workflow Execution Errors

```bash

# Find workflow timeout errors

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep -E "timeout\|exceeded.*3600"

# Find workflow retry attempts

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep -E "retry attempt [0-9]\|retrying in"

# Find workflow failures by ID

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep "exec_7h3j6k9m2n"
```

#### Rate Limiting

```bash

# Find rate limit events

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "429\|rate.*limit\|throttle"

# Count rate limit errors by hour

kubectl logs -n cloudflow deployment/cloudflow-api --since=24h \| \
  grep "rate_limit_exceeded" \| \
  awk '{print $1}' \| \
  cut -d'T' -f1-2 \| \
  sort \| uniq -c
```

### Advanced Log Analysis



#### Using jq for JSON Logs

```bash

# Parse JSON logs and filter by level

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  jq 'select(.level == "ERROR")'

# Extract specific fields

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  jq '{timestamp: .timestamp, level: .level, message: .message, execution_id: .context.execution_id}'

# Filter by workflow ID

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  jq 'select(.workflow_id == "wf_9k2n4m8p1q")'

# Count errors by type

kubectl logs -n cloudflow deployment/cloudflow-api --since=1h \| \
  jq -r 'select(.level == "ERROR") \| .error_type' \| \
  sort \| uniq -c \| sort -rn
```

#### Correlation IDs

CloudFlow uses correlation IDs to trace requests across services.

```bash

# Extract correlation ID from error

CORRELATION_ID="corr_8h4j9k2m5n"

# Trace request across all services

for pod in $(kubectl get pods -n cloudflow -l tier=backend -o name); do
  echo "=== $pod ==="
  kubectl logs -n cloudflow $pod \| grep $CORRELATION_ID
done

# Export full trace to file

cloudflow debug trace $CORRELATION_ID --output trace-$CORRELATION_ID.json
```

### Debugging Commands



#### Enable Debug Mode for Workflow Execution

```bash

# Execute workflow with debug logging

cloudflow workflows execute wf_9k2n4m8p1q \
  --debug \
  --log-level TRACE \
  --output-logs /tmp/workflow-debug.log

# Enable step-by-step execution

cloudflow workflows execute wf_9k2n4m8p1q \
  --step-mode interactive \
  --breakpoint-on-error

# Capture full execution context

cloudflow workflows execute wf_9k2n4m8p1q \
  --capture-context \
  --context-output /tmp/execution-context.json
```

#### Database Query Debugging

```bash

# Enable query logging

cloudflow db config set log_statement all
cloudflow db config set log_duration on
cloudflow db config set log_min_duration_statement 1000  # Log queries > 1s

# Capture query plan for slow endpoint

cloudflow debug capture-queries \
  --endpoint "/api/v1/workflows/list" \
  --duration 60 \
  --output query-analysis.txt

# Analyze query performance

cloudflow db analyze-performance --last 1h
```

#### Network Debugging

```bash

# Test connectivity from CloudFlow pod

kubectl run -n cloudflow netdebug --rm -i --tty \
  --image=nicolaka/netshoot -- /bin/bash

# Inside pod:



# Check DNS resolution

nslookup api.cloudflow.io

# Check connectivity to database

nc -zv cloudflow-db.internal.company.com 5432

# Trace route

traceroute api.cloudflow.io

# Capture packets

tcpdump -i any -w /tmp/capture.pcap port 5432
```

---

## Escalation Procedures



### Severity Levels

CloudFlow incidents are classified into four severity levels:

5. **[troubleshooting_guide]**

   #### RBAC Policy Violations

CloudFlow uses role-based access control (RBAC) with the following hierarchy:
- `viewer` - Read-only access
- `developer` - Create and modify workflows (non-production)
- `operator` - Execute workflows, view logs
- `admin` - Full access to workspace
- `platform-admin` - Cross-workspace administration

**Verify resource permissions:**
```bash

# Check effective permissions

cloudflow rbac check \
  --user john.doe@company.com \
  --resource workflow:prod-pipeline \
  --action execute

# List all policies affecting user

cloudflow rbac policies --user john.doe@company.com --verbose
```

---

## Performance Problems



### Slow Query Performance



#### Symptoms

- API response times > 5000ms
- Database query latency warnings in logs
- CloudFlow UI becomes unresponsive
- Timeout errors: `Request exceeded maximum duration of 30000ms`

#### Diagnosis Steps

**1. Identify Slow Queries**

```bash

# Query CloudFlow metrics

cloudflow metrics query --metric api_request_duration_ms \
  --filter "p95 > 5000" \
  --last 1h

---

### Question 3: How do I handle rate limit errors in my code?

**Expected Answer**: Implement exponential backoff, monitor `X-RateLimit-Remaining` header values, cache responses when appropriate, and consider upgrading to Enterprise tier for higher limits

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[troubleshooting_guide]**

   # Configure error handling

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_validation \
  --on-error continue \
  --error-threshold 5%  # Fail if > 5% of records invalid
```

**2. External API Failures**
```
ExternalAPIError: API request to https://partner-api.example.com failed with status 502
```

**Resolution:**
```bash

# Add circuit breaker

cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --circuit-breaker-enabled true \
  --circuit-breaker-threshold 5 \
  --circuit-breaker-timeout 30000

# Configure fallback behavior

cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --fallback-action use_cached_data \
  --cache-ttl 3600
```

---

## Rate Limiting & Throttling



### 429 Too Many Requests



#### Error Response

```json
{
  "error": "rate_limit_exceeded",
  "message": "API rate limit exceeded. Retry after 45 seconds.",
  "status": 429,
  "headers": {
    "X-RateLimit-Limit": "1000",
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1706112345",
    "Retry-After": "45"
  }
}
```

2. **[troubleshooting_guide]**

   #### Handling Rate Limits in Code

**Python example with retry logic:**
```python
import time
import requests

def cloudflow_api_call_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        if remaining < 10:
            print(f"Warning: Only {remaining} requests remaining")
            
        return response
    
    raise Exception("Max retries exceeded due to rate limiting")
```

**Bash script with rate limit checking:**
```bash
#!/bin/bash

check_rate_limit() {
    local remaining=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \
        -H "Authorization: Bearer $CF_ACCESS_TOKEN" \| \
        grep -i "x-ratelimit-remaining" \| \
        awk '{print $2}' \| tr -d '\r')
    
    if [ "$remaining" -lt 10 ]; then
        echo "Warning: Only $remaining requests remaining"
        local reset=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \
            -H "Authorization: Bearer $CF_ACCESS_TOKEN" \| \
            grep -i "x-ratelimit-reset" \| \
            awk '{print $2}' \| tr -d '\r')
        local wait_time=$((reset - $(date +%s)))
        echo "Rate limit resets in $wait_time seconds"
        sleep $wait_time
    fi
}

# Use before API calls

check_rate_limit
cloudflow workflows execute wf_9k2n4m8p1q
```

#### Optimization Strategies

**1. Implement request batching:**
```bash

# Batch multiple workflow executions

cloudflow workflows execute-batch \
  --workflow-ids "wf_id1,wf_id2,wf_id3,wf_id4,wf_id5" \
  --batch-size 5

# This counts as 1 API request instead of 5

```

**2. Use webhooks instead of polling:**
```bash

# Configure webhook for workflow completion

cloudflow webhooks create \
  --event workflow.completed \
  --url https://your-service.com/webhooks/cloudflow \
  --secret $WEBHOOK_SECRET

# Verify webhook

cloudflow webhooks test webhook_abc123
```

**3. Leverage caching:**
```bash

# Enable client-side caching

export CLOUDFLOW_CACHE_ENABLED=true
export CLOUDFLOW_CACHE_TTL=300

# Cache workflow metadata

cloudflow workflows list --use-cache --cache-ttl 600
```

---

## Log Analysis & Debugging



### Accessing CloudFlow Logs



#### Kubernetes Deployments

```bash

# List all CloudFlow pods

kubectl get pods -n cloudflow

# Tail logs from API server

kubectl logs -f -n cloudflow deployment/cloudflow-api --tail=100

# Get logs from specific pod

kubectl logs -n cloudflow cloudflow-api-7d4f6b8c9d-x7k2m

# Get logs from previous crashed pod

kubectl logs -n cloudflow cloudflow-api-7d4f6b8c9d-x7k2m --previous

# Get logs from all pods in deployment

kubectl logs -n cloudflow deployment/cloudflow-api --all-containers=true

# Stream logs from multiple pods

kubectl logs -n cloudflow -l app=cloudflow-api -f --max-log-requests=10
```

#### Log Levels

CloudFlow supports the following log levels:
- `TRACE` - Very detailed debugging information
- `DEBUG` - Detailed debugging information
- `INFO` - Informational messages (default)
- `WARN` - Warning messages
- `ERROR` - Error messages
- `FATAL` - Fatal errors causing shutdown

**Changing log levels:**
```bash

# Set global log level

cloudflow config set logging.level DEBUG

# Set log level for specific component

cloudflow config set logging.components.database DEBUG
cloudflow config set logging.components.auth INFO
cloudflow config set logging.components.workflows TRACE

# Temporary log level increase (resets after 1 hour)

cloudflow debug set-log-level DEBUG --duration 3600

# View current log configuration

cloudflow config get logging --format json
```

### Grep Patterns for Common Issues



#### Authentication Failures

```bash

# Find all authentication errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "authentication\\|401\\|unauthorized" \| \
  tail -n 50

# Find JWT validation failures

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "JWT\|token.*invalid\|signature.*failed"

# Find clock skew issues

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "clock skew\\|time.*sync\\|nbf\\|exp"
```

#### Database Errors

```bash

# Find database connection errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "connection.*pool\|could not connect\|database.*timeout"

# Find slow queries

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep "slow query" \| \
  awk '{print $NF}' \| \
  sort -n \| \
  tail -n 20

# Find deadlock errors

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -i "deadlock detected"
```

#### Workflow Execution Errors

```bash

# Find workflow timeout errors

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep -E "timeout\|exceeded.*3600"

# Find workflow retry attempts

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep -E "retry attempt [0-9]\|retrying in"

# Find workflow failures by ID

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  grep "exec_7h3j6k9m2n"
```

#### Rate Limiting

```bash

# Find rate limit events

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  grep -E "429\|rate.*limit\|throttle"

# Count rate limit errors by hour

kubectl logs -n cloudflow deployment/cloudflow-api --since=24h \| \
  grep "rate_limit_exceeded" \| \
  awk '{print $1}' \| \
  cut -d'T' -f1-2 \| \
  sort \| uniq -c
```

### Advanced Log Analysis



#### Using jq for JSON Logs

```bash

# Parse JSON logs and filter by level

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  jq 'select(.level == "ERROR")'

# Extract specific fields

kubectl logs -n cloudflow deployment/cloudflow-api \| \
  jq '{timestamp: .timestamp, level: .level, message: .message, execution_id: .context.execution_id}'

# Filter by workflow ID

kubectl logs -n cloudflow deployment/cloudflow-workflow-engine \| \
  jq 'select(.workflow_id == "wf_9k2n4m8p1q")'

# Count errors by type

kubectl logs -n cloudflow deployment/cloudflow-api --since=1h \| \
  jq -r 'select(.level == "ERROR") \| .error_type' \| \
  sort \| uniq -c \| sort -rn
```

#### Correlation IDs

CloudFlow uses correlation IDs to trace requests across services.

```bash

# Extract correlation ID from error

CORRELATION_ID="corr_8h4j9k2m5n"

# Trace request across all services

for pod in $(kubectl get pods -n cloudflow -l tier=backend -o name); do
  echo "=== $pod ==="
  kubectl logs -n cloudflow $pod \| grep $CORRELATION_ID
done

# Export full trace to file

cloudflow debug trace $CORRELATION_ID --output trace-$CORRELATION_ID.json
```

### Debugging Commands



#### Enable Debug Mode for Workflow Execution

```bash

# Execute workflow with debug logging

cloudflow workflows execute wf_9k2n4m8p1q \
  --debug \
  --log-level TRACE \
  --output-logs /tmp/workflow-debug.log

# Enable step-by-step execution

cloudflow workflows execute wf_9k2n4m8p1q \
  --step-mode interactive \
  --breakpoint-on-error

# Capture full execution context

cloudflow workflows execute wf_9k2n4m8p1q \
  --capture-context \
  --context-output /tmp/execution-context.json
```

#### Database Query Debugging

```bash

# Enable query logging

cloudflow db config set log_statement all
cloudflow db config set log_duration on
cloudflow db config set log_min_duration_statement 1000  # Log queries > 1s

# Capture query plan for slow endpoint

cloudflow debug capture-queries \
  --endpoint "/api/v1/workflows/list" \
  --duration 60 \
  --output query-analysis.txt

# Analyze query performance

cloudflow db analyze-performance --last 1h
```

#### Network Debugging

```bash

# Test connectivity from CloudFlow pod

kubectl run -n cloudflow netdebug --rm -i --tty \
  --image=nicolaka/netshoot -- /bin/bash

# Inside pod:



# Check DNS resolution

nslookup api.cloudflow.io

# Check connectivity to database

nc -zv cloudflow-db.internal.company.com 5432

# Trace route

traceroute api.cloudflow.io

# Capture packets

tcpdump -i any -w /tmp/capture.pcap port 5432
```

---

## Escalation Procedures



### Severity Levels

CloudFlow incidents are classified into four severity levels:

3. **[user_guide]**

   ## Error Handling

Robust error handling ensures your workflows are resilient and reliable.

### Retry Policies

Configure automatic retries for failed actions:

```yaml
- id: api_call
  action: http_request
  config:
    url: "https://api.example.com/data"
  retry:
    max_attempts: 3
    backoff_type: "exponential"  # or "fixed", "linear"
    initial_interval: 1000       # milliseconds
    max_interval: 30000
    multiplier: 2.0
    retry_on:
      - timeout
      - network_error
      - status: [500, 502, 503, 504]
```

**Backoff Strategies:**

- **Fixed**: Wait the same amount of time between retries
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 1s
  - Attempt 3: Wait 1s

- **Linear**: Increase wait time by a fixed amount
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 2s
  - Attempt 3: Wait 3s

- **Exponential**: Double the wait time with each retry (recommended)
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 2s
  - Attempt 3: Wait 4s

**Retry Conditions:**
Control which errors trigger retries:
- `timeout`: Request timeout
- `network_error`: Connection failures
- `status`: Specific HTTP status codes
- `error_code`: Application-specific error codes

4. **[api_reference]**

   ## Rate Limiting

To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.

**Default Limits:**
- 100 requests per minute per authenticated user
- 20 requests per minute for unauthenticated requests
- Burst allowance: 150 requests in a 10-second window

### Rate Limit Headers

Every API response includes rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1640995200
```

When you exceed the rate limit, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please retry after 42 seconds.",
    "retry_after": 42
  }
}
```

**Best Practices:**
- Monitor `X-RateLimit-Remaining` header values
- Implement exponential backoff when receiving 429 responses
- Cache responses when appropriate to reduce API calls
- Consider upgrading to Enterprise tier for higher limits (1000 req/min)

## Pagination

List endpoints return paginated results to optimize performance. Use `limit` and `offset` parameters to navigate through result sets.

**Parameters:**
- `limit`: Number of items per page (default: 20, max: 100)
- `offset`: Number of items to skip (default: 0)

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://api.cloudflow.io/v2/workflows?limit=50&offset=100"
```

**Response Structure:**

```json
{
  "data": [...],
  "pagination": {
    "total": 347,
    "limit": 50,
    "offset": 100,
    "has_more": true
  }
}
```

**Python Example:**

```python
import requests

def fetch_all_workflows(api_key):
    base_url = "https://api.cloudflow.io/v2/workflows"
    headers = {"X-API-Key": api_key}
    all_workflows = []
    offset = 0
    limit = 100
    
    while True:
        response = requests.get(
            base_url,
            headers=headers,
            params={"limit": limit, "offset": offset}
        )
        data = response.json()
        all_workflows.extend(data['data'])
        
        if not data['pagination']['has_more']:
            break
        
        offset += limit
    
    return all_workflows
```

## Endpoints



### Workflows



#### List Workflows

Retrieve a paginated list of all workflows in your account.

**Endpoint:** `GET /workflows`

**Query Parameters:**
- `status` (optional): Filter by status (`active`, `paused`, `archived`)
- `created_after` (optional): ISO 8601 timestamp
- `limit` (optional): Items per page (max 100)
- `offset` (optional): Pagination offset

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://api.cloudflow.io/v2/workflows?status=active&limit=25"
```

**Example Response (200 OK):**

```json
{
  "data": [
    {
      "id": "wf_8x7k2m9p",
      "name": "Data Processing Pipeline",
      "description": "Processes customer data every hour",
      "status": "active",
      "trigger": {
        "type": "schedule",
        "cron": "0 * * * *"
      },
      "steps": 12,
      "created_at": "2026-01-15T10:30:00Z",
      "updated_at": "2026-01-23T14:22:00Z",
      "last_run": "2026-01-24T09:00:00Z"
    }
  ],
  "pagination": {
    "total": 1,
    "limit": 25,
    "offset": 0,
    "has_more": false
  }
}
```

#### Create Workflow

Create a new workflow with specified configuration.

**Endpoint:** `POST /workflows`

**Request Body:**

```json
{
  "name": "Email Campaign Automation",
  "description": "Sends personalized emails based on user behavior",
  "trigger": {
    "type": "webhook",
    "url": "https://your-app.com/webhook"
  },
  "steps": [
    {
      "type": "fetch_data",
      "source": "users_table",
      "filters": {"active": true}
    },
    {
      "type": "transform",
      "script": "data.map(user => ({ ...user, segment: calculateSegment(user) }))"
    },
    {
      "type": "send_email",
      "template_id": "tpl_welcome_v2"
    }
  ]
}
```

**Example Request:**

```bash
curl -X POST https://api.cloudflow.io/v2/workflows \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Email Campaign Automation",
    "trigger": {"type": "webhook"},
    "steps": [{"type": "fetch_data", "source": "users_table"}]
  }'
```

**Example Response (201 Created):**

```json
{
  "id": "wf_9k3m7n2q",
  "name": "Email Campaign Automation",
  "status": "active",
  "created_at": "2026-01-24T10:15:00Z",
  "webhook_url": "https://api.cloudflow.io/v2/webhooks/wf_9k3m7n2q/trigger"
}
```

#### Get Workflow Details

Retrieve detailed information about a specific workflow.

**Endpoint:** `GET /workflows/{workflow_id}`

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p
```

**Example Response (200 OK):**

```json
{
  "id": "wf_8x7k2m9p",
  "name": "Data Processing Pipeline",
  "description": "Processes customer data every hour",
  "status": "active",
  "trigger": {
    "type": "schedule",
    "cron": "0 * * * *",
    "timezone": "UTC"
  },
  "steps": [
    {
      "id": "step_1",
      "type": "fetch_data",
      "source": "customer_db",
      "query": "SELECT * FROM customers WHERE updated_at > :last_run"
    },
    {
      "id": "step_2",
      "type": "transform",
      "script_id": "scr_transform_v3"
    }
  ],
  "metrics": {
    "total_runs": 1247,
    "success_rate": 98.7,
    "avg_duration_ms": 3420,
    "last_error": null
  },
  "created_at": "2026-01-15T10:30:00Z",
  "updated_at": "2026-01-23T14:22:00Z"
}
```

#### Update Workflow

Modify an existing workflow's configuration.

**Endpoint:** `PATCH /workflows/{workflow_id}`

**Example Request:**

```bash
curl -X PATCH https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"status": "paused"}'
```

**Example Response (200 OK):**

```json
{
  "id": "wf_8x7k2m9p",
  "status": "paused",
  "updated_at": "2026-01-24T10:30:00Z"
}
```

#### Delete Workflow

Permanently delete a workflow.

**Endpoint:** `DELETE /workflows/{workflow_id}`

**Example Request:**

```bash
curl -X DELETE https://api.cloudflow.io/v2/workflows/wf_8x7k2m9p \
  -H "X-API-Key: YOUR_API_KEY"
```

**Example Response (204 No Content)**

5. **[api_reference]**

   ### Error Codes

CloudFlow returns specific error codes to help you identify and resolve issues:

- `invalid_parameter`: One or more request parameters are invalid
- `missing_required_field`: Required field is missing from request body
- `authentication_failed`: Invalid API key or token
- `insufficient_permissions`: User lacks required scope or permission
- `resource_not_found`: Requested resource does not exist
- `rate_limit_exceeded`: Too many requests, see rate limiting section
- `workflow_execution_failed`: Workflow execution encountered an error
- `invalid_json`: Request body contains malformed JSON
- `duplicate_resource`: Resource with same identifier already exists
- `quota_exceeded`: Account quota limit reached

### Error Handling Best Practices

**Python Example:**

```python
import requests
import time

def make_api_request(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(response.headers.get('X-RateLimit-Reset', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            elif response.status_code >= 500:
                # Server error - retry with exponential backoff
                wait_time = 2 ** attempt
                print(f"Server error. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            else:
                # Client error - don't retry
                error_data = response.json()
                raise Exception(f"API Error: {error_data['error']['message']}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception("Max retries exceeded")
```

## Webhooks

CloudFlow can send webhook notifications when specific events occur in your workflows.

**Supported Events:**
- `workflow.started`
- `workflow.completed`
- `workflow.failed`
- `pipeline.completed`
- `pipeline.failed`

**Webhook Payload Example:**

```json
{
  "event": "workflow.completed",
  "timestamp": "2026-01-24T10:45:00Z",
  "data": {
    "workflow_id": "wf_8x7k2m9p",
    "execution_id": "exec_9k3m7n2q",
    "status": "completed",
    "duration_ms": 3420,
    "records_processed": 1247
  }
}
```

Configure webhooks in your account settings or via the API:

```bash
curl -X POST https://api.cloudflow.io/v2/webhooks \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/cloudflow-webhook",
    "events": ["workflow.completed", "workflow.failed"],
    "secret": "whsec_your_webhook_secret"
  }'
```

## Support

For additional help and resources:

- **Documentation:** https://docs.cloudflow.io
- **API Status:** https://status.cloudflow.io
- **Support Email:** support@cloudflow.io
- **Community Forum:** https://community.cloudflow.io

Enterprise customers have access to 24/7 priority support via phone and dedicated Slack channels.

---

### Question 4: What are the CloudFlow API authentication methods?

**Expected Answer**: CloudFlow supports three authentication methods: API Keys, OAuth 2.0, and JWT Tokens

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[api_reference]**

   ## Authentication

CloudFlow supports three authentication methods to suit different use cases and security requirements.

### API Keys

API keys provide simple authentication for server-to-server communication. Include your API key in the request header:

```bash
curl -H "X-API-Key: cf_live_a1b2c3d4e5f6g7h8i9j0" \
  https://api.cloudflow.io/v2/workflows
```

**Security Notes:**
- Never expose API keys in client-side code
- Rotate keys every 90 days
- Use separate keys for development and production environments

2. **[api_reference]**

   ### OAuth 2.0

OAuth 2.0 is recommended for applications that access CloudFlow on behalf of users. We support the Authorization Code flow with PKCE.

**Authorization Endpoint:** `https://auth.cloudflow.io/oauth/authorize`

**Token Endpoint:** `https://auth.cloudflow.io/oauth/token`

**Supported Scopes:**
- `workflows:read` - Read workflow configurations
- `workflows:write` - Create and modify workflows
- `pipelines:read` - Read pipeline data
- `pipelines:write` - Create and manage pipelines
- `analytics:read` - Access analytics and metrics
- `admin:full` - Full administrative access

Example authorization request:

```bash
curl -X POST https://auth.cloudflow.io/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "code=AUTH_CODE_HERE" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "redirect_uri=https://yourapp.com/callback"
```

3. **[api_reference]**

   # CloudFlow API Reference

Version 2.1.0 \| Last Updated: January 2026

## Overview

The CloudFlow API is a RESTful service that enables developers to programmatically manage cloud workflows, data pipelines, and automation tasks. This documentation provides comprehensive details on authentication, endpoints, request/response formats, error handling, and best practices.

**Base URL:** `https://api.cloudflow.io/v2`

**API Status:** https://status.cloudflow.io

4. **[troubleshooting_guide]**

   # CloudFlow Platform Troubleshooting Guide

**Version:** 3.2.1  
**Last Updated:** January 2026  
**Audience:** Platform Engineers, DevOps, Support Teams

## Table of Contents

1. [Overview](#overview)
2. [Authentication & Authorization Issues](#authentication--authorization-issues)
3. [Performance Problems](#performance-problems)
4. [Database Connection Issues](#database-connection-issues)
5. [Workflow Execution Failures](#workflow-execution-failures)
6. [Rate Limiting & Throttling](#rate-limiting--throttling)
7. [Log Analysis & Debugging](#log-analysis--debugging)
8. [Escalation Procedures](#escalation-procedures)

---

5. **[user_guide]**

   ## Available Actions

CloudFlow provides a comprehensive library of actions to build powerful automations.

### HTTP Requests

Make HTTP requests to any API endpoint:

**Configuration:**
- **Method**: GET, POST, PUT, PATCH, DELETE, HEAD
- **URL**: Full endpoint URL (supports variable interpolation)
- **Headers**: Custom headers as key-value pairs
- **Query Parameters**: URL parameters
- **Body**: JSON, form data, or raw text
- **Authentication**: Basic Auth, Bearer Token, API Key, OAuth 2.0

**Example:**
```yaml
- id: fetch_user
  action: http_request
  config:
    method: GET
    url: "https://api.example.com/users/{{user_id}}"
    headers:
      Authorization: "Bearer {{secrets.API_TOKEN}}"
      Content-Type: "application/json"
    timeout: 30
```

**Response Handling:**
Access the response in subsequent steps:
- `{{steps.fetch_user.status}}` - HTTP status code
- `{{steps.fetch_user.body}}` - Response body
- `{{steps.fetch_user.headers}}` - Response headers

---

### Question 5: How long do JWT tokens last in CloudFlow?

**Expected Answer**: All tokens expire after 3600 seconds (1 hour). Implement token refresh logic in your application.

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[api_reference]**

   ### JWT Tokens

For advanced use cases, CloudFlow supports JSON Web Tokens (JWT) with RS256 signing algorithm. JWTs must include the following claims:

- `iss` (issuer): Your application identifier
- `sub` (subject): User or service account ID
- `aud` (audience): `https://api.cloudflow.io`
- `exp` (expiration): Unix timestamp (max 3600 seconds from `iat`)
- `iat` (issued at): Unix timestamp
- `scope`: Space-separated list of requested scopes

Example JWT header:

```python
import jwt
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Load your private key

with open('private_key.pem', 'rb') as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=None,
        backend=default_backend()
    )

# Create JWT payload

payload = {
    'iss': 'your-app-id',
    'sub': 'user-12345',
    'aud': 'https://api.cloudflow.io',
    'exp': int(time.time()) + 3600,
    'iat': int(time.time()),
    'scope': 'workflows:read workflows:write'
}

# Generate token

token = jwt.encode(payload, private_key, algorithm='RS256')

# Use in API request

headers = {'Authorization': f'Bearer {token}'}
```

**Token Expiration:** All tokens expire after 3600 seconds (1 hour). Implement token refresh logic in your application.

2. **[troubleshooting_guide]**

   ## Overview

This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analysis, resolution steps, and preventive measures.

### Quick Diagnostic Checklist

Before diving into specific issues, perform these initial checks:

- Verify service health: `cloudflow status --all`
- Check API connectivity: `curl -I https://api.cloudflow.io/health`
- Review recent deployments: `kubectl get deployments -n cloudflow --sort-by=.metadata.creationTimestamp`
- Inspect platform metrics: `cloudflow metrics --last 1h`

---

## Authentication & Authorization Issues



### 401 Unauthorized Errors



#### Symptoms

- API requests return `401 Unauthorized`
- Error message: `Authentication credentials were not provided or are invalid`
- Frontend displays "Session expired" message

#### Common Causes

**1. Token Expiration**

CloudFlow access tokens expire after 3600 seconds (1 hour) by default. Refresh tokens are valid for 30 days.

**Verification:**
```bash

# Decode JWT to check expiration

echo $CF_ACCESS_TOKEN \| cut -d'.' -f2 \| base64 -d \| jq '.exp'

# Compare with current time

date +%s
```

**Resolution:**
```bash

# Refresh the access token

curl -X POST https://api.cloudflow.io/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "'$CF_REFRESH_TOKEN'"}'

# Update environment variable

export CF_ACCESS_TOKEN="<new_token>"
```

**2. Invalid JWT Signature**

Error: `JWT signature verification failed`

**Causes:**
- Token was modified or corrupted
- Using wrong signing key
- Token generated with different secret

**Resolution:**
```bash

# Validate token structure

cloudflow auth validate --token $CF_ACCESS_TOKEN

# Generate new token

cloudflow auth login --username <user> --password <pass>

# For service accounts

cloudflow auth service-account --create --name <sa-name> --scopes "workflow:read,workflow:write"
```

**3. Clock Skew Issues**

JWT validation fails when system clocks are out of sync (tolerance: ±300 seconds).

**Diagnosis:**
```bash

# Check system time

timedatectl status

# Compare with NTP server

ntpdate -q pool.ntp.org

# Check JWT issued time vs current time

jwt_iat=$(echo $CF_ACCESS_TOKEN \| cut -d'.' -f2 \| base64 -d \| jq '.iat')
current_time=$(date +%s)
skew=$((current_time - jwt_iat))
echo "Clock skew: $skew seconds"
```

**Resolution:**
```bash

# Sync with NTP server

sudo ntpdate -s pool.ntp.org

# Enable automatic time sync

sudo timedatectl set-ntp true

# Restart CloudFlow client

cloudflow restart
```

**4. Insufficient Permissions**

Error: `User does not have required permissions for this operation`

**Check user roles:**
```bash
cloudflow auth whoami --verbose

# Expected output:



# User: john.doe@company.com



# Roles: developer, workflow-admin



# Scopes: workflow:*, data:read, metrics:read

```

**Request permission elevation:**
```bash

# Submit access request

cloudflow auth request-access \
  --resource "workflow:production:deploy" \
  --justification "Deploy critical hotfix for TICKET-1234"

# Check pending approvals

cloudflow auth list-requests --status pending
```

### 403 Forbidden Errors



3. **[architecture_overview]**

   ### Authentication & Authorization

**JWT Token Validation**:
- Algorithm: RS256 (asymmetric signing)
- Key rotation: Every 30 days with 7-day overlap period
- Public key distribution: JWKS endpoint cached in Redis
- Validation: Signature, expiry, issuer, audience claims
- Token revocation: Blacklist in Redis for compromised tokens

**Permission Model**:
```
User → Roles → Permissions
     ↘       ↗
      Tenants (Multi-tenancy isolation)
```

Example permissions:
- `workflow:read` - View workflows
- `workflow:write` - Create/update workflows
- `workflow:execute` - Trigger workflow execution
- `workflow:delete` - Delete workflows
- `admin:*` - All administrative operations

**API Key Management**:
- Format: `cfk_live_<32-char-random>` (production), `cfk_test_<32-char-random>` (sandbox)
- Hashing: SHA-256 before storage in PostgreSQL
- Scoping: API keys can be scoped to specific workflows or operations
- Rate limits: Configurable per API key (default: 1000 RPM)

4. **[architecture_overview]**

   ## Microservices Breakdown



### API Gateway

**Purpose**: Single entry point for all client requests, providing authentication, rate limiting, request routing, and protocol translation.

**Technology**: Node.js with Express.js framework  
**Replicas**: 12 pods (production), auto-scaling 8-20 based on CPU  
**Resource Allocation**: 2 vCPU, 4GB RAM per pod

**Key Responsibilities**:
- JWT token validation (delegated to Auth Service for initial validation)
- Rate limiting: 1000 requests per minute per API key (sliding window)
- Request/response transformation and validation using JSON Schema
- Routing to downstream services based on URL path patterns
- CORS handling for web clients
- Request/response logging and correlation ID injection

**Critical Endpoints**:
- `POST /api/v1/workflows` - Create new workflow
- `GET /api/v1/workflows/:id` - Retrieve workflow status
- `POST /api/v1/workflows/:id/execute` - Trigger workflow execution
- `GET /api/v1/workflows/:id/history` - Get execution history

**Dependencies**:
- Auth Service (for token validation)
- Redis (for rate limiting counters)
- All downstream microservices

**Performance Targets**:
- P50 latency: < 50ms
- P99 latency: < 200ms
- Throughput: 10,000 RPS sustained

---

5. **[architecture_overview]**

   ### Auth Service

**Purpose**: Centralized authentication and authorization service handling user identity, token generation, and permission validation.

**Technology**: Go with gRPC for internal communication, REST for external  
**Replicas**: 8 pods (production), auto-scaling 6-12  
**Resource Allocation**: 1 vCPU, 2GB RAM per pod

**Key Responsibilities**:
- User authentication via multiple providers (OAuth2, SAML, local credentials)
- JWT token generation and validation (RS256 algorithm)
- Role-based access control (RBAC) with fine-grained permissions
- Session management with Redis-backed storage
- API key generation and validation for service accounts
- MFA enforcement for administrative operations

**Authentication Flow**:
```
Client Request → API Gateway → Auth Service
                                     │
                                     ├─ Validate credentials
                                     ├─ Check MFA if required
                                     ├─ Generate JWT (15min expiry)
                                     ├─ Generate refresh token (7 days)
                                     └─ Store session in Redis
```

**Token Structure**:
- Access Token: JWT with 15-minute expiry
- Refresh Token: Opaque token with 7-day expiry, stored in PostgreSQL
- Claims: user_id, email, roles[], permissions[], tenant_id

**Security Features**:
- Password hashing: Argon2id with 64MB memory, 4 iterations
- Token rotation on refresh to prevent replay attacks
- Brute force protection: 5 failed attempts → 15-minute lockout
- Secrets stored in HashiCorp Vault, rotated every 90 days

**Performance Targets**:
- Token validation: < 10ms (P99)
- Token generation: < 50ms (P99)
- Throughput: 5,000 RPS for validation operations

---

---

### Question 6: What should I do if my workflow exceeds the maximum execution timeout?

**Expected Answer**: Workflows have a default timeout of 3600 seconds (60 minutes). You can increase the timeout, optimize workflow steps, or split the workflow into smaller workflows

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[user_guide]**

   ## Error Handling

Robust error handling ensures your workflows are resilient and reliable.

### Retry Policies

Configure automatic retries for failed actions:

```yaml
- id: api_call
  action: http_request
  config:
    url: "https://api.example.com/data"
  retry:
    max_attempts: 3
    backoff_type: "exponential"  # or "fixed", "linear"
    initial_interval: 1000       # milliseconds
    max_interval: 30000
    multiplier: 2.0
    retry_on:
      - timeout
      - network_error
      - status: [500, 502, 503, 504]
```

**Backoff Strategies:**

- **Fixed**: Wait the same amount of time between retries
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 1s
  - Attempt 3: Wait 1s

- **Linear**: Increase wait time by a fixed amount
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 2s
  - Attempt 3: Wait 3s

- **Exponential**: Double the wait time with each retry (recommended)
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 2s
  - Attempt 3: Wait 4s

**Retry Conditions:**
Control which errors trigger retries:
- `timeout`: Request timeout
- `network_error`: Connection failures
- `status`: Specific HTTP status codes
- `error_code`: Application-specific error codes

2. **[user_guide]**

   ### Steps Per Workflow

- **Maximum**: 50 steps per workflow
- **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via webhooks.

### Execution Timeout

- **Default**: 3600 seconds (60 minutes)
- **Behavior**: Workflows exceeding this timeout are automatically terminated
- **Custom Timeouts**: Enterprise plans can request custom timeout limits

**Setting Step-Level Timeouts:**
```yaml
- id: long_running_task
  action: http_request
  config:
    url: "https://api.example.com/process"
    timeout: 300  # 5 minutes for this specific step
```

3. **[troubleshooting_guide]**

   # Create sub-workflows

cloudflow workflows create data-pipeline-part1 \
  --steps "data_ingestion,data_validation" \
  --timeout 1800

cloudflow workflows create data-pipeline-part2 \
  --steps "data_transformation,data_export" \
  --timeout 3600 \
  --trigger workflow_completed \
  --trigger-workflow data-pipeline-part1
```

### Retry Logic and Exponential Backoff

CloudFlow implements automatic retry with exponential backoff for transient failures:
- Max retries: 3
- Initial delay: 1 second
- Backoff multiplier: 2
- Max delay: 60 seconds

#### Retry Sequence

```
Attempt 1: Immediate
Attempt 2: Wait 1s  (2^0 * 1s)
Attempt 3: Wait 2s  (2^1 * 1s)
Attempt 4: Wait 4s  (2^2 * 1s)
```

#### Configuration

```bash

# View current retry settings

cloudflow workflows get wf_9k2n4m8p1q --format json \| jq '.retry_policy'

# Customize retry behavior

cloudflow workflows update wf_9k2n4m8p1q \
  --retry-max-attempts 5 \
  --retry-initial-delay 2000 \
  --retry-backoff-multiplier 2 \
  --retry-max-delay 120000 \
  --retry-on-errors "NETWORK_ERROR,TIMEOUT,RATE_LIMIT"

# Disable retry for specific step

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_export \
  --retry-enabled false
```

#### Monitoring Retries

```bash

# List failed executions with retry information

cloudflow workflows executions list \
  --status FAILED \
  --show-retries \
  --last 7d

# Get retry history for specific execution

cloudflow workflows executions retries exec_7h3j6k9m2n

# Output:



# Execution: exec_7h3j6k9m2n



# Attempt 1: FAILED - NetworkError: Connection refused (delay: 0ms)



# Attempt 2: FAILED - NetworkError: Connection timeout (delay: 1000ms)



# Attempt 3: FAILED - NetworkError: Connection timeout (delay: 2000ms)



# Attempt 4: FAILED - NetworkError: Connection timeout (delay: 4000ms)



# Final Status: FAILED_AFTER_RETRIES

```

### Workflow Step Failures



#### Common Error Patterns

**1. Data Validation Errors**
```
ValidationError: Field 'customer_id' is required but missing in 234 records
```

**Resolution:**
```bash

# Add data quality checks

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_ingestion \
  --add-validator required_fields \
  --validator-config '{"fields": ["customer_id", "timestamp", "amount"]}'

4. **[user_guide]**

   ### Data Limits

- **Maximum request/response size**: 10MB per action
- **Maximum execution payload**: 50MB total
- **Variable value size**: 1MB per variable

### Enterprise Plan Limits

Enterprise customers can request increased limits:
- Up to 100 steps per workflow
- Up to 10,000 executions per day
- Up to 7200 second timeout (2 hours)
- Priority execution queue
- Dedicated capacity allocation

Contact sales@cloudflow.io for Enterprise pricing and custom limits.

## Best Practices

Follow these best practices to build reliable, maintainable workflows:

### 1. Use Descriptive Names

**Good:**
- Workflow: "Sync Customer Data from Salesforce to Database"
- Step: "validate_customer_email"

**Bad:**
- Workflow: "Workflow 1"
- Step: "step3"

### 2. Handle Errors Gracefully

Always implement error handling for external API calls and database operations:

```yaml
- id: fetch_data
  action: http_request
  config:
    url: "https://api.example.com/data"
  retry:
    max_attempts: 3
    backoff_type: exponential
  on_error:
    - id: log_error
      action: database_query
      config:
        query: "INSERT INTO error_log (workflow_id, error) VALUES ($1, $2)"
        parameters:
          - "{{workflow.id}}"
          - "{{error.message}}"
```

### 3. Use Secrets for Sensitive Data

Never hardcode API keys, passwords, or tokens in workflows:

**Bad:**
```yaml
headers:
  Authorization: "Bearer sk_live_abc123xyz789"
```

**Good:**
```yaml
headers:
  Authorization: "Bearer {{secrets.API_TOKEN}}"
```

Store secrets in **Settings** > **Secrets** with encryption at rest.

### 4. Validate Input Data

Always validate trigger data before processing:

```yaml
- id: validate_input
  action: javascript
  code: \|
    const required_fields = ['email', 'name', 'order_id'];
    for (const field of required_fields) {
      if (!input[field]) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
    
    // Validate email format
    const email_regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email_regex.test(input.email)) {
      throw new Error('Invalid email format');
    }
    
    return { validated: true };
```

### 5. Use Idempotency Keys

For operations that shouldn't be repeated (payments, record creation), use idempotency keys:

```yaml
- id: create_charge
  action: http_request
  config:
    url: "https://api.stripe.com/v1/charges"
    method: POST
    headers:
      Idempotency-Key: "{{workflow.id}}-{{execution.id}}"
    body:
      amount: "{{amount}}"
```

### 6. Monitor and Log

Add logging steps for important workflow milestones:

```yaml
- id: log_start
  action: database_query
  config:
    query: "INSERT INTO workflow_audit (execution_id, step, timestamp) VALUES ($1, $2, $3)"
    parameters:
      - "{{execution.id}}"
      - "workflow_started"
      - "{{now()}}"
```

### 7. Keep Workflows Modular

Break complex workflows into smaller, reusable components:

- Use sub-workflows for repeated logic
- Trigger child workflows via webhooks
- Share common configurations via templates

5. **[troubleshooting_guide]**

   #### RBAC Policy Violations

CloudFlow uses role-based access control (RBAC) with the following hierarchy:
- `viewer` - Read-only access
- `developer` - Create and modify workflows (non-production)
- `operator` - Execute workflows, view logs
- `admin` - Full access to workspace
- `platform-admin` - Cross-workspace administration

**Verify resource permissions:**
```bash

# Check effective permissions

cloudflow rbac check \
  --user john.doe@company.com \
  --resource workflow:prod-pipeline \
  --action execute

# List all policies affecting user

cloudflow rbac policies --user john.doe@company.com --verbose
```

---

## Performance Problems



### Slow Query Performance



#### Symptoms

- API response times > 5000ms
- Database query latency warnings in logs
- CloudFlow UI becomes unresponsive
- Timeout errors: `Request exceeded maximum duration of 30000ms`

#### Diagnosis Steps

**1. Identify Slow Queries**

```bash

# Query CloudFlow metrics

cloudflow metrics query --metric api_request_duration_ms \
  --filter "p95 > 5000" \
  --last 1h

---

### Question 7: How do I restart a failed workflow execution?

**Expected Answer**: View failed executions in the Dead Letter Queue, inspect the execution context and error details, and use the 'Retry' button to reprocess with the same input data

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[user_guide]**

   ## Error Handling

Robust error handling ensures your workflows are resilient and reliable.

### Retry Policies

Configure automatic retries for failed actions:

```yaml
- id: api_call
  action: http_request
  config:
    url: "https://api.example.com/data"
  retry:
    max_attempts: 3
    backoff_type: "exponential"  # or "fixed", "linear"
    initial_interval: 1000       # milliseconds
    max_interval: 30000
    multiplier: 2.0
    retry_on:
      - timeout
      - network_error
      - status: [500, 502, 503, 504]
```

**Backoff Strategies:**

- **Fixed**: Wait the same amount of time between retries
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 1s
  - Attempt 3: Wait 1s

- **Linear**: Increase wait time by a fixed amount
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 2s
  - Attempt 3: Wait 3s

- **Exponential**: Double the wait time with each retry (recommended)
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 2s
  - Attempt 3: Wait 4s

**Retry Conditions:**
Control which errors trigger retries:
- `timeout`: Request timeout
- `network_error`: Connection failures
- `status`: Specific HTTP status codes
- `error_code`: Application-specific error codes

2. **[troubleshooting_guide]**

   # Create sub-workflows

cloudflow workflows create data-pipeline-part1 \
  --steps "data_ingestion,data_validation" \
  --timeout 1800

cloudflow workflows create data-pipeline-part2 \
  --steps "data_transformation,data_export" \
  --timeout 3600 \
  --trigger workflow_completed \
  --trigger-workflow data-pipeline-part1
```

### Retry Logic and Exponential Backoff

CloudFlow implements automatic retry with exponential backoff for transient failures:
- Max retries: 3
- Initial delay: 1 second
- Backoff multiplier: 2
- Max delay: 60 seconds

#### Retry Sequence

```
Attempt 1: Immediate
Attempt 2: Wait 1s  (2^0 * 1s)
Attempt 3: Wait 2s  (2^1 * 1s)
Attempt 4: Wait 4s  (2^2 * 1s)
```

#### Configuration

```bash

# View current retry settings

cloudflow workflows get wf_9k2n4m8p1q --format json \| jq '.retry_policy'

# Customize retry behavior

cloudflow workflows update wf_9k2n4m8p1q \
  --retry-max-attempts 5 \
  --retry-initial-delay 2000 \
  --retry-backoff-multiplier 2 \
  --retry-max-delay 120000 \
  --retry-on-errors "NETWORK_ERROR,TIMEOUT,RATE_LIMIT"

# Disable retry for specific step

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_export \
  --retry-enabled false
```

#### Monitoring Retries

```bash

# List failed executions with retry information

cloudflow workflows executions list \
  --status FAILED \
  --show-retries \
  --last 7d

# Get retry history for specific execution

cloudflow workflows executions retries exec_7h3j6k9m2n

# Output:



# Execution: exec_7h3j6k9m2n



# Attempt 1: FAILED - NetworkError: Connection refused (delay: 0ms)



# Attempt 2: FAILED - NetworkError: Connection timeout (delay: 1000ms)



# Attempt 3: FAILED - NetworkError: Connection timeout (delay: 2000ms)



# Attempt 4: FAILED - NetworkError: Connection timeout (delay: 4000ms)



# Final Status: FAILED_AFTER_RETRIES

```

### Workflow Step Failures



#### Common Error Patterns

**1. Data Validation Errors**
```
ValidationError: Field 'customer_id' is required but missing in 234 records
```

**Resolution:**
```bash

# Add data quality checks

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_ingestion \
  --add-validator required_fields \
  --validator-config '{"fields": ["customer_id", "timestamp", "amount"]}'

3. **[architecture_overview]**

   ### Scheduler Service

**Purpose**: Time-based workflow triggering system supporting cron-like schedules and one-time delayed executions.

**Technology**: Go with distributed locking via Redis  
**Replicas**: 4 pods (production), active-passive with leader election  
**Resource Allocation**: 2 vCPU, 4GB RAM per pod

**Key Responsibilities**:
- Parse and validate cron expressions (extended format supporting seconds)
- Maintain schedule registry in PostgreSQL
- Distributed scheduling with leader election (one active scheduler)
- Missed execution handling with configurable catch-up policy
- Schedule conflict detection and resolution
- Time zone support for international schedules

**Scheduling Architecture**:
```
PostgreSQL Schedules Table
         │
         ▼
  Leader Scheduler (elected via Redis)
         │
         ├─ Scan for due schedules (every 10 seconds)
         ├─ Acquire lock per schedule (prevents duplicates)
         ├─ Publish to Kafka → workflow.events topic
         └─ Update last_run timestamp
```

**Schedule Types**:
- **Cron-based**: `0 */5 * * * *` (every 5 minutes)
- **One-time**: Specific timestamp for delayed execution
- **Interval-based**: Every N seconds/minutes/hours

**Reliability Features**:
- Leader election using Redis with 30-second lease
- Heartbeat mechanism to detect leader failure (5-second interval)
- Automatic failover to standby scheduler (< 10 seconds)
- Schedule versioning to handle updates during execution
- Missed execution policy: SKIP, RUN_ONCE, or RUN_ALL

**Performance Targets**:
- Schedule evaluation: < 100ms per cycle
- Accuracy: ± 1 second for schedule triggers
- Capacity: 100,000 active schedules

---

4. **[troubleshooting_guide]**

   # Configure error handling

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_validation \
  --on-error continue \
  --error-threshold 5%  # Fail if > 5% of records invalid
```

**2. External API Failures**
```
ExternalAPIError: API request to https://partner-api.example.com failed with status 502
```

**Resolution:**
```bash

# Add circuit breaker

cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --circuit-breaker-enabled true \
  --circuit-breaker-threshold 5 \
  --circuit-breaker-timeout 30000

# Configure fallback behavior

cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --fallback-action use_cached_data \
  --cache-ttl 3600
```

---

## Rate Limiting & Throttling



### 429 Too Many Requests



#### Error Response

```json
{
  "error": "rate_limit_exceeded",
  "message": "API rate limit exceeded. Retry after 45 seconds.",
  "status": 429,
  "headers": {
    "X-RateLimit-Limit": "1000",
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1706112345",
    "Retry-After": "45"
  }
}
```

5. **[user_guide]**

   ### Fallback Actions

Execute alternative actions when the primary action fails:

```yaml
- id: primary_payment
  action: http_request
  config:
    url: "https://primary-payment-gateway.com/charge"
    method: POST
    body:
      amount: "{{amount}}"
  on_error:
    - id: fallback_payment
      action: http_request
      config:
        url: "https://backup-payment-gateway.com/charge"
        method: POST
        body:
          amount: "{{amount}}"
    - id: notify_admin
      action: email
      config:
        to: "admin@company.com"
        subject: "Payment Gateway Failure"
        body: "Primary gateway failed, switched to backup"
```

**Error Object:**
Access error details in fallback actions:
```
{{error.message}}        # Error message
{{error.code}}          # Error code
{{error.step_id}}       # Failed step ID
{{error.timestamp}}     # When the error occurred
{{error.attempts}}      # Number of retry attempts
```

### Dead Letter Queue

When all retries and fallbacks fail, CloudFlow can route failed executions to a Dead Letter Queue (DLQ) for manual review:

**Enable DLQ:**
```yaml
workflow:
  error_handling:
    dead_letter_queue:
      enabled: true
      retain_days: 30
```

**DLQ Features:**
- View failed executions in the dashboard
- Inspect complete execution context and error details
- Retry individual executions after fixing issues
- Export failed executions for analysis
- Set up alerts for DLQ threshold breaches

**Accessing the DLQ:**
1. Navigate to **Workflows** > **[Your Workflow]** > **Dead Letter Queue**
2. Filter by error type, date range, or execution ID
3. Click an execution to view full details
4. Click **"Retry"** to reprocess with the same input data

### Error Notifications

Get notified when workflows fail:

```yaml
workflow:
  notifications:
    on_failure:
      - type: email
        to: "ops-team@company.com"
      - type: slack
        channel: "#alerts"
        message: "Workflow {{workflow.name}} failed: {{error.message}}"
    on_success_after_retry:
      - type: slack
        channel: "#monitoring"
        message: "Workflow recovered after {{error.attempts}} attempts"
```

## Workflow Limits

CloudFlow enforces the following limits to ensure platform stability and performance:

---

### Question 8: What are the recommended steps for troubleshooting a workflow failure?

**Expected Answer**: Verify service health, check API connectivity, review recent deployments, inspect platform metrics, check logs, and follow the escalation procedure based on the severity level

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[troubleshooting_guide]**

   ## Overview

This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analysis, resolution steps, and preventive measures.

### Quick Diagnostic Checklist

Before diving into specific issues, perform these initial checks:

- Verify service health: `cloudflow status --all`
- Check API connectivity: `curl -I https://api.cloudflow.io/health`
- Review recent deployments: `kubectl get deployments -n cloudflow --sort-by=.metadata.creationTimestamp`
- Inspect platform metrics: `cloudflow metrics --last 1h`

---

## Authentication & Authorization Issues



### 401 Unauthorized Errors



#### Symptoms

- API requests return `401 Unauthorized`
- Error message: `Authentication credentials were not provided or are invalid`
- Frontend displays "Session expired" message

#### Common Causes

**1. Token Expiration**

CloudFlow access tokens expire after 3600 seconds (1 hour) by default. Refresh tokens are valid for 30 days.

**Verification:**
```bash

# Decode JWT to check expiration

echo $CF_ACCESS_TOKEN \| cut -d'.' -f2 \| base64 -d \| jq '.exp'

# Compare with current time

date +%s
```

**Resolution:**
```bash

# Refresh the access token

curl -X POST https://api.cloudflow.io/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "'$CF_REFRESH_TOKEN'"}'

# Update environment variable

export CF_ACCESS_TOKEN="<new_token>"
```

**2. Invalid JWT Signature**

Error: `JWT signature verification failed`

**Causes:**
- Token was modified or corrupted
- Using wrong signing key
- Token generated with different secret

**Resolution:**
```bash

# Validate token structure

cloudflow auth validate --token $CF_ACCESS_TOKEN

# Generate new token

cloudflow auth login --username <user> --password <pass>

# For service accounts

cloudflow auth service-account --create --name <sa-name> --scopes "workflow:read,workflow:write"
```

**3. Clock Skew Issues**

JWT validation fails when system clocks are out of sync (tolerance: ±300 seconds).

**Diagnosis:**
```bash

# Check system time

timedatectl status

# Compare with NTP server

ntpdate -q pool.ntp.org

# Check JWT issued time vs current time

jwt_iat=$(echo $CF_ACCESS_TOKEN \| cut -d'.' -f2 \| base64 -d \| jq '.iat')
current_time=$(date +%s)
skew=$((current_time - jwt_iat))
echo "Clock skew: $skew seconds"
```

**Resolution:**
```bash

# Sync with NTP server

sudo ntpdate -s pool.ntp.org

# Enable automatic time sync

sudo timedatectl set-ntp true

# Restart CloudFlow client

cloudflow restart
```

**4. Insufficient Permissions**

Error: `User does not have required permissions for this operation`

**Check user roles:**
```bash
cloudflow auth whoami --verbose

# Expected output:



# User: john.doe@company.com



# Roles: developer, workflow-admin



# Scopes: workflow:*, data:read, metrics:read

```

**Request permission elevation:**
```bash

# Submit access request

cloudflow auth request-access \
  --resource "workflow:production:deploy" \
  --justification "Deploy critical hotfix for TICKET-1234"

# Check pending approvals

cloudflow auth list-requests --status pending
```

### 403 Forbidden Errors



2. **[troubleshooting_guide]**

   ### Getting Help

If this troubleshooting guide doesn't resolve your issue:

1. Search the knowledge base: `cloudflow kb search "your issue"`
2. Check community forum for similar issues
3. Contact support with detailed logs and reproduction steps
4. For urgent issues, use emergency escalation procedures

**Remember:** Always capture logs, metrics, and reproduction steps before escalating!

---

*Last updated: January 24, 2026*  
*Document version: 3.2.1*  
*Feedback: docs-feedback@cloudflow.io*

3. **[troubleshooting_guide]**

   # CloudFlow Platform Troubleshooting Guide

**Version:** 3.2.1  
**Last Updated:** January 2026  
**Audience:** Platform Engineers, DevOps, Support Teams

## Table of Contents

1. [Overview](#overview)
2. [Authentication & Authorization Issues](#authentication--authorization-issues)
3. [Performance Problems](#performance-problems)
4. [Database Connection Issues](#database-connection-issues)
5. [Workflow Execution Failures](#workflow-execution-failures)
6. [Rate Limiting & Throttling](#rate-limiting--throttling)
7. [Log Analysis & Debugging](#log-analysis--debugging)
8. [Escalation Procedures](#escalation-procedures)

---

4. **[troubleshooting_guide]**

   #### Step 2: Gather Information (5-15 minutes)

Create incident document with:
- Incident timestamp and duration
- Affected services and endpoints
- Error rates and user impact
- Recent changes or deployments
- Relevant log excerpts
- Correlation IDs for failed requests

```bash

# Generate incident report

cloudflow debug incident-report \
  --start "2026-01-24T10:30:00Z" \
  --end "2026-01-24T11:00:00Z" \
  --output incident-report.md

# Capture system snapshot

cloudflow debug snapshot --output snapshot-$(date +%Y%m%d-%H%M%S).tar.gz
```

#### Step 3: Escalate Based on Severity

**For SEV-1 (Critical):**
```bash

# Page on-call engineer

cloudflow incident create \
  --severity SEV-1 \
  --title "Complete API outage" \
  --description "All API requests returning 503" \
  --page-oncall

# Notify status page

cloudflow status update \
  --status "major_outage" \
  --message "We are investigating a complete service outage"

# Create war room

cloudflow incident war-room create incident-2024012401
```

**For SEV-2 (High):**
```bash

# Create incident and notify

cloudflow incident create \
  --severity SEV-2 \
  --title "High rate of workflow failures" \
  --description "Workflow success rate dropped to 75%" \
  --notify-oncall

# Update status page

cloudflow status update \
  --status "partial_outage" \
  --message "We are experiencing elevated error rates"
```

**For SEV-3/SEV-4:**
```bash

# Create support ticket

cloudflow support ticket create \
  --priority MEDIUM \
  --subject "Intermittent timeout errors" \
  --description "See attached logs and reproduction steps"
```

### Contact Information

- **On-call hotline:** +1-888-CLOUDFLOW (24/7)
- **Support email:** support@cloudflow.io
- **Incident Slack channel:** #cloudflow-incidents
- **Status page:** https://status.cloudflow.io
- **Documentation:** https://docs.cloudflow.io

### Post-Incident Review

After resolving incidents, complete a postmortem:

```bash

# Generate postmortem template

cloudflow incident postmortem incident-2024012401 \
  --template standard \
  --output postmortem-2024012401.md

# Required sections:



# 1. Incident summary



# 2. Impact assessment



# 3. Timeline of events



# 4. Root cause analysis



# 5. Resolution steps



# 6. Action items



# 7. Lessons learned

```

---

## Additional Resources

- **CloudFlow Documentation:** https://docs.cloudflow.io
- **API Reference:** https://api-docs.cloudflow.io
- **Community Forum:** https://community.cloudflow.io
- **GitHub Issues:** https://github.com/cloudflow/platform/issues
- **Training Portal:** https://training.cloudflow.io
- **Monitoring Dashboard:** https://monitoring.cloudflow.io

5. **[user_guide]**

   ### Steps Per Workflow

- **Maximum**: 50 steps per workflow
- **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via webhooks.

### Execution Timeout

- **Default**: 3600 seconds (60 minutes)
- **Behavior**: Workflows exceeding this timeout are automatically terminated
- **Custom Timeouts**: Enterprise plans can request custom timeout limits

**Setting Step-Level Timeouts:**
```yaml
- id: long_running_task
  action: http_request
  config:
    url: "https://api.example.com/process"
    timeout: 300  # 5 minutes for this specific step
```

---

### Question 9: How do I set up database connection pooling in CloudFlow?

**Expected Answer**: Use PgBouncer for connection pooling, configure CloudFlow to use PgBouncer, set connection pool modes, and add read replicas to optimize database performance

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[deployment_guide]**

   ### Environment Configuration

CloudFlow requires the following environment variables:

\| Variable \| Description \| Example \| Required \|
\|----------\|-------------\|---------\|----------\|
\| `DATABASE_URL` \| PostgreSQL connection string \| `postgresql://user:pass@host:5432/cloudflow` \| Yes \|
\| `REDIS_URL` \| Redis connection string \| `redis://redis-master.cloudflow-prod.svc.cluster.local:6379` \| Yes \|
\| `JWT_SECRET` \| Secret key for JWT token signing \| `<generated-secret-256-bit>` \| Yes \|
\| `LOG_LEVEL` \| Application log verbosity \| `info`, `debug`, `warn`, `error` \| No (default: `info`) \|
\| `API_PORT` \| API server port \| `3000` \| No (default: `3000`) \|
\| `WORKER_CONCURRENCY` \| Number of concurrent workers \| `10` \| No (default: `5`) \|
\| `SESSION_SECRET` \| Session encryption key \| `<generated-secret-256-bit>` \| Yes \|
\| `AWS_REGION` \| AWS region for services \| `us-east-1` \| Yes \|
\| `S3_BUCKET` \| S3 bucket for file storage \| `cloudflow-prod-storage` \| Yes \|
\| `SMTP_HOST` \| Email server hostname \| `smtp.sendgrid.net` \| No \|
\| `SMTP_PORT` \| Email server port \| `587` \| No \|
\| `METRICS_ENABLED` \| Enable Prometheus metrics \| `true` \| No (default: `false`) \|

Create a Kubernetes secret for sensitive variables:

```bash
kubectl create secret generic cloudflow-secrets \
  --namespace cloudflow-prod \
  --from-literal=DATABASE_URL="postgresql://cloudflow:$(cat db-password.txt)@pgbouncer.cloudflow-prod.svc.cluster.local:5432/cloudflow" \
  --from-literal=REDIS_URL="redis://redis-master.cloudflow-prod.svc.cluster.local:6379" \
  --from-literal=JWT_SECRET="$(openssl rand -base64 32)" \
  --from-literal=SESSION_SECRET="$(openssl rand -base64 32)"
```

### Helm Chart Deployment

Add the CloudFlow Helm repository:

```bash
helm repo add cloudflow https://charts.cloudflow.io
helm repo update
```

Create a values file for production configuration:

```yaml

# values-production.yaml

replicaCount: 3

image:
  repository: 123456789012.dkr.ecr.us-east-1.amazonaws.com/cloudflow
  tag: "2.4.0"
  pullPolicy: IfNotPresent

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

service:
  type: ClusterIP
  port: 80
  targetPort: 3000
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: api.cloudflow.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cloudflow-tls
      hosts:
        - api.cloudflow.io

healthCheck:
  enabled: true
  livenessProbe:
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  readinessProbe:
    path: /ready
    initialDelaySeconds: 10
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3

env:
  - name: NODE_ENV
    value: "production"
  - name: LOG_LEVEL
    value: "info"
  - name: API_PORT
    value: "3000"
  - name: WORKER_CONCURRENCY
    value: "10"
  - name: AWS_REGION
    value: "us-east-1"
  - name: METRICS_ENABLED
    value: "true"

envFrom:
  - secretRef:
      name: cloudflow-secrets

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true

persistence:
  enabled: true
  storageClass: gp3
  accessMode: ReadWriteOnce
  size: 50Gi

redis:
  enabled: true
  architecture: replication
  auth:
    enabled: true
  master:
    persistence:
      size: 20Gi
  replica:
    replicaCount: 2
    persistence:
      size: 20Gi
```

Deploy CloudFlow using Helm:

```bash
helm install cloudflow cloudflow/cloudflow \
  --namespace cloudflow-prod \
  --values values-production.yaml \
  --wait \
  --timeout 10m
```

### Deployment Verification

Verify the deployment status:

```bash

# Check pod status

kubectl get pods -n cloudflow-prod

# Expected output:



# NAME                              READY   STATUS    RESTARTS   AGE



# cloudflow-api-7d8f9c5b6d-4xk2p   1/1     Running   0          2m



# cloudflow-api-7d8f9c5b6d-9hj5m   1/1     Running   0          2m



# cloudflow-api-7d8f9c5b6d-tn8wq   1/1     Running   0          2m



# Check service endpoints

kubectl get svc -n cloudflow-prod

# Check ingress

kubectl get ingress -n cloudflow-prod
```

Test health endpoints:

```bash

# Port forward for testing

kubectl port-forward -n cloudflow-prod svc/cloudflow-api 8080:80

# Test health endpoint

curl http://localhost:8080/health

# Expected: {"status":"healthy","timestamp":"2026-01-24T10:30:00Z"}



# Test readiness endpoint

curl http://localhost:8080/ready

# Expected: {"status":"ready","dependencies":{"database":"connected","redis":"connected"}}

```

---

## Database Configuration



### PostgreSQL Setup

CloudFlow uses PostgreSQL 14 as its primary data store. Deploy PostgreSQL using the Bitnami Helm chart:

```yaml

2. **[deployment_guide]**

   ### Required Tools

- `kubectl` (v1.28 or later)
- `helm` (v3.12 or later)
- `aws-cli` (v2.13 or later)
- `eksctl` (v0.165 or later)
- `terraform` (v1.6 or later) - for infrastructure provisioning

### Access Requirements

- AWS account with appropriate IAM permissions
- EKS cluster admin access
- Container registry access (ECR)
- Domain name and SSL certificates
- Secrets management access (AWS Secrets Manager or Vault)

### Network Requirements

- VPC with at least 3 public and 3 private subnets
- NAT Gateway configured
- Security groups allowing:
  - Ingress: HTTPS (443), HTTP (80)
  - Internal: PostgreSQL (5432), Redis (6379)
  - Monitoring: Prometheus (9090), Grafana (3000)

---

## Infrastructure Setup



### EKS Cluster Creation

Use the following `eksctl` configuration to create the EKS cluster:

```yaml

# cluster-config.yaml

apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: cloudflow-production
  region: us-east-1
  version: "1.28"

availabilityZones:
  - us-east-1a
  - us-east-1b
  - us-east-1c

vpc:
  cidr: 10.0.0.0/16
  nat:
    gateway: HighlyAvailable

managedNodeGroups:
  - name: cloudflow-workers
    instanceType: m5.xlarge
    minSize: 3
    maxSize: 10
    desiredCapacity: 5
    volumeSize: 100
    privateNetworking: true
    labels:
      role: worker
      environment: production
    tags:
      Environment: production
      Application: cloudflow
    iam:
      withAddonPolicies:
        ebs: true
        efs: true
        albIngress: true
        cloudWatch: true

cloudWatch:
  clusterLogging:
    enableTypes:
      - api
      - audit
      - authenticator
      - controllerManager
      - scheduler
```

Create the cluster:

```bash
eksctl create cluster -f cluster-config.yaml
```

### Storage Configuration

Install the EBS CSI driver for persistent volumes:

```bash
eksctl create addon --name aws-ebs-csi-driver \
  --cluster cloudflow-production \
  --service-account-role-arn arn:aws:iam::ACCOUNT_ID:role/AmazonEKS_EBS_CSI_DriverRole \
  --force
```

---

## Kubernetes Deployment



### Namespace Setup

Create the CloudFlow namespace and configure resource quotas:

```yaml

# namespace.yaml

apiVersion: v1
kind: Namespace
metadata:
  name: cloudflow-prod
  labels:
    name: cloudflow-prod
    environment: production

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: cloudflow-quota
  namespace: cloudflow-prod
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
```

Apply the namespace configuration:

```bash
kubectl apply -f namespace.yaml
```

3. **[architecture_overview]**

   ### Architecture Diagram (Conceptual)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer (ALB)                       │
│                     (TLS Termination - 443)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                         │
│              (Rate Limiting, Auth, Request Routing)              │
└─┬───────┬──────────┬────────────┬──────────────┬────────────────┘
  │       │          │            │              │
  ▼       ▼          ▼            ▼              ▼
┌────┐ ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│Auth│ │Workflow│ │Scheduler│ │Notification│ │User Service│
│Svc │ │ Engine │ │ Service │ │  Service  │ │            │
└─┬──┘ └───┬────┘ └────┬─────┘ └─────┬────┘ └──────┬─────┘
  │        │           │             │              │
  └────────┴───────────┴─────────────┴──────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
  ┌─────────┐    ┌──────────┐    ┌─────────┐
  │PostgreSQL│    │  Redis   │    │  Kafka  │
  │ Cluster │    │  Cluster │    │ Cluster │
  └─────────┘    └──────────┘    └─────────┘
```

### Technology Stack

- **Runtime**: Node.js 20.x (API Gateway, Workflow Engine), Go 1.21 (Auth Service, Scheduler)
- **Container Orchestration**: Kubernetes 1.28 (EKS)
- **Service Mesh**: Istio 1.20 for service-to-service communication
- **Databases**: PostgreSQL 15.4, Redis 7.2
- **Message Broker**: Apache Kafka 3.6
- **Monitoring**: Prometheus, Grafana, Jaeger for distributed tracing
- **Secrets Management**: HashiCorp Vault 1.15

---

4. **[deployment_guide]**

   # postgres-values.yaml

global:
  postgresql:
    auth:
      username: cloudflow
      database: cloudflow
      existingSecret: postgres-credentials

image:
  tag: "14.10.0"

primary:
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 2000m
      memory: 4Gi
  
  persistence:
    enabled: true
    size: 100Gi
    storageClass: gp3
  
  extendedConfiguration: \|
    max_connections = 100
    shared_buffers = 2GB
    effective_cache_size = 6GB
    maintenance_work_mem = 512MB
    checkpoint_completion_target = 0.9
    wal_buffers = 16MB
    default_statistics_target = 100
    random_page_cost = 1.1
    effective_io_concurrency = 200
    work_mem = 10MB
    min_wal_size = 1GB
    max_wal_size = 4GB

readReplicas:
  replicaCount: 2
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 1000m
      memory: 2Gi

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
```

Install PostgreSQL:

```bash

# Create password secret

kubectl create secret generic postgres-credentials \
  --namespace cloudflow-prod \
  --from-literal=postgres-password="$(openssl rand -base64 32)" \
  --from-literal=password="$(openssl rand -base64 32)"

# Install PostgreSQL

helm install postgresql bitnami/postgresql \
  --namespace cloudflow-prod \
  --values postgres-values.yaml \
  --wait
```

### PgBouncer Configuration

Deploy PgBouncer for connection pooling to handle up to 100 connections efficiently:

```yaml

# pgbouncer.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: pgbouncer-config
  namespace: cloudflow-prod
data:
  pgbouncer.ini: \|
    [databases]
    cloudflow = host=postgresql.cloudflow-prod.svc.cluster.local port=5432 dbname=cloudflow
    
    [pgbouncer]
    listen_addr = 0.0.0.0
    listen_port = 5432
    auth_type = md5
    auth_file = /etc/pgbouncer/userlist.txt
    pool_mode = transaction
    max_client_conn = 1000
    default_pool_size = 25
    reserve_pool_size = 5
    reserve_pool_timeout = 3
    max_db_connections = 100
    max_user_connections = 100
    server_lifetime = 3600
    server_idle_timeout = 600
    log_connections = 1
    log_disconnections = 1
    log_pooler_errors = 1

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: cloudflow-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: pgbouncer/pgbouncer:1.21.0
        ports:
        - containerPort: 5432
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        volumeMounts:
        - name: config
          mountPath: /etc/pgbouncer
      volumes:
      - name: config
        configMap:
          name: pgbouncer-config

---
apiVersion: v1
kind: Service
metadata:
  name: pgbouncer
  namespace: cloudflow-prod
spec:
  selector:
    app: pgbouncer
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

Apply the PgBouncer configuration:

```bash
kubectl apply -f pgbouncer.yaml
```

### Database Migrations

Run database migrations before deploying new versions:

```bash

# Create migration job

kubectl create job cloudflow-migrate-$(date +%s) \
  --namespace cloudflow-prod \
  --image=123456789012.dkr.ecr.us-east-1.amazonaws.com/cloudflow:2.4.0 \
  -- npm run migrate

# Monitor migration progress

kubectl logs -f job/cloudflow-migrate-<timestamp> -n cloudflow-prod
```

---

## Monitoring and Observability



5. **[troubleshooting_guide]**

   # Check database slow query log

kubectl logs -n cloudflow deploy/cloudflow-db-primary \| \
  grep "slow query" \| \
  tail -n 50

# Analyze query patterns

cloudflow db analyze-queries --min-duration 5000 --limit 20
```

**2. Review Query Execution Plans**

```sql
-- Connect to CloudFlow database
cloudflow db connect --readonly

-- Explain slow query
EXPLAIN ANALYZE
SELECT w.*, e.status, e.error_message
FROM workflows w
LEFT JOIN executions e ON w.id = e.workflow_id
WHERE w.workspace_id = 'ws_abc123'
  AND e.created_at > NOW() - INTERVAL '7 days'
ORDER BY e.created_at DESC;

-- Check for missing indexes
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('workflows', 'executions', 'workflow_steps')
ORDER BY tablename, indexname;
```

**3. Optimize Queries**

Common optimization techniques:

**Add missing indexes:**
```sql
-- Index for workflow lookup by workspace
CREATE INDEX CONCURRENTLY idx_workflows_workspace_created 
ON workflows(workspace_id, created_at DESC);

-- Index for execution status queries
CREATE INDEX CONCURRENTLY idx_executions_status_created
ON executions(workflow_id, status, created_at DESC)
WHERE status IN ('running', 'pending');

-- Composite index for common filter combinations
CREATE INDEX CONCURRENTLY idx_executions_workspace_status
ON executions(workspace_id, status, created_at)
INCLUDE (error_message, retry_count);
```

**Use query result caching:**
```bash

# Enable query cache for workspace metadata

cloudflow config set cache.workspace.ttl 3600

# Configure Redis cache backend

cloudflow config set cache.backend redis
cloudflow config set cache.redis.host redis.cloudflow.svc.cluster.local
cloudflow config set cache.redis.port 6379
```

### High API Latency



#### Latency Breakdown Analysis

```bash

# Generate latency report

cloudflow metrics latency-report --endpoint "/api/v1/workflows" --last 24h

# Sample output:



# Endpoint: POST /api/v1/workflows/execute



# P50: 245ms \| P95: 1823ms \| P99: 4521ms



# Breakdown:



# - Auth: 45ms (18%)



# - DB Query: 156ms (64%)



# - Business Logic: 32ms (13%)



# - Response Serialization: 12ms (5%)

```

**Network latency issues:**
```bash

# Test connectivity to CloudFlow API

time curl -w "@curl-format.txt" -o /dev/null -s https://api.cloudflow.io/health

# Create curl-format.txt:

cat > curl-format.txt << EOF
    time_namelookup:  %{time_namelookup}s\n
       time_connect:  %{time_connect}s\n
    time_appconnect:  %{time_appconnect}s\n
   time_pretransfer:  %{time_pretransfer}s\n
      time_redirect:  %{time_redirect}s\n
 time_starttransfer:  %{time_starttransfer}s\n
                    ----------\n
         time_total:  %{time_total}s\n
EOF

# Trace route to API endpoint

traceroute api.cloudflow.io

# Check DNS resolution time

dig api.cloudflow.io \| grep "Query time"
```

### Memory Leaks



#### Detection

```bash

# Monitor CloudFlow service memory usage

kubectl top pods -n cloudflow --sort-by=memory

# Get detailed memory metrics for specific pod

kubectl exec -n cloudflow deploy/cloudflow-api -- \
  curl localhost:9090/metrics \| grep memory

# Check for OOMKilled pods

kubectl get pods -n cloudflow --field-selector=status.phase=Failed \| \
  grep OOMKilled

# Review memory limits and requests

kubectl describe deployment cloudflow-api -n cloudflow \| \
  grep -A 5 "Limits\\|Requests"
```

#### Common Causes

**1. Workflow Context Accumulation**

Large workflow executions may accumulate state in memory.

**Solution:**
```bash

# Configure context cleanup

cloudflow config set workflow.context.max_size_mb 100
cloudflow config set workflow.context.cleanup_threshold 0.8

# Enable context persistence to disk

cloudflow config set workflow.context.persistence.enabled true
cloudflow config set workflow.context.persistence.backend s3
```

**2. Connection Pool Leaks**

**Diagnosis:**
```bash

# Check active connections

cloudflow db connections --verbose

# Expected output:



# Active: 45/100



# Idle: 23



# Waiting: 2



# Average age: 245s

```

**Resolution:**
```bash

# Adjust connection pool settings

cloudflow config set db.pool.max_connections 100
cloudflow config set db.pool.min_connections 10
cloudflow config set db.pool.idle_timeout 300
cloudflow config set db.pool.max_lifetime 1800

# Force connection pool reset

cloudflow db pool reset --confirm
```

**3. Event Stream Buffers**

Unbounded event buffers can cause memory exhaustion.

```bash

# Configure event buffer limits

cloudflow config set events.buffer.max_size 10000
cloudflow config set events.buffer.overflow_strategy drop_oldest

# Enable event streaming to external sink

cloudflow config set events.sink.type kafka
cloudflow config set events.sink.kafka.brokers "kafka-1:9092,kafka-2:9092"
cloudflow config set events.sink.kafka.topic cloudflow-events
```

---

## Database Connection Issues



### Connection Pool Exhaustion



#### Symptoms

- Error: `could not obtain connection from pool within 5000ms`
- Error: `connection pool exhausted (100/100 connections in use)`
- API requests fail with `503 Service Unavailable`
- Database CPU usage normal, but connection count at maximum

#### Investigation

```bash

# Check current connection pool status

cloudflow db pool status --detailed

# Output example:



# Pool Statistics:



# Total Connections: 100/100 (100%)



# Active: 87



# Idle: 13



# Waiting Requests: 45



# Average Wait Time: 3420ms



# Max Wait Time: 8234ms



# Identify long-running queries

cloudflow db queries --status running --min-duration 30000

# Check connection distribution by client

SELECT application_name, state, COUNT(*) as conn_count,
       AVG(EXTRACT(EPOCH FROM (NOW() - state_change))) as avg_duration_sec
FROM pg_stat_activity
WHERE datname = 'cloudflow_production'
GROUP BY application_name, state
ORDER BY conn_count DESC;
```

#### Resolution

**Immediate mitigation:**
```bash

# Temporarily increase connection limit (requires database restart)

cloudflow db config set max_connections 150

# Kill idle connections older than 5 minutes

cloudflow db connections kill --idle-timeout 300

# Restart connection pool without downtime

kubectl rollout restart deployment/cloudflow-api -n cloudflow
kubectl rollout status deployment/cloudflow-api -n cloudflow
```

**Long-term solutions:**

1. **Implement connection pooling optimization:**
```bash

# Use PgBouncer for connection pooling

kubectl apply -f cloudflow-pgbouncer.yaml

# Configure CloudFlow to use PgBouncer

cloudflow config set db.host pgbouncer.cloudflow.svc.cluster.local
cloudflow config set db.port 6432
cloudflow config set db.pool.mode transaction
```

2. **Add read replicas:**
```bash

# Route read-only queries to replicas

cloudflow db replicas add --count 2
cloudflow config set db.read_replicas "replica-1:5432,replica-2:5432"
cloudflow config set db.read_write_split true
```

### Connection Timeout Errors



#### Error Messages

- `connection timeout after 30000ms`
- `could not connect to database server at 10.0.2.45:5432`
- `database server unreachable`

#### Troubleshooting Steps

```bash

# Test network connectivity

telnet cloudflow-db.internal.company.com 5432

# Check DNS resolution

nslookup cloudflow-db.internal.company.com

# Verify database is accepting connections

pg_isready -h cloudflow-db.internal.company.com -p 5432 -U cloudflow

# Check firewall rules

sudo iptables -L -n \| grep 5432

# Test from CloudFlow pod network

kubectl run -n cloudflow debug-pod --rm -i --tty \
  --image=postgres:14 -- \
  psql -h cloudflow-db.internal.company.com -U cloudflow -d cloudflow_production

# Review database logs for connection rejections

kubectl logs -n cloudflow statefulset/cloudflow-db --tail=100 \| \
  grep -i "connection\\|reject\\|authentication"
```

### Maximum Connection Limit (100) Reached

This is a hard limit in CloudFlow's database tier.

#### Permanent Solutions

**Option 1: Upgrade database tier**
```bash

# Check available tiers

cloudflow db tiers list

# Upgrade to higher tier (supports 200 connections)

cloudflow db upgrade --tier standard-plus --confirm

# Monitor migration progress

cloudflow db migration status
```

**Option 2: Implement aggressive connection reuse**
```bash

# Reduce connection lifetime

cloudflow config set db.pool.max_lifetime 600  # 10 minutes

# Enable prepared statement caching

cloudflow config set db.prepared_statements.cache true
cloudflow config set db.prepared_statements.max_size 250

# Reduce idle connection timeout

cloudflow config set db.pool.idle_timeout 120  # 2 minutes
```

---

## Workflow Execution Failures



### Timeout Errors (3600 second limit)



#### Error Message

```
WorkflowExecutionError: Workflow exceeded maximum execution time of 3600 seconds
Status: TIMEOUT
Workflow ID: wf_9k2n4m8p1q
Execution ID: exec_7h3j6k9m2n
```

#### Analysis

```bash

# Get workflow execution details

cloudflow workflows executions get exec_7h3j6k9m2n --verbose

# Check step-by-step breakdown

cloudflow workflows executions steps exec_7h3j6k9m2n

# Sample output:



# Step 1: data_ingestion     - Duration: 245s    - Status: SUCCESS



# Step 2: data_validation    - Duration: 123s    - Status: SUCCESS



# Step 3: data_transformation - Duration: 3189s   - Status: TIMEOUT



# Step 4: data_export        - Duration: 0s      - Status: SKIPPED



# Identify bottleneck step

cloudflow workflows analyze exec_7h3j6k9m2n --show-bottlenecks
```

#### Solutions

**1. Increase workflow timeout (if justified):**
```bash

# Update workflow configuration

cloudflow workflows update wf_9k2n4m8p1q \
  --timeout 7200 \
  --reason "Large dataset processing requires extended time"

# Verify update

cloudflow workflows get wf_9k2n4m8p1q \| grep timeout
```

**2. Optimize slow steps:**
```bash

# Enable parallel processing

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_transformation \
  --parallel-workers 8 \
  --batch-size 1000

# Add checkpointing for long operations

cloudflow workflows update wf_9k2n4m8p1q \
  --step data_transformation \
  --enable-checkpointing \
  --checkpoint-interval 300
```

**3. Split workflow into smaller workflows:**
```bash

---

### Question 10: What Kubernetes resources are needed to deploy CloudFlow?

**Expected Answer**: Deploy an EKS cluster with managed node groups, configure storage with EBS CSI driver, set up a namespace with resource quotas, and use Helm charts for deployment

**Your Score**: ___ / 10

**Your Notes**: 

**Retrieved Chunks**:

1. **[deployment_guide]**

   ## Overview

CloudFlow is a cloud-native workflow orchestration platform designed for high-availability production environments. This guide provides comprehensive instructions for deploying and operating CloudFlow on Amazon EKS (Elastic Kubernetes Service).

### Architecture Summary

CloudFlow consists of the following components:

- **API Server**: REST API for workflow management (Node.js/Express)
- **Worker Service**: Background job processor (Node.js)
- **Scheduler**: Cron-based task scheduler (Node.js)
- **PostgreSQL**: Primary data store (version 14)
- **Redis**: Cache and message queue (version 7.0)
- **PgBouncer**: Database connection pooler

### Deployment Model

- **Namespace**: `cloudflow-prod`
- **Cluster**: AWS EKS 1.28
- **Region**: us-east-1 (primary), us-west-2 (disaster recovery)
- **High Availability**: Multi-AZ deployment across 3 availability zones

---

## Prerequisites

Before beginning the deployment process, ensure you have the following:

2. **[deployment_guide]**

   ### Required Tools

- `kubectl` (v1.28 or later)
- `helm` (v3.12 or later)
- `aws-cli` (v2.13 or later)
- `eksctl` (v0.165 or later)
- `terraform` (v1.6 or later) - for infrastructure provisioning

### Access Requirements

- AWS account with appropriate IAM permissions
- EKS cluster admin access
- Container registry access (ECR)
- Domain name and SSL certificates
- Secrets management access (AWS Secrets Manager or Vault)

### Network Requirements

- VPC with at least 3 public and 3 private subnets
- NAT Gateway configured
- Security groups allowing:
  - Ingress: HTTPS (443), HTTP (80)
  - Internal: PostgreSQL (5432), Redis (6379)
  - Monitoring: Prometheus (9090), Grafana (3000)

---

## Infrastructure Setup



### EKS Cluster Creation

Use the following `eksctl` configuration to create the EKS cluster:

```yaml

# cluster-config.yaml

apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: cloudflow-production
  region: us-east-1
  version: "1.28"

availabilityZones:
  - us-east-1a
  - us-east-1b
  - us-east-1c

vpc:
  cidr: 10.0.0.0/16
  nat:
    gateway: HighlyAvailable

managedNodeGroups:
  - name: cloudflow-workers
    instanceType: m5.xlarge
    minSize: 3
    maxSize: 10
    desiredCapacity: 5
    volumeSize: 100
    privateNetworking: true
    labels:
      role: worker
      environment: production
    tags:
      Environment: production
      Application: cloudflow
    iam:
      withAddonPolicies:
        ebs: true
        efs: true
        albIngress: true
        cloudWatch: true

cloudWatch:
  clusterLogging:
    enableTypes:
      - api
      - audit
      - authenticator
      - controllerManager
      - scheduler
```

Create the cluster:

```bash
eksctl create cluster -f cluster-config.yaml
```

### Storage Configuration

Install the EBS CSI driver for persistent volumes:

```bash
eksctl create addon --name aws-ebs-csi-driver \
  --cluster cloudflow-production \
  --service-account-role-arn arn:aws:iam::ACCOUNT_ID:role/AmazonEKS_EBS_CSI_DriverRole \
  --force
```

---

## Kubernetes Deployment



### Namespace Setup

Create the CloudFlow namespace and configure resource quotas:

```yaml

# namespace.yaml

apiVersion: v1
kind: Namespace
metadata:
  name: cloudflow-prod
  labels:
    name: cloudflow-prod
    environment: production

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: cloudflow-quota
  namespace: cloudflow-prod
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
```

Apply the namespace configuration:

```bash
kubectl apply -f namespace.yaml
```

3. **[deployment_guide]**

   ### Environment Configuration

CloudFlow requires the following environment variables:

\| Variable \| Description \| Example \| Required \|
\|----------\|-------------\|---------\|----------\|
\| `DATABASE_URL` \| PostgreSQL connection string \| `postgresql://user:pass@host:5432/cloudflow` \| Yes \|
\| `REDIS_URL` \| Redis connection string \| `redis://redis-master.cloudflow-prod.svc.cluster.local:6379` \| Yes \|
\| `JWT_SECRET` \| Secret key for JWT token signing \| `<generated-secret-256-bit>` \| Yes \|
\| `LOG_LEVEL` \| Application log verbosity \| `info`, `debug`, `warn`, `error` \| No (default: `info`) \|
\| `API_PORT` \| API server port \| `3000` \| No (default: `3000`) \|
\| `WORKER_CONCURRENCY` \| Number of concurrent workers \| `10` \| No (default: `5`) \|
\| `SESSION_SECRET` \| Session encryption key \| `<generated-secret-256-bit>` \| Yes \|
\| `AWS_REGION` \| AWS region for services \| `us-east-1` \| Yes \|
\| `S3_BUCKET` \| S3 bucket for file storage \| `cloudflow-prod-storage` \| Yes \|
\| `SMTP_HOST` \| Email server hostname \| `smtp.sendgrid.net` \| No \|
\| `SMTP_PORT` \| Email server port \| `587` \| No \|
\| `METRICS_ENABLED` \| Enable Prometheus metrics \| `true` \| No (default: `false`) \|

Create a Kubernetes secret for sensitive variables:

```bash
kubectl create secret generic cloudflow-secrets \
  --namespace cloudflow-prod \
  --from-literal=DATABASE_URL="postgresql://cloudflow:$(cat db-password.txt)@pgbouncer.cloudflow-prod.svc.cluster.local:5432/cloudflow" \
  --from-literal=REDIS_URL="redis://redis-master.cloudflow-prod.svc.cluster.local:6379" \
  --from-literal=JWT_SECRET="$(openssl rand -base64 32)" \
  --from-literal=SESSION_SECRET="$(openssl rand -base64 32)"
```

### Helm Chart Deployment

Add the CloudFlow Helm repository:

```bash
helm repo add cloudflow https://charts.cloudflow.io
helm repo update
```

Create a values file for production configuration:

```yaml

# values-production.yaml

replicaCount: 3

image:
  repository: 123456789012.dkr.ecr.us-east-1.amazonaws.com/cloudflow
  tag: "2.4.0"
  pullPolicy: IfNotPresent

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

service:
  type: ClusterIP
  port: 80
  targetPort: 3000
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: api.cloudflow.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cloudflow-tls
      hosts:
        - api.cloudflow.io

healthCheck:
  enabled: true
  livenessProbe:
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  readinessProbe:
    path: /ready
    initialDelaySeconds: 10
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3

env:
  - name: NODE_ENV
    value: "production"
  - name: LOG_LEVEL
    value: "info"
  - name: API_PORT
    value: "3000"
  - name: WORKER_CONCURRENCY
    value: "10"
  - name: AWS_REGION
    value: "us-east-1"
  - name: METRICS_ENABLED
    value: "true"

envFrom:
  - secretRef:
      name: cloudflow-secrets

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true

persistence:
  enabled: true
  storageClass: gp3
  accessMode: ReadWriteOnce
  size: 50Gi

redis:
  enabled: true
  architecture: replication
  auth:
    enabled: true
  master:
    persistence:
      size: 20Gi
  replica:
    replicaCount: 2
    persistence:
      size: 20Gi
```

Deploy CloudFlow using Helm:

```bash
helm install cloudflow cloudflow/cloudflow \
  --namespace cloudflow-prod \
  --values values-production.yaml \
  --wait \
  --timeout 10m
```

### Deployment Verification

Verify the deployment status:

```bash

# Check pod status

kubectl get pods -n cloudflow-prod

# Expected output:



# NAME                              READY   STATUS    RESTARTS   AGE



# cloudflow-api-7d8f9c5b6d-4xk2p   1/1     Running   0          2m



# cloudflow-api-7d8f9c5b6d-9hj5m   1/1     Running   0          2m



# cloudflow-api-7d8f9c5b6d-tn8wq   1/1     Running   0          2m



# Check service endpoints

kubectl get svc -n cloudflow-prod

# Check ingress

kubectl get ingress -n cloudflow-prod
```

Test health endpoints:

```bash

# Port forward for testing

kubectl port-forward -n cloudflow-prod svc/cloudflow-api 8080:80

# Test health endpoint

curl http://localhost:8080/health

# Expected: {"status":"healthy","timestamp":"2026-01-24T10:30:00Z"}



# Test readiness endpoint

curl http://localhost:8080/ready

# Expected: {"status":"ready","dependencies":{"database":"connected","redis":"connected"}}

```

---

## Database Configuration



### PostgreSQL Setup

CloudFlow uses PostgreSQL 14 as its primary data store. Deploy PostgreSQL using the Bitnami Helm chart:

```yaml

4. **[deployment_guide]**

   # CloudFlow Platform - Deployment and Operations Guide

**Version:** 2.4.0  
**Last Updated:** January 2026  
**Target Environment:** Production (AWS EKS)

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Database Configuration](#database-configuration)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
8. [Scaling and Performance](#scaling-and-performance)
9. [Security and Compliance](#security-and-compliance)
10. [Troubleshooting](#troubleshooting)

---

5. **[deployment_guide]**

   # postgres-values.yaml

global:
  postgresql:
    auth:
      username: cloudflow
      database: cloudflow
      existingSecret: postgres-credentials

image:
  tag: "14.10.0"

primary:
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 2000m
      memory: 4Gi
  
  persistence:
    enabled: true
    size: 100Gi
    storageClass: gp3
  
  extendedConfiguration: \|
    max_connections = 100
    shared_buffers = 2GB
    effective_cache_size = 6GB
    maintenance_work_mem = 512MB
    checkpoint_completion_target = 0.9
    wal_buffers = 16MB
    default_statistics_target = 100
    random_page_cost = 1.1
    effective_io_concurrency = 200
    work_mem = 10MB
    min_wal_size = 1GB
    max_wal_size = 4GB

readReplicas:
  replicaCount: 2
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 1000m
      memory: 2Gi

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
```

Install PostgreSQL:

```bash

# Create password secret

kubectl create secret generic postgres-credentials \
  --namespace cloudflow-prod \
  --from-literal=postgres-password="$(openssl rand -base64 32)" \
  --from-literal=password="$(openssl rand -base64 32)"

# Install PostgreSQL

helm install postgresql bitnami/postgresql \
  --namespace cloudflow-prod \
  --values postgres-values.yaml \
  --wait
```

### PgBouncer Configuration

Deploy PgBouncer for connection pooling to handle up to 100 connections efficiently:

```yaml

# pgbouncer.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: pgbouncer-config
  namespace: cloudflow-prod
data:
  pgbouncer.ini: \|
    [databases]
    cloudflow = host=postgresql.cloudflow-prod.svc.cluster.local port=5432 dbname=cloudflow
    
    [pgbouncer]
    listen_addr = 0.0.0.0
    listen_port = 5432
    auth_type = md5
    auth_file = /etc/pgbouncer/userlist.txt
    pool_mode = transaction
    max_client_conn = 1000
    default_pool_size = 25
    reserve_pool_size = 5
    reserve_pool_timeout = 3
    max_db_connections = 100
    max_user_connections = 100
    server_lifetime = 3600
    server_idle_timeout = 600
    log_connections = 1
    log_disconnections = 1
    log_pooler_errors = 1

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: cloudflow-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: pgbouncer/pgbouncer:1.21.0
        ports:
        - containerPort: 5432
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        volumeMounts:
        - name: config
          mountPath: /etc/pgbouncer
      volumes:
      - name: config
        configMap:
          name: pgbouncer-config

---
apiVersion: v1
kind: Service
metadata:
  name: pgbouncer
  namespace: cloudflow-prod
spec:
  selector:
    app: pgbouncer
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

Apply the PgBouncer configuration:

```bash
kubectl apply -f pgbouncer.yaml
```

### Database Migrations

Run database migrations before deploying new versions:

```bash

# Create migration job

kubectl create job cloudflow-migrate-$(date +%s) \
  --namespace cloudflow-prod \
  --image=123456789012.dkr.ecr.us-east-1.amazonaws.com/cloudflow:2.4.0 \
  -- npm run migrate

# Monitor migration progress

kubectl logs -f job/cloudflow-migrate-<timestamp> -n cloudflow-prod
```

---

## Monitoring and Observability



---
