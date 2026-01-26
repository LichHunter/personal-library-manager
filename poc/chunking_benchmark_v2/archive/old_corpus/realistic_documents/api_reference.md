# CloudFlow API Reference

Version 2.1.0 | Last Updated: January 2026

## Overview

The CloudFlow API is a RESTful service that enables developers to programmatically manage cloud workflows, data pipelines, and automation tasks. This documentation provides comprehensive details on authentication, endpoints, request/response formats, error handling, and best practices.

**Base URL:** `https://api.cloudflow.io/v2`

**API Status:** https://status.cloudflow.io

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

### Pipelines

#### List Pipeline Executions

Retrieve execution history for data pipelines.

**Endpoint:** `GET /pipelines/{pipeline_id}/executions`

**Query Parameters:**
- `status` (optional): Filter by status (`running`, `completed`, `failed`)
- `start_date` (optional): ISO 8601 timestamp
- `end_date` (optional): ISO 8601 timestamp
- `limit` (optional): Items per page (max 100)
- `offset` (optional): Pagination offset

**Example Request:**

```python
import requests

api_key = "YOUR_API_KEY"
pipeline_id = "pipe_4x9k2m"
url = f"https://api.cloudflow.io/v2/pipelines/{pipeline_id}/executions"

headers = {"X-API-Key": api_key}
params = {
    "status": "completed",
    "start_date": "2026-01-20T00:00:00Z",
    "limit": 50
}

response = requests.get(url, headers=headers, params=params)
executions = response.json()

for execution in executions['data']:
    print(f"Execution {execution['id']}: {execution['status']} - {execution['duration_ms']}ms")
```

**Example Response (200 OK):**

```json
{
  "data": [
    {
      "id": "exec_7m3k9x2p",
      "pipeline_id": "pipe_4x9k2m",
      "status": "completed",
      "started_at": "2026-01-23T15:00:00Z",
      "completed_at": "2026-01-23T15:03:42Z",
      "duration_ms": 222000,
      "records_processed": 15420,
      "records_failed": 3,
      "error_rate": 0.02
    }
  ],
  "pagination": {
    "total": 156,
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

### Analytics

#### Get Workflow Metrics

Retrieve performance metrics and analytics for workflows.

**Endpoint:** `GET /analytics/workflows/{workflow_id}`

**Query Parameters:**
- `period` (required): Time period (`1h`, `24h`, `7d`, `30d`)
- `metrics` (optional): Comma-separated list of metrics

**Available Metrics:**
- `execution_count`: Total number of executions
- `success_rate`: Percentage of successful executions
- `avg_duration`: Average execution duration in milliseconds
- `error_count`: Total number of errors
- `throughput`: Records processed per hour

**Example Request:**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://api.cloudflow.io/v2/analytics/workflows/wf_8x7k2m9p?period=7d&metrics=execution_count,success_rate,avg_duration"
```

**Example Response (200 OK):**

```json
{
  "workflow_id": "wf_8x7k2m9p",
  "period": "7d",
  "start_date": "2026-01-17T10:30:00Z",
  "end_date": "2026-01-24T10:30:00Z",
  "metrics": {
    "execution_count": 168,
    "success_rate": 97.6,
    "avg_duration": 3285,
    "error_count": 4,
    "throughput": 2847
  },
  "timeseries": [
    {
      "timestamp": "2026-01-17T00:00:00Z",
      "execution_count": 24,
      "success_rate": 100.0
    },
    {
      "timestamp": "2026-01-18T00:00:00Z",
      "execution_count": 24,
      "success_rate": 95.8
    }
  ]
}
```

## Error Handling

CloudFlow uses standard HTTP status codes and returns detailed error information in JSON format.

### Error Response Format

```json
{
  "error": {
    "code": "invalid_parameter",
    "message": "The 'limit' parameter must be between 1 and 100",
    "field": "limit",
    "request_id": "req_8k3m9x2p"
  }
}
```

### HTTP Status Codes

| Status Code | Description | Common Causes |
|------------|-------------|---------------|
| 400 | Bad Request | Invalid parameters, malformed JSON, validation errors |
| 401 | Unauthorized | Missing or invalid authentication credentials |
| 403 | Forbidden | Insufficient permissions for requested resource |
| 404 | Not Found | Resource does not exist or has been deleted |
| 429 | Too Many Requests | Rate limit exceeded, retry after specified period |
| 500 | Internal Server Error | Unexpected server error, contact support if persists |
| 502 | Bad Gateway | Temporary service issue, retry with exponential backoff |
| 503 | Service Unavailable | Scheduled maintenance or temporary outage |

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
