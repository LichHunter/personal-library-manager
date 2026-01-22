# API Reference

Complete reference for the CloudFlow REST API.

## Authentication

All API requests require authentication via Bearer token or API key.

### Bearer Token

```bash
curl -H "Authorization: Bearer <token>" https://api.cloudflow.io/v1/workflows
```

### API Key

```bash
curl -H "X-API-Key: <api-key>" https://api.cloudflow.io/v1/workflows
```

## Endpoints

### Workflows

#### List Workflows

```
GET /v1/workflows
```

Query parameters:
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20, max: 100)
- `status` (string): Filter by status

Response:

```json
{
  "data": [
    {
      "id": "wf_123",
      "name": "My Workflow",
      "status": "active",
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "meta": {
    "total": 42,
    "page": 1,
    "limit": 20
  }
}
```

#### Create Workflow

```
POST /v1/workflows
```

Request body:

```json
{
  "name": "My Workflow",
  "description": "Does something useful",
  "definition": {
    "trigger": { "type": "manual" },
    "steps": [...]
  }
}
```

#### Get Workflow

```
GET /v1/workflows/{id}
```

#### Update Workflow

```
PUT /v1/workflows/{id}
```

#### Delete Workflow

```
DELETE /v1/workflows/{id}
```

#### Execute Workflow

```
POST /v1/workflows/{id}/execute
```

Request body (optional):

```json
{
  "variables": {
    "input_data": "value"
  }
}
```

### Executions

#### List Executions

```
GET /v1/executions
```

Query parameters:
- `workflow_id` (string): Filter by workflow
- `status` (string): Filter by status
- `from` (datetime): Start date filter
- `to` (datetime): End date filter

#### Get Execution

```
GET /v1/executions/{id}
```

Response includes:
- Execution status
- Step results
- Logs
- Timing information

#### Cancel Execution

```
POST /v1/executions/{id}/cancel
```

### Users

#### Get Current User

```
GET /v1/users/me
```

#### Update User

```
PATCH /v1/users/me
```

### API Keys

#### List API Keys

```
GET /v1/api-keys
```

#### Create API Key

```
POST /v1/api-keys
```

#### Revoke API Key

```
DELETE /v1/api-keys/{id}
```

## Webhooks

Configure webhooks to receive events:

```
POST /v1/webhooks
```

```json
{
  "url": "https://your-server.com/webhook",
  "events": ["execution.completed", "execution.failed"]
}
```

## Rate Limits

| Plan | Requests/minute |
|------|-----------------|
| Free | 60 |
| Pro | 600 |
| Enterprise | Custom |

## SDKs

Official SDKs:
- Python: `pip install cloudflow`
- Node.js: `npm install @cloudflow/sdk`
- Go: `go get github.com/cloudflow/sdk-go`
