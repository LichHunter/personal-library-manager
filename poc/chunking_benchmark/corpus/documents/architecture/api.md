# API Architecture

This document describes the API architecture of CloudFlow.

## REST API Design

Our API follows REST principles with some pragmatic deviations.

### Endpoints

Base URL: `https://api.cloudflow.io/v1`

#### Workflows

- `GET /workflows` - List all workflows
- `POST /workflows` - Create a new workflow
- `GET /workflows/{id}` - Get workflow details
- `PUT /workflows/{id}` - Update workflow
- `DELETE /workflows/{id}` - Delete workflow
- `POST /workflows/{id}/execute` - Execute workflow

#### Executions

- `GET /executions` - List executions
- `GET /executions/{id}` - Get execution details
- `POST /executions/{id}/cancel` - Cancel execution

### Request Format

All requests must include:
- `Authorization: Bearer <token>` header
- `Content-Type: application/json` for POST/PUT requests

### Response Format

All responses follow this structure:

```json
{
  "data": { ... },
  "meta": {
    "request_id": "uuid",
    "timestamp": "ISO8601"
  }
}
```

### Error Handling

Errors return appropriate HTTP status codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

Error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human readable message",
    "details": { ... }
  }
}
```

## GraphQL API

We also offer a GraphQL API for complex queries.

Endpoint: `https://api.cloudflow.io/graphql`

### Schema

The GraphQL schema includes:
- Query: workflows, executions, users
- Mutation: createWorkflow, updateWorkflow, executeWorkflow
- Subscription: executionUpdates

## Rate Limiting

Rate limits are applied per API key:
- Standard: 100 requests/minute
- Pro: 1000 requests/minute
- Enterprise: Custom limits

Rate limit headers are included in responses:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`
