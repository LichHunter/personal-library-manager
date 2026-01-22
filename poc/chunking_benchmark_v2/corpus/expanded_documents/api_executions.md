# Executions API Reference

## Overview

The Executions API provides programmatic access to executions functionality.
Base URL: `https://api.cloudflow.io/v1/executions`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /executions

List workflow executions.

**Parameters:**

- `workflow_id` (string): Filter by workflow ID
- `since` (datetime): Filter executions after this timestamp

**Response:**
- Success: `200` - Returns array of execution objects
- Error: `400` - Bad Request

**Example:**
```json
{"executions": [{"id": "ex_789", "status": "completed", "duration_ms": 1523}]}
```

### POST /executions/{id}/cancel

Cancel a running execution.

**Parameters:**

- `id` (string): Execution ID
- `reason` (string): Cancellation reason (optional)

**Response:**
- Success: `200` - Execution cancelled
- Error: `400` - Bad Request

**Example:**
```json
{"id": "ex_789", "status": "cancelled", "cancelled_at": "2024-01-15T10:05:00Z"}
```
