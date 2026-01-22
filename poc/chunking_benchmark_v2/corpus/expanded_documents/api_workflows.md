# Workflows API Reference

## Overview

The Workflows API provides programmatic access to workflows functionality.
Base URL: `https://api.cloudflow.io/v1/workflows`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /workflows

List all workflows for the authenticated user.

**Parameters:**

- `limit` (integer): Maximum number of results (default: 50, max: 200)
- `status` (string): Filter by status: active, paused, archived

**Response:**
- Success: `200` - Returns array of workflow objects
- Error: `400` - Bad Request

**Example:**
```json
{"workflows": [{"id": "wf_123", "name": "Daily Report", "status": "active"}]}
```

### POST /workflows

Create a new workflow.

**Parameters:**

- `name` (string): Workflow name (required)
- `definition` (object): Workflow definition in JSON format

**Response:**
- Success: `201` - Workflow created successfully
- Error: `400` - Bad Request

**Example:**
```json
{"id": "wf_456", "name": "New Workflow", "created_at": "2024-01-15T10:00:00Z"}
```

### DELETE /workflows/{id}

Delete a workflow permanently.

**Parameters:**

- `id` (string): Workflow ID (required)

**Response:**
- Success: `204` - Workflow deleted
- Error: `404` - Workflow not found

**Example:**
```json
{}
```
