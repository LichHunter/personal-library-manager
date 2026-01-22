# Secrets API Reference

## Overview

The Secrets API provides programmatic access to secrets functionality.
Base URL: `https://api.cloudflow.io/v1/secrets`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /secrets

List secret names (values not returned).

**Parameters:**


**Response:**
- Success: `200` - Returns array of secret metadata
- Error: `400` - Bad Request

**Example:**
```json
{"secrets": [{"name": "DATABASE_URL", "created_at": "2024-01-01", "updated_at": "2024-01-10"}]}
```

### PUT /secrets/{name}

Create or update a secret.

**Parameters:**

- `value` (string): Secret value (encrypted at rest)

**Response:**
- Success: `200` - Secret saved
- Error: `400` - Bad Request

**Example:**
```json
{"name": "DATABASE_URL", "updated_at": "2024-01-15T10:00:00Z"}
```

### DELETE /secrets/{name}

Delete a secret.

**Parameters:**


**Response:**
- Success: `204` - Secret deleted
- Error: `400` - Bad Request

**Example:**
```json
{}
```
