# Users API Reference

## Overview

The Users API provides programmatic access to users functionality.
Base URL: `https://api.cloudflow.io/v1/users`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /users/me

Get current user profile.

**Parameters:**


**Response:**
- Success: `200` - Returns user object
- Error: `400` - Bad Request

**Example:**
```json
{"id": "usr_001", "email": "user@example.com", "role": "admin", "created_at": "2023-06-01"}
```

### PATCH /users/me

Update current user profile.

**Parameters:**

- `name` (string): Display name
- `timezone` (string): Timezone (e.g., America/New_York)

**Response:**
- Success: `200` - User updated
- Error: `400` - Bad Request

**Example:**
```json
{"id": "usr_001", "name": "John Doe", "timezone": "America/New_York"}
```
