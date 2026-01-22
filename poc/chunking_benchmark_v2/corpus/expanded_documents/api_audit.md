# Audit API Reference

## Overview

The Audit API provides programmatic access to audit functionality.
Base URL: `https://api.cloudflow.io/v1/audit`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /audit/logs

Retrieve audit logs for compliance.

**Parameters:**

- `start_date` (date): Start date for log range
- `end_date` (date): End date for log range
- `actor` (string): Filter by user ID

**Response:**
- Success: `200` - Returns array of audit log entries
- Error: `400` - Bad Request

**Example:**
```json
{"logs": [{"timestamp": "2024-01-15T10:00:00Z", "actor": "usr_001", "action": "workflow.created", "resource": "wf_123"}]}
```
