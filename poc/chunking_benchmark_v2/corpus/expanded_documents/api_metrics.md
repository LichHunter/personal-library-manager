# Metrics API Reference

## Overview

The Metrics API provides programmatic access to metrics functionality.
Base URL: `https://api.cloudflow.io/v1/metrics`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /metrics/workflows

Get workflow performance metrics.

**Parameters:**

- `period` (string): Time period: hour, day, week, month
- `workflow_id` (string): Filter by workflow ID

**Response:**
- Success: `200` - Returns metrics data
- Error: `400` - Bad Request

**Example:**
```json
{"success_rate": 0.985, "avg_duration_ms": 1250, "p99_duration_ms": 5000, "total_executions": 10000}
```
