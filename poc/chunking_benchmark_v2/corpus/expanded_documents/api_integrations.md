# Integrations API Reference

## Overview

The Integrations API provides programmatic access to integrations functionality.
Base URL: `https://api.cloudflow.io/v1/integrations`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /integrations

List available integrations.

**Parameters:**

- `category` (string): Filter by category: storage, notification, database

**Response:**
- Success: `200` - Returns array of integration objects
- Error: `400` - Bad Request

**Example:**
```json
{"integrations": [{"id": "int_s3", "name": "Amazon S3", "category": "storage", "status": "connected"}]}
```

### POST /integrations/{id}/connect

Connect an integration.

**Parameters:**

- `credentials` (object): Integration-specific credentials

**Response:**
- Success: `200` - Integration connected
- Error: `400` - Bad Request

**Example:**
```json
{"id": "int_s3", "status": "connected", "connected_at": "2024-01-15T10:00:00Z"}
```
