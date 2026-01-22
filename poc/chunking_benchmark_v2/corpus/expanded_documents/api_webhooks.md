# Webhooks API Reference

## Overview

The Webhooks API provides programmatic access to webhooks functionality.
Base URL: `https://api.cloudflow.io/v1/webhooks`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /webhooks

List configured webhooks.

**Parameters:**

- `active` (boolean): Filter by active status

**Response:**
- Success: `200` - Returns array of webhook objects
- Error: `400` - Bad Request

**Example:**
```json
{"webhooks": [{"id": "wh_01", "url": "https://example.com/hook", "events": ["workflow.completed"]}]}
```

### POST /webhooks

Create a new webhook endpoint.

**Parameters:**

- `url` (string): Webhook URL (must be HTTPS)
- `events` (array): Events to subscribe to
- `secret` (string): Signing secret for verification

**Response:**
- Success: `201` - Webhook created
- Error: `400` - Bad Request

**Example:**
```json
{"id": "wh_02", "url": "https://example.com/new-hook", "secret": "whsec_..."}
```
