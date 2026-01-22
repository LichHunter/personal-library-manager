# Billing API Reference

## Overview

The Billing API provides programmatic access to billing functionality.
Base URL: `https://api.cloudflow.io/v1/billing`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /billing/usage

Get current billing period usage.

**Parameters:**

- `breakdown` (boolean): Include detailed breakdown by resource

**Response:**
- Success: `200` - Returns usage summary
- Error: `400` - Bad Request

**Example:**
```json
{"period": "2024-01", "executions": 15000, "compute_minutes": 2500, "storage_gb": 50}
```

### GET /billing/invoices

List past invoices.

**Parameters:**

- `year` (integer): Filter by year

**Response:**
- Success: `200` - Returns array of invoice objects
- Error: `400` - Bad Request

**Example:**
```json
{"invoices": [{"id": "inv_202401", "amount": 299.00, "status": "paid", "date": "2024-01-01"}]}
```
