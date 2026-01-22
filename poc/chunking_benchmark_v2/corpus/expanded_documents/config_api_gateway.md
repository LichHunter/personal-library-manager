# API Gateway Configuration Reference

## Overview

This document describes all configuration options for API Gateway.

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GATEWAY_PORT` | integer | 8000 | Gateway listen port |
| `GATEWAY_RATE_LIMIT` | integer | 100 | Requests per minute per user |
| `GATEWAY_TIMEOUT_MS` | integer | 30000 | Request timeout in milliseconds |
| `GATEWAY_MAX_BODY_SIZE` | string | 10MB | Maximum request body size |
| `GATEWAY_CORS_ORIGINS` | string | * | Allowed CORS origins |
| `GATEWAY_SSL_ENABLED` | boolean | true | Enable TLS termination |

## Configuration File

Configuration can also be provided via `api gateway.yaml`:

```yaml
gateway_port: 8000
gateway_rate_limit: 100
gateway_timeout_ms: 30000
gateway_max_body_size: 10MB
gateway_cors_origins: *
```

## Validation

Configuration is validated at startup. Invalid configuration will prevent the service from starting.
