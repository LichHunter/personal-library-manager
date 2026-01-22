# Cache Configuration Reference

## Overview

This document describes all configuration options for Cache.

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | string | required | Redis connection string |
| `REDIS_POOL_SIZE` | integer | 10 | Connection pool size |
| `CACHE_TTL_SECONDS` | integer | 300 | Default cache TTL |
| `CACHE_PREFIX` | string | cf: | Key prefix for namespacing |
| `CACHE_COMPRESSION` | boolean | true | Enable value compression |
| `CACHE_MAX_SIZE_MB` | integer | 100 | Maximum cache size per pod |

## Configuration File

Configuration can also be provided via `cache.yaml`:

```yaml
redis_url: required
redis_pool_size: 10
cache_ttl_seconds: 300
cache_prefix: cf:
cache_compression: true
```

## Validation

Configuration is validated at startup. Invalid configuration will prevent the service from starting.
