# Database Configuration Reference

## Overview

This document describes all configuration options for Database.

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | string | required | PostgreSQL connection string |
| `DATABASE_POOL_MIN` | integer | 5 | Minimum connection pool size |
| `DATABASE_POOL_MAX` | integer | 20 | Maximum connection pool size |
| `DATABASE_STATEMENT_TIMEOUT` | integer | 30000 | Query timeout in milliseconds |
| `DATABASE_SSL_MODE` | string | require | SSL mode: disable, require, verify-full |
| `DATABASE_MIGRATION_AUTO` | boolean | false | Run migrations on startup |

## Configuration File

Configuration can also be provided via `database.yaml`:

```yaml
database_url: required
database_pool_min: 5
database_pool_max: 20
database_statement_timeout: 30000
database_ssl_mode: require
```

## Validation

Configuration is validated at startup. Invalid configuration will prevent the service from starting.
