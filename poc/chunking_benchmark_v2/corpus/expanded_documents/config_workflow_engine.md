# Workflow Engine Configuration Reference

## Overview

This document describes all configuration options for Workflow Engine.

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WORKFLOW_MAX_STEPS` | integer | 100 | Maximum steps per workflow |
| `WORKFLOW_TIMEOUT_SECONDS` | integer | 3600 | Maximum workflow execution time |
| `WORKFLOW_RETRY_LIMIT` | integer | 3 | Default retry attempts for failed steps |
| `WORKFLOW_CONCURRENT_LIMIT` | integer | 10 | Maximum concurrent executions per workflow |
| `EXECUTOR_POOL_SIZE` | integer | 50 | Number of executor workers |
| `EXECUTOR_MEMORY_LIMIT` | string | 512Mi | Memory limit per executor |

## Configuration File

Configuration can also be provided via `workflow engine.yaml`:

```yaml
workflow_max_steps: 100
workflow_timeout_seconds: 3600
workflow_retry_limit: 3
workflow_concurrent_limit: 10
executor_pool_size: 50
```

## Validation

Configuration is validated at startup. Invalid configuration will prevent the service from starting.
