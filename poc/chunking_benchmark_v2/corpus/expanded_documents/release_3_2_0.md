# Release Notes - v3.2.0

**Release Date:** 2024-01-15

## New Features

- Workflow versioning with rollback support
- New Python SDK with async support
- Bulk operations API for workflows
- Custom retry policies per step

## Bug Fixes

- Fixed race condition in concurrent execution handler
- Resolved memory leak in long-running workflows
- Fixed timezone handling in scheduled triggers

## Breaking Changes

- **BREAKING:** API v1 endpoints deprecated, will be removed in v4.0
- **BREAKING:** Minimum Python version now 3.9

## Upgrade Instructions

```bash
cloudflow upgrade --version 3.2.0
```

## Known Issues

None at this time.
