# Changelog

All notable changes to CloudFlow.

## [2.5.0] - 2024-01-15

### Added

- New GraphQL API endpoint
- Workflow versioning support
- Custom retry policies
- S3 event triggers

### Changed

- Improved execution performance by 40%
- Updated to PostgreSQL 15
- New rate limiting algorithm

### Fixed

- Fixed memory leak in workflow engine
- Resolved race condition in parallel steps
- Fixed timezone handling in schedules

## [2.4.0] - 2023-12-01

### Added

- Conditional step execution
- Loop step type
- Slack integration
- Workflow templates

### Changed

- Redesigned workflow editor UI
- Improved error messages
- Updated API documentation

### Fixed

- Fixed webhook retry logic
- Resolved connection pool exhaustion
- Fixed variable interpolation bugs

## [2.3.0] - 2023-10-15

### Added

- API key authentication
- Execution history export
- Custom webhook headers
- Environment variables in workflows

### Deprecated

- Legacy authentication endpoint (use OAuth 2.0)
- v0 API endpoints (use v1)

### Security

- Fixed XSS vulnerability in workflow editor
- Updated dependencies with security patches
