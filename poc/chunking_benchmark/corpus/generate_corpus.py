#!/usr/bin/env python3
"""Generate synthetic SaaS company documentation corpus."""

import json
import os
from pathlib import Path


# Document templates for a fictional SaaS company "CloudFlow"
DOCUMENTS = [
    # Architecture documents (deeply nested)
    {
        "id": "arch_overview",
        "title": "System Architecture Overview",
        "filename": "architecture/overview.md",
        "content": """# System Architecture Overview

CloudFlow is a cloud-native SaaS platform for workflow automation. This document provides a high-level overview of our system architecture.

## Core Components

The system consists of several key components that work together to provide workflow automation capabilities.

### API Gateway

The API Gateway is the entry point for all client requests. It handles authentication, rate limiting, and request routing.

Key responsibilities:
- JWT token validation
- Rate limiting (100 requests/minute per user)
- Request routing to appropriate services
- SSL termination

The gateway is implemented using Kong and runs on Kubernetes.

### Workflow Engine

The Workflow Engine is the heart of CloudFlow. It executes workflow definitions and manages state.

#### Execution Model

Workflows are executed using an event-driven model. Each step in a workflow produces events that trigger subsequent steps.

The engine supports:
- Sequential execution
- Parallel execution
- Conditional branching
- Error handling and retries

#### State Management

Workflow state is persisted in PostgreSQL. Each workflow instance has a unique ID and maintains its current state, variables, and execution history.

### Data Layer

We use a polyglot persistence approach:
- PostgreSQL for transactional data
- Redis for caching and session storage
- S3 for file storage
- Elasticsearch for search and analytics

## Infrastructure

### Kubernetes Deployment

All services run on Kubernetes (EKS). We use:
- Horizontal Pod Autoscaler for scaling
- Istio for service mesh
- Prometheus + Grafana for monitoring

### Database Architecture

PostgreSQL runs in a primary-replica configuration with automated failover.

Connection pooling is handled by PgBouncer with a pool size of 100 connections per service.

## Security

### Authentication

We use OAuth 2.0 with JWT tokens. Tokens expire after 1 hour and can be refreshed using refresh tokens.

### Authorization

Role-based access control (RBAC) is implemented at the API Gateway level. Roles include:
- Admin: Full access
- Editor: Can create and modify workflows
- Viewer: Read-only access

### Data Encryption

All data is encrypted at rest using AES-256. Data in transit uses TLS 1.3.
"""
    },
    {
        "id": "arch_api",
        "title": "API Architecture",
        "filename": "architecture/api.md",
        "content": """# API Architecture

This document describes the API architecture of CloudFlow.

## REST API Design

Our API follows REST principles with some pragmatic deviations.

### Endpoints

Base URL: `https://api.cloudflow.io/v1`

#### Workflows

- `GET /workflows` - List all workflows
- `POST /workflows` - Create a new workflow
- `GET /workflows/{id}` - Get workflow details
- `PUT /workflows/{id}` - Update workflow
- `DELETE /workflows/{id}` - Delete workflow
- `POST /workflows/{id}/execute` - Execute workflow

#### Executions

- `GET /executions` - List executions
- `GET /executions/{id}` - Get execution details
- `POST /executions/{id}/cancel` - Cancel execution

### Request Format

All requests must include:
- `Authorization: Bearer <token>` header
- `Content-Type: application/json` for POST/PUT requests

### Response Format

All responses follow this structure:

```json
{
  "data": { ... },
  "meta": {
    "request_id": "uuid",
    "timestamp": "ISO8601"
  }
}
```

### Error Handling

Errors return appropriate HTTP status codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

Error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human readable message",
    "details": { ... }
  }
}
```

## GraphQL API

We also offer a GraphQL API for complex queries.

Endpoint: `https://api.cloudflow.io/graphql`

### Schema

The GraphQL schema includes:
- Query: workflows, executions, users
- Mutation: createWorkflow, updateWorkflow, executeWorkflow
- Subscription: executionUpdates

## Rate Limiting

Rate limits are applied per API key:
- Standard: 100 requests/minute
- Pro: 1000 requests/minute
- Enterprise: Custom limits

Rate limit headers are included in responses:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`
"""
    },
    {
        "id": "arch_database",
        "title": "Database Design",
        "filename": "architecture/database.md",
        "content": """# Database Design

This document describes the database architecture and schema for CloudFlow.

## Overview

We use PostgreSQL 15 as our primary database. The database is deployed in a primary-replica configuration on AWS RDS.

## Schema Design

### Core Tables

#### users

Stores user account information.

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'viewer',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### workflows

Stores workflow definitions.

```sql
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    definition JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### executions

Tracks workflow executions.

```sql
CREATE TABLE executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES workflows(id),
    status VARCHAR(50) NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    result JSONB,
    error TEXT
);
```

### Indexes

Key indexes for performance:

```sql
CREATE INDEX idx_workflows_user_id ON workflows(user_id);
CREATE INDEX idx_executions_workflow_id ON executions(workflow_id);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_workflows_status ON workflows(status);
```

## Connection Management

### Connection Pooling

We use PgBouncer for connection pooling:
- Pool mode: transaction
- Max connections per pool: 100
- Default pool size: 20

### Connection Strings

Environment-specific connection strings:
- Development: `postgresql://dev:password@localhost:5432/cloudflow_dev`
- Staging: `postgresql://app:${DB_PASSWORD}@staging-db.internal:5432/cloudflow`
- Production: `postgresql://app:${DB_PASSWORD}@prod-db.internal:5432/cloudflow`

## Migrations

We use Flyway for database migrations. Migration files are in `db/migrations/`.

Naming convention: `V{version}__{description}.sql`

## Backup Strategy

- Full backup: Daily at 02:00 UTC
- Point-in-time recovery: Enabled with 7-day retention
- Cross-region replication: To us-west-2 for disaster recovery
"""
    },
    # ADRs (Decision Records)
    {
        "id": "adr_001",
        "title": "ADR-001: Use PostgreSQL as Primary Database",
        "filename": "decisions/adr-001-postgresql.md",
        "content": """# ADR-001: Use PostgreSQL as Primary Database

## Status

Accepted

## Context

We need to choose a primary database for CloudFlow. The database must support:
- ACID transactions
- JSON storage for workflow definitions
- Good performance for read-heavy workloads
- Mature ecosystem and tooling

## Options Considered

### Option 1: PostgreSQL

Pros:
- Excellent JSONB support for flexible schema
- Strong ACID compliance
- Mature replication and failover
- Rich ecosystem (extensions, tools)
- Team has experience

Cons:
- Requires careful tuning for high scale
- No built-in horizontal scaling

### Option 2: MongoDB

Pros:
- Native JSON document storage
- Horizontal scaling built-in
- Flexible schema

Cons:
- Weaker consistency guarantees
- Less mature tooling
- Team would need training

### Option 3: CockroachDB

Pros:
- PostgreSQL compatible
- Distributed by design
- Strong consistency

Cons:
- Higher operational complexity
- Less mature ecosystem
- Higher cost

## Decision

We will use **PostgreSQL** as our primary database.

## Rationale

1. JSONB support covers our need for flexible workflow definitions
2. Team already has PostgreSQL expertise
3. Proven reliability at scale with proper architecture
4. Strong ecosystem for monitoring, backups, and migrations
5. Can add read replicas for scaling reads
6. Can consider Citus for horizontal scaling if needed later

## Consequences

- Need to implement application-level sharding if we outgrow single-node
- Must set up proper connection pooling (PgBouncer)
- Need to invest in monitoring and query optimization
- Database migrations require careful planning
"""
    },
    {
        "id": "adr_002",
        "title": "ADR-002: Choose Kubernetes for Container Orchestration",
        "filename": "decisions/adr-002-kubernetes.md",
        "content": """# ADR-002: Choose Kubernetes for Container Orchestration

## Status

Accepted

## Context

CloudFlow needs a container orchestration platform. Requirements:
- Auto-scaling based on load
- Service discovery
- Rolling deployments
- Secret management
- Multi-environment support

## Options Considered

### Option 1: Kubernetes (EKS)

Pros:
- Industry standard
- Rich ecosystem
- Auto-scaling (HPA, VPA)
- Strong community

Cons:
- Complex to operate
- Learning curve
- Can be expensive at small scale

### Option 2: AWS ECS

Pros:
- Simpler than Kubernetes
- Native AWS integration
- Lower operational overhead

Cons:
- AWS lock-in
- Less flexible
- Smaller ecosystem

### Option 3: Docker Swarm

Pros:
- Simple to set up
- Built into Docker

Cons:
- Limited features
- Smaller community
- Less active development

## Decision

We will use **Kubernetes (EKS)** for container orchestration.

## Rationale

1. Industry standard - easy to hire engineers with experience
2. Rich ecosystem of tools (Helm, Istio, Prometheus)
3. Cloud-agnostic - can migrate to GKE/AKS if needed
4. Powerful auto-scaling capabilities
5. Strong secret management with external-secrets operator

## Consequences

- Need to invest in Kubernetes training
- Higher initial setup complexity
- Need dedicated DevOps capacity
- Must establish GitOps practices
- Consider managed add-ons to reduce operational burden
"""
    },
    {
        "id": "adr_003",
        "title": "ADR-003: API Authentication Strategy",
        "filename": "decisions/adr-003-authentication.md",
        "content": """# ADR-003: API Authentication Strategy

## Status

Accepted

## Context

We need to implement authentication for the CloudFlow API. Requirements:
- Secure token-based authentication
- Support for API keys (machine-to-machine)
- Single sign-on capability
- Token refresh mechanism

## Options Considered

### Option 1: JWT with OAuth 2.0

Pros:
- Industry standard
- Stateless verification
- Works with SSO providers
- Supports refresh tokens

Cons:
- Token revocation is complex
- Larger token size

### Option 2: Session-based with Redis

Pros:
- Easy revocation
- Small token size

Cons:
- Requires session storage
- Not as scalable
- Doesn't work well with microservices

### Option 3: API Keys only

Pros:
- Simple implementation
- Easy to understand

Cons:
- No user identity
- Hard to rotate
- No SSO support

## Decision

We will use **JWT with OAuth 2.0** for authentication.

## Implementation Details

### Token Structure

```json
{
  "sub": "user_id",
  "email": "user@example.com",
  "role": "editor",
  "iat": 1234567890,
  "exp": 1234571490,
  "iss": "cloudflow"
}
```

### Token Lifetimes

- Access token: 1 hour
- Refresh token: 7 days

### API Keys

For machine-to-machine auth, we'll also support API keys:
- Stored hashed in database
- Scoped to specific permissions
- Can be rotated without downtime

## Consequences

- Need to implement token refresh flow
- Must handle token revocation via short expiry + deny list
- Need to integrate with identity provider (Auth0)
- API keys require separate management interface
"""
    },
    # How-to Guides
    {
        "id": "howto_quickstart",
        "title": "Quick Start Guide",
        "filename": "guides/quickstart.md",
        "content": """# Quick Start Guide

Get started with CloudFlow in 5 minutes.

## Prerequisites

- CloudFlow account (sign up at cloudflow.io)
- API key (generate in Settings > API Keys)

## Step 1: Install the CLI

```bash
# macOS
brew install cloudflow-cli

# Linux
curl -fsSL https://get.cloudflow.io | bash

# Windows
choco install cloudflow-cli
```

## Step 2: Configure Authentication

```bash
cloudflow auth login
# Follow the browser prompt to authenticate
```

Or use an API key:

```bash
export CLOUDFLOW_API_KEY=your-api-key
```

## Step 3: Create Your First Workflow

Create a file `hello-world.yaml`:

```yaml
name: Hello World
trigger:
  type: manual
steps:
  - id: greet
    type: log
    message: "Hello, CloudFlow!"
```

Deploy it:

```bash
cloudflow workflow create -f hello-world.yaml
```

## Step 4: Execute the Workflow

```bash
cloudflow workflow execute hello-world
```

## Step 5: Check the Results

```bash
cloudflow execution list --workflow hello-world
cloudflow execution logs <execution-id>
```

## Next Steps

- Read the [Workflow Syntax Guide](workflow-syntax.md)
- Explore [Built-in Actions](actions.md)
- Set up [Scheduled Triggers](triggers.md)
"""
    },
    {
        "id": "howto_deploy",
        "title": "Deployment Guide",
        "filename": "guides/deployment.md",
        "content": """# Deployment Guide

This guide covers deploying CloudFlow to production.

## Deployment Environments

We maintain three environments:
- **Development**: For local development
- **Staging**: For pre-production testing
- **Production**: Live environment

## Prerequisites

- kubectl configured for target cluster
- Helm 3.x installed
- Access to container registry
- Database credentials

## Deployment Process

### 1. Build Container Images

```bash
# Build all services
make build-all

# Or build specific service
make build SERVICE=api-gateway
```

Images are tagged with git SHA:

```bash
docker tag cloudflow/api-gateway:latest \
  registry.cloudflow.io/api-gateway:$(git rev-parse --short HEAD)
```

### 2. Run Database Migrations

```bash
# Connect to database
kubectl exec -it postgres-0 -- psql -U cloudflow

# Or use Flyway
flyway -url=jdbc:postgresql://db:5432/cloudflow migrate
```

### 3. Deploy with Helm

```bash
# Update dependencies
helm dependency update ./helm/cloudflow

# Deploy to staging
helm upgrade --install cloudflow ./helm/cloudflow \
  --namespace staging \
  --values ./helm/values-staging.yaml

# Deploy to production
helm upgrade --install cloudflow ./helm/cloudflow \
  --namespace production \
  --values ./helm/values-production.yaml
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -n production

# Check service health
curl https://api.cloudflow.io/health

# Run smoke tests
make test-smoke ENV=production
```

## Rollback Procedure

If issues are detected:

```bash
# Rollback Helm release
helm rollback cloudflow -n production

# Or rollback to specific revision
helm rollback cloudflow 5 -n production
```

## Environment Variables

Required environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://...` |
| `JWT_SECRET` | Secret for JWT signing | `random-string` |
| `AWS_REGION` | AWS region | `us-east-1` |

## Monitoring

After deployment, verify:
- Prometheus metrics at `/metrics`
- Grafana dashboards
- Error rates in Sentry
- Logs in CloudWatch
"""
    },
    {
        "id": "howto_workflows",
        "title": "Creating Workflows",
        "filename": "guides/creating-workflows.md",
        "content": """# Creating Workflows

Learn how to create and manage workflows in CloudFlow.

## Workflow Structure

A workflow consists of:
- **Trigger**: What starts the workflow
- **Steps**: Actions to perform
- **Variables**: Data passed between steps

## Basic Workflow

```yaml
name: My Workflow
description: A simple example workflow

trigger:
  type: webhook
  path: /my-workflow

variables:
  greeting: "Hello"

steps:
  - id: step1
    type: http
    url: https://api.example.com/data
    method: GET
    
  - id: step2
    type: transform
    input: "{{ steps.step1.response }}"
    expression: "data.items.map(i => i.name)"
    
  - id: step3
    type: log
    message: "{{ variables.greeting }}, found {{ steps.step2.output.length }} items"
```

## Trigger Types

### Manual Trigger

Execute via API or CLI:

```yaml
trigger:
  type: manual
```

### Webhook Trigger

HTTP endpoint that starts workflow:

```yaml
trigger:
  type: webhook
  path: /process-order
  method: POST
```

### Schedule Trigger

Cron-based scheduling:

```yaml
trigger:
  type: schedule
  cron: "0 9 * * *"  # Daily at 9 AM
```

### Event Trigger

React to system events:

```yaml
trigger:
  type: event
  source: s3
  event: object.created
```

## Step Types

### HTTP Request

```yaml
- id: fetch_data
  type: http
  url: "{{ variables.api_url }}"
  method: POST
  headers:
    Authorization: "Bearer {{ secrets.API_TOKEN }}"
  body:
    query: "{{ trigger.payload.query }}"
```

### Conditional

```yaml
- id: check_status
  type: condition
  if: "{{ steps.fetch_data.response.status == 'success' }}"
  then:
    - id: process
      type: log
      message: "Processing successful data"
  else:
    - id: handle_error
      type: notify
      channel: "#alerts"
```

### Loop

```yaml
- id: process_items
  type: loop
  items: "{{ steps.fetch_data.response.items }}"
  step:
    id: process_item
    type: http
    url: "https://api.example.com/process"
    body:
      item: "{{ item }}"
```

## Error Handling

### Retry Configuration

```yaml
- id: unreliable_api
  type: http
  url: https://flaky-api.example.com
  retry:
    attempts: 3
    delay: 5s
    backoff: exponential
```

### Error Handlers

```yaml
- id: risky_step
  type: http
  url: https://api.example.com
  on_error:
    - id: notify_failure
      type: notify
      channel: "#alerts"
      message: "Step failed: {{ error.message }}"
```

## Best Practices

1. **Use meaningful step IDs**: Makes debugging easier
2. **Add descriptions**: Document what each step does
3. **Handle errors**: Always plan for failure
4. **Use secrets**: Never hardcode credentials
5. **Test locally**: Use `cloudflow workflow test` before deploying
"""
    },
    # API Documentation
    {
        "id": "api_reference",
        "title": "API Reference",
        "filename": "api/reference.md",
        "content": """# API Reference

Complete reference for the CloudFlow REST API.

## Authentication

All API requests require authentication via Bearer token or API key.

### Bearer Token

```bash
curl -H "Authorization: Bearer <token>" https://api.cloudflow.io/v1/workflows
```

### API Key

```bash
curl -H "X-API-Key: <api-key>" https://api.cloudflow.io/v1/workflows
```

## Endpoints

### Workflows

#### List Workflows

```
GET /v1/workflows
```

Query parameters:
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20, max: 100)
- `status` (string): Filter by status

Response:

```json
{
  "data": [
    {
      "id": "wf_123",
      "name": "My Workflow",
      "status": "active",
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "meta": {
    "total": 42,
    "page": 1,
    "limit": 20
  }
}
```

#### Create Workflow

```
POST /v1/workflows
```

Request body:

```json
{
  "name": "My Workflow",
  "description": "Does something useful",
  "definition": {
    "trigger": { "type": "manual" },
    "steps": [...]
  }
}
```

#### Get Workflow

```
GET /v1/workflows/{id}
```

#### Update Workflow

```
PUT /v1/workflows/{id}
```

#### Delete Workflow

```
DELETE /v1/workflows/{id}
```

#### Execute Workflow

```
POST /v1/workflows/{id}/execute
```

Request body (optional):

```json
{
  "variables": {
    "input_data": "value"
  }
}
```

### Executions

#### List Executions

```
GET /v1/executions
```

Query parameters:
- `workflow_id` (string): Filter by workflow
- `status` (string): Filter by status
- `from` (datetime): Start date filter
- `to` (datetime): End date filter

#### Get Execution

```
GET /v1/executions/{id}
```

Response includes:
- Execution status
- Step results
- Logs
- Timing information

#### Cancel Execution

```
POST /v1/executions/{id}/cancel
```

### Users

#### Get Current User

```
GET /v1/users/me
```

#### Update User

```
PATCH /v1/users/me
```

### API Keys

#### List API Keys

```
GET /v1/api-keys
```

#### Create API Key

```
POST /v1/api-keys
```

#### Revoke API Key

```
DELETE /v1/api-keys/{id}
```

## Webhooks

Configure webhooks to receive events:

```
POST /v1/webhooks
```

```json
{
  "url": "https://your-server.com/webhook",
  "events": ["execution.completed", "execution.failed"]
}
```

## Rate Limits

| Plan | Requests/minute |
|------|-----------------|
| Free | 60 |
| Pro | 600 |
| Enterprise | Custom |

## SDKs

Official SDKs:
- Python: `pip install cloudflow`
- Node.js: `npm install @cloudflow/sdk`
- Go: `go get github.com/cloudflow/sdk-go`
"""
    },
    # Troubleshooting / Runbook
    {
        "id": "runbook_incidents",
        "title": "Incident Response Runbook",
        "filename": "runbooks/incidents.md",
        "content": """# Incident Response Runbook

Procedures for handling production incidents.

## Severity Levels

### P1 - Critical

- Service completely down
- Data loss occurring
- Security breach

Response time: Immediate
Resolution target: 1 hour

### P2 - High

- Major feature unavailable
- Significant performance degradation
- Affecting >10% of users

Response time: 15 minutes
Resolution target: 4 hours

### P3 - Medium

- Minor feature issues
- Workaround available

Response time: 1 hour
Resolution target: 24 hours

## Common Issues

### API Gateway 502 Errors

Symptoms:
- Users receiving 502 Bad Gateway
- Increased error rate in monitoring

Diagnosis:

```bash
# Check Kong logs
kubectl logs -l app=kong -n production --tail=100

# Check upstream services
kubectl get pods -n production
kubectl describe pod <pod-name>
```

Resolution:

1. Check if pods are healthy
2. Verify service endpoints: `kubectl get endpoints`
3. Check resource limits - pods may be OOMKilled
4. Restart unhealthy pods: `kubectl delete pod <pod>`

### Database Connection Errors

Symptoms:
- "Connection refused" or timeout errors
- Increased latency

Diagnosis:

```bash
# Check PgBouncer stats
kubectl exec -it pgbouncer-0 -- psql -p 6432 pgbouncer -c "SHOW POOLS"

# Check connection count
kubectl exec -it postgres-0 -- psql -c "SELECT count(*) FROM pg_stat_activity"
```

Resolution:

1. Check if connection limit reached
2. Verify PgBouncer is running
3. Check for long-running queries: `SELECT * FROM pg_stat_activity WHERE state != 'idle'`
4. Kill stuck queries if necessary

### High Memory Usage

Symptoms:
- OOMKilled pods
- Slow response times

Diagnosis:

```bash
# Check memory usage
kubectl top pods -n production

# Check for memory leaks
kubectl exec -it <pod> -- cat /proc/meminfo
```

Resolution:

1. Increase memory limits if justified
2. Check for memory leaks in application code
3. Scale horizontally if needed

### Workflow Execution Stuck

Symptoms:
- Executions in "running" state for too long
- No progress in step execution

Diagnosis:

```bash
# Check workflow engine logs
kubectl logs -l app=workflow-engine --tail=200

# Check execution in database
kubectl exec -it postgres-0 -- psql -c "SELECT * FROM executions WHERE status = 'running' AND started_at < NOW() - INTERVAL '1 hour'"
```

Resolution:

1. Check if external service is responding
2. Verify network connectivity
3. Cancel stuck executions if necessary
4. Restart workflow engine pod

## Escalation

If issue cannot be resolved:

1. P1: Page on-call engineer immediately
2. P2: Page on-call within 15 minutes
3. P3: Create ticket, notify team lead

On-call schedule: opsgenie.com/cloudflow

## Post-Incident

After resolution:

1. Update status page
2. Notify affected users
3. Create incident report within 24 hours
4. Schedule post-mortem for P1/P2
"""
    },
    # Additional documents for variety
    {
        "id": "security_policy",
        "title": "Security Policy",
        "filename": "security/policy.md",
        "content": """# Security Policy

CloudFlow security practices and policies.

## Data Protection

### Encryption

All data is encrypted:
- At rest: AES-256
- In transit: TLS 1.3

### Data Retention

- Execution logs: 90 days
- Audit logs: 1 year
- User data: Until account deletion + 30 days

## Access Control

### Authentication

- Passwords: bcrypt with cost factor 12
- MFA: TOTP-based, required for admin accounts
- Sessions: 24-hour timeout, secure cookies

### Authorization

RBAC roles:
- **Owner**: Full access, billing management
- **Admin**: User management, all workflows
- **Editor**: Create/edit workflows
- **Viewer**: Read-only access

## Vulnerability Management

### Reporting

Report vulnerabilities to: security@cloudflow.io

### Bug Bounty

We offer rewards for responsible disclosure:
- Critical: $5,000
- High: $2,500
- Medium: $1,000
- Low: $250

### Patching

- Critical: Within 24 hours
- High: Within 7 days
- Medium: Within 30 days

## Compliance

CloudFlow is compliant with:
- SOC 2 Type II
- GDPR
- HIPAA (Enterprise plan)

## Incident Response

See [Incident Response Runbook](../runbooks/incidents.md) for procedures.
"""
    },
    {
        "id": "changelog",
        "title": "Changelog",
        "filename": "changelog.md",
        "content": """# Changelog

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
"""
    },
    {
        "id": "meeting_notes_q1",
        "title": "Q1 Planning Meeting Notes",
        "filename": "meetings/2024-q1-planning.md",
        "content": """# Q1 2024 Planning Meeting

Date: January 5, 2024
Attendees: Sarah (PM), Mike (Eng Lead), Lisa (Design), Tom (DevOps)

## Agenda

1. Review Q4 results
2. Q1 priorities
3. Resource allocation
4. Timeline

## Q4 Review

### Accomplishments

- Launched GraphQL API
- Reduced p99 latency by 35%
- Achieved SOC 2 compliance
- 50% growth in active users

### Challenges

- Scaling issues during Black Friday
- Customer complaints about documentation
- Technical debt in workflow engine

## Q1 Priorities

### P0 - Must Have

1. **Workflow versioning** - Customers need to track changes
   - Owner: Mike
   - Target: End of February
   
2. **Improved monitoring** - Need better visibility
   - Owner: Tom
   - Target: End of January

### P1 - Should Have

3. **New workflow editor** - Current one is clunky
   - Owner: Lisa
   - Target: End of March
   
4. **Python SDK v2** - Based on customer feedback
   - Owner: Mike
   - Target: End of February

### P2 - Nice to Have

5. **AI-assisted workflow creation**
6. **Mobile app**

## Resource Allocation

| Project | Engineers | Duration |
|---------|-----------|----------|
| Versioning | 2 | 8 weeks |
| Monitoring | 1 | 4 weeks |
| Editor | 1 + Design | 10 weeks |
| SDK | 1 | 4 weeks |

## Risks

- Mike taking paternity leave in March
- Dependency on external API for new feature
- Design resources stretched thin

## Action Items

- [ ] Sarah: Create Q1 roadmap doc
- [ ] Mike: Break down versioning into epics
- [ ] Tom: Evaluate monitoring tools
- [ ] Lisa: Start editor wireframes

## Next Meeting

January 12, 2024 - Sprint planning
"""
    },
    {
        "id": "onboarding",
        "title": "Engineering Onboarding",
        "filename": "guides/onboarding.md",
        "content": """# Engineering Onboarding

Welcome to CloudFlow! This guide will help you get set up.

## Day 1

### Accounts Setup

1. **GitHub**: Accept org invite, enable 2FA
2. **Slack**: Join #engineering, #incidents, #random
3. **AWS**: Request access via IT ticket
4. **Datadog**: Get invite from your manager

### Local Development

1. Clone the monorepo:
   ```bash
   git clone git@github.com:cloudflow/cloudflow.git
   cd cloudflow
   ```

2. Install dependencies:
   ```bash
   make setup
   ```

3. Start local environment:
   ```bash
   docker-compose up -d
   make dev
   ```

4. Run tests:
   ```bash
   make test
   ```

### Access the Local App

- API: http://localhost:8080
- UI: http://localhost:3000
- Postgres: localhost:5432
- Redis: localhost:6379

## Week 1

### Codebase Tour

- `api/` - REST API service (Go)
- `web/` - Frontend (React)
- `engine/` - Workflow engine (Go)
- `infra/` - Terraform & Kubernetes configs
- `docs/` - Documentation

### Key Concepts

Read these docs:
1. [Architecture Overview](../architecture/overview.md)
2. [API Architecture](../architecture/api.md)
3. [Database Design](../architecture/database.md)

### First Task

Your manager will assign a "good first issue". These are scoped tasks to help you learn the codebase.

## Week 2-4

### Deep Dives

Schedule 1:1s with:
- Product Manager - understand roadmap
- DevOps - deployment process
- Senior Engineer - code review practices

### Shadowing

- Shadow on-call engineer for a day
- Join incident response call (if one happens)
- Attend sprint planning and retro

## Resources

- Internal wiki: wiki.cloudflow.io
- Runbooks: [runbooks/](../runbooks/)
- ADRs: [decisions/](../decisions/)
- Team calendar: calendar.cloudflow.io/engineering

## Questions?

Ask in #engineering or your onboarding buddy!
"""
    },
    {
        "id": "testing_guide",
        "title": "Testing Guide",
        "filename": "guides/testing.md",
        "content": """# Testing Guide

How we test CloudFlow.

## Test Types

### Unit Tests

Test individual functions and methods.

Location: `*_test.go` files alongside source

```bash
# Run all unit tests
make test-unit

# Run specific package
go test ./api/handlers/...

# With coverage
go test -cover ./...
```

### Integration Tests

Test service interactions.

Location: `tests/integration/`

```bash
# Requires local environment running
make test-integration
```

### End-to-End Tests

Test full user flows.

Location: `tests/e2e/`

```bash
# Uses Playwright
make test-e2e
```

## Writing Tests

### Unit Test Example

```go
func TestWorkflowValidation(t *testing.T) {
    tests := []struct {
        name    string
        input   Workflow
        wantErr bool
    }{
        {
            name: "valid workflow",
            input: Workflow{
                Name: "test",
                Steps: []Step{{ID: "step1"}},
            },
            wantErr: false,
        },
        {
            name: "missing name",
            input: Workflow{
                Steps: []Step{{ID: "step1"}},
            },
            wantErr: true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := tt.input.Validate()
            if (err != nil) != tt.wantErr {
                t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### Integration Test Example

```go
func TestCreateWorkflowAPI(t *testing.T) {
    // Setup
    db := setupTestDB(t)
    defer db.Close()
    
    server := setupTestServer(db)
    
    // Test
    resp, err := server.Client().Post(
        "/v1/workflows",
        "application/json",
        strings.NewReader(`{"name": "test"}`),
    )
    
    // Assert
    require.NoError(t, err)
    assert.Equal(t, 201, resp.StatusCode)
}
```

## Test Data

### Fixtures

Test fixtures are in `tests/fixtures/`:

- `workflows/` - Sample workflow definitions
- `users/` - Test user data
- `executions/` - Sample execution data

### Factories

Use factories for dynamic test data:

```go
user := factory.NewUser().
    WithRole("admin").
    Build()
```

## Coverage

We aim for:
- Unit tests: >80% coverage
- Integration tests: Critical paths covered
- E2E tests: Main user flows covered

Check coverage:

```bash
make coverage
# Opens coverage report in browser
```

## CI Pipeline

Tests run on every PR:

1. Lint
2. Unit tests
3. Integration tests
4. E2E tests (on main branch only)

All tests must pass before merge.

## Mocking

Use interfaces for dependencies, mock in tests:

```go
type WorkflowStore interface {
    Get(id string) (*Workflow, error)
    Create(w *Workflow) error
}

// In tests
type mockStore struct {
    workflows map[string]*Workflow
}

func (m *mockStore) Get(id string) (*Workflow, error) {
    return m.workflows[id], nil
}
```
"""
    }
]


def generate_corpus(output_dir: Path) -> list[dict]:
    """Generate the corpus and return document metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    for doc in DOCUMENTS:
        # Create directory structure
        filepath = output_dir / doc["filename"]
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write document
        filepath.write_text(doc["content"])
        
        # Track metadata
        metadata.append({
            "id": doc["id"],
            "title": doc["title"],
            "filename": doc["filename"],
            "path": str(filepath),
            "word_count": len(doc["content"].split()),
            "char_count": len(doc["content"])
        })
        
        print(f"Created: {doc['filename']}")
    
    return metadata


def main():
    """Generate corpus and save metadata."""
    script_dir = Path(__file__).parent
    docs_dir = script_dir / "documents"
    
    print("Generating CloudFlow SaaS documentation corpus...")
    print("=" * 50)
    
    metadata = generate_corpus(docs_dir)
    
    # Save metadata
    metadata_path = script_dir / "corpus_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("=" * 50)
    print(f"Generated {len(metadata)} documents")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print summary
    total_words = sum(m["word_count"] for m in metadata)
    print(f"Total words: {total_words:,}")


if __name__ == "__main__":
    main()
