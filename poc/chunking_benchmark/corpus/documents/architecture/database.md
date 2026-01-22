# Database Design

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
