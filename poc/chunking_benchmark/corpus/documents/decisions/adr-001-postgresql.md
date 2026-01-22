# ADR-001: Use PostgreSQL as Primary Database

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
