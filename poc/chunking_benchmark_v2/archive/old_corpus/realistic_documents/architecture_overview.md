# CloudFlow Platform - System Architecture Overview

**Document Version:** 2.3.1  
**Last Updated:** January 15, 2026  
**Owner:** Platform Architecture Team  
**Status:** Production

## Executive Summary

CloudFlow is a distributed, cloud-native workflow automation platform designed to orchestrate complex business processes at scale. The platform processes over 2.5 million workflow executions daily with an average P99 latency of 180ms for API operations and 4.2 seconds for workflow execution. This document provides a comprehensive overview of the system architecture, including microservices design, data flow patterns, infrastructure decisions, and operational considerations.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Microservices Breakdown](#microservices-breakdown)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Database Architecture](#database-architecture)
5. [Message Queue Patterns](#message-queue-patterns)
6. [Caching Strategy](#caching-strategy)
7. [Security Architecture](#security-architecture)
8. [Performance Characteristics](#performance-characteristics)
9. [Disaster Recovery](#disaster-recovery)

---

## High-Level Architecture

CloudFlow follows a microservices architecture pattern deployed across multiple availability zones in AWS. The platform is designed with the following core principles:

- **Scalability**: Horizontal scaling for all services with auto-scaling groups
- **Resilience**: Circuit breakers, bulkheads, and graceful degradation
- **Observability**: Distributed tracing, centralized logging, and comprehensive metrics
- **Security**: Zero-trust network architecture with mTLS encryption

### Architecture Diagram (Conceptual)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer (ALB)                       │
│                     (TLS Termination - 443)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                         │
│              (Rate Limiting, Auth, Request Routing)              │
└─┬───────┬──────────┬────────────┬──────────────┬────────────────┘
  │       │          │            │              │
  ▼       ▼          ▼            ▼              ▼
┌────┐ ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│Auth│ │Workflow│ │Scheduler│ │Notification│ │User Service│
│Svc │ │ Engine │ │ Service │ │  Service  │ │            │
└─┬──┘ └───┬────┘ └────┬─────┘ └─────┬────┘ └──────┬─────┘
  │        │           │             │              │
  └────────┴───────────┴─────────────┴──────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
  ┌─────────┐    ┌──────────┐    ┌─────────┐
  │PostgreSQL│    │  Redis   │    │  Kafka  │
  │ Cluster │    │  Cluster │    │ Cluster │
  └─────────┘    └──────────┘    └─────────┘
```

### Technology Stack

- **Runtime**: Node.js 20.x (API Gateway, Workflow Engine), Go 1.21 (Auth Service, Scheduler)
- **Container Orchestration**: Kubernetes 1.28 (EKS)
- **Service Mesh**: Istio 1.20 for service-to-service communication
- **Databases**: PostgreSQL 15.4, Redis 7.2
- **Message Broker**: Apache Kafka 3.6
- **Monitoring**: Prometheus, Grafana, Jaeger for distributed tracing
- **Secrets Management**: HashiCorp Vault 1.15

---

## Microservices Breakdown

### API Gateway

**Purpose**: Single entry point for all client requests, providing authentication, rate limiting, request routing, and protocol translation.

**Technology**: Node.js with Express.js framework  
**Replicas**: 12 pods (production), auto-scaling 8-20 based on CPU  
**Resource Allocation**: 2 vCPU, 4GB RAM per pod

**Key Responsibilities**:
- JWT token validation (delegated to Auth Service for initial validation)
- Rate limiting: 1000 requests per minute per API key (sliding window)
- Request/response transformation and validation using JSON Schema
- Routing to downstream services based on URL path patterns
- CORS handling for web clients
- Request/response logging and correlation ID injection

**Critical Endpoints**:
- `POST /api/v1/workflows` - Create new workflow
- `GET /api/v1/workflows/:id` - Retrieve workflow status
- `POST /api/v1/workflows/:id/execute` - Trigger workflow execution
- `GET /api/v1/workflows/:id/history` - Get execution history

**Dependencies**:
- Auth Service (for token validation)
- Redis (for rate limiting counters)
- All downstream microservices

**Performance Targets**:
- P50 latency: < 50ms
- P99 latency: < 200ms
- Throughput: 10,000 RPS sustained

---

### Auth Service

**Purpose**: Centralized authentication and authorization service handling user identity, token generation, and permission validation.

**Technology**: Go with gRPC for internal communication, REST for external  
**Replicas**: 8 pods (production), auto-scaling 6-12  
**Resource Allocation**: 1 vCPU, 2GB RAM per pod

**Key Responsibilities**:
- User authentication via multiple providers (OAuth2, SAML, local credentials)
- JWT token generation and validation (RS256 algorithm)
- Role-based access control (RBAC) with fine-grained permissions
- Session management with Redis-backed storage
- API key generation and validation for service accounts
- MFA enforcement for administrative operations

**Authentication Flow**:
```
Client Request → API Gateway → Auth Service
                                     │
                                     ├─ Validate credentials
                                     ├─ Check MFA if required
                                     ├─ Generate JWT (15min expiry)
                                     ├─ Generate refresh token (7 days)
                                     └─ Store session in Redis
```

**Token Structure**:
- Access Token: JWT with 15-minute expiry
- Refresh Token: Opaque token with 7-day expiry, stored in PostgreSQL
- Claims: user_id, email, roles[], permissions[], tenant_id

**Security Features**:
- Password hashing: Argon2id with 64MB memory, 4 iterations
- Token rotation on refresh to prevent replay attacks
- Brute force protection: 5 failed attempts → 15-minute lockout
- Secrets stored in HashiCorp Vault, rotated every 90 days

**Performance Targets**:
- Token validation: < 10ms (P99)
- Token generation: < 50ms (P99)
- Throughput: 5,000 RPS for validation operations

---

### Workflow Engine

**Purpose**: Core orchestration service that executes workflow definitions, manages state transitions, and coordinates task execution across distributed systems.

**Technology**: Node.js with TypeScript, Bull queue library  
**Replicas**: 16 pods (production), auto-scaling 12-24  
**Resource Allocation**: 4 vCPU, 8GB RAM per pod

**Key Responsibilities**:
- Parse and validate workflow definitions (JSON-based DSL)
- Execute workflow steps with state machine pattern
- Handle retries with exponential backoff (max 3 retries, backoff: 2^n seconds)
- Coordinate parallel and sequential task execution
- Manage workflow state persistence and recovery
- Support for conditional branching and loops
- Sub-workflow invocation and composition

**Workflow Execution Model**:
```
Workflow Submitted → Validation → Queue in Kafka
                                        │
                                        ▼
                            Workflow Engine picks up
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              [Task 1 Exec]       [Task 2 Exec]       [Task 3 Exec]
                    │                   │                   │
                    └───────────────────┴───────────────────┘
                                        │
                                        ▼
                            State Update → PostgreSQL
                                        │
                                        ▼
                            Publish Event → Kafka
```

**Workflow DSL Example**:
```json
{
  "workflow_id": "customer-onboarding-v2",
  "version": "2.1.0",
  "steps": [
    {
      "id": "validate-customer",
      "type": "http_request",
      "config": {
        "url": "https://validation.internal/verify",
        "method": "POST",
        "timeout": 5000,
        "retry": {"max_attempts": 3, "backoff": "exponential"}
      }
    },
    {
      "id": "create-account",
      "type": "database_operation",
      "depends_on": ["validate-customer"]
    }
  ]
}
```

**State Management**:
- Workflow states: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
- Step states: QUEUED, EXECUTING, SUCCESS, FAILED, SKIPPED
- State transitions stored in PostgreSQL with event sourcing pattern
- Checkpointing every 10 steps for long-running workflows

**Performance Targets**:
- Simple workflow (3-5 steps): < 2 seconds (P99)
- Complex workflow (20+ steps): < 5 seconds (P99)
- Throughput: 500 concurrent workflow executions per pod

---

### Scheduler Service

**Purpose**: Time-based workflow triggering system supporting cron-like schedules and one-time delayed executions.

**Technology**: Go with distributed locking via Redis  
**Replicas**: 4 pods (production), active-passive with leader election  
**Resource Allocation**: 2 vCPU, 4GB RAM per pod

**Key Responsibilities**:
- Parse and validate cron expressions (extended format supporting seconds)
- Maintain schedule registry in PostgreSQL
- Distributed scheduling with leader election (one active scheduler)
- Missed execution handling with configurable catch-up policy
- Schedule conflict detection and resolution
- Time zone support for international schedules

**Scheduling Architecture**:
```
PostgreSQL Schedules Table
         │
         ▼
  Leader Scheduler (elected via Redis)
         │
         ├─ Scan for due schedules (every 10 seconds)
         ├─ Acquire lock per schedule (prevents duplicates)
         ├─ Publish to Kafka → workflow.events topic
         └─ Update last_run timestamp
```

**Schedule Types**:
- **Cron-based**: `0 */5 * * * *` (every 5 minutes)
- **One-time**: Specific timestamp for delayed execution
- **Interval-based**: Every N seconds/minutes/hours

**Reliability Features**:
- Leader election using Redis with 30-second lease
- Heartbeat mechanism to detect leader failure (5-second interval)
- Automatic failover to standby scheduler (< 10 seconds)
- Schedule versioning to handle updates during execution
- Missed execution policy: SKIP, RUN_ONCE, or RUN_ALL

**Performance Targets**:
- Schedule evaluation: < 100ms per cycle
- Accuracy: ± 1 second for schedule triggers
- Capacity: 100,000 active schedules

---

### Notification Service

**Purpose**: Multi-channel notification delivery system supporting email, SMS, webhooks, and in-app notifications.

**Technology**: Node.js with worker pool pattern  
**Replicas**: 8 pods (production), auto-scaling 6-16  
**Resource Allocation**: 2 vCPU, 4GB RAM per pod

**Key Responsibilities**:
- Consume notification events from Kafka
- Template rendering with Handlebars (cached in Redis)
- Multi-channel delivery (Email via SendGrid, SMS via Twilio, Webhooks)
- Retry logic with dead letter queue for failed deliveries
- Delivery status tracking and analytics
- User preference management (opt-in/opt-out)

**Notification Flow**:
```
Event Source → Kafka (notifications.email topic)
                        │
                        ▼
              Notification Service Consumer
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   [Email Queue]   [SMS Queue]   [Webhook Queue]
        │               │               │
        ▼               ▼               ▼
   [SendGrid]      [Twilio]      [HTTP POST]
        │               │               │
        └───────────────┴───────────────┘
                        │
                        ▼
            Delivery Status → PostgreSQL
```

**Channel Configuration**:
- Email: SendGrid with IP warming, DKIM/SPF configured
- SMS: Twilio with fallback to AWS SNS
- Webhook: Signed payloads (HMAC-SHA256), retry 3 times with exponential backoff
- In-app: WebSocket connections via Socket.io

**Template Management**:
- Templates stored in PostgreSQL with versioning
- Compiled templates cached in Redis (1-hour TTL)
- Support for localization (i18n) with 12 languages
- A/B testing capability for email campaigns

**Performance Targets**:
- Email delivery: < 5 seconds (P99) from event to SendGrid
- SMS delivery: < 3 seconds (P99)
- Throughput: 10,000 notifications per minute

---

## Data Flow Architecture

### Synchronous Request Flow

Client requests follow a synchronous path through the API Gateway to backend services:

```
1. Client → Load Balancer (TLS termination)
2. Load Balancer → API Gateway (HTTP/2)
3. API Gateway → Auth Service (JWT validation via gRPC)
4. API Gateway → Target Service (HTTP/REST or gRPC)
5. Target Service → Database/Cache (data retrieval)
6. Response propagates back through the chain
```

**Timeout Configuration**:
- Client → Load Balancer: 60 seconds
- Load Balancer → API Gateway: 55 seconds
- API Gateway → Services: 45 seconds
- Service → Database: 10 seconds

### Asynchronous Event Flow

Long-running operations and inter-service communication use event-driven patterns:

```
1. Service publishes event → Kafka topic
2. Kafka persists event (replication factor: 3)
3. Consumer groups subscribe to topics
4. Consumers process events with at-least-once delivery
5. State updates written to PostgreSQL
6. Success/failure events published back to Kafka
```

**Event Schema**:
```json
{
  "event_id": "uuid-v4",
  "event_type": "workflow.execution.completed",
  "timestamp": "2026-01-15T10:30:00.000Z",
  "correlation_id": "request-trace-id",
  "payload": {
    "workflow_id": "wf-12345",
    "execution_id": "exec-67890",
    "status": "COMPLETED",
    "duration_ms": 4230
  },
  "metadata": {
    "source_service": "workflow-engine",
    "schema_version": "1.0"
  }
}
```

### Inter-Service Communication

Services communicate using two primary patterns:

**Synchronous (gRPC)**:
- Auth Service validation calls from API Gateway
- User Service profile lookups
- Low latency requirement (< 50ms)
- Request-response pattern with circuit breaker

**Asynchronous (Kafka)**:
- Workflow execution events
- Notification triggers
- Audit log events
- Eventual consistency acceptable

---

## Database Architecture

### PostgreSQL Primary Database

**Cluster Configuration**:
- Primary-replica setup with 1 primary + 2 read replicas
- Instance type: db.r6g.2xlarge (8 vCPU, 64GB RAM)
- Storage: 2TB gp3 SSD with 12,000 IOPS
- Multi-AZ deployment for high availability
- Automated backups: Daily snapshots, 30-day retention
- Point-in-time recovery: 5-minute granularity

**Database Schema Design**:

```
Core Tables:
- users (5M rows): User accounts and profiles
- workflows (2M rows): Workflow definitions
- workflow_executions (500M rows, partitioned): Execution history
- workflow_steps (2B rows, partitioned): Individual step records
- schedules (100K rows): Scheduled workflow triggers
- notifications (1B rows, partitioned): Notification delivery log

Partitioning Strategy:
- workflow_executions: Monthly partitions by created_at
- workflow_steps: Monthly partitions by created_at
- notifications: Weekly partitions by created_at
- Automatic partition creation via cron job
- Retention policy: Drop partitions older than 12 months
```

**Indexing Strategy**:
- Primary keys: UUIDs with B-tree indexes
- Frequently queried columns: Compound indexes (e.g., user_id + status + created_at)
- JSONB columns: GIN indexes for workflow definitions
- Full-text search: GiST indexes on description fields

**Connection Pooling**:
- PgBouncer in transaction mode
- Pool size: 100 connections per service
- Max client connections: 2000
- Connection timeout: 30 seconds

**Performance Characteristics**:
- Read query P99: < 20ms
- Write query P99: < 50ms
- Transaction throughput: 15,000 TPS
- Replication lag: < 500ms

---

### Redis Caching Layer

**Cluster Configuration**:
- Redis Cluster with 6 nodes (3 primary + 3 replica)
- Instance type: cache.r6g.xlarge (4 vCPU, 26GB RAM)
- Total memory: 78GB usable cache space
- Persistence: RDB snapshots every 5 minutes + AOF
- Eviction policy: allkeys-lru

**Cache Usage Patterns**:

1. **Session Storage**:
   - Key pattern: `session:{user_id}`
   - TTL: 15 minutes (aligned with JWT expiry)
   - Data: User session state, preferences
   - Invalidation: On logout or password change

2. **Rate Limiting Counters**:
   - Key pattern: `ratelimit:{api_key}:{window}`
   - TTL: 60 seconds (sliding window)
   - Data: Request count per window
   - Algorithm: Token bucket with Redis INCR

3. **Compiled Templates**:
   - Key pattern: `template:{template_id}:{version}`
   - TTL: 1 hour
   - Data: Compiled Handlebars template
   - Invalidation: On template update

4. **Workflow Definitions**:
   - Key pattern: `workflow:def:{workflow_id}`
   - TTL: 1 hour
   - Data: Parsed workflow JSON
   - Invalidation: On workflow update or manual flush

5. **User Profiles**:
   - Key pattern: `user:profile:{user_id}`
   - TTL: 30 minutes
   - Data: User metadata (name, email, roles)
   - Invalidation: On profile update

**Cache Hit Rates** (Production Metrics):
- Session lookups: 98.5%
- Workflow definitions: 94.2%
- Template cache: 99.1%
- User profiles: 91.8%

**Performance Characteristics**:
- GET operation P99: < 2ms
- SET operation P99: < 3ms
- Throughput: 100,000 operations per second
- Network latency: < 1ms (same AZ)

---

### Kafka Event Streaming

**Cluster Configuration**:
- 5 broker nodes (distributed across 3 AZs)
- Instance type: kafka.m5.2xlarge (8 vCPU, 32GB RAM)
- Storage: 10TB per broker (gp3 SSD)
- ZooKeeper ensemble: 3 nodes for cluster coordination
- Replication factor: 3 (min in-sync replicas: 2)

**Topic Architecture**:

```
workflow.events (32 partitions):
  - Workflow lifecycle events (created, started, completed, failed)
  - Retention: 7 days
  - Message rate: 5,000/sec peak
  - Consumer groups: workflow-engine, analytics-pipeline

notifications.email (16 partitions):
  - Email notification triggers
  - Retention: 3 days
  - Message rate: 2,000/sec peak
  - Consumer groups: notification-service

notifications.sms (8 partitions):
  - SMS notification triggers
  - Retention: 3 days
  - Message rate: 500/sec peak
  - Consumer groups: notification-service

audit.logs (24 partitions):
  - Security and compliance audit events
  - Retention: 90 days (compliance requirement)
  - Message rate: 3,000/sec peak
  - Consumer groups: audit-processor, security-monitor

dead-letter-queue (8 partitions):
  - Failed message processing
  - Retention: 30 days
  - Manual intervention required
  - Consumer groups: ops-team-alerts
```

**Producer Configuration**:
- Acknowledgment: `acks=all` (wait for all in-sync replicas)
- Compression: LZ4 (reduces network bandwidth by 60%)
- Batching: 100ms linger time, 100KB batch size
- Idempotence: Enabled to prevent duplicates

**Consumer Configuration**:
- Auto-commit: Disabled (manual commit after processing)
- Offset reset: Earliest (replay from beginning on new consumer group)
- Max poll records: 500
- Session timeout: 30 seconds

---

## Message Queue Patterns

### Pub/Sub Pattern

Used for broadcasting events to multiple interested consumers:

```
Workflow Engine publishes workflow.execution.completed
                        │
        ┌───────────────┼───────────────┬──────────────┐
        ▼               ▼               ▼              ▼
  Analytics      Notification    Audit Logger    Billing
   Service         Service         Service       Service
```

Each service maintains its own consumer group and processes events independently. Failures in one consumer don't affect others.

### Request-Reply Pattern

Synchronous communication over async messaging (used sparingly):

```
API Gateway publishes request → Kafka (reply-to: temp-queue-123)
                                          │
                                          ▼
                                  Service processes request
                                          │
                                          ▼
                        Service publishes response → temp-queue-123
                                          │
                                          ▼
                            API Gateway receives response
```

Timeout: 5 seconds, fallback to direct HTTP call if no response.

### Saga Pattern

Distributed transaction management for multi-service workflows:

```
Step 1: Create Order → SUCCESS → Step 2: Reserve Inventory
                                              │
                                         FAILURE
                                              │
                                              ▼
                                  Compensating Transaction
                                              │
                                              ▼
                                    Cancel Order (Rollback)
```

Orchestrated by Workflow Engine with compensation logic defined in workflow DSL.

### Dead Letter Queue (DLQ)

Failed messages after all retry attempts are routed to DLQ:

```
Message processing fails (3 retries with exponential backoff)
                        │
                        ▼
          Publish to dead-letter-queue topic
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  Alert to PagerDuty        Store in PostgreSQL for analysis
        │                               │
        ▼                               ▼
  Manual investigation      Automated pattern detection
```

DLQ Processing SLA: < 4 hours for critical events, < 24 hours for non-critical.

---

## Caching Strategy

### Cache-Aside Pattern (Lazy Loading)

Primary caching pattern used across all services:

```
1. Application checks cache (Redis GET)
2. If HIT → Return cached data
3. If MISS:
   a. Query database (PostgreSQL)
   b. Store result in cache with TTL
   c. Return data to application
```

**Implementation Example** (Workflow Definition):
```javascript
async function getWorkflowDefinition(workflowId) {
  const cacheKey = `workflow:def:${workflowId}`;
  
  // Step 1: Check cache
  let workflow = await redis.get(cacheKey);
  if (workflow) {
    metrics.increment('cache.hit.workflow_def');
    return JSON.parse(workflow);
  }
  
  // Step 2: Cache miss - query database
  metrics.increment('cache.miss.workflow_def');
  workflow = await db.query(
    'SELECT * FROM workflows WHERE id = $1',
    [workflowId]
  );
  
  // Step 3: Store in cache (1 hour TTL)
  await redis.setex(cacheKey, 3600, JSON.stringify(workflow));
  
  return workflow;
}
```

### Write-Through Pattern

Used for critical data where cache consistency is paramount:

```
1. Application writes to database
2. Database transaction commits
3. Application updates cache
4. Return success to client
```

Applied to: User profiles, authentication sessions, system configuration.

### Cache Invalidation Strategies

**Time-Based Expiration (TTL)**:
- Short-lived data: 5-15 minutes (session tokens, rate limit counters)
- Medium-lived data: 1 hour (workflow definitions, templates)
- Long-lived data: 24 hours (static configuration)

**Event-Based Invalidation**:
```
Database Update Event → Kafka (cache.invalidation topic)
                              │
                              ▼
                    All service instances consume event
                              │
                              ▼
                    Redis DEL for affected keys
```

**Pattern-Based Invalidation**:
- Use Redis SCAN + DEL for wildcard patterns
- Example: Invalidate all user caches: `user:*:{user_id}`

**Cache Warming**:
- Scheduled job runs every hour to pre-populate frequently accessed data
- Targets: Top 1000 workflows, system configuration, active user profiles
- Reduces cache miss rate during traffic spikes

### Multi-Level Caching

Application implements L1 (in-memory) and L2 (Redis) caching:

```
Request → L1 Cache (Node.js Map, 1000 entry LRU)
              │
          MISS │
              ▼
       L2 Cache (Redis, distributed)
              │
          MISS │
              ▼
       Database (PostgreSQL)
```

L1 cache reduces Redis network calls by 40% for hot data.

---

## Security Architecture

### Network Security

**Zero-Trust Network Model**:
- All service-to-service communication encrypted with mTLS
- Certificate rotation: Every 90 days (automated via cert-manager)
- Certificate authority: Internal PKI with HashiCorp Vault
- Network segmentation: Private subnets for services, public subnet for ALB only

**Service Mesh (Istio)**:
```
Service A → Envoy Sidecar (mTLS client cert) 
                  │
            Encrypted channel
                  │
            Envoy Sidecar (mTLS server cert) → Service B
```

**Firewall Rules**:
- Security groups: Deny all by default, explicit allow rules
- Ingress: Only ALB can reach API Gateway (port 8080)
- Egress: Services can only reach specific dependencies
- Database access: Limited to application subnets only

### Authentication & Authorization

**JWT Token Validation**:
- Algorithm: RS256 (asymmetric signing)
- Key rotation: Every 30 days with 7-day overlap period
- Public key distribution: JWKS endpoint cached in Redis
- Validation: Signature, expiry, issuer, audience claims
- Token revocation: Blacklist in Redis for compromised tokens

**Permission Model**:
```
User → Roles → Permissions
     ↘       ↗
      Tenants (Multi-tenancy isolation)
```

Example permissions:
- `workflow:read` - View workflows
- `workflow:write` - Create/update workflows
- `workflow:execute` - Trigger workflow execution
- `workflow:delete` - Delete workflows
- `admin:*` - All administrative operations

**API Key Management**:
- Format: `cfk_live_<32-char-random>` (production), `cfk_test_<32-char-random>` (sandbox)
- Hashing: SHA-256 before storage in PostgreSQL
- Scoping: API keys can be scoped to specific workflows or operations
- Rate limits: Configurable per API key (default: 1000 RPM)

### Secrets Management

**HashiCorp Vault Integration**:
- Dynamic database credentials: Generated on-demand, 1-hour TTL
- Encryption keys: Transit secrets engine for encryption-as-a-service
- API keys for external services: Stored with versioning
- Rotation policy: Automated rotation every 90 days with notification

**Secret Access Pattern**:
```
Service starts → Vault authentication (Kubernetes service account)
                        │
                        ▼
              Request secret lease (1-hour TTL)
                        │
                        ▼
              Vault returns dynamic credentials
                        │
                        ▼
              Service connects to PostgreSQL
                        │
              Renew lease every 30 minutes
```

### Data Protection

**Encryption at Rest**:
- PostgreSQL: AES-256 encryption enabled at volume level
- Redis: Encryption enabled using AWS KMS
- Kafka: Encryption at rest for all data on brokers
- S3 backups: Server-side encryption with KMS (SSE-KMS)

**Encryption in Transit**:
- External traffic: TLS 1.3 only (TLS 1.2 deprecated)
- Internal traffic: mTLS via Istio service mesh
- Database connections: SSL/TLS enforced (sslmode=require)

**Data Masking**:
- Sensitive fields (email, phone) masked in logs
- PII redacted from error messages and stack traces
- Audit logs: Full data retention with access controls

**Compliance**:
- GDPR: User data export and deletion workflows
- SOC 2 Type II: Audit logging for all data access
- HIPAA: PHI isolation in dedicated tenants with enhanced encryption

---

## Performance Characteristics

### Latency Targets

**API Operations** (P99 latency):
- `GET /workflows/{id}`: < 50ms (cache hit), < 150ms (cache miss)
- `POST /workflows`: < 200ms (includes validation and database write)
- `POST /workflows/{id}/execute`: < 100ms (async, returns execution ID)
- `GET /workflows/{id}/history`: < 300ms (paginated, 50 records per page)

**Workflow Execution** (P99 latency):
- Simple workflow (< 5 steps): < 2 seconds
- Medium workflow (5-15 steps): < 5 seconds
- Complex workflow (> 15 steps): < 15 seconds
- Parallel execution: Linear scaling up to 10 concurrent branches

**Database Operations** (P99 latency):
- Simple SELECT: < 5ms
- JOIN query: < 20ms
- INSERT/UPDATE: < 10ms
- Batch operations: < 100ms (batch size: 100 records)

**Cache Operations** (P99 latency):
- Redis GET: < 2ms
- Redis SET: < 3ms
- Redis DEL: < 2ms

### Throughput Capacity

**API Gateway**:
- Sustained: 10,000 requests per second
- Peak: 25,000 requests per second (5-minute burst)
- Rate limiting: 1,000 requests per minute per API key

**Workflow Engine**:
- Concurrent executions: 8,000 workflows (across 16 pods)
- Execution start rate: 500 per second
- Completion rate: 450 per second (average 2-second execution time)

**Database**:
- Read throughput: 50,000 queries per second (across replicas)
- Write throughput: 15,000 transactions per second
- Connection capacity: 2,000 concurrent connections

**Kafka**:
- Message ingestion: 100,000 messages per second
- Consumer throughput: 80,000 messages per second (aggregated)
- End-to-end latency: < 100ms (P99)

### Resource Utilization

**CPU Utilization** (Target: 60-70% average):
- API Gateway: 55% average, 80% peak
- Workflow Engine: 65% average, 85% peak
- Auth Service: 40% average, 60% peak

**Memory Utilization** (Target: < 80%):
- API Gateway: 2.5GB average per pod (4GB allocated)
- Workflow Engine: 6GB average per pod (8GB allocated)
- Notification Service: 2.8GB average per pod (4GB allocated)

**Network Throughput**:
- Ingress: 2 Gbps average, 5 Gbps peak
- Egress: 1.5 Gbps average, 4 Gbps peak
- Internal (service mesh): 8 Gbps average

### Monitoring & Alerting

**Key Metrics** (Monitored via Prometheus):
- Request rate (per service, per endpoint)
- Error rate (4xx, 5xx responses)
- Latency percentiles (P50, P95, P99)
- Resource utilization (CPU, memory, disk)
- Cache hit/miss ratios
- Database connection pool saturation
- Kafka consumer lag

**Alerts** (via PagerDuty):
- P1 (Immediate): API error rate > 1%, database connection failure
- P2 (< 15 min): P99 latency > 500ms, cache hit rate < 80%
- P3 (< 1 hour): Resource utilization > 85%, replica count < minimum

**SLI/SLO/SLA**:
- SLI: API success rate (non-5xx responses)
- SLO: 99.9% uptime per month (43 minutes downtime allowance)
- SLA: 99.5% uptime guarantee (customer-facing SLA with credits)

---

## Disaster Recovery

### Recovery Objectives

**RPO (Recovery Point Objective): 1 hour**
- Maximum acceptable data loss: 1 hour of transactions
- Achieved through: Continuous database replication + hourly snapshots
- Kafka retention: 7 days allows event replay

**RTO (Recovery Time Objective): 4 hours**
- Maximum acceptable downtime: 4 hours for full system recovery
- Includes: Failover, data verification, and service restoration
- Automated runbooks reduce RTO to < 2 hours for common scenarios

### High Availability Architecture

**Multi-AZ Deployment**:
```
Region: us-east-1

AZ-1a:                   AZ-1b:                   AZ-1c:
- API Gateway (4)        - API Gateway (4)        - API Gateway (4)
- Workflow Engine (6)    - Workflow Engine (5)    - Workflow Engine (5)
- Auth Service (3)       - Auth Service (3)       - Auth Service (2)
- PostgreSQL Primary     - PostgreSQL Replica     - PostgreSQL Replica
- Redis Primary (2)      - Redis Replica (2)      - Redis Replica (2)
- Kafka Broker (2)       - Kafka Broker (2)       - Kafka Broker (1)
```

**Automatic Failover**:
- Database: 30-60 seconds (automatic promotion of replica)
- Redis: < 10 seconds (Sentinel-based failover)
- Kafka: < 30 seconds (controller election)
- Services: Kubernetes health checks with 10-second liveness probes

### Backup Strategy

**Database Backups**:
- Automated snapshots: Daily at 02:00 UTC
- Retention: 30 days for daily, 90 days for monthly
- Cross-region replication: Async replication to us-west-2 (15-minute lag)
- Backup verification: Weekly automated restore test in staging environment

**Backup Schedule**:
```
Daily:    Full snapshot → S3 (encrypted)
Hourly:   WAL archives → S3 (point-in-time recovery)
Weekly:   Backup validation test
Monthly:  Long-term archive to Glacier
```

**Configuration Backups**:
- Kubernetes manifests: Stored in Git (GitOps with ArgoCD)
- Vault secrets: Automated snapshot every 6 hours
- Kafka topic configurations: Exported daily to S3

### Disaster Recovery Procedures

**Scenario 1: Single AZ Failure**
- Detection: Health checks fail for entire AZ (< 30 seconds)
- Action: Traffic automatically routed to healthy AZs by ALB
- Recovery time: < 5 minutes (no manual intervention)
- Data loss: None (multi-AZ replication)

**Scenario 2: Database Primary Failure**
- Detection: Health check fails for primary database (< 30 seconds)
- Action: Automatic promotion of read replica to primary
- Recovery time: 30-60 seconds
- Data loss: Minimal (< 1 second due to synchronous replication)

**Scenario 3: Full Region Failure**
- Detection: Multiple health check failures across all AZs (< 2 minutes)
- Action: Manual failover to DR region (us-west-2)
- Procedure:
  1. Update DNS to point to DR region (TTL: 60 seconds)
  2. Promote DR database replica to primary
  3. Scale up DR region services to production capacity
  4. Verify data consistency and integrity
  5. Update monitoring dashboards
- Recovery time: 2-4 hours (includes verification)
- Data loss: < 1 hour (last cross-region replication)

**Scenario 4: Data Corruption**
- Detection: Data validation checks or user report
- Action: Point-in-time recovery from WAL archives
- Procedure:
  1. Identify corruption time window
  2. Restore from snapshot prior to corruption
  3. Replay WAL logs up to corruption point
  4. Verify data integrity
  5. Resume normal operations
- Recovery time: 1-3 hours depending on data volume
- Data loss: None if corruption detected quickly

### Testing & Validation

**DR Drill Schedule**:
- Monthly: Automated failover test (single AZ failure simulation)
- Quarterly: Full DR region failover (non-production hours)
- Annually: Complete disaster simulation with all stakeholders

**Last DR Test Results** (Dec 15, 2025):
- Scenario: Full region failover
- Actual RTO: 2 hours 23 minutes (target: 4 hours)
- Actual RPO: 42 minutes (target: 1 hour)
- Issues identified: DNS propagation slower than expected (resolved)
- Success criteria: Met all recovery objectives

### Business Continuity

**Communication Plan**:
- Status page: status.cloudflow.com (updated every 15 minutes during incident)
- Customer notifications: Email + SMS for all P1 incidents
- Internal escalation: PagerDuty → Incident Commander → Engineering Manager → CTO

**Data Retention Policy**:
- Active data: 12 months in hot storage (PostgreSQL)
- Archived data: 7 years in cold storage (S3 Glacier)
- Audit logs: 7 years (compliance requirement)
- Backup retention: 30 days standard, 90 days monthly snapshots

---

## Appendix

### Service Dependency Matrix

| Service | Depends On | Critical Path |
|---------|-----------|---------------|
| API Gateway | Auth Service, Redis | Yes |
| Auth Service | PostgreSQL, Redis, Vault | Yes |
| Workflow Engine | PostgreSQL, Kafka, Redis | Yes |
| Scheduler | PostgreSQL, Redis, Kafka | No |
| Notification Service | Kafka, PostgreSQL, SendGrid, Twilio | No |

### Capacity Planning

**Current Capacity** (Jan 2026):
- Daily workflow executions: 2.5M
- Active users: 150,000
- API requests per day: 50M
- Database size: 1.2TB
- Kafka throughput: 100K msg/sec peak

**6-Month Projection** (Jul 2026):
- Daily workflow executions: 4M (+60%)
- Active users: 225,000 (+50%)
- API requests per day: 80M (+60%)
- Database size: 2TB (+67%)
- Required scaling: +30% compute, +50% storage

### Contact Information

- **Architecture Team**: architecture@cloudflow.internal
- **On-Call Engineer**: PagerDuty escalation
- **Security Team**: security@cloudflow.internal
- **Documentation**: https://wiki.cloudflow.internal/architecture

---

**Document Revision History**:
- v2.3.1 (Jan 15, 2026): Updated performance metrics, added DR test results
- v2.3.0 (Dec 1, 2025): Added multi-AZ deployment details
- v2.2.0 (Oct 15, 2025): Security architecture overhaul
- v2.1.0 (Sep 1, 2025): Initial Kafka migration documentation
- v2.0.0 (Jul 1, 2025): Complete microservices rewrite
