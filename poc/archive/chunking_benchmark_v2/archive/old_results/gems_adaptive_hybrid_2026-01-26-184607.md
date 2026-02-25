# Gem Strategy Test Results

**Strategy**: adaptive_hybrid
**Date**: 2026-01-26T18:46:07.346361
**Queries Tested**: 15

## Query: mh_002
**Type**: multi-hop
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND

**Query**: If I'm hitting connection pool exhaustion, should I use PgBouncer or add read replicas?

**Expected Answer**: Both are valid. PgBouncer for connection pooling (max_db_connections=100, pool_mode=transaction). Read replicas for read-heavy workloads. Troubleshooting guide recommends PgBouncer first.

**Retrieved Chunks**:
1. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_9, score: 1.000]
   > #### Pods in CrashLoopBackOff
   > 
   > **Symptoms**: Pods continuously restart
   > **Diagnosis**:
   > ```bash
   > kubectl logs -n cloudflow-prod <pod-name> --previous
   > kubectl describe pod -n cloudflow-prod <pod-name>
   > ```
   > 
   > **Common Causes**:
   > - Database connection failure
   > - Invalid environment variables
   > - Insufficient resources
   > 
   > #### High Memory Usage
   > 
   > **Symptoms**: Pods being OOMKilled
   > **Diagnosis**:
   > ```bash
   > kubectl top pods -n cloudflow-prod
   > ```
   > 
   > **Resolution**:
   > - Increase memory limits in deployment
   > - Check for me...

2. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_10, score: 0.500]
   > ## Database Architecture
   > 
   > 
   > 
   > ### PostgreSQL Primary Database
   > 
   > **Cluster Configuration**:
   > - Primary-replica setup with 1 primary + 2 read replicas
   > - Instance type: db.r6g.2xlarge (8 vCPU, 64GB RAM)
   > - Storage: 2TB gp3 SSD with 12,000 IOPS
   > - Multi-AZ deployment for high availability
   > - Automated backups: Daily snapshots, 30-day retention
   > - Point-in-time recovery: 5-minute granularity
   > 
   > **Database Schema Design**:
   > 
   > ```
   > Core Tables:
   > - users (5M rows): User accounts and profiles
   > - workflows (2M rows): Wo...

3. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_4, score: 0.333]
   > # postgres-values.yaml
   > 
   > global:
   >   postgresql:
   >     auth:
   >       username: cloudflow
   >       database: cloudflow
   >       existingSecret: postgres-credentials
   > 
   > image:
   >   tag: "14.10.0"
   > 
   > primary:
   >   resources:
   >     limits:
   >       cpu: 4000m
   >       memory: 8Gi
   >     requests:
   >       cpu: 2000m
   >       memory: 4Gi
   >   
   >   persistence:
   >     enabled: true
   >     size: 100Gi
   >     storageClass: gp3
   >   
   >   extendedConfiguration: |
   >     max_connections = 100
   >     shared_buffers = 2GB
   >     effective_cache_size = 6GB
   >     maintenance_wor...

4. [doc_id: user_guide, chunk_id: user_guide_mdsem_7, score: 0.250]
   > ### Database Queries
   > 
   > Execute queries against supported databases (PostgreSQL, MySQL, MongoDB, Redis):
   > 
   > **SQL Databases (PostgreSQL, MySQL):**
   > ```yaml
   > - id: get_orders
   >   action: database_query
   >   config:
   >     connection: "{{secrets.DB_CONNECTION_STRING}}"
   >     query: |
   >       SELECT * FROM orders 
   >       WHERE customer_id = $1 
   >       AND status = $2
   >       ORDER BY created_at DESC
   >       LIMIT 10
   >     parameters:
   >       - "{{trigger.customer_id}}"
   >       - "pending"
   > ```
   > 
   > **MongoDB:**
   > ```yaml
   > - id: find_do...

5. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_18, score: 0.200]
   > ### Throughput Capacity
   > 
   > **API Gateway**:
   > - Sustained: 10,000 requests per second
   > - Peak: 25,000 requests per second (5-minute burst)
   > - Rate limiting: 1,000 requests per minute per API key
   > 
   > **Workflow Engine**:
   > - Concurrent executions: 8,000 workflows (across 16 pods)
   > - Execution start rate: 500 per second
   > - Completion rate: 450 per second (average 2-second execution time)
   > 
   > **Database**:
   > - Read throughput: 50,000 queries per second (across replicas)
   > - Write throughput: 15,000 transactions per se...

**Baseline Score**: 5/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: mh_004
**Type**: multi-hop
**Root Causes**: YAML_BLIND, EMBEDDING_BLIND

**Query**: How do the HPA scaling parameters relate to the API Gateway resource requirements?

**Expected Answer**: HPA: minReplicas=3, maxReplicas=10, targetCPU=70%. API Gateway: 2 vCPU, 4GB RAM per pod. Scales when CPU exceeds 70% of 2 vCPU.

**Retrieved Chunks**:
1. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_8, score: 1.000]
   > ## Data Flow Architecture
   > 
   > 
   > 
   > ### Synchronous Request Flow
   > 
   > Client requests follow a synchronous path through the API Gateway to backend services:
   > 
   > ```
   > 1. Client → Load Balancer (TLS termination)
   > 2. Load Balancer → API Gateway (HTTP/2)
   > 3. API Gateway → Auth Service (JWT validation via gRPC)
   > 4. API Gateway → Target Service (HTTP/REST or gRPC)
   > 5. Target Service → Database/Cache (data retrieval)
   > 6. Response propagates back through the chain
   > ```
   > 
   > **Timeout Configuration**:
   > - Client → Load Balancer: 6...

2. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_22, score: 0.500]
   > ### High Availability Architecture
   > 
   > **Multi-AZ Deployment**:
   > ```
   > Region: us-east-1
   > 
   > AZ-1a:                   AZ-1b:                   AZ-1c:
   > - API Gateway (4)        - API Gateway (4)        - API Gateway (4)
   > - Workflow Engine (6)    - Workflow Engine (5)    - Workflow Engine (5)
   > - Auth Service (3)       - Auth Service (3)       - Auth Service (2)
   > - PostgreSQL Primary     - PostgreSQL Replica     - PostgreSQL Replica
   > - Redis Primary (2)      - Redis Replica (2)      - Redis Replica (2)
   > - Kafka B...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_17, score: 0.333]
   > ## Performance Characteristics
   > 
   > 
   > 
   > ### Latency Targets
   > 
   > **API Operations** (P99 latency):
   > - `GET /workflows/{id}`: < 50ms (cache hit), < 150ms (cache miss)
   > - `POST /workflows`: < 200ms (includes validation and database write)
   > - `POST /workflows/{id}/execute`: < 100ms (async, returns execution ID)
   > - `GET /workflows/{id}/history`: < 300ms (paginated, 50 records per page)
   > 
   > **Workflow Execution** (P99 latency):
   > - Simple workflow (< 5 steps): < 2 seconds
   > - Medium workflow (5-15 steps): < 5 seconds
   > - C...

4. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_1, score: 0.250]
   > ## Table of Contents
   > 
   > 1. [High-Level Architecture](#high-level-architecture)
   > 2. [Microservices Breakdown](#microservices-breakdown)
   > 3. [Data Flow Architecture](#data-flow-architecture)
   > 4. [Database Architecture](#database-architecture)
   > 5. [Message Queue Patterns](#message-queue-patterns)
   > 6. [Caching Strategy](#caching-strategy)
   > 7. [Security Architecture](#security-architecture)
   > 8. [Performance Characteristics](#performance-characteristics)
   > 9. [Disaster Recovery](#disaster-recovery)
   > 
   > ---
   > 
   > ## High...

5. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_3, score: 0.200]
   > ## Microservices Breakdown
   > 
   > 
   > 
   > ### API Gateway
   > 
   > **Purpose**: Single entry point for all client requests, providing authentication, rate limiting, request routing, and protocol translation.
   > 
   > **Technology**: Node.js with Express.js framework  
   > **Replicas**: 12 pods (production), auto-scaling 8-20 based on CPU  
   > **Resource Allocation**: 2 vCPU, 4GB RAM per pod
   > 
   > **Key Responsibilities**:
   > - JWT token validation (delegated to Auth Service for initial validation)
   > - Rate limiting: 1000 requests per minut...

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: tmp_003
**Type**: temporal
**Root Causes**: EMBEDDING_BLIND

**Query**: What's the sequence of events when a workflow execution times out?

**Expected Answer**: Workflow runs up to 3600s. If exceeded, automatically terminated. Error: 'exceeded maximum execution time of 3600 seconds'. Status: TIMEOUT. Can request custom timeout up to 7200s on Enterprise.

**Retrieved Chunks**:
1. [doc_id: user_guide, chunk_id: user_guide_mdsem_13, score: 1.000]
   > ### Fallback Actions
   > 
   > Execute alternative actions when the primary action fails:
   > 
   > ```yaml
   > - id: primary_payment
   >   action: http_request
   >   config:
   >     url: "https://primary-payment-gateway.com/charge"
   >     method: POST
   >     body:
   >       amount: "{{amount}}"
   >   on_error:
   >     - id: fallback_payment
   >       action: http_request
   >       config:
   >         url: "https://backup-payment-gateway.com/charge"
   >         method: POST
   >         body:
   >           amount: "{{amount}}"
   >     - id: notify_admin
   >       action: email
   >  ...

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_4, score: 0.500]
   > # Create sub-workflows
   > 
   > cloudflow workflows create data-pipeline-part1 \
   >   --steps "data_ingestion,data_validation" \
   >   --timeout 1800
   > 
   > cloudflow workflows create data-pipeline-part2 \
   >   --steps "data_transformation,data_export" \
   >   --timeout 3600 \
   >   --trigger workflow_completed \
   >   --trigger-workflow data-pipeline-part1
   > ```
   > 
   > ### Retry Logic and Exponential Backoff
   > 
   > CloudFlow implements automatic retry with exponential backoff for transient failures:
   > - Max retries: 3
   > - Initial delay: 1 second
   > -...

3. [doc_id: user_guide, chunk_id: user_guide_mdsem_15, score: 0.333]
   > ### Execution Limits
   > 
   > - **Maximum**: 1000 executions per day (per workflow)
   > - **Rate Limiting**: 100 concurrent executions per workflow
   > - **Burst Limit**: 10 executions per second
   > 
   > **What happens when limits are reached:**
   > - New executions are queued automatically
   > - Webhook triggers return HTTP 429 (Too Many Requests)
   > - Scheduled executions are skipped (logged in audit trail)
   > - Email notifications sent to workflow owner
   > 
   > **Monitoring Usage:**
   > View real-time metrics in your workflow dashboard:
   > - ...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_11, score: 0.250]
   > #### Step 2: Gather Information (5-15 minutes)
   > 
   > Create incident document with:
   > - Incident timestamp and duration
   > - Affected services and endpoints
   > - Error rates and user impact
   > - Recent changes or deployments
   > - Relevant log excerpts
   > - Correlation IDs for failed requests
   > 
   > ```bash
   > 
   > # Generate incident report
   > 
   > cloudflow debug incident-report \
   >   --start "2026-01-24T10:30:00Z" \
   >   --end "2026-01-24T11:00:00Z" \
   >   --output incident-report.md
   > 
   > # Capture system snapshot
   > 
   > cloudflow debug snapshot --outp...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_5, score: 0.200]
   > ### Triggers
   > 
   > Triggers determine when your workflow runs. CloudFlow supports several trigger types:
   > 
   > **Webhook Triggers**
   > Receive HTTP requests at a unique URL to start your workflow:
   > - Support GET, POST, PUT, PATCH, DELETE methods
   > - Automatically parse JSON and form data
   > - Access headers, query parameters, and body in your workflow
   > 
   > **Schedule Triggers**
   > Run workflows on a recurring schedule (see [Scheduling](#scheduling) for details)
   > 
   > **Event Triggers**
   > Respond to events from integrated applic...

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: tmp_004
**Type**: temporal
**Root Causes**: EMBEDDING_BLIND

**Query**: How long does it take for workflow definition cache changes to propagate?

**Expected Answer**: Workflow definitions cached in Redis with TTL of 1 hour. Cache invalidated on workflow update or manual flush. Cache hit rate is 94.2%.

**Retrieved Chunks**:
1. [doc_id: user_guide, chunk_id: user_guide_mdsem_17, score: 1.000]
   > ### 8. Test Thoroughly
   > 
   > Before activating a workflow:
   > 1. Use test mode with sample data
   > 2. Verify all actions execute correctly
   > 3. Test error handling paths
   > 4. Review execution logs
   > 5. Start with a limited scope (e.g., test channel, small dataset)
   > 
   > ### 9. Document Your Workflows
   > 
   > Add descriptions to workflows and steps:
   > 
   > ```yaml
   > workflow:
   >   name: "Daily Sales Report"
   >   description: |
   >     Generates a daily sales report and distributes it to the sales team.
   >     Runs at 8:00 AM EST Monday-Friday.
   >  ...

2. [doc_id: user_guide, chunk_id: user_guide_mdsem_10, score: 0.500]
   > ### Timezone Handling
   > 
   > All scheduled workflows run in **UTC by default**. To account for your local timezone:
   > 
   > **Option 1: Convert to UTC**
   > If you want a workflow to run at 9:00 AM EST (UTC-5), schedule it for 14:00 UTC:
   > ```
   > 0 14 * * *  # 9:00 AM EST = 14:00 UTC
   > ```
   > 
   > **Option 2: Use Timezone Configuration**
   > Specify a timezone in your workflow configuration:
   > ```yaml
   > schedule:
   >   cron: "0 9 * * *"
   >   timezone: "America/New_York"  # IANA timezone identifier
   > ```
   > 
   > **Supported Timezones:**
   > CloudFlow sup...

3. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_3, score: 0.333]
   > # Check database slow query log
   > 
   > kubectl logs -n cloudflow deploy/cloudflow-db-primary | \
   >   grep "slow query" | \
   >   tail -n 50
   > 
   > # Analyze query patterns
   > 
   > cloudflow db analyze-queries --min-duration 5000 --limit 20
   > ```
   > 
   > **2. Review Query Execution Plans**
   > 
   > ```sql
   > -- Connect to CloudFlow database
   > cloudflow db connect --readonly
   > 
   > -- Explain slow query
   > EXPLAIN ANALYZE
   > SELECT w.*, e.status, e.error_message
   > FROM workflows w
   > LEFT JOIN executions e ON w.id = e.workflow_id
   > WHERE w.workspace_id = 'ws_abc...

4. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_13, score: 0.250]
   > ### Cache Invalidation Strategies
   > 
   > **Time-Based Expiration (TTL)**:
   > - Short-lived data: 5-15 minutes (session tokens, rate limit counters)
   > - Medium-lived data: 1 hour (workflow definitions, templates)
   > - Long-lived data: 24 hours (static configuration)
   > 
   > **Event-Based Invalidation**:
   > ```
   > Database Update Event → Kafka (cache.invalidation topic)
   >                               │
   >                               ▼
   >                     All service instances consume event
   >                               │
   >     ...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_4, score: 0.200]
   > ### YAML Definition
   > 
   > For advanced users and version control integration, CloudFlow supports YAML-based workflow definitions:
   > 
   > ```yaml
   > name: "Process Customer Orders"
   > description: "Validates and processes new customer orders"
   > version: "1.0"
   > 
   > trigger:
   >   type: webhook
   >   method: POST
   >   path: /orders/new
   > 
   > steps:
   >   - id: validate_order
   >     name: "Validate Order Data"
   >     action: javascript
   >     code: |
   >       if (!input.order_id || !input.customer_email) {
   >         throw new Error("Missing required field...

**Baseline Score**: 4/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: tmp_005
**Type**: temporal
**Root Causes**: EMBEDDING_BLIND

**Query**: What's the timeline for automatic failover when the database primary fails?

**Expected Answer**: Database primary failure: 30-60 seconds for automatic promotion of replica. Redis failover: <10 seconds. Kafka controller election: <30 seconds.

**Retrieved Chunks**:
1. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_23, score: 1.000]
   > ### Disaster Recovery Procedures
   > 
   > **Scenario 1: Single AZ Failure**
   > - Detection: Health checks fail for entire AZ (< 30 seconds)
   > - Action: Traffic automatically routed to healthy AZs by ALB
   > - Recovery time: < 5 minutes (no manual intervention)
   > - Data loss: None (multi-AZ replication)
   > 
   > **Scenario 2: Database Primary Failure**
   > - Detection: Health check fails for primary database (< 30 seconds)
   > - Action: Automatic promotion of read replica to primary
   > - Recovery time: 30-60 seconds
   > - Data loss: Mini...

2. [doc_id: user_guide, chunk_id: user_guide_mdsem_13, score: 0.500]
   > ### Fallback Actions
   > 
   > Execute alternative actions when the primary action fails:
   > 
   > ```yaml
   > - id: primary_payment
   >   action: http_request
   >   config:
   >     url: "https://primary-payment-gateway.com/charge"
   >     method: POST
   >     body:
   >       amount: "{{amount}}"
   >   on_error:
   >     - id: fallback_payment
   >       action: http_request
   >       config:
   >         url: "https://backup-payment-gateway.com/charge"
   >         method: POST
   >         body:
   >           amount: "{{amount}}"
   >     - id: notify_admin
   >       action: email
   >  ...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_10, score: 0.333]
   > ## Database Architecture
   > 
   > 
   > 
   > ### PostgreSQL Primary Database
   > 
   > **Cluster Configuration**:
   > - Primary-replica setup with 1 primary + 2 read replicas
   > - Instance type: db.r6g.2xlarge (8 vCPU, 64GB RAM)
   > - Storage: 2TB gp3 SSD with 12,000 IOPS
   > - Multi-AZ deployment for high availability
   > - Automated backups: Daily snapshots, 30-day retention
   > - Point-in-time recovery: 5-minute granularity
   > 
   > **Database Schema Design**:
   > 
   > ```
   > Core Tables:
   > - users (5M rows): User accounts and profiles
   > - workflows (2M rows): Wo...

4. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_24, score: 0.250]
   > ### Testing & Validation
   > 
   > **DR Drill Schedule**:
   > - Monthly: Automated failover test (single AZ failure simulation)
   > - Quarterly: Full DR region failover (non-production hours)
   > - Annually: Complete disaster simulation with all stakeholders
   > 
   > **Last DR Test Results** (Dec 15, 2025):
   > - Scenario: Full region failover
   > - Actual RTO: 2 hours 23 minutes (target: 4 hours)
   > - Actual RPO: 42 minutes (target: 1 hour)
   > - Issues identified: DNS propagation slower than expected (resolved)
   > - Success criteria: Met a...

5. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_6, score: 0.200]
   > ### Scheduler Service
   > 
   > **Purpose**: Time-based workflow triggering system supporting cron-like schedules and one-time delayed executions.
   > 
   > **Technology**: Go with distributed locking via Redis  
   > **Replicas**: 4 pods (production), active-passive with leader election  
   > **Resource Allocation**: 2 vCPU, 4GB RAM per pod
   > 
   > **Key Responsibilities**:
   > - Parse and validate cron expressions (extended format supporting seconds)
   > - Maintain schedule registry in PostgreSQL
   > - Distributed scheduling with leader e...

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: cmp_001
**Type**: comparative
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND

**Query**: What's the difference between PgBouncer connection pooling and direct PostgreSQL connections?

**Expected Answer**: PgBouncer: pool_mode=transaction, default_pool_size=25, max_db_connections=100. Allows 1000 client connections with only 100 actual DB connections. Direct: limited to max_connections=100.

**Retrieved Chunks**:
1. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_4, score: 1.000]
   > # postgres-values.yaml
   > 
   > global:
   >   postgresql:
   >     auth:
   >       username: cloudflow
   >       database: cloudflow
   >       existingSecret: postgres-credentials
   > 
   > image:
   >   tag: "14.10.0"
   > 
   > primary:
   >   resources:
   >     limits:
   >       cpu: 4000m
   >       memory: 8Gi
   >     requests:
   >       cpu: 2000m
   >       memory: 4Gi
   >   
   >   persistence:
   >     enabled: true
   >     size: 100Gi
   >     storageClass: gp3
   >   
   >   extendedConfiguration: |
   >     max_connections = 100
   >     shared_buffers = 2GB
   >     effective_cache_size = 6GB
   >     maintenance_wor...

2. [doc_id: deployment_guide, chunk_id: deployment_guide_mdsem_9, score: 0.500]
   > #### Pods in CrashLoopBackOff
   > 
   > **Symptoms**: Pods continuously restart
   > **Diagnosis**:
   > ```bash
   > kubectl logs -n cloudflow-prod <pod-name> --previous
   > kubectl describe pod -n cloudflow-prod <pod-name>
   > ```
   > 
   > **Common Causes**:
   > - Database connection failure
   > - Invalid environment variables
   > - Insufficient resources
   > 
   > #### High Memory Usage
   > 
   > **Symptoms**: Pods being OOMKilled
   > **Diagnosis**:
   > ```bash
   > kubectl top pods -n cloudflow-prod
   > ```
   > 
   > **Resolution**:
   > - Increase memory limits in deployment
   > - Check for me...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_10, score: 0.333]
   > ## Database Architecture
   > 
   > 
   > 
   > ### PostgreSQL Primary Database
   > 
   > **Cluster Configuration**:
   > - Primary-replica setup with 1 primary + 2 read replicas
   > - Instance type: db.r6g.2xlarge (8 vCPU, 64GB RAM)
   > - Storage: 2TB gp3 SSD with 12,000 IOPS
   > - Multi-AZ deployment for high availability
   > - Automated backups: Daily snapshots, 30-day retention
   > - Point-in-time recovery: 5-minute granularity
   > 
   > **Database Schema Design**:
   > 
   > ```
   > Core Tables:
   > - users (5M rows): User accounts and profiles
   > - workflows (2M rows): Wo...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_3, score: 0.250]
   > # Check database slow query log
   > 
   > kubectl logs -n cloudflow deploy/cloudflow-db-primary | \
   >   grep "slow query" | \
   >   tail -n 50
   > 
   > # Analyze query patterns
   > 
   > cloudflow db analyze-queries --min-duration 5000 --limit 20
   > ```
   > 
   > **2. Review Query Execution Plans**
   > 
   > ```sql
   > -- Connect to CloudFlow database
   > cloudflow db connect --readonly
   > 
   > -- Explain slow query
   > EXPLAIN ANALYZE
   > SELECT w.*, e.status, e.error_message
   > FROM workflows w
   > LEFT JOIN executions e ON w.id = e.workflow_id
   > WHERE w.workspace_id = 'ws_abc...

5. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_12, score: 0.200]
   > ### Kafka Event Streaming
   > 
   > **Cluster Configuration**:
   > - 5 broker nodes (distributed across 3 AZs)
   > - Instance type: kafka.m5.2xlarge (8 vCPU, 32GB RAM)
   > - Storage: 10TB per broker (gp3 SSD)
   > - ZooKeeper ensemble: 3 nodes for cluster coordination
   > - Replication factor: 3 (min in-sync replicas: 2)
   > 
   > **Topic Architecture**:
   > 
   > ```
   > workflow.events (32 partitions):
   >   - Workflow lifecycle events (created, started, completed, failed)
   >   - Retention: 7 days
   >   - Message rate: 5,000/sec peak
   >   - Consumer groups: ...

**Baseline Score**: 2/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: cmp_002
**Type**: comparative
**Root Causes**: EMBEDDING_BLIND

**Query**: How do fixed, linear, and exponential backoff strategies differ for retries?

**Expected Answer**: Fixed: same wait time (1s, 1s, 1s). Linear: increase by fixed amount (1s, 2s, 3s). Exponential: double each time (1s, 2s, 4s). Exponential is recommended.

**Retrieved Chunks**:
1. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_13, score: 1.000]
   > ### Cache Invalidation Strategies
   > 
   > **Time-Based Expiration (TTL)**:
   > - Short-lived data: 5-15 minutes (session tokens, rate limit counters)
   > - Medium-lived data: 1 hour (workflow definitions, templates)
   > - Long-lived data: 24 hours (static configuration)
   > 
   > **Event-Based Invalidation**:
   > ```
   > Database Update Event → Kafka (cache.invalidation topic)
   >                               │
   >                               ▼
   >                     All service instances consume event
   >                               │
   >     ...

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_4, score: 0.500]
   > # Create sub-workflows
   > 
   > cloudflow workflows create data-pipeline-part1 \
   >   --steps "data_ingestion,data_validation" \
   >   --timeout 1800
   > 
   > cloudflow workflows create data-pipeline-part2 \
   >   --steps "data_transformation,data_export" \
   >   --timeout 3600 \
   >   --trigger workflow_completed \
   >   --trigger-workflow data-pipeline-part1
   > ```
   > 
   > ### Retry Logic and Exponential Backoff
   > 
   > CloudFlow implements automatic retry with exponential backoff for transient failures:
   > - Max retries: 3
   > - Initial delay: 1 second
   > -...

3. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 0.333]
   > #### Handling Rate Limits in Code
   > 
   > **Python example with retry logic:**
   > ```python
   > import time
   > import requests
   > 
   > def cloudflow_api_call_with_retry(url, headers, max_retries=3):
   >     for attempt in range(max_retries):
   >         response = requests.get(url, headers=headers)
   >         
   >         if response.status_code == 429:
   >             retry_after = int(response.headers.get('Retry-After', 60))
   >             print(f"Rate limited. Waiting {retry_after} seconds...")
   >             time.sleep(retry_after)
   >        ...

4. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 0.250]
   > ## Rate Limiting
   > 
   > To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.
   > 
   > **Default Limits:**
   > - 100 requests per minute per authenticated user
   > - 20 requests per minute for unauthenticated requests
   > - Burst allowance: 150 requests in a 10-second window
   > 
   > ### Rate Limit Headers
   > 
   > Every API response includes rate limit information:
   > 
   > ```
   > X-RateLimit-Limit: 100
   > X-RateLimit-Remaining: 87
   > X-RateLimit-Reset: 1640995200
   > ```
   > 
   > When you exceed the rate limit, you'll rec...

5. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_12, score: 0.200]
   > ### Getting Help
   > 
   > If this troubleshooting guide doesn't resolve your issue:
   > 
   > 1. Search the knowledge base: `cloudflow kb search "your issue"`
   > 2. Check community forum for similar issues
   > 3. Contact support with detailed logs and reproduction steps
   > 4. For urgent issues, use emergency escalation procedures
   > 
   > **Remember:** Always capture logs, metrics, and reproduction steps before escalating!
   > 
   > ---
   > 
   > *Last updated: January 24, 2026*  
   > *Document version: 3.2.1*  
   > *Feedback: docs-feedback@cloudflow.io*

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: cmp_003
**Type**: comparative
**Root Causes**: EMBEDDING_BLIND

**Query**: What's the difference between /health and /ready endpoints?

**Expected Answer**: /health: liveness check, returns basic status. /ready: readiness check, checks dependencies like database and redis connectivity.

**Retrieved Chunks**:
1. [doc_id: user_guide, chunk_id: user_guide_mdsem_12, score: 1.000]
   > ## Error Handling
   > 
   > Robust error handling ensures your workflows are resilient and reliable.
   > 
   > ### Retry Policies
   > 
   > Configure automatic retries for failed actions:
   > 
   > ```yaml
   > - id: api_call
   >   action: http_request
   >   config:
   >     url: "https://api.example.com/data"
   >   retry:
   >     max_attempts: 3
   >     backoff_type: "exponential"  # or "fixed", "linear"
   >     initial_interval: 1000       # milliseconds
   >     max_interval: 30000
   >     multiplier: 2.0
   >     retry_on:
   >       - timeout
   >       - network_error
   >       - statu...

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_6, score: 0.500]
   > ### Analytics
   > 
   > 
   > 
   > #### Get Workflow Metrics
   > 
   > Retrieve performance metrics and analytics for workflows.
   > 
   > **Endpoint:** `GET /analytics/workflows/{workflow_id}`
   > 
   > **Query Parameters:**
   > - `period` (required): Time period (`1h`, `24h`, `7d`, `30d`)
   > - `metrics` (optional): Comma-separated list of metrics
   > 
   > **Available Metrics:**
   > - `execution_count`: Total number of executions
   > - `success_rate`: Percentage of successful executions
   > - `avg_duration`: Average execution duration in milliseconds
   > - `error_count...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_5, score: 0.333]
   > ### Workflow Engine
   > 
   > **Purpose**: Core orchestration service that executes workflow definitions, manages state transitions, and coordinates task execution across distributed systems.
   > 
   > **Technology**: Node.js with TypeScript, Bull queue library  
   > **Replicas**: 16 pods (production), auto-scaling 12-24  
   > **Resource Allocation**: 4 vCPU, 8GB RAM per pod
   > 
   > **Key Responsibilities**:
   > - Parse and validate workflow definitions (JSON-based DSL)
   > - Execute workflow steps with state machine pattern
   > - Handle r...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_10, score: 0.250]
   > #### SEV-3: Medium (P3)
   > 
   > - **Definition:** Partial functionality degraded affecting some users
   > - **Examples:**
   >   - Intermittent failures for specific workflow types
   >   - Minor performance issues
   >   - Non-critical feature unavailable
   > - **Response Time:** < 4 hours
   > - **Escalation:** Create ticket, normal business hours support
   > 
   > #### SEV-4: Low (P4)
   > 
   > - **Definition:** Minor issues with minimal user impact
   > - **Examples:**
   >   - Cosmetic issues
   >   - Documentation errors
   >   - Feature requests
   > - **Response T...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_6, score: 0.200]
   > ## Available Actions
   > 
   > CloudFlow provides a comprehensive library of actions to build powerful automations.
   > 
   > ### HTTP Requests
   > 
   > Make HTTP requests to any API endpoint:
   > 
   > **Configuration:**
   > - **Method**: GET, POST, PUT, PATCH, DELETE, HEAD
   > - **URL**: Full endpoint URL (supports variable interpolation)
   > - **Headers**: Custom headers as key-value pairs
   > - **Query Parameters**: URL parameters
   > - **Body**: JSON, form data, or raw text
   > - **Authentication**: Basic Auth, Bearer Token, API Key, OAuth 2.0
   > 
   > **E...

**Baseline Score**: 5/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_001
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: What should I NOT do when I'm rate limited?

**Expected Answer**: Don't keep hammering the API. Instead: check Retry-After header, implement exponential backoff, monitor X-RateLimit-Remaining, cache responses, consider upgrading tier.

**Retrieved Chunks**:
1. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 1.000]
   > ## Rate Limiting
   > 
   > To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.
   > 
   > **Default Limits:**
   > - 100 requests per minute per authenticated user
   > - 20 requests per minute for unauthenticated requests
   > - Burst allowance: 150 requests in a 10-second window
   > 
   > ### Rate Limit Headers
   > 
   > Every API response includes rate limit information:
   > 
   > ```
   > X-RateLimit-Limit: 100
   > X-RateLimit-Remaining: 87
   > X-RateLimit-Reset: 1640995200
   > ```
   > 
   > When you exceed the rate limit, you'll rec...

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 0.500]
   > #### Handling Rate Limits in Code
   > 
   > **Python example with retry logic:**
   > ```python
   > import time
   > import requests
   > 
   > def cloudflow_api_call_with_retry(url, headers, max_retries=3):
   >     for attempt in range(max_retries):
   >         response = requests.get(url, headers=headers)
   >         
   >         if response.status_code == 429:
   >             retry_after = int(response.headers.get('Retry-After', 60))
   >             print(f"Rate limited. Waiting {retry_after} seconds...")
   >             time.sleep(retry_after)
   >        ...

3. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.333]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_6, score: 0.250]
   > #### Rate Limit Tiers
   > 
   > CloudFlow enforces the following rate limits per workspace:
   > 
   > | Tier | Requests/Minute | Requests/Hour | Concurrent Workflows |
   > |------|-----------------|---------------|----------------------|
   > | Free | 60 | 1,000 | 5 |
   > | Standard | 1,000 | 50,000 | 50 |
   > | Premium | 5,000 | 250,000 | 200 |
   > | Enterprise | Custom | Custom | Unlimited |
   > 
   > #### Checking Rate Limit Status
   > 
   > ```bash
   > 
   > # Check current rate limit status
   > 
   > curl -I https://api.cloudflow.io/api/v1/workflows \
   >   -H "Author...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_13, score: 0.200]
   > ### Fallback Actions
   > 
   > Execute alternative actions when the primary action fails:
   > 
   > ```yaml
   > - id: primary_payment
   >   action: http_request
   >   config:
   >     url: "https://primary-payment-gateway.com/charge"
   >     method: POST
   >     body:
   >       amount: "{{amount}}"
   >   on_error:
   >     - id: fallback_payment
   >       action: http_request
   >       config:
   >         url: "https://backup-payment-gateway.com/charge"
   >         method: POST
   >         body:
   >           amount: "{{amount}}"
   >     - id: notify_admin
   >       action: email
   >  ...

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_002
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: Why doesn't HS256 work for JWT token validation in CloudFlow?

**Expected Answer**: CloudFlow uses RS256 (asymmetric) not HS256 (symmetric). RS256 requires private key for signing, public key for validation. HS256 would fail with algorithm mismatch error.

**Retrieved Chunks**:
1. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_1, score: 1.000]
   > ## Overview
   > 
   > This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analysis, resolution steps, and preventive measures.
   > 
   > ### Quick Diagnostic Checklist
   > 
   > Before diving into specific issues, perform these initial checks:
   > 
   > - Verify service health: `cloudflow status --all`
   > - Check API connectivity: `curl -I https://api.cloudflow.io/health`
   > - Review recent deployments: `kube...

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_3, score: 0.500]
   > ### JWT Tokens
   > 
   > For advanced use cases, CloudFlow supports JSON Web Tokens (JWT) with RS256 signing algorithm. JWTs must include the following claims:
   > 
   > - `iss` (issuer): Your application identifier
   > - `sub` (subject): User or service account ID
   > - `aud` (audience): `https://api.cloudflow.io`
   > - `exp` (expiration): Unix timestamp (max 3600 seconds from `iat`)
   > - `iat` (issued at): Unix timestamp
   > - `scope`: Space-separated list of requested scopes
   > 
   > Example JWT header:
   > 
   > ```python
   > import jwt
   > import time...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_15, score: 0.333]
   > ### Authentication & Authorization
   > 
   > **JWT Token Validation**:
   > - Algorithm: RS256 (asymmetric signing)
   > - Key rotation: Every 30 days with 7-day overlap period
   > - Public key distribution: JWKS endpoint cached in Redis
   > - Validation: Signature, expiry, issuer, audience claims
   > - Token revocation: Blacklist in Redis for compromised tokens
   > 
   > **Permission Model**:
   > ```
   > User → Roles → Permissions
   >      ↘       ↗
   >       Tenants (Multi-tenancy isolation)
   > ```
   > 
   > Example permissions:
   > - `workflow:read` - View workfl...

4. [doc_id: user_guide, chunk_id: user_guide_mdsem_18, score: 0.250]
   > ### Pattern 5: Error Aggregation and Alerting
   > 
   > Aggregate errors and send smart alerts:
   > 
   > ```yaml
   > name: "Application Error Monitor"
   > schedule:
   >   cron: "*/5 * * * *"
   >   timezone: "UTC"
   > 
   > steps:
   >   - id: fetch_recent_errors
   >     action: database_query
   >     config:
   >       query: |
   >         SELECT error_type, COUNT(*) as count, MAX(created_at) as last_seen
   >         FROM error_logs
   >         WHERE created_at > NOW() - INTERVAL '5 minutes'
   >         AND alerted = false
   >         GROUP BY error_type
   >         HAVING COUN...

5. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.200]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_003
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: Why can't I schedule workflows more frequently than every minute?

**Expected Answer**: Minimum scheduling interval is 1 minute. Expressions evaluating to more frequent executions will be rejected. For near real-time, use webhook or event-based triggers instead.

**Retrieved Chunks**:
1. [doc_id: user_guide, chunk_id: user_guide_mdsem_9, score: 1.000]
   > ## Scheduling
   > 
   > CloudFlow supports powerful scheduling options for recurring workflows.
   > 
   > ### Cron Syntax
   > 
   > Use standard cron expressions to define schedules:
   > 
   > ```
   > *    *    *    *    *
   > ┬    ┬    ┬    ┬    ┬
   > │    │    │    │    │
   > │    │    │    │    └─── Day of Week (0-6, Sunday=0)
   > │    │    │    └──────── Month (1-12)
   > │    │    └───────────── Day of Month (1-31)
   > │    └────────────────── Hour (0-23)
   > └─────────────────────── Minute (0-59)
   > ```
   > 
   > **Common Cron Patterns:**
   > 
   > | Pattern | Description |
   > |--...

2. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_6, score: 0.500]
   > ### Scheduler Service
   > 
   > **Purpose**: Time-based workflow triggering system supporting cron-like schedules and one-time delayed executions.
   > 
   > **Technology**: Go with distributed locking via Redis  
   > **Replicas**: 4 pods (production), active-passive with leader election  
   > **Resource Allocation**: 2 vCPU, 4GB RAM per pod
   > 
   > **Key Responsibilities**:
   > - Parse and validate cron expressions (extended format supporting seconds)
   > - Maintain schedule registry in PostgreSQL
   > - Distributed scheduling with leader e...

3. [doc_id: user_guide, chunk_id: user_guide_mdsem_11, score: 0.333]
   > ### Schedule Management
   > 
   > **Creating a Schedule:**
   > 1. Open your workflow in the editor
   > 2. Click the **"Trigger"** section
   > 3. Select **"Schedule"** as the trigger type
   > 4. Enter your cron expression or use the visual schedule builder
   > 5. Select your timezone
   > 6. Save and activate
   > 
   > **Testing Schedules:**
   > Use the built-in schedule calculator to preview upcoming executions:
   > ```
   > Next 5 executions:
   > 1. 2026-01-24 14:00:00 UTC
   > 2. 2026-01-25 14:00:00 UTC
   > 3. 2026-01-26 14:00:00 UTC
   > 4. 2026-01-27 14:00:00 UTC
   > ...

4. [doc_id: user_guide, chunk_id: user_guide_mdsem_14, score: 0.250]
   > ### Steps Per Workflow
   > 
   > - **Maximum**: 50 steps per workflow
   > - **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via webhooks.
   > 
   > ### Execution Timeout
   > 
   > - **Default**: 3600 seconds (60 minutes)
   > - **Behavior**: Workflows exceeding this timeout are automatically terminated
   > - **Custom Timeouts**: Enterprise plans can request custom timeout limits
   > 
   > **Setting Step-Level Timeouts:**
   > ```yaml
   > - id: long_running_task
   >   actio...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_10, score: 0.200]
   > ### Timezone Handling
   > 
   > All scheduled workflows run in **UTC by default**. To account for your local timezone:
   > 
   > **Option 1: Convert to UTC**
   > If you want a workflow to run at 9:00 AM EST (UTC-5), schedule it for 14:00 UTC:
   > ```
   > 0 14 * * *  # 9:00 AM EST = 14:00 UTC
   > ```
   > 
   > **Option 2: Use Timezone Configuration**
   > Specify a timezone in your workflow configuration:
   > ```yaml
   > schedule:
   >   cron: "0 9 * * *"
   >   timezone: "America/New_York"  # IANA timezone identifier
   > ```
   > 
   > **Supported Timezones:**
   > CloudFlow sup...

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_004
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: What happens if I don't implement token refresh logic?

**Expected Answer**: Tokens expire after 3600 seconds (1 hour). Without refresh logic, authentication will fail after expiry. Need to implement refresh using refresh token (valid 7-30 days).

**Retrieved Chunks**:
1. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_1, score: 1.000]
   > ## Overview
   > 
   > This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analysis, resolution steps, and preventive measures.
   > 
   > ### Quick Diagnostic Checklist
   > 
   > Before diving into specific issues, perform these initial checks:
   > 
   > - Verify service health: `cloudflow status --all`
   > - Check API connectivity: `curl -I https://api.cloudflow.io/health`
   > - Review recent deployments: `kube...

2. [doc_id: api_reference, chunk_id: api_reference_mdsem_3, score: 0.500]
   > ### JWT Tokens
   > 
   > For advanced use cases, CloudFlow supports JSON Web Tokens (JWT) with RS256 signing algorithm. JWTs must include the following claims:
   > 
   > - `iss` (issuer): Your application identifier
   > - `sub` (subject): User or service account ID
   > - `aud` (audience): `https://api.cloudflow.io`
   > - `exp` (expiration): Unix timestamp (max 3600 seconds from `iat`)
   > - `iat` (issued at): Unix timestamp
   > - `scope`: Space-separated list of requested scopes
   > 
   > Example JWT header:
   > 
   > ```python
   > import jwt
   > import time...

3. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.333]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

4. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_12, score: 0.250]
   > ### Getting Help
   > 
   > If this troubleshooting guide doesn't resolve your issue:
   > 
   > 1. Search the knowledge base: `cloudflow kb search "your issue"`
   > 2. Check community forum for similar issues
   > 3. Contact support with detailed logs and reproduction steps
   > 4. For urgent issues, use emergency escalation procedures
   > 
   > **Remember:** Always capture logs, metrics, and reproduction steps before escalating!
   > 
   > ---
   > 
   > *Last updated: January 24, 2026*  
   > *Document version: 3.2.1*  
   > *Feedback: docs-feedback@cloudflow.io*

5. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_15, score: 0.200]
   > ### Authentication & Authorization
   > 
   > **JWT Token Validation**:
   > - Algorithm: RS256 (asymmetric signing)
   > - Key rotation: Every 30 days with 7-day overlap period
   > - Public key distribution: JWKS endpoint cached in Redis
   > - Validation: Signature, expiry, issuer, audience claims
   > - Token revocation: Blacklist in Redis for compromised tokens
   > 
   > **Permission Model**:
   > ```
   > User → Roles → Permissions
   >      ↘       ↗
   >       Tenants (Multi-tenancy isolation)
   > ```
   > 
   > Example permissions:
   > - `workflow:read` - View workfl...

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: neg_005
**Type**: negation
**Root Causes**: NEGATION_BLIND

**Query**: Why shouldn't I hardcode API keys in workflow definitions?

**Expected Answer**: Security risk - keys could be exposed. Use secrets instead: {{secrets.API_TOKEN}}. Secrets are encrypted at rest. Store in Settings > Secrets.

**Retrieved Chunks**:
1. [doc_id: api_reference, chunk_id: api_reference_mdsem_1, score: 1.000]
   > ## Authentication
   > 
   > CloudFlow supports three authentication methods to suit different use cases and security requirements.
   > 
   > ### API Keys
   > 
   > API keys provide simple authentication for server-to-server communication. Include your API key in the request header:
   > 
   > ```bash
   > curl -H "X-API-Key: cf_live_a1b2c3d4e5f6g7h8i9j0" \
   >   https://api.cloudflow.io/v2/workflows
   > ```
   > 
   > **Security Notes:**
   > - Never expose API keys in client-side code
   > - Rotate keys every 90 days
   > - Use separate keys for development and produc...

2. [doc_id: user_guide, chunk_id: user_guide_mdsem_16, score: 0.500]
   > ### Data Limits
   > 
   > - **Maximum request/response size**: 10MB per action
   > - **Maximum execution payload**: 50MB total
   > - **Variable value size**: 1MB per variable
   > 
   > ### Enterprise Plan Limits
   > 
   > Enterprise customers can request increased limits:
   > - Up to 100 steps per workflow
   > - Up to 10,000 executions per day
   > - Up to 7200 second timeout (2 hours)
   > - Priority execution queue
   > - Dedicated capacity allocation
   > 
   > Contact sales@cloudflow.io for Enterprise pricing and custom limits.
   > 
   > ## Best Practices
   > 
   > Follow the...

3. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_15, score: 0.333]
   > ### Authentication & Authorization
   > 
   > **JWT Token Validation**:
   > - Algorithm: RS256 (asymmetric signing)
   > - Key rotation: Every 30 days with 7-day overlap period
   > - Public key distribution: JWKS endpoint cached in Redis
   > - Validation: Signature, expiry, issuer, audience claims
   > - Token revocation: Blacklist in Redis for compromised tokens
   > 
   > **Permission Model**:
   > ```
   > User → Roles → Permissions
   >      ↘       ↗
   >       Tenants (Multi-tenancy isolation)
   > ```
   > 
   > Example permissions:
   > - `workflow:read` - View workfl...

4. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.250]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

5. [doc_id: api_reference, chunk_id: api_reference_mdsem_0, score: 0.200]
   > # CloudFlow API Reference
   > 
   > Version 2.1.0 | Last Updated: January 2026
   > 
   > ## Overview
   > 
   > The CloudFlow API is a RESTful service that enables developers to programmatically manage cloud workflows, data pipelines, and automation tasks. This documentation provides comprehensive details on authentication, endpoints, request/response formats, error handling, and best practices.
   > 
   > **Base URL:** `https://api.cloudflow.io/v2`
   > 
   > **API Status:** https://status.cloudflow.io

**Baseline Score**: 5/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: imp_001
**Type**: implicit
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND

**Query**: Best practice for handling long-running data processing that might exceed time limits

**Expected Answer**: Workflow timeout is 3600s. Solutions: split into smaller workflows, enable checkpointing (every 300s), use parallel workers, request custom timeout (up to 7200s on Enterprise).

**Retrieved Chunks**:
1. [doc_id: user_guide, chunk_id: user_guide_mdsem_16, score: 1.000]
   > ### Data Limits
   > 
   > - **Maximum request/response size**: 10MB per action
   > - **Maximum execution payload**: 50MB total
   > - **Variable value size**: 1MB per variable
   > 
   > ### Enterprise Plan Limits
   > 
   > Enterprise customers can request increased limits:
   > - Up to 100 steps per workflow
   > - Up to 10,000 executions per day
   > - Up to 7200 second timeout (2 hours)
   > - Priority execution queue
   > - Dedicated capacity allocation
   > 
   > Contact sales@cloudflow.io for Enterprise pricing and custom limits.
   > 
   > ## Best Practices
   > 
   > Follow the...

2. [doc_id: user_guide, chunk_id: user_guide_mdsem_12, score: 0.500]
   > ## Error Handling
   > 
   > Robust error handling ensures your workflows are resilient and reliable.
   > 
   > ### Retry Policies
   > 
   > Configure automatic retries for failed actions:
   > 
   > ```yaml
   > - id: api_call
   >   action: http_request
   >   config:
   >     url: "https://api.example.com/data"
   >   retry:
   >     max_attempts: 3
   >     backoff_type: "exponential"  # or "fixed", "linear"
   >     initial_interval: 1000       # milliseconds
   >     max_interval: 30000
   >     multiplier: 2.0
   >     retry_on:
   >       - timeout
   >       - network_error
   >       - statu...

3. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 0.333]
   > ## Rate Limiting
   > 
   > To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.
   > 
   > **Default Limits:**
   > - 100 requests per minute per authenticated user
   > - 20 requests per minute for unauthenticated requests
   > - Burst allowance: 150 requests in a 10-second window
   > 
   > ### Rate Limit Headers
   > 
   > Every API response includes rate limit information:
   > 
   > ```
   > X-RateLimit-Limit: 100
   > X-RateLimit-Remaining: 87
   > X-RateLimit-Reset: 1640995200
   > ```
   > 
   > When you exceed the rate limit, you'll rec...

4. [doc_id: user_guide, chunk_id: user_guide_mdsem_14, score: 0.250]
   > ### Steps Per Workflow
   > 
   > - **Maximum**: 50 steps per workflow
   > - **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via webhooks.
   > 
   > ### Execution Timeout
   > 
   > - **Default**: 3600 seconds (60 minutes)
   > - **Behavior**: Workflows exceeding this timeout are automatically terminated
   > - **Custom Timeouts**: Enterprise plans can request custom timeout limits
   > 
   > **Setting Step-Level Timeouts:**
   > ```yaml
   > - id: long_running_task
   >   actio...

5. [doc_id: architecture_overview, chunk_id: architecture_overview_mdsem_23, score: 0.200]
   > ### Disaster Recovery Procedures
   > 
   > **Scenario 1: Single AZ Failure**
   > - Detection: Health checks fail for entire AZ (< 30 seconds)
   > - Action: Traffic automatically routed to healthy AZs by ALB
   > - Recovery time: < 5 minutes (no manual intervention)
   > - Data loss: None (multi-AZ replication)
   > 
   > **Scenario 2: Database Primary Failure**
   > - Detection: Health check fails for primary database (< 30 seconds)
   > - Action: Automatic promotion of read replica to primary
   > - Recovery time: 30-60 seconds
   > - Data loss: Mini...

**Baseline Score**: 6/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---

## Query: imp_003
**Type**: implicit
**Root Causes**: VOCABULARY_MISMATCH, EMBEDDING_BLIND

**Query**: How to debug why my API calls are slow

**Expected Answer**: Check latency breakdown: Auth (18%), DB Query (64%), Business Logic (13%), Serialization (5%). Use cloudflow metrics latency-report. Check slow query log. Review connection pool status.

**Retrieved Chunks**:
1. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_7, score: 1.000]
   > #### Handling Rate Limits in Code
   > 
   > **Python example with retry logic:**
   > ```python
   > import time
   > import requests
   > 
   > def cloudflow_api_call_with_retry(url, headers, max_retries=3):
   >     for attempt in range(max_retries):
   >         response = requests.get(url, headers=headers)
   >         
   >         if response.status_code == 429:
   >             retry_after = int(response.headers.get('Retry-After', 60))
   >             print(f"Rate limited. Waiting {retry_after} seconds...")
   >             time.sleep(retry_after)
   >        ...

2. [doc_id: troubleshooting_guide, chunk_id: troubleshooting_guide_mdsem_3, score: 0.500]
   > # Check database slow query log
   > 
   > kubectl logs -n cloudflow deploy/cloudflow-db-primary | \
   >   grep "slow query" | \
   >   tail -n 50
   > 
   > # Analyze query patterns
   > 
   > cloudflow db analyze-queries --min-duration 5000 --limit 20
   > ```
   > 
   > **2. Review Query Execution Plans**
   > 
   > ```sql
   > -- Connect to CloudFlow database
   > cloudflow db connect --readonly
   > 
   > -- Explain slow query
   > EXPLAIN ANALYZE
   > SELECT w.*, e.status, e.error_message
   > FROM workflows w
   > LEFT JOIN executions e ON w.id = e.workflow_id
   > WHERE w.workspace_id = 'ws_abc...

3. [doc_id: api_reference, chunk_id: api_reference_mdsem_4, score: 0.333]
   > ## Rate Limiting
   > 
   > To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints.
   > 
   > **Default Limits:**
   > - 100 requests per minute per authenticated user
   > - 20 requests per minute for unauthenticated requests
   > - Burst allowance: 150 requests in a 10-second window
   > 
   > ### Rate Limit Headers
   > 
   > Every API response includes rate limit information:
   > 
   > ```
   > X-RateLimit-Limit: 100
   > X-RateLimit-Remaining: 87
   > X-RateLimit-Reset: 1640995200
   > ```
   > 
   > When you exceed the rate limit, you'll rec...

4. [doc_id: api_reference, chunk_id: api_reference_mdsem_8, score: 0.250]
   > ### Error Codes
   > 
   > CloudFlow returns specific error codes to help you identify and resolve issues:
   > 
   > - `invalid_parameter`: One or more request parameters are invalid
   > - `missing_required_field`: Required field is missing from request body
   > - `authentication_failed`: Invalid API key or token
   > - `insufficient_permissions`: User lacks required scope or permission
   > - `resource_not_found`: Requested resource does not exist
   > - `rate_limit_exceeded`: Too many requests, see rate limiting section
   > - `workflow_ex...

5. [doc_id: user_guide, chunk_id: user_guide_mdsem_16, score: 0.200]
   > ### Data Limits
   > 
   > - **Maximum request/response size**: 10MB per action
   > - **Maximum execution payload**: 50MB total
   > - **Variable value size**: 1MB per variable
   > 
   > ### Enterprise Plan Limits
   > 
   > Enterprise customers can request increased limits:
   > - Up to 100 steps per workflow
   > - Up to 10,000 executions per day
   > - Up to 7200 second timeout (2 hours)
   > - Priority execution queue
   > - Dedicated capacity allocation
   > 
   > Contact sales@cloudflow.io for Enterprise pricing and custom limits.
   > 
   > ## Best Practices
   > 
   > Follow the...

**Baseline Score**: 7/10
**New Score**: ___/10 (FILL IN)
**Notes**: _______________

---
