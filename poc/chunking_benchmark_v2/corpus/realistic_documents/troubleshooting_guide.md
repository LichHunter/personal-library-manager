# CloudFlow Platform Troubleshooting Guide

**Version:** 3.2.1  
**Last Updated:** January 2026  
**Audience:** Platform Engineers, DevOps, Support Teams

## Table of Contents

1. [Overview](#overview)
2. [Authentication & Authorization Issues](#authentication--authorization-issues)
3. [Performance Problems](#performance-problems)
4. [Database Connection Issues](#database-connection-issues)
5. [Workflow Execution Failures](#workflow-execution-failures)
6. [Rate Limiting & Throttling](#rate-limiting--throttling)
7. [Log Analysis & Debugging](#log-analysis--debugging)
8. [Escalation Procedures](#escalation-procedures)

---

## Overview

This guide provides comprehensive troubleshooting steps for common CloudFlow platform issues encountered in production environments. Each section includes error symptoms, root cause analysis, resolution steps, and preventive measures.

### Quick Diagnostic Checklist

Before diving into specific issues, perform these initial checks:

- Verify service health: `cloudflow status --all`
- Check API connectivity: `curl -I https://api.cloudflow.io/health`
- Review recent deployments: `kubectl get deployments -n cloudflow --sort-by=.metadata.creationTimestamp`
- Inspect platform metrics: `cloudflow metrics --last 1h`

---

## Authentication & Authorization Issues

### 401 Unauthorized Errors

#### Symptoms
- API requests return `401 Unauthorized`
- Error message: `Authentication credentials were not provided or are invalid`
- Frontend displays "Session expired" message

#### Common Causes

**1. Token Expiration**

CloudFlow access tokens expire after 3600 seconds (1 hour) by default. Refresh tokens are valid for 30 days.

**Verification:**
```bash
# Decode JWT to check expiration
echo $CF_ACCESS_TOKEN | cut -d'.' -f2 | base64 -d | jq '.exp'

# Compare with current time
date +%s
```

**Resolution:**
```bash
# Refresh the access token
curl -X POST https://api.cloudflow.io/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "'$CF_REFRESH_TOKEN'"}'

# Update environment variable
export CF_ACCESS_TOKEN="<new_token>"
```

**2. Invalid JWT Signature**

Error: `JWT signature verification failed`

**Causes:**
- Token was modified or corrupted
- Using wrong signing key
- Token generated with different secret

**Resolution:**
```bash
# Validate token structure
cloudflow auth validate --token $CF_ACCESS_TOKEN

# Generate new token
cloudflow auth login --username <user> --password <pass>

# For service accounts
cloudflow auth service-account --create --name <sa-name> --scopes "workflow:read,workflow:write"
```

**3. Clock Skew Issues**

JWT validation fails when system clocks are out of sync (tolerance: ±300 seconds).

**Diagnosis:**
```bash
# Check system time
timedatectl status

# Compare with NTP server
ntpdate -q pool.ntp.org

# Check JWT issued time vs current time
jwt_iat=$(echo $CF_ACCESS_TOKEN | cut -d'.' -f2 | base64 -d | jq '.iat')
current_time=$(date +%s)
skew=$((current_time - jwt_iat))
echo "Clock skew: $skew seconds"
```

**Resolution:**
```bash
# Sync with NTP server
sudo ntpdate -s pool.ntp.org

# Enable automatic time sync
sudo timedatectl set-ntp true

# Restart CloudFlow client
cloudflow restart
```

**4. Insufficient Permissions**

Error: `User does not have required permissions for this operation`

**Check user roles:**
```bash
cloudflow auth whoami --verbose

# Expected output:
# User: john.doe@company.com
# Roles: developer, workflow-admin
# Scopes: workflow:*, data:read, metrics:read
```

**Request permission elevation:**
```bash
# Submit access request
cloudflow auth request-access \
  --resource "workflow:production:deploy" \
  --justification "Deploy critical hotfix for TICKET-1234"

# Check pending approvals
cloudflow auth list-requests --status pending
```

### 403 Forbidden Errors

#### RBAC Policy Violations

CloudFlow uses role-based access control (RBAC) with the following hierarchy:
- `viewer` - Read-only access
- `developer` - Create and modify workflows (non-production)
- `operator` - Execute workflows, view logs
- `admin` - Full access to workspace
- `platform-admin` - Cross-workspace administration

**Verify resource permissions:**
```bash
# Check effective permissions
cloudflow rbac check \
  --user john.doe@company.com \
  --resource workflow:prod-pipeline \
  --action execute

# List all policies affecting user
cloudflow rbac policies --user john.doe@company.com --verbose
```

---

## Performance Problems

### Slow Query Performance

#### Symptoms
- API response times > 5000ms
- Database query latency warnings in logs
- CloudFlow UI becomes unresponsive
- Timeout errors: `Request exceeded maximum duration of 30000ms`

#### Diagnosis Steps

**1. Identify Slow Queries**

```bash
# Query CloudFlow metrics
cloudflow metrics query --metric api_request_duration_ms \
  --filter "p95 > 5000" \
  --last 1h

# Check database slow query log
kubectl logs -n cloudflow deploy/cloudflow-db-primary | \
  grep "slow query" | \
  tail -n 50

# Analyze query patterns
cloudflow db analyze-queries --min-duration 5000 --limit 20
```

**2. Review Query Execution Plans**

```sql
-- Connect to CloudFlow database
cloudflow db connect --readonly

-- Explain slow query
EXPLAIN ANALYZE
SELECT w.*, e.status, e.error_message
FROM workflows w
LEFT JOIN executions e ON w.id = e.workflow_id
WHERE w.workspace_id = 'ws_abc123'
  AND e.created_at > NOW() - INTERVAL '7 days'
ORDER BY e.created_at DESC;

-- Check for missing indexes
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('workflows', 'executions', 'workflow_steps')
ORDER BY tablename, indexname;
```

**3. Optimize Queries**

Common optimization techniques:

**Add missing indexes:**
```sql
-- Index for workflow lookup by workspace
CREATE INDEX CONCURRENTLY idx_workflows_workspace_created 
ON workflows(workspace_id, created_at DESC);

-- Index for execution status queries
CREATE INDEX CONCURRENTLY idx_executions_status_created
ON executions(workflow_id, status, created_at DESC)
WHERE status IN ('running', 'pending');

-- Composite index for common filter combinations
CREATE INDEX CONCURRENTLY idx_executions_workspace_status
ON executions(workspace_id, status, created_at)
INCLUDE (error_message, retry_count);
```

**Use query result caching:**
```bash
# Enable query cache for workspace metadata
cloudflow config set cache.workspace.ttl 3600

# Configure Redis cache backend
cloudflow config set cache.backend redis
cloudflow config set cache.redis.host redis.cloudflow.svc.cluster.local
cloudflow config set cache.redis.port 6379
```

### High API Latency

#### Latency Breakdown Analysis

```bash
# Generate latency report
cloudflow metrics latency-report --endpoint "/api/v1/workflows" --last 24h

# Sample output:
# Endpoint: POST /api/v1/workflows/execute
# P50: 245ms | P95: 1823ms | P99: 4521ms
# Breakdown:
#   - Auth: 45ms (18%)
#   - DB Query: 156ms (64%)
#   - Business Logic: 32ms (13%)
#   - Response Serialization: 12ms (5%)
```

**Network latency issues:**
```bash
# Test connectivity to CloudFlow API
time curl -w "@curl-format.txt" -o /dev/null -s https://api.cloudflow.io/health

# Create curl-format.txt:
cat > curl-format.txt << EOF
    time_namelookup:  %{time_namelookup}s\n
       time_connect:  %{time_connect}s\n
    time_appconnect:  %{time_appconnect}s\n
   time_pretransfer:  %{time_pretransfer}s\n
      time_redirect:  %{time_redirect}s\n
 time_starttransfer:  %{time_starttransfer}s\n
                    ----------\n
         time_total:  %{time_total}s\n
EOF

# Trace route to API endpoint
traceroute api.cloudflow.io

# Check DNS resolution time
dig api.cloudflow.io | grep "Query time"
```

### Memory Leaks

#### Detection

```bash
# Monitor CloudFlow service memory usage
kubectl top pods -n cloudflow --sort-by=memory

# Get detailed memory metrics for specific pod
kubectl exec -n cloudflow deploy/cloudflow-api -- \
  curl localhost:9090/metrics | grep memory

# Check for OOMKilled pods
kubectl get pods -n cloudflow --field-selector=status.phase=Failed | \
  grep OOMKilled

# Review memory limits and requests
kubectl describe deployment cloudflow-api -n cloudflow | \
  grep -A 5 "Limits\|Requests"
```

#### Common Causes

**1. Workflow Context Accumulation**

Large workflow executions may accumulate state in memory.

**Solution:**
```bash
# Configure context cleanup
cloudflow config set workflow.context.max_size_mb 100
cloudflow config set workflow.context.cleanup_threshold 0.8

# Enable context persistence to disk
cloudflow config set workflow.context.persistence.enabled true
cloudflow config set workflow.context.persistence.backend s3
```

**2. Connection Pool Leaks**

**Diagnosis:**
```bash
# Check active connections
cloudflow db connections --verbose

# Expected output:
# Active: 45/100
# Idle: 23
# Waiting: 2
# Average age: 245s
```

**Resolution:**
```bash
# Adjust connection pool settings
cloudflow config set db.pool.max_connections 100
cloudflow config set db.pool.min_connections 10
cloudflow config set db.pool.idle_timeout 300
cloudflow config set db.pool.max_lifetime 1800

# Force connection pool reset
cloudflow db pool reset --confirm
```

**3. Event Stream Buffers**

Unbounded event buffers can cause memory exhaustion.

```bash
# Configure event buffer limits
cloudflow config set events.buffer.max_size 10000
cloudflow config set events.buffer.overflow_strategy drop_oldest

# Enable event streaming to external sink
cloudflow config set events.sink.type kafka
cloudflow config set events.sink.kafka.brokers "kafka-1:9092,kafka-2:9092"
cloudflow config set events.sink.kafka.topic cloudflow-events
```

---

## Database Connection Issues

### Connection Pool Exhaustion

#### Symptoms
- Error: `could not obtain connection from pool within 5000ms`
- Error: `connection pool exhausted (100/100 connections in use)`
- API requests fail with `503 Service Unavailable`
- Database CPU usage normal, but connection count at maximum

#### Investigation

```bash
# Check current connection pool status
cloudflow db pool status --detailed

# Output example:
# Pool Statistics:
#   Total Connections: 100/100 (100%)
#   Active: 87
#   Idle: 13
#   Waiting Requests: 45
#   Average Wait Time: 3420ms
#   Max Wait Time: 8234ms

# Identify long-running queries
cloudflow db queries --status running --min-duration 30000

# Check connection distribution by client
SELECT application_name, state, COUNT(*) as conn_count,
       AVG(EXTRACT(EPOCH FROM (NOW() - state_change))) as avg_duration_sec
FROM pg_stat_activity
WHERE datname = 'cloudflow_production'
GROUP BY application_name, state
ORDER BY conn_count DESC;
```

#### Resolution

**Immediate mitigation:**
```bash
# Temporarily increase connection limit (requires database restart)
cloudflow db config set max_connections 150

# Kill idle connections older than 5 minutes
cloudflow db connections kill --idle-timeout 300

# Restart connection pool without downtime
kubectl rollout restart deployment/cloudflow-api -n cloudflow
kubectl rollout status deployment/cloudflow-api -n cloudflow
```

**Long-term solutions:**

1. **Implement connection pooling optimization:**
```bash
# Use PgBouncer for connection pooling
kubectl apply -f cloudflow-pgbouncer.yaml

# Configure CloudFlow to use PgBouncer
cloudflow config set db.host pgbouncer.cloudflow.svc.cluster.local
cloudflow config set db.port 6432
cloudflow config set db.pool.mode transaction
```

2. **Add read replicas:**
```bash
# Route read-only queries to replicas
cloudflow db replicas add --count 2
cloudflow config set db.read_replicas "replica-1:5432,replica-2:5432"
cloudflow config set db.read_write_split true
```

### Connection Timeout Errors

#### Error Messages
- `connection timeout after 30000ms`
- `could not connect to database server at 10.0.2.45:5432`
- `database server unreachable`

#### Troubleshooting Steps

```bash
# Test network connectivity
telnet cloudflow-db.internal.company.com 5432

# Check DNS resolution
nslookup cloudflow-db.internal.company.com

# Verify database is accepting connections
pg_isready -h cloudflow-db.internal.company.com -p 5432 -U cloudflow

# Check firewall rules
sudo iptables -L -n | grep 5432

# Test from CloudFlow pod network
kubectl run -n cloudflow debug-pod --rm -i --tty \
  --image=postgres:14 -- \
  psql -h cloudflow-db.internal.company.com -U cloudflow -d cloudflow_production

# Review database logs for connection rejections
kubectl logs -n cloudflow statefulset/cloudflow-db --tail=100 | \
  grep -i "connection\|reject\|authentication"
```

### Maximum Connection Limit (100) Reached

This is a hard limit in CloudFlow's database tier.

#### Permanent Solutions

**Option 1: Upgrade database tier**
```bash
# Check available tiers
cloudflow db tiers list

# Upgrade to higher tier (supports 200 connections)
cloudflow db upgrade --tier standard-plus --confirm

# Monitor migration progress
cloudflow db migration status
```

**Option 2: Implement aggressive connection reuse**
```bash
# Reduce connection lifetime
cloudflow config set db.pool.max_lifetime 600  # 10 minutes

# Enable prepared statement caching
cloudflow config set db.prepared_statements.cache true
cloudflow config set db.prepared_statements.max_size 250

# Reduce idle connection timeout
cloudflow config set db.pool.idle_timeout 120  # 2 minutes
```

---

## Workflow Execution Failures

### Timeout Errors (3600 second limit)

#### Error Message
```
WorkflowExecutionError: Workflow exceeded maximum execution time of 3600 seconds
Status: TIMEOUT
Workflow ID: wf_9k2n4m8p1q
Execution ID: exec_7h3j6k9m2n
```

#### Analysis

```bash
# Get workflow execution details
cloudflow workflows executions get exec_7h3j6k9m2n --verbose

# Check step-by-step breakdown
cloudflow workflows executions steps exec_7h3j6k9m2n

# Sample output:
# Step 1: data_ingestion     - Duration: 245s    - Status: SUCCESS
# Step 2: data_validation    - Duration: 123s    - Status: SUCCESS
# Step 3: data_transformation - Duration: 3189s   - Status: TIMEOUT
# Step 4: data_export        - Duration: 0s      - Status: SKIPPED

# Identify bottleneck step
cloudflow workflows analyze exec_7h3j6k9m2n --show-bottlenecks
```

#### Solutions

**1. Increase workflow timeout (if justified):**
```bash
# Update workflow configuration
cloudflow workflows update wf_9k2n4m8p1q \
  --timeout 7200 \
  --reason "Large dataset processing requires extended time"

# Verify update
cloudflow workflows get wf_9k2n4m8p1q | grep timeout
```

**2. Optimize slow steps:**
```bash
# Enable parallel processing
cloudflow workflows update wf_9k2n4m8p1q \
  --step data_transformation \
  --parallel-workers 8 \
  --batch-size 1000

# Add checkpointing for long operations
cloudflow workflows update wf_9k2n4m8p1q \
  --step data_transformation \
  --enable-checkpointing \
  --checkpoint-interval 300
```

**3. Split workflow into smaller workflows:**
```bash
# Create sub-workflows
cloudflow workflows create data-pipeline-part1 \
  --steps "data_ingestion,data_validation" \
  --timeout 1800

cloudflow workflows create data-pipeline-part2 \
  --steps "data_transformation,data_export" \
  --timeout 3600 \
  --trigger workflow_completed \
  --trigger-workflow data-pipeline-part1
```

### Retry Logic and Exponential Backoff

CloudFlow implements automatic retry with exponential backoff for transient failures:
- Max retries: 3
- Initial delay: 1 second
- Backoff multiplier: 2
- Max delay: 60 seconds

#### Retry Sequence
```
Attempt 1: Immediate
Attempt 2: Wait 1s  (2^0 * 1s)
Attempt 3: Wait 2s  (2^1 * 1s)
Attempt 4: Wait 4s  (2^2 * 1s)
```

#### Configuration

```bash
# View current retry settings
cloudflow workflows get wf_9k2n4m8p1q --format json | jq '.retry_policy'

# Customize retry behavior
cloudflow workflows update wf_9k2n4m8p1q \
  --retry-max-attempts 5 \
  --retry-initial-delay 2000 \
  --retry-backoff-multiplier 2 \
  --retry-max-delay 120000 \
  --retry-on-errors "NETWORK_ERROR,TIMEOUT,RATE_LIMIT"

# Disable retry for specific step
cloudflow workflows update wf_9k2n4m8p1q \
  --step data_export \
  --retry-enabled false
```

#### Monitoring Retries

```bash
# List failed executions with retry information
cloudflow workflows executions list \
  --status FAILED \
  --show-retries \
  --last 7d

# Get retry history for specific execution
cloudflow workflows executions retries exec_7h3j6k9m2n

# Output:
# Execution: exec_7h3j6k9m2n
# Attempt 1: FAILED - NetworkError: Connection refused (delay: 0ms)
# Attempt 2: FAILED - NetworkError: Connection timeout (delay: 1000ms)
# Attempt 3: FAILED - NetworkError: Connection timeout (delay: 2000ms)
# Attempt 4: FAILED - NetworkError: Connection timeout (delay: 4000ms)
# Final Status: FAILED_AFTER_RETRIES
```

### Workflow Step Failures

#### Common Error Patterns

**1. Data Validation Errors**
```
ValidationError: Field 'customer_id' is required but missing in 234 records
```

**Resolution:**
```bash
# Add data quality checks
cloudflow workflows update wf_9k2n4m8p1q \
  --step data_ingestion \
  --add-validator required_fields \
  --validator-config '{"fields": ["customer_id", "timestamp", "amount"]}'

# Configure error handling
cloudflow workflows update wf_9k2n4m8p1q \
  --step data_validation \
  --on-error continue \
  --error-threshold 5%  # Fail if > 5% of records invalid
```

**2. External API Failures**
```
ExternalAPIError: API request to https://partner-api.example.com failed with status 502
```

**Resolution:**
```bash
# Add circuit breaker
cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --circuit-breaker-enabled true \
  --circuit-breaker-threshold 5 \
  --circuit-breaker-timeout 30000

# Configure fallback behavior
cloudflow workflows update wf_9k2n4m8p1q \
  --step external_api_call \
  --fallback-action use_cached_data \
  --cache-ttl 3600
```

---

## Rate Limiting & Throttling

### 429 Too Many Requests

#### Error Response
```json
{
  "error": "rate_limit_exceeded",
  "message": "API rate limit exceeded. Retry after 45 seconds.",
  "status": 429,
  "headers": {
    "X-RateLimit-Limit": "1000",
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1706112345",
    "Retry-After": "45"
  }
}
```

#### Rate Limit Tiers

CloudFlow enforces the following rate limits per workspace:

| Tier | Requests/Minute | Requests/Hour | Concurrent Workflows |
|------|-----------------|---------------|----------------------|
| Free | 60 | 1,000 | 5 |
| Standard | 1,000 | 50,000 | 50 |
| Premium | 5,000 | 250,000 | 200 |
| Enterprise | Custom | Custom | Unlimited |

#### Checking Rate Limit Status

```bash
# Check current rate limit status
curl -I https://api.cloudflow.io/api/v1/workflows \
  -H "Authorization: Bearer $CF_ACCESS_TOKEN"

# Extract rate limit headers
curl -s -I https://api.cloudflow.io/api/v1/workflows \
  -H "Authorization: Bearer $CF_ACCESS_TOKEN" | \
  grep -i "x-ratelimit"

# Output:
# X-RateLimit-Limit: 1000
# X-RateLimit-Remaining: 247
# X-RateLimit-Reset: 1706112400

# Monitor rate limit usage
cloudflow metrics query \
  --metric rate_limit_remaining \
  --workspace ws_abc123 \
  --last 1h \
  --interval 1m
```

#### Handling Rate Limits in Code

**Python example with retry logic:**
```python
import time
import requests

def cloudflow_api_call_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        if remaining < 10:
            print(f"Warning: Only {remaining} requests remaining")
            
        return response
    
    raise Exception("Max retries exceeded due to rate limiting")
```

**Bash script with rate limit checking:**
```bash
#!/bin/bash

check_rate_limit() {
    local remaining=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \
        -H "Authorization: Bearer $CF_ACCESS_TOKEN" | \
        grep -i "x-ratelimit-remaining" | \
        awk '{print $2}' | tr -d '\r')
    
    if [ "$remaining" -lt 10 ]; then
        echo "Warning: Only $remaining requests remaining"
        local reset=$(curl -s -I https://api.cloudflow.io/api/v1/workflows \
            -H "Authorization: Bearer $CF_ACCESS_TOKEN" | \
            grep -i "x-ratelimit-reset" | \
            awk '{print $2}' | tr -d '\r')
        local wait_time=$((reset - $(date +%s)))
        echo "Rate limit resets in $wait_time seconds"
        sleep $wait_time
    fi
}

# Use before API calls
check_rate_limit
cloudflow workflows execute wf_9k2n4m8p1q
```

#### Optimization Strategies

**1. Implement request batching:**
```bash
# Batch multiple workflow executions
cloudflow workflows execute-batch \
  --workflow-ids "wf_id1,wf_id2,wf_id3,wf_id4,wf_id5" \
  --batch-size 5

# This counts as 1 API request instead of 5
```

**2. Use webhooks instead of polling:**
```bash
# Configure webhook for workflow completion
cloudflow webhooks create \
  --event workflow.completed \
  --url https://your-service.com/webhooks/cloudflow \
  --secret $WEBHOOK_SECRET

# Verify webhook
cloudflow webhooks test webhook_abc123
```

**3. Leverage caching:**
```bash
# Enable client-side caching
export CLOUDFLOW_CACHE_ENABLED=true
export CLOUDFLOW_CACHE_TTL=300

# Cache workflow metadata
cloudflow workflows list --use-cache --cache-ttl 600
```

---

## Log Analysis & Debugging

### Accessing CloudFlow Logs

#### Kubernetes Deployments

```bash
# List all CloudFlow pods
kubectl get pods -n cloudflow

# Tail logs from API server
kubectl logs -f -n cloudflow deployment/cloudflow-api --tail=100

# Get logs from specific pod
kubectl logs -n cloudflow cloudflow-api-7d4f6b8c9d-x7k2m

# Get logs from previous crashed pod
kubectl logs -n cloudflow cloudflow-api-7d4f6b8c9d-x7k2m --previous

# Get logs from all pods in deployment
kubectl logs -n cloudflow deployment/cloudflow-api --all-containers=true

# Stream logs from multiple pods
kubectl logs -n cloudflow -l app=cloudflow-api -f --max-log-requests=10
```

#### Log Levels

CloudFlow supports the following log levels:
- `TRACE` - Very detailed debugging information
- `DEBUG` - Detailed debugging information
- `INFO` - Informational messages (default)
- `WARN` - Warning messages
- `ERROR` - Error messages
- `FATAL` - Fatal errors causing shutdown

**Changing log levels:**
```bash
# Set global log level
cloudflow config set logging.level DEBUG

# Set log level for specific component
cloudflow config set logging.components.database DEBUG
cloudflow config set logging.components.auth INFO
cloudflow config set logging.components.workflows TRACE

# Temporary log level increase (resets after 1 hour)
cloudflow debug set-log-level DEBUG --duration 3600

# View current log configuration
cloudflow config get logging --format json
```

### Grep Patterns for Common Issues

#### Authentication Failures
```bash
# Find all authentication errors
kubectl logs -n cloudflow deployment/cloudflow-api | \
  grep -i "authentication\|401\|unauthorized" | \
  tail -n 50

# Find JWT validation failures
kubectl logs -n cloudflow deployment/cloudflow-api | \
  grep -E "JWT|token.*invalid|signature.*failed"

# Find clock skew issues
kubectl logs -n cloudflow deployment/cloudflow-api | \
  grep -i "clock skew\|time.*sync\|nbf\|exp"
```

#### Database Errors
```bash
# Find database connection errors
kubectl logs -n cloudflow deployment/cloudflow-api | \
  grep -E "connection.*pool|could not connect|database.*timeout"

# Find slow queries
kubectl logs -n cloudflow deployment/cloudflow-api | \
  grep "slow query" | \
  awk '{print $NF}' | \
  sort -n | \
  tail -n 20

# Find deadlock errors
kubectl logs -n cloudflow deployment/cloudflow-api | \
  grep -i "deadlock detected"
```

#### Workflow Execution Errors
```bash
# Find workflow timeout errors
kubectl logs -n cloudflow deployment/cloudflow-workflow-engine | \
  grep -E "timeout|exceeded.*3600"

# Find workflow retry attempts
kubectl logs -n cloudflow deployment/cloudflow-workflow-engine | \
  grep -E "retry attempt [0-9]|retrying in"

# Find workflow failures by ID
kubectl logs -n cloudflow deployment/cloudflow-workflow-engine | \
  grep "exec_7h3j6k9m2n"
```

#### Rate Limiting
```bash
# Find rate limit events
kubectl logs -n cloudflow deployment/cloudflow-api | \
  grep -E "429|rate.*limit|throttle"

# Count rate limit errors by hour
kubectl logs -n cloudflow deployment/cloudflow-api --since=24h | \
  grep "rate_limit_exceeded" | \
  awk '{print $1}' | \
  cut -d'T' -f1-2 | \
  sort | uniq -c
```

### Advanced Log Analysis

#### Using jq for JSON Logs

```bash
# Parse JSON logs and filter by level
kubectl logs -n cloudflow deployment/cloudflow-api | \
  jq 'select(.level == "ERROR")'

# Extract specific fields
kubectl logs -n cloudflow deployment/cloudflow-api | \
  jq '{timestamp: .timestamp, level: .level, message: .message, execution_id: .context.execution_id}'

# Filter by workflow ID
kubectl logs -n cloudflow deployment/cloudflow-workflow-engine | \
  jq 'select(.workflow_id == "wf_9k2n4m8p1q")'

# Count errors by type
kubectl logs -n cloudflow deployment/cloudflow-api --since=1h | \
  jq -r 'select(.level == "ERROR") | .error_type' | \
  sort | uniq -c | sort -rn
```

#### Correlation IDs

CloudFlow uses correlation IDs to trace requests across services.

```bash
# Extract correlation ID from error
CORRELATION_ID="corr_8h4j9k2m5n"

# Trace request across all services
for pod in $(kubectl get pods -n cloudflow -l tier=backend -o name); do
  echo "=== $pod ==="
  kubectl logs -n cloudflow $pod | grep $CORRELATION_ID
done

# Export full trace to file
cloudflow debug trace $CORRELATION_ID --output trace-$CORRELATION_ID.json
```

### Debugging Commands

#### Enable Debug Mode for Workflow Execution

```bash
# Execute workflow with debug logging
cloudflow workflows execute wf_9k2n4m8p1q \
  --debug \
  --log-level TRACE \
  --output-logs /tmp/workflow-debug.log

# Enable step-by-step execution
cloudflow workflows execute wf_9k2n4m8p1q \
  --step-mode interactive \
  --breakpoint-on-error

# Capture full execution context
cloudflow workflows execute wf_9k2n4m8p1q \
  --capture-context \
  --context-output /tmp/execution-context.json
```

#### Database Query Debugging

```bash
# Enable query logging
cloudflow db config set log_statement all
cloudflow db config set log_duration on
cloudflow db config set log_min_duration_statement 1000  # Log queries > 1s

# Capture query plan for slow endpoint
cloudflow debug capture-queries \
  --endpoint "/api/v1/workflows/list" \
  --duration 60 \
  --output query-analysis.txt

# Analyze query performance
cloudflow db analyze-performance --last 1h
```

#### Network Debugging

```bash
# Test connectivity from CloudFlow pod
kubectl run -n cloudflow netdebug --rm -i --tty \
  --image=nicolaka/netshoot -- /bin/bash

# Inside pod:
# Check DNS resolution
nslookup api.cloudflow.io

# Check connectivity to database
nc -zv cloudflow-db.internal.company.com 5432

# Trace route
traceroute api.cloudflow.io

# Capture packets
tcpdump -i any -w /tmp/capture.pcap port 5432
```

---

## Escalation Procedures

### Severity Levels

CloudFlow incidents are classified into four severity levels:

#### SEV-1: Critical (P1)
- **Definition:** Complete service outage or severe degradation affecting all users
- **Examples:**
  - API returns 5xx errors for > 5 minutes
  - Database completely unavailable
  - Authentication system down
  - Data loss or corruption
- **Response Time:** Immediate (< 15 minutes)
- **Escalation:** Page on-call engineer immediately

#### SEV-2: High (P2)
- **Definition:** Major functionality impaired affecting multiple users
- **Examples:**
  - Workflow execution success rate < 90%
  - Significant performance degradation (p95 latency > 10s)
  - Rate limiting affecting large customer segment
- **Response Time:** < 1 hour
- **Escalation:** Create incident ticket and notify on-call

#### SEV-3: Medium (P3)
- **Definition:** Partial functionality degraded affecting some users
- **Examples:**
  - Intermittent failures for specific workflow types
  - Minor performance issues
  - Non-critical feature unavailable
- **Response Time:** < 4 hours
- **Escalation:** Create ticket, normal business hours support

#### SEV-4: Low (P4)
- **Definition:** Minor issues with minimal user impact
- **Examples:**
  - Cosmetic issues
  - Documentation errors
  - Feature requests
- **Response Time:** < 2 business days
- **Escalation:** Standard support ticket

### Escalation Steps

#### Step 1: Initial Assessment (0-5 minutes)

```bash
# Run health check
cloudflow health check --comprehensive

# Check status page
curl https://status.cloudflow.io/api/v1/status.json

# Review recent changes
cloudflow audit log --last 2h --event-type "deployment,configuration"

# Check metrics dashboard
cloudflow metrics dashboard --incident-mode
```

#### Step 2: Gather Information (5-15 minutes)

Create incident document with:
- Incident timestamp and duration
- Affected services and endpoints
- Error rates and user impact
- Recent changes or deployments
- Relevant log excerpts
- Correlation IDs for failed requests

```bash
# Generate incident report
cloudflow debug incident-report \
  --start "2026-01-24T10:30:00Z" \
  --end "2026-01-24T11:00:00Z" \
  --output incident-report.md

# Capture system snapshot
cloudflow debug snapshot --output snapshot-$(date +%Y%m%d-%H%M%S).tar.gz
```

#### Step 3: Escalate Based on Severity

**For SEV-1 (Critical):**
```bash
# Page on-call engineer
cloudflow incident create \
  --severity SEV-1 \
  --title "Complete API outage" \
  --description "All API requests returning 503" \
  --page-oncall

# Notify status page
cloudflow status update \
  --status "major_outage" \
  --message "We are investigating a complete service outage"

# Create war room
cloudflow incident war-room create incident-2024012401
```

**For SEV-2 (High):**
```bash
# Create incident and notify
cloudflow incident create \
  --severity SEV-2 \
  --title "High rate of workflow failures" \
  --description "Workflow success rate dropped to 75%" \
  --notify-oncall

# Update status page
cloudflow status update \
  --status "partial_outage" \
  --message "We are experiencing elevated error rates"
```

**For SEV-3/SEV-4:**
```bash
# Create support ticket
cloudflow support ticket create \
  --priority MEDIUM \
  --subject "Intermittent timeout errors" \
  --description "See attached logs and reproduction steps"
```

### Contact Information

- **On-call hotline:** +1-888-CLOUDFLOW (24/7)
- **Support email:** support@cloudflow.io
- **Incident Slack channel:** #cloudflow-incidents
- **Status page:** https://status.cloudflow.io
- **Documentation:** https://docs.cloudflow.io

### Post-Incident Review

After resolving incidents, complete a postmortem:

```bash
# Generate postmortem template
cloudflow incident postmortem incident-2024012401 \
  --template standard \
  --output postmortem-2024012401.md

# Required sections:
# 1. Incident summary
# 2. Impact assessment
# 3. Timeline of events
# 4. Root cause analysis
# 5. Resolution steps
# 6. Action items
# 7. Lessons learned
```

---

## Additional Resources

- **CloudFlow Documentation:** https://docs.cloudflow.io
- **API Reference:** https://api-docs.cloudflow.io
- **Community Forum:** https://community.cloudflow.io
- **GitHub Issues:** https://github.com/cloudflow/platform/issues
- **Training Portal:** https://training.cloudflow.io
- **Monitoring Dashboard:** https://monitoring.cloudflow.io

### Getting Help

If this troubleshooting guide doesn't resolve your issue:

1. Search the knowledge base: `cloudflow kb search "your issue"`
2. Check community forum for similar issues
3. Contact support with detailed logs and reproduction steps
4. For urgent issues, use emergency escalation procedures

**Remember:** Always capture logs, metrics, and reproduction steps before escalating!

---

*Last updated: January 24, 2026*  
*Document version: 3.2.1*  
*Feedback: docs-feedback@cloudflow.io*
