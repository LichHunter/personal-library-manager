# Runbook: Database Connection Exhaustion

## Severity

**Level:** P1 - Critical
**Response Time:** 15 minutes

## Symptoms

- Connection pool at 100%
- Queries timing out
- 'too many connections' errors in logs

## Diagnosis

### Step 1

Check active connections: `SELECT count(*) FROM pg_stat_activity`

### Step 2

Identify long-running queries: `SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE state != 'idle' ORDER BY duration DESC`

### Step 3

Check PgBouncer stats: `SHOW POOLS`

### Step 4

Verify connection limits: `SHOW max_connections`

## Resolution

### Step 1

Kill long-running queries: `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE duration > interval '5 minutes'`

### Step 2

Increase PgBouncer pool size temporarily: update pgbouncer.ini

### Step 3

Restart affected services to release connections: `kubectl rollout restart deployment/api`

### Step 4

Scale down non-critical services to free connections

## Escalation

If connections remain exhausted, escalate to DBA. Consider enabling connection queueing.

## Post-Incident

- Document timeline in incident log
- Schedule post-mortem within 48 hours
- Update runbook if new patterns discovered
