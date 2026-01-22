# Runbook: High CPU Usage

## Severity

**Level:** P1 - Critical
**Response Time:** 15 minutes

## Symptoms

- CPU utilization > 80% sustained
- Increased response latency
- Autoscaler at maximum replicas

## Diagnosis

### Step 1

Check current CPU usage: `kubectl top pods -n production`

### Step 2

Identify hot pods: `kubectl top pods --sort-by=cpu -n production | head -10`

### Step 3

Check for recent deployments: `kubectl rollout history deployment/api -n production`

### Step 4

Profile the application: `cloudflow debug profile --pod api-xyz --duration 60s`

## Resolution

### Step 1

Scale up if traffic-related: `kubectl scale deployment/api --replicas=10 -n production`

### Step 2

Restart pods if memory leak suspected: `kubectl rollout restart deployment/api -n production`

### Step 3

Roll back recent deployment if regression: `kubectl rollout undo deployment/api -n production`

### Step 4

Add resource limits if unbounded: update deployment with CPU limits

## Escalation

If CPU remains high after scaling, escalate to platform team. Page on-call if affecting users.

## Post-Incident

- Document timeline in incident log
- Schedule post-mortem within 48 hours
- Update runbook if new patterns discovered
