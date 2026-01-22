# Runbook: Out of Memory (OOM) Kills

## Severity

**Level:** P3 - Medium
**Response Time:** 15 minutes

## Symptoms

- Pods restarting with OOMKilled status
- Memory usage at limit
- Application slowdown before restart

## Diagnosis

### Step 1

Check OOM events: `kubectl get events -n production --field-selector reason=OOMKilled`

### Step 2

Review memory usage: `kubectl top pods -n production --sort-by=memory`

### Step 3

Check container limits: `kubectl describe pod <pod-name> -n production | grep -A5 Limits`

### Step 4

Analyze heap dumps if available: `cloudflow debug heapdump --pod <pod-name>`

## Resolution

### Step 1

Increase memory limit: update deployment resources.limits.memory

### Step 2

Add memory request equal to limit to guarantee allocation

### Step 3

Enable JVM heap dump on OOM: add -XX:+HeapDumpOnOutOfMemoryError

### Step 4

Investigate memory leaks with profiling tools

### Step 5

Consider horizontal scaling instead of vertical

## Escalation

If OOMs continue after limit increase, escalate to development team for memory leak investigation.

## Post-Incident

- Document timeline in incident log
- Schedule post-mortem within 48 hours
- Update runbook if new patterns discovered
