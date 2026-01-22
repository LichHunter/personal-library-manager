# Runbook: API Gateway 5xx Errors

## Severity

**Level:** P3 - Medium
**Response Time:** 1 hour

## Symptoms

- Error rate > 1%
- 5xx responses in access logs
- Downstream service failures

## Diagnosis

### Step 1

Check Kong error logs: `kubectl logs -l app=kong -n kong --tail=100 | grep -E '5[0-9]{2}'`

### Step 2

Verify backend health: `kubectl get pods -n production -o wide`

### Step 3

Check service endpoints: `kubectl get endpoints -n production`

### Step 4

Test backend directly: `kubectl exec -it kong-xxx -n kong -- curl http://api.production.svc:8080/health`

## Resolution

### Step 1

Restart unhealthy backends: `kubectl rollout restart deployment/api -n production`

### Step 2

Enable circuit breaker if cascading: update Kong circuit-breaker plugin

### Step 3

Scale up backends: `kubectl scale deployment/api --replicas=5 -n production`

### Step 4

Check for resource exhaustion on backend pods

### Step 5

Verify network policies allow traffic

## Escalation

If 5xx persists > 15 minutes, page on-call. May indicate infrastructure issue.

## Post-Incident

- Document timeline in incident log
- Schedule post-mortem within 48 hours
- Update runbook if new patterns discovered
