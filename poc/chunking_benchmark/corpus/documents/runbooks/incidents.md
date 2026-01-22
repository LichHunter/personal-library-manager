# Incident Response Runbook

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
