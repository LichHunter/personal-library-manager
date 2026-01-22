# Runbook: Kafka Consumer Lag

## Severity

**Level:** P1 - Critical
**Response Time:** 1 hour

## Symptoms

- Consumer lag > 10000 messages
- Processing delays reported
- Alerts from lag monitor

## Diagnosis

### Step 1

Check consumer group lag: `kafka-consumer-groups.sh --bootstrap-server kafka:9092 --describe --group workflow-executor`

### Step 2

Verify consumer health: `kubectl get pods -l app=workflow-executor -n production`

### Step 3

Check for processing errors: `kubectl logs -l app=workflow-executor --tail=100 -n production | grep ERROR`

### Step 4

Monitor throughput: `cloudflow metrics get kafka.consumer.records_per_second`

## Resolution

### Step 1

Scale up consumers: `kubectl scale deployment/workflow-executor --replicas=10 -n production`

### Step 2

Reset offset if messages are stale: `kafka-consumer-groups.sh --bootstrap-server kafka:9092 --group workflow-executor --reset-offsets --to-latest --execute --topic workflows`

### Step 3

Pause producer if backpressure needed: `cloudflow workflows pause-triggers`

### Step 4

Increase partition count for parallelism (requires coordination)

## Escalation

If lag persists > 30 minutes, page platform team. May need Kafka cluster scaling.

## Post-Incident

- Document timeline in incident log
- Schedule post-mortem within 48 hours
- Update runbook if new patterns discovered
