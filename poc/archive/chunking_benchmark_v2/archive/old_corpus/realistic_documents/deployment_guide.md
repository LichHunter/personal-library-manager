# CloudFlow Platform - Deployment and Operations Guide

**Version:** 2.4.0  
**Last Updated:** January 2026  
**Target Environment:** Production (AWS EKS)

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Database Configuration](#database-configuration)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
8. [Scaling and Performance](#scaling-and-performance)
9. [Security and Compliance](#security-and-compliance)
10. [Troubleshooting](#troubleshooting)

---

## Overview

CloudFlow is a cloud-native workflow orchestration platform designed for high-availability production environments. This guide provides comprehensive instructions for deploying and operating CloudFlow on Amazon EKS (Elastic Kubernetes Service).

### Architecture Summary

CloudFlow consists of the following components:

- **API Server**: REST API for workflow management (Node.js/Express)
- **Worker Service**: Background job processor (Node.js)
- **Scheduler**: Cron-based task scheduler (Node.js)
- **PostgreSQL**: Primary data store (version 14)
- **Redis**: Cache and message queue (version 7.0)
- **PgBouncer**: Database connection pooler

### Deployment Model

- **Namespace**: `cloudflow-prod`
- **Cluster**: AWS EKS 1.28
- **Region**: us-east-1 (primary), us-west-2 (disaster recovery)
- **High Availability**: Multi-AZ deployment across 3 availability zones

---

## Prerequisites

Before beginning the deployment process, ensure you have the following:

### Required Tools

- `kubectl` (v1.28 or later)
- `helm` (v3.12 or later)
- `aws-cli` (v2.13 or later)
- `eksctl` (v0.165 or later)
- `terraform` (v1.6 or later) - for infrastructure provisioning

### Access Requirements

- AWS account with appropriate IAM permissions
- EKS cluster admin access
- Container registry access (ECR)
- Domain name and SSL certificates
- Secrets management access (AWS Secrets Manager or Vault)

### Network Requirements

- VPC with at least 3 public and 3 private subnets
- NAT Gateway configured
- Security groups allowing:
  - Ingress: HTTPS (443), HTTP (80)
  - Internal: PostgreSQL (5432), Redis (6379)
  - Monitoring: Prometheus (9090), Grafana (3000)

---

## Infrastructure Setup

### EKS Cluster Creation

Use the following `eksctl` configuration to create the EKS cluster:

```yaml
# cluster-config.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: cloudflow-production
  region: us-east-1
  version: "1.28"

availabilityZones:
  - us-east-1a
  - us-east-1b
  - us-east-1c

vpc:
  cidr: 10.0.0.0/16
  nat:
    gateway: HighlyAvailable

managedNodeGroups:
  - name: cloudflow-workers
    instanceType: m5.xlarge
    minSize: 3
    maxSize: 10
    desiredCapacity: 5
    volumeSize: 100
    privateNetworking: true
    labels:
      role: worker
      environment: production
    tags:
      Environment: production
      Application: cloudflow
    iam:
      withAddonPolicies:
        ebs: true
        efs: true
        albIngress: true
        cloudWatch: true

cloudWatch:
  clusterLogging:
    enableTypes:
      - api
      - audit
      - authenticator
      - controllerManager
      - scheduler
```

Create the cluster:

```bash
eksctl create cluster -f cluster-config.yaml
```

### Storage Configuration

Install the EBS CSI driver for persistent volumes:

```bash
eksctl create addon --name aws-ebs-csi-driver \
  --cluster cloudflow-production \
  --service-account-role-arn arn:aws:iam::ACCOUNT_ID:role/AmazonEKS_EBS_CSI_DriverRole \
  --force
```

---

## Kubernetes Deployment

### Namespace Setup

Create the CloudFlow namespace and configure resource quotas:

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cloudflow-prod
  labels:
    name: cloudflow-prod
    environment: production

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: cloudflow-quota
  namespace: cloudflow-prod
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
```

Apply the namespace configuration:

```bash
kubectl apply -f namespace.yaml
```

### Environment Configuration

CloudFlow requires the following environment variables:

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/cloudflow` | Yes |
| `REDIS_URL` | Redis connection string | `redis://redis-master.cloudflow-prod.svc.cluster.local:6379` | Yes |
| `JWT_SECRET` | Secret key for JWT token signing | `<generated-secret-256-bit>` | Yes |
| `LOG_LEVEL` | Application log verbosity | `info`, `debug`, `warn`, `error` | No (default: `info`) |
| `API_PORT` | API server port | `3000` | No (default: `3000`) |
| `WORKER_CONCURRENCY` | Number of concurrent workers | `10` | No (default: `5`) |
| `SESSION_SECRET` | Session encryption key | `<generated-secret-256-bit>` | Yes |
| `AWS_REGION` | AWS region for services | `us-east-1` | Yes |
| `S3_BUCKET` | S3 bucket for file storage | `cloudflow-prod-storage` | Yes |
| `SMTP_HOST` | Email server hostname | `smtp.sendgrid.net` | No |
| `SMTP_PORT` | Email server port | `587` | No |
| `METRICS_ENABLED` | Enable Prometheus metrics | `true` | No (default: `false`) |

Create a Kubernetes secret for sensitive variables:

```bash
kubectl create secret generic cloudflow-secrets \
  --namespace cloudflow-prod \
  --from-literal=DATABASE_URL="postgresql://cloudflow:$(cat db-password.txt)@pgbouncer.cloudflow-prod.svc.cluster.local:5432/cloudflow" \
  --from-literal=REDIS_URL="redis://redis-master.cloudflow-prod.svc.cluster.local:6379" \
  --from-literal=JWT_SECRET="$(openssl rand -base64 32)" \
  --from-literal=SESSION_SECRET="$(openssl rand -base64 32)"
```

### Helm Chart Deployment

Add the CloudFlow Helm repository:

```bash
helm repo add cloudflow https://charts.cloudflow.io
helm repo update
```

Create a values file for production configuration:

```yaml
# values-production.yaml
replicaCount: 3

image:
  repository: 123456789012.dkr.ecr.us-east-1.amazonaws.com/cloudflow
  tag: "2.4.0"
  pullPolicy: IfNotPresent

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

service:
  type: ClusterIP
  port: 80
  targetPort: 3000
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: api.cloudflow.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cloudflow-tls
      hosts:
        - api.cloudflow.io

healthCheck:
  enabled: true
  livenessProbe:
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  readinessProbe:
    path: /ready
    initialDelaySeconds: 10
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3

env:
  - name: NODE_ENV
    value: "production"
  - name: LOG_LEVEL
    value: "info"
  - name: API_PORT
    value: "3000"
  - name: WORKER_CONCURRENCY
    value: "10"
  - name: AWS_REGION
    value: "us-east-1"
  - name: METRICS_ENABLED
    value: "true"

envFrom:
  - secretRef:
      name: cloudflow-secrets

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true

persistence:
  enabled: true
  storageClass: gp3
  accessMode: ReadWriteOnce
  size: 50Gi

redis:
  enabled: true
  architecture: replication
  auth:
    enabled: true
  master:
    persistence:
      size: 20Gi
  replica:
    replicaCount: 2
    persistence:
      size: 20Gi
```

Deploy CloudFlow using Helm:

```bash
helm install cloudflow cloudflow/cloudflow \
  --namespace cloudflow-prod \
  --values values-production.yaml \
  --wait \
  --timeout 10m
```

### Deployment Verification

Verify the deployment status:

```bash
# Check pod status
kubectl get pods -n cloudflow-prod

# Expected output:
# NAME                              READY   STATUS    RESTARTS   AGE
# cloudflow-api-7d8f9c5b6d-4xk2p   1/1     Running   0          2m
# cloudflow-api-7d8f9c5b6d-9hj5m   1/1     Running   0          2m
# cloudflow-api-7d8f9c5b6d-tn8wq   1/1     Running   0          2m

# Check service endpoints
kubectl get svc -n cloudflow-prod

# Check ingress
kubectl get ingress -n cloudflow-prod
```

Test health endpoints:

```bash
# Port forward for testing
kubectl port-forward -n cloudflow-prod svc/cloudflow-api 8080:80

# Test health endpoint
curl http://localhost:8080/health
# Expected: {"status":"healthy","timestamp":"2026-01-24T10:30:00Z"}

# Test readiness endpoint
curl http://localhost:8080/ready
# Expected: {"status":"ready","dependencies":{"database":"connected","redis":"connected"}}
```

---

## Database Configuration

### PostgreSQL Setup

CloudFlow uses PostgreSQL 14 as its primary data store. Deploy PostgreSQL using the Bitnami Helm chart:

```yaml
# postgres-values.yaml
global:
  postgresql:
    auth:
      username: cloudflow
      database: cloudflow
      existingSecret: postgres-credentials

image:
  tag: "14.10.0"

primary:
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 2000m
      memory: 4Gi
  
  persistence:
    enabled: true
    size: 100Gi
    storageClass: gp3
  
  extendedConfiguration: |
    max_connections = 100
    shared_buffers = 2GB
    effective_cache_size = 6GB
    maintenance_work_mem = 512MB
    checkpoint_completion_target = 0.9
    wal_buffers = 16MB
    default_statistics_target = 100
    random_page_cost = 1.1
    effective_io_concurrency = 200
    work_mem = 10MB
    min_wal_size = 1GB
    max_wal_size = 4GB

readReplicas:
  replicaCount: 2
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 1000m
      memory: 2Gi

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
```

Install PostgreSQL:

```bash
# Create password secret
kubectl create secret generic postgres-credentials \
  --namespace cloudflow-prod \
  --from-literal=postgres-password="$(openssl rand -base64 32)" \
  --from-literal=password="$(openssl rand -base64 32)"

# Install PostgreSQL
helm install postgresql bitnami/postgresql \
  --namespace cloudflow-prod \
  --values postgres-values.yaml \
  --wait
```

### PgBouncer Configuration

Deploy PgBouncer for connection pooling to handle up to 100 connections efficiently:

```yaml
# pgbouncer.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pgbouncer-config
  namespace: cloudflow-prod
data:
  pgbouncer.ini: |
    [databases]
    cloudflow = host=postgresql.cloudflow-prod.svc.cluster.local port=5432 dbname=cloudflow
    
    [pgbouncer]
    listen_addr = 0.0.0.0
    listen_port = 5432
    auth_type = md5
    auth_file = /etc/pgbouncer/userlist.txt
    pool_mode = transaction
    max_client_conn = 1000
    default_pool_size = 25
    reserve_pool_size = 5
    reserve_pool_timeout = 3
    max_db_connections = 100
    max_user_connections = 100
    server_lifetime = 3600
    server_idle_timeout = 600
    log_connections = 1
    log_disconnections = 1
    log_pooler_errors = 1

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: cloudflow-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: pgbouncer/pgbouncer:1.21.0
        ports:
        - containerPort: 5432
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        volumeMounts:
        - name: config
          mountPath: /etc/pgbouncer
      volumes:
      - name: config
        configMap:
          name: pgbouncer-config

---
apiVersion: v1
kind: Service
metadata:
  name: pgbouncer
  namespace: cloudflow-prod
spec:
  selector:
    app: pgbouncer
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

Apply the PgBouncer configuration:

```bash
kubectl apply -f pgbouncer.yaml
```

### Database Migrations

Run database migrations before deploying new versions:

```bash
# Create migration job
kubectl create job cloudflow-migrate-$(date +%s) \
  --namespace cloudflow-prod \
  --image=123456789012.dkr.ecr.us-east-1.amazonaws.com/cloudflow:2.4.0 \
  -- npm run migrate

# Monitor migration progress
kubectl logs -f job/cloudflow-migrate-<timestamp> -n cloudflow-prod
```

---

## Monitoring and Observability

### Prometheus Setup

Install Prometheus using the kube-prometheus-stack:

```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi
```

CloudFlow exposes Prometheus metrics on port 9090 at the `/metrics` endpoint. The metrics include:

- **HTTP Metrics**: Request rate, latency, error rate
- **Database Metrics**: Connection pool status, query duration
- **Worker Metrics**: Job queue size, processing time, failure rate
- **System Metrics**: CPU usage, memory usage, heap statistics

Example metrics exposed:

```
# HELP cloudflow_http_requests_total Total number of HTTP requests
# TYPE cloudflow_http_requests_total counter
cloudflow_http_requests_total{method="GET",route="/api/workflows",status="200"} 15420

# HELP cloudflow_http_request_duration_seconds HTTP request duration in seconds
# TYPE cloudflow_http_request_duration_seconds histogram
cloudflow_http_request_duration_seconds_bucket{method="POST",route="/api/workflows",le="0.1"} 8234

# HELP cloudflow_worker_jobs_processed_total Total number of jobs processed
# TYPE cloudflow_worker_jobs_processed_total counter
cloudflow_worker_jobs_processed_total{status="success"} 45230

# HELP cloudflow_database_connections Current database connections
# TYPE cloudflow_database_connections gauge
cloudflow_database_connections{state="active"} 23
```

### Grafana Dashboards

Access Grafana to view CloudFlow dashboards:

```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

Import the CloudFlow dashboard (ID: 15847) or use the provided JSON template. Key dashboard panels include:

1. **API Performance**: Request rate, P95/P99 latency, error rate
2. **Resource Usage**: CPU, memory, disk I/O per pod
3. **Database Health**: Connection pool utilization, query performance
4. **Worker Status**: Queue depth, processing rate, job success/failure ratio
5. **System Overview**: Pod status, replica count, autoscaling events

### Logging with CloudWatch

CloudFlow logs are automatically shipped to CloudWatch Logs. Configure log aggregation:

```bash
# Install Fluent Bit for log forwarding
helm install fluent-bit fluent/fluent-bit \
  --namespace logging \
  --create-namespace \
  --set cloudWatch.enabled=true \
  --set cloudWatch.region=us-east-1 \
  --set cloudWatch.logGroupName=/aws/eks/cloudflow-production/application
```

Query logs using CloudWatch Insights:

```sql
fields @timestamp, @message, level, requestId, userId
| filter namespace = "cloudflow-prod"
| filter level = "error"
| sort @timestamp desc
| limit 100
```

---

## Backup and Disaster Recovery

### Database Backup Strategy

CloudFlow implements a comprehensive backup strategy with the following retention policy:

- **Daily snapshots**: Retained for 30 days
- **Weekly backups**: Retained for 90 days
- **Monthly backups**: Retained for 1 year

#### Automated Backup with Velero

Install Velero for cluster-wide backups:

```bash
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket cloudflow-velero-backups \
  --backup-location-config region=us-east-1 \
  --snapshot-location-config region=us-east-1 \
  --secret-file ./credentials-velero
```

Create a daily backup schedule:

```yaml
# backup-schedule.yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: cloudflow-daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"  # 2 AM UTC daily
  template:
    includedNamespaces:
    - cloudflow-prod
    ttl: 720h0m0s  # 30 days
    storageLocation: default
    snapshotVolumes: true
```

Apply the backup schedule:

```bash
kubectl apply -f backup-schedule.yaml
```

#### PostgreSQL Backup

Configure PostgreSQL continuous archiving with WAL-G:

```bash
# Create backup cronjob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: cloudflow-prod
spec:
  schedule: "0 1 * * *"  # 1 AM UTC daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:14
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-credentials
                  key: password
            command:
            - /bin/bash
            - -c
            - |
              TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
              pg_dump -h postgresql.cloudflow-prod.svc.cluster.local \
                -U cloudflow -d cloudflow \
                | gzip > /backups/cloudflow_\${TIMESTAMP}.sql.gz
              aws s3 cp /backups/cloudflow_\${TIMESTAMP}.sql.gz \
                s3://cloudflow-db-backups/daily/
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
EOF
```

### Disaster Recovery Procedure

In case of catastrophic failure, follow these steps to restore CloudFlow:

1. **Restore EKS Cluster** (if necessary):
   ```bash
   eksctl create cluster -f cluster-config.yaml
   ```

2. **Restore Velero Backup**:
   ```bash
   # List available backups
   velero backup get
   
   # Restore from backup
   velero restore create --from-backup cloudflow-daily-backup-20260124
   ```

3. **Restore Database**:
   ```bash
   # Download backup from S3
   aws s3 cp s3://cloudflow-db-backups/daily/cloudflow_20260124_010000.sql.gz .
   
   # Restore database
   gunzip -c cloudflow_20260124_010000.sql.gz | \
     psql -h postgresql.cloudflow-prod.svc.cluster.local -U cloudflow -d cloudflow
   ```

4. **Verify Service Health**:
   ```bash
   kubectl get pods -n cloudflow-prod
   curl https://api.cloudflow.io/health
   ```

**Recovery Time Objective (RTO)**: 4 hours  
**Recovery Point Objective (RPO)**: 24 hours

---

## Scaling and Performance

### Horizontal Pod Autoscaling

CloudFlow is configured with Horizontal Pod Autoscaler (HPA) to automatically scale based on resource utilization:

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cloudflow-api-hpa
  namespace: cloudflow-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cloudflow-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

Monitor autoscaling events:

```bash
kubectl get hpa -n cloudflow-prod -w
kubectl describe hpa cloudflow-api-hpa -n cloudflow-prod
```

### Performance Optimization

#### Connection Pooling

Ensure optimal database connection pooling settings in PgBouncer:

- **Pool Mode**: Transaction (optimal for microservices)
- **Default Pool Size**: 25 connections per user
- **Max DB Connections**: 100 (matches PostgreSQL `max_connections`)

#### Redis Caching

Implement Redis caching for frequently accessed data:

```javascript
// Example caching strategy
const CACHE_TTL = {
  workflows: 300,      // 5 minutes
  userSessions: 3600,  // 1 hour
  apiResults: 60       // 1 minute
};
```

#### Load Testing

Perform regular load testing to validate scaling configuration:

```bash
# Install k6 for load testing
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: loadtest-script
  namespace: cloudflow-prod
data:
  script.js: |
    import http from 'k6/http';
    import { check, sleep } from 'k6';
    
    export let options = {
      stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 0 },
      ],
      thresholds: {
        http_req_duration: ['p(95)<500'],
      },
    };
    
    export default function () {
      let res = http.get('https://api.cloudflow.io/api/workflows');
      check(res, { 'status is 200': (r) => r.status === 200 });
      sleep(1);
    }
EOF
```

**Performance Targets**:
- P95 API latency: < 500ms
- P99 API latency: < 1000ms
- Error rate: < 0.1%
- Throughput: > 1000 requests/second

---

## Security and Compliance

### Network Policies

Implement network policies to restrict pod-to-pod communication:

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cloudflow-network-policy
  namespace: cloudflow-prod
spec:
  podSelector:
    matchLabels:
      app: cloudflow
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: cloudflow-prod
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 3000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Pod Security Standards

Enforce Pod Security Standards at the namespace level:

```bash
kubectl label namespace cloudflow-prod \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

### Secrets Management

Use AWS Secrets Manager for sensitive credentials:

```bash
# Install External Secrets Operator
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets-system \
  --create-namespace

# Create SecretStore
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: cloudflow-prod
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
EOF
```

---

## Troubleshooting

### Common Issues

#### Pods in CrashLoopBackOff

**Symptoms**: Pods continuously restart
**Diagnosis**:
```bash
kubectl logs -n cloudflow-prod <pod-name> --previous
kubectl describe pod -n cloudflow-prod <pod-name>
```

**Common Causes**:
- Database connection failure
- Invalid environment variables
- Insufficient resources

#### High Memory Usage

**Symptoms**: Pods being OOMKilled
**Diagnosis**:
```bash
kubectl top pods -n cloudflow-prod
```

**Resolution**:
- Increase memory limits in deployment
- Check for memory leaks in application logs
- Review heap dump for Node.js processes

#### Database Connection Pool Exhausted

**Symptoms**: "Too many connections" errors
**Diagnosis**:
```bash
# Check PgBouncer stats
kubectl exec -it pgbouncer-xxx -n cloudflow-prod -- \
  psql -p 5432 -U pgbouncer pgbouncer -c "SHOW POOLS;"
```

**Resolution**:
- Increase `max_db_connections` in PgBouncer
- Optimize application connection usage
- Add more PostgreSQL replicas

### Support Contacts

- **On-call Engineer**: pagerduty.com/cloudflow-oncall
- **Slack Channel**: #cloudflow-operations
- **Documentation**: https://docs.cloudflow.io
- **Status Page**: https://status.cloudflow.io

---

## Appendix

### Useful Commands

```bash
# View all resources in namespace
kubectl get all -n cloudflow-prod

# Check resource usage
kubectl top pods -n cloudflow-prod
kubectl top nodes

# View logs
kubectl logs -f deployment/cloudflow-api -n cloudflow-prod

# Execute command in pod
kubectl exec -it <pod-name> -n cloudflow-prod -- /bin/sh

# Port forward for local testing
kubectl port-forward svc/cloudflow-api 8080:80 -n cloudflow-prod

# Rollback deployment
helm rollback cloudflow -n cloudflow-prod

# Update deployment
helm upgrade cloudflow cloudflow/cloudflow \
  --namespace cloudflow-prod \
  --values values-production.yaml
```

### Maintenance Windows

Scheduled maintenance occurs during the following windows:

- **Primary**: Sunday 02:00-06:00 UTC
- **Secondary**: Wednesday 10:00-12:00 UTC

All deployments should target these windows to minimize user impact.

---

**Document Version**: 2.4.0  
**Maintained by**: Platform Engineering Team  
**Last Review**: January 24, 2026
