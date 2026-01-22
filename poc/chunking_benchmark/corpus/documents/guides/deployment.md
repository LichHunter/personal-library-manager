# Deployment Guide

This guide covers deploying CloudFlow to production.

## Deployment Environments

We maintain three environments:
- **Development**: For local development
- **Staging**: For pre-production testing
- **Production**: Live environment

## Prerequisites

- kubectl configured for target cluster
- Helm 3.x installed
- Access to container registry
- Database credentials

## Deployment Process

### 1. Build Container Images

```bash
# Build all services
make build-all

# Or build specific service
make build SERVICE=api-gateway
```

Images are tagged with git SHA:

```bash
docker tag cloudflow/api-gateway:latest   registry.cloudflow.io/api-gateway:$(git rev-parse --short HEAD)
```

### 2. Run Database Migrations

```bash
# Connect to database
kubectl exec -it postgres-0 -- psql -U cloudflow

# Or use Flyway
flyway -url=jdbc:postgresql://db:5432/cloudflow migrate
```

### 3. Deploy with Helm

```bash
# Update dependencies
helm dependency update ./helm/cloudflow

# Deploy to staging
helm upgrade --install cloudflow ./helm/cloudflow   --namespace staging   --values ./helm/values-staging.yaml

# Deploy to production
helm upgrade --install cloudflow ./helm/cloudflow   --namespace production   --values ./helm/values-production.yaml
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -n production

# Check service health
curl https://api.cloudflow.io/health

# Run smoke tests
make test-smoke ENV=production
```

## Rollback Procedure

If issues are detected:

```bash
# Rollback Helm release
helm rollback cloudflow -n production

# Or rollback to specific revision
helm rollback cloudflow 5 -n production
```

## Environment Variables

Required environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://...` |
| `JWT_SECRET` | Secret for JWT signing | `random-string` |
| `AWS_REGION` | AWS region | `us-east-1` |

## Monitoring

After deployment, verify:
- Prometheus metrics at `/metrics`
- Grafana dashboards
- Error rates in Sentry
- Logs in CloudWatch
