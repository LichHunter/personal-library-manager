# Runbook: SSL Certificate Expiry

## Severity

**Level:** P3 - Medium
**Response Time:** 15 minutes

## Symptoms

- Certificate expiring within 7 days
- SSL handshake failures
- Browser security warnings

## Diagnosis

### Step 1

Check certificate expiry: `echo | openssl s_client -connect api.cloudflow.io:443 2>/dev/null | openssl x509 -noout -dates`

### Step 2

Verify cert-manager status: `kubectl get certificates -n production`

### Step 3

Check cert-manager logs: `kubectl logs -l app=cert-manager -n cert-manager --tail=50`

### Step 4

Verify DNS is correct: `dig +short api.cloudflow.io`

## Resolution

### Step 1

Trigger manual renewal: `kubectl delete certificate api-cert -n production && kubectl apply -f certificate.yaml`

### Step 2

Check rate limits: Let's Encrypt has 50 certs/domain/week limit

### Step 3

Use staging issuer for testing: switch to letsencrypt-staging ClusterIssuer

### Step 4

If Let's Encrypt fails, manually upload certificate

## Escalation

If certificate cannot be renewed, escalate to security team immediately. Consider using backup certificate.

## Post-Incident

- Document timeline in incident log
- Schedule post-mortem within 48 hours
- Update runbook if new patterns discovered
