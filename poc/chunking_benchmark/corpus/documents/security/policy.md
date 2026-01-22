# Security Policy

CloudFlow security practices and policies.

## Data Protection

### Encryption

All data is encrypted:
- At rest: AES-256
- In transit: TLS 1.3

### Data Retention

- Execution logs: 90 days
- Audit logs: 1 year
- User data: Until account deletion + 30 days

## Access Control

### Authentication

- Passwords: bcrypt with cost factor 12
- MFA: TOTP-based, required for admin accounts
- Sessions: 24-hour timeout, secure cookies

### Authorization

RBAC roles:
- **Owner**: Full access, billing management
- **Admin**: User management, all workflows
- **Editor**: Create/edit workflows
- **Viewer**: Read-only access

## Vulnerability Management

### Reporting

Report vulnerabilities to: security@cloudflow.io

### Bug Bounty

We offer rewards for responsible disclosure:
- Critical: $5,000
- High: $2,500
- Medium: $1,000
- Low: $250

### Patching

- Critical: Within 24 hours
- High: Within 7 days
- Medium: Within 30 days

## Compliance

CloudFlow is compliant with:
- SOC 2 Type II
- GDPR
- HIPAA (Enterprise plan)

## Incident Response

See [Incident Response Runbook](../runbooks/incidents.md) for procedures.
