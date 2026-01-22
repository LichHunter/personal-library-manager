# How to Set Up Data Encryption

## Prerequisites

- CloudFlow Enterprise plan

## Steps

### Step 1: Enable encryption at rest

All data is encrypted at rest using AES-256 by default.

> **Note:** Encryption is automatic for all plans

### Step 2: Configure KMS

Bring your own encryption keys using AWS KMS or GCP KMS.

```bash
cloudflow encryption configure --kms-key arn:aws:kms:...
```

### Step 3: Enable field-level encryption

Encrypt sensitive fields within your workflow data.

```bash
cloudflow encryption add-field --path $.user.ssn --key-alias sensitive-data
```

### Step 4: Verify encryption

Confirm encryption is active for your workspace.

```bash
cloudflow encryption status
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
