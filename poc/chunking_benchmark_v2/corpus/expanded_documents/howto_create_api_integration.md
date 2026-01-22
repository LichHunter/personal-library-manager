# How to Create API Integration

## Prerequisites

- CloudFlow account

## Steps

### Step 1: Generate API key

Create an API key for programmatic access.

```bash
cloudflow apikeys create --name 'CI Integration' --scopes workflows:read,workflows:execute
```

### Step 2: Store key securely

Never commit API keys to source control. Use environment variables or secret management.

> **Note:** Rotate keys every 90 days

### Step 3: Test authentication

Verify the API key works correctly.

```bash
curl -H 'Authorization: Bearer cf_...' https://api.cloudflow.io/v1/me
```

### Step 4: Implement rate limiting

Handle rate limit responses (HTTP 429) with exponential backoff.

> **Note:** Default limit is 100 requests per minute

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
