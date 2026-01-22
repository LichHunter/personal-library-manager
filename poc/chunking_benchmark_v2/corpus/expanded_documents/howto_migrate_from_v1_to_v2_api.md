# How to Migrate from v1 to v2 API

## Prerequisites

- Existing v1 integration
- v2 API access enabled

## Steps

### Step 1: Review breaking changes

The v2 API changes authentication from API keys to OAuth 2.0. Review the migration guide.

### Step 2: Update client library

Upgrade to the latest SDK version.

```bash
npm install @cloudflow/sdk@latest
```

### Step 3: Update authentication

Replace API key authentication with OAuth 2.0 client credentials flow.

> **Note:** Generate new credentials in the dashboard

### Step 4: Test in staging

Run your integration tests against the v2 staging endpoint.

```bash
CLOUDFLOW_API_URL=https://api-v2-staging.cloudflow.io npm test
```

### Step 5: Switch production

Update production configuration to use v2 API.

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
