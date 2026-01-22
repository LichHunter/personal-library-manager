# How to Set Up Database Connection

## Prerequisites

- Database with network access to CloudFlow
- Connection credentials

## Steps

### Step 1: Add connection secret

Store your database connection string as a secret.

```bash
cloudflow secrets set DATABASE_URL postgresql://user:pass@host:5432/db
```

### Step 2: Configure connection pool

Set pool size based on your plan limits. Free tier allows 5 connections, Pro allows 20.

> **Note:** Pool exhaustion will queue requests

### Step 3: Test connection

Verify the database is accessible from CloudFlow.

```bash
cloudflow db test --secret DATABASE_URL
```

### Step 4: Enable SSL

For production, always require SSL connections.

```bash
cloudflow db configure --require-ssl
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
