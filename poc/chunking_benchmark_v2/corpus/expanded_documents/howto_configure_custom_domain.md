# How to Configure Custom Domain

## Prerequisites

- Domain with DNS access
- CloudFlow Pro plan or higher

## Steps

### Step 1: Verify domain ownership

Add a TXT record to verify domain ownership.

```bash
cloudflow domains verify --domain app.example.com
```

### Step 2: Configure DNS

Add a CNAME record pointing to your CloudFlow endpoint.

> **Note:** CNAME should point to ingress.cloudflow.io

### Step 3: Enable SSL

CloudFlow automatically provisions SSL certificates via Let's Encrypt.

```bash
cloudflow domains ssl --domain app.example.com
```

### Step 4: Set as primary

Make this the primary domain for your application.

```bash
cloudflow domains set-primary --domain app.example.com
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
