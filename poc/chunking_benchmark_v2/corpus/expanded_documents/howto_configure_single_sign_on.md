# How to Configure Single Sign-On

## Prerequisites

- Admin access
- Enterprise plan
- Identity provider access

## Steps

### Step 1: Choose provider

CloudFlow supports SAML 2.0, OpenID Connect, and OAuth 2.0 providers.

### Step 2: Get metadata

Download CloudFlow's SAML metadata for your IdP configuration.

```bash
cloudflow sso metadata --format xml > cloudflow-metadata.xml
```

### Step 3: Configure IdP

Add CloudFlow as a service provider in your identity provider.

> **Note:** ACS URL: https://auth.cloudflow.io/saml/callback

### Step 4: Upload IdP metadata

Provide your IdP's metadata to CloudFlow.

```bash
cloudflow sso configure --idp-metadata ./idp-metadata.xml
```

### Step 5: Test login

Verify SSO login works correctly.

```bash
cloudflow sso test
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
