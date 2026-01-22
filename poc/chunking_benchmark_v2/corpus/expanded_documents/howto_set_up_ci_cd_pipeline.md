# How to Set Up CI/CD Pipeline

## Prerequisites

- GitHub account
- CloudFlow API key
- Repository admin access

## Steps

### Step 1: Create GitHub Actions workflow

Create a new file at `.github/workflows/cloudflow.yml` with the deployment configuration.

```bash
mkdir -p .github/workflows && touch .github/workflows/cloudflow.yml
```

### Step 2: Configure secrets

Add your CloudFlow API key to GitHub secrets.

```bash
gh secret set CLOUDFLOW_API_KEY
```

### Step 3: Add deployment step

Add the CloudFlow deployment action to your workflow.

> **Note:** Use the official cloudflow/deploy-action@v2

### Step 4: Test the pipeline

Push a commit to trigger the workflow.

```bash
git push origin main
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
