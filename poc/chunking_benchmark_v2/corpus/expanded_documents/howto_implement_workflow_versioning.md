# How to Implement Workflow Versioning

## Prerequisites

- Existing workflow

## Steps

### Step 1: Enable versioning

Turn on versioning for your workflow. This is required for the feature.

```bash
cloudflow workflows update wf_123 --enable-versioning
```

### Step 2: Create new version

Save changes as a new version instead of overwriting.

```bash
cloudflow workflows publish wf_123 --version 2.0.0
```

### Step 3: Set active version

Choose which version handles new executions.

```bash
cloudflow workflows activate wf_123 --version 2.0.0
```

### Step 4: Roll back if needed

Quickly revert to a previous version.

```bash
cloudflow workflows rollback wf_123 --to-version 1.0.0
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
