# How to Debug Failed Execution

## Prerequisites

- Failed execution ID

## Steps

### Step 1: Get execution details

Retrieve the full execution record including step outputs.

```bash
cloudflow executions get ex_123 --include-steps
```

### Step 2: View step logs

Check logs for the failed step.

```bash
cloudflow executions logs ex_123 --step 3
```

### Step 3: Inspect input/output

Review the data passed between steps.

```bash
cloudflow executions inspect ex_123 --step 3
```

### Step 4: Enable debug mode

Re-run with verbose logging enabled.

```bash
cloudflow workflows execute wf_123 --debug
```

### Step 5: Check error patterns

Look for similar failures in recent executions.

```bash
cloudflow executions list --status failed --since 24h
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
