# How to Implement Retry Logic

## Prerequisites

- Existing workflow

## Steps

### Step 1: Define retry policy

Add retry configuration to your workflow step.

> **Note:** Maximum 5 retry attempts allowed

### Step 2: Configure backoff

Set exponential backoff with base delay of 1 second. The formula is: delay = base * 2^attempt.

### Step 3: Add timeout

Set maximum execution time to prevent infinite loops. Default timeout is 30 seconds.

### Step 4: Handle final failure

Configure on_failure handler to notify or escalate when all retries are exhausted.

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
