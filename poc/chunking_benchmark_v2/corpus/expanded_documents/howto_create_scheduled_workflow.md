# How to Create Scheduled Workflow

## Prerequisites

- Workflow definition

## Steps

### Step 1: Define schedule

Use cron expression to set the schedule. Example: '0 9 * * MON-FRI' runs at 9 AM on weekdays.

### Step 2: Set timezone

Specify timezone for the schedule. Default is UTC.

> **Note:** Use IANA timezone names like America/New_York

### Step 3: Configure overlap handling

Choose behavior when previous execution is still running: skip, queue, or cancel.

### Step 4: Enable notifications

Get notified on schedule failures.

```bash
cloudflow workflows notify --on-failure
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
