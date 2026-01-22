# How to Set Up Monitoring Alerts

## Prerequisites

- CloudFlow Pro plan or higher

## Steps

### Step 1: Define alert rules

Create alert conditions based on metrics thresholds.

```bash
cloudflow alerts create --name 'High Error Rate' --condition 'error_rate > 0.05'
```

### Step 2: Configure channels

Set up notification channels: email, Slack, or PagerDuty.

```bash
cloudflow alerts channel add --type slack --webhook https://hooks.slack.com/...
```

### Step 3: Set severity levels

Assign severity to each alert: critical, warning, or info.

> **Note:** Critical alerts bypass quiet hours

### Step 4: Test alerts

Trigger a test alert to verify routing.

```bash
cloudflow alerts test --name 'High Error Rate'
```

## Verification

To verify the setup completed successfully:

```bash
cloudflow status
```

Expected output: `Status: OK`
