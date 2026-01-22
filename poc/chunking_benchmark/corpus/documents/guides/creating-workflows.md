# Creating Workflows

Learn how to create and manage workflows in CloudFlow.

## Workflow Structure

A workflow consists of:
- **Trigger**: What starts the workflow
- **Steps**: Actions to perform
- **Variables**: Data passed between steps

## Basic Workflow

```yaml
name: My Workflow
description: A simple example workflow

trigger:
  type: webhook
  path: /my-workflow

variables:
  greeting: "Hello"

steps:
  - id: step1
    type: http
    url: https://api.example.com/data
    method: GET
    
  - id: step2
    type: transform
    input: "{{ steps.step1.response }}"
    expression: "data.items.map(i => i.name)"
    
  - id: step3
    type: log
    message: "{{ variables.greeting }}, found {{ steps.step2.output.length }} items"
```

## Trigger Types

### Manual Trigger

Execute via API or CLI:

```yaml
trigger:
  type: manual
```

### Webhook Trigger

HTTP endpoint that starts workflow:

```yaml
trigger:
  type: webhook
  path: /process-order
  method: POST
```

### Schedule Trigger

Cron-based scheduling:

```yaml
trigger:
  type: schedule
  cron: "0 9 * * *"  # Daily at 9 AM
```

### Event Trigger

React to system events:

```yaml
trigger:
  type: event
  source: s3
  event: object.created
```

## Step Types

### HTTP Request

```yaml
- id: fetch_data
  type: http
  url: "{{ variables.api_url }}"
  method: POST
  headers:
    Authorization: "Bearer {{ secrets.API_TOKEN }}"
  body:
    query: "{{ trigger.payload.query }}"
```

### Conditional

```yaml
- id: check_status
  type: condition
  if: "{{ steps.fetch_data.response.status == 'success' }}"
  then:
    - id: process
      type: log
      message: "Processing successful data"
  else:
    - id: handle_error
      type: notify
      channel: "#alerts"
```

### Loop

```yaml
- id: process_items
  type: loop
  items: "{{ steps.fetch_data.response.items }}"
  step:
    id: process_item
    type: http
    url: "https://api.example.com/process"
    body:
      item: "{{ item }}"
```

## Error Handling

### Retry Configuration

```yaml
- id: unreliable_api
  type: http
  url: https://flaky-api.example.com
  retry:
    attempts: 3
    delay: 5s
    backoff: exponential
```

### Error Handlers

```yaml
- id: risky_step
  type: http
  url: https://api.example.com
  on_error:
    - id: notify_failure
      type: notify
      channel: "#alerts"
      message: "Step failed: {{ error.message }}"
```

## Best Practices

1. **Use meaningful step IDs**: Makes debugging easier
2. **Add descriptions**: Document what each step does
3. **Handle errors**: Always plan for failure
4. **Use secrets**: Never hardcode credentials
5. **Test locally**: Use `cloudflow workflow test` before deploying
