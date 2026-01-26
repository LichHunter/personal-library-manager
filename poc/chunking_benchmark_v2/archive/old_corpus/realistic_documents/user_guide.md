# CloudFlow User Guide

Welcome to CloudFlow, the modern workflow automation platform that helps you connect your apps, automate repetitive tasks, and build powerful integrations without writing code.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Workflow Creation](#workflow-creation)
3. [Available Actions](#available-actions)
4. [Variables and Expressions](#variables-and-expressions)
5. [Scheduling](#scheduling)
6. [Error Handling](#error-handling)
7. [Workflow Limits](#workflow-limits)
8. [Best Practices](#best-practices)
9. [Common Workflow Patterns](#common-workflow-patterns)

## Getting Started

### Account Setup

Creating your CloudFlow account takes just a few minutes:

1. Navigate to https://app.cloudflow.io/signup
2. Enter your email address and create a strong password
3. Verify your email address by clicking the link sent to your inbox
4. Complete your profile by providing your organization name and use case
5. Choose your plan (Free, Professional, or Enterprise)

Once your account is created, you'll be redirected to your CloudFlow dashboard where you can start building workflows immediately.

### Your First Workflow

Let's create a simple workflow that sends you a Slack notification when a new file is uploaded to your Google Drive:

1. Click the **"Create Workflow"** button in the top-right corner
2. Give your workflow a name: "Google Drive to Slack Notifier"
3. Select **"Google Drive - New File"** as your trigger
4. Authenticate your Google Drive account when prompted
5. Choose the folder you want to monitor
6. Click **"Add Action"** and select **"Slack - Send Message"**
7. Authenticate your Slack account
8. Select the channel where you want to receive notifications
9. Customize your message using variables: `New file uploaded: {{trigger.file.name}}`
10. Click **"Save & Activate"** to enable your workflow

Congratulations! Your first workflow is now live. Every time a file is added to the specified Google Drive folder, you'll receive a Slack notification.

## Workflow Creation

CloudFlow offers multiple ways to create and manage workflows, giving you the flexibility to work in the way that suits you best.

### Visual Editor

The Visual Editor is CloudFlow's drag-and-drop interface for building workflows without code. It's perfect for users who prefer a graphical approach:

**Key Features:**
- Drag-and-drop action blocks from the sidebar
- Visual connections between steps show your workflow logic
- Inline configuration for each action
- Real-time validation and error checking
- Test mode to run your workflow with sample data

**Using the Visual Editor:**

1. Open the Visual Editor by clicking **"Create Workflow"** or editing an existing workflow
2. Add a trigger by clicking the **"Add Trigger"** button
3. Configure your trigger settings in the right panel
4. Add actions by clicking the **"+"** button below any step
5. Connect conditional branches by adding **"Condition"** blocks
6. Use the **"Test"** button to validate your workflow with sample data

### YAML Definition

For advanced users and version control integration, CloudFlow supports YAML-based workflow definitions:

```yaml
name: "Process Customer Orders"
description: "Validates and processes new customer orders"
version: "1.0"

trigger:
  type: webhook
  method: POST
  path: /orders/new

steps:
  - id: validate_order
    name: "Validate Order Data"
    action: javascript
    code: |
      if (!input.order_id || !input.customer_email) {
        throw new Error("Missing required fields");
      }
      return { valid: true };
    
  - id: check_inventory
    name: "Check Inventory"
    action: http_request
    config:
      method: GET
      url: "https://api.inventory.example.com/check"
      params:
        product_id: "{{trigger.body.product_id}}"
      headers:
        Authorization: "Bearer {{secrets.INVENTORY_API_KEY}}"
    
  - id: send_confirmation
    name: "Send Confirmation Email"
    action: email
    config:
      to: "{{trigger.body.customer_email}}"
      subject: "Order Confirmation - #{{trigger.body.order_id}}"
      body: "Thank you for your order! Your order #{{trigger.body.order_id}} has been confirmed."
```

**Benefits of YAML Definitions:**
- Version control friendly (commit to Git)
- Easy to share and duplicate workflows
- Supports comments and documentation
- Can be generated programmatically
- Enables infrastructure-as-code practices

To import a YAML workflow, click **"Import"** > **"From YAML"** in your dashboard.

### Triggers

Triggers determine when your workflow runs. CloudFlow supports several trigger types:

**Webhook Triggers**
Receive HTTP requests at a unique URL to start your workflow:
- Support GET, POST, PUT, PATCH, DELETE methods
- Automatically parse JSON and form data
- Access headers, query parameters, and body in your workflow

**Schedule Triggers**
Run workflows on a recurring schedule (see [Scheduling](#scheduling) for details)

**Event Triggers**
Respond to events from integrated applications:
- New email received (Gmail, Outlook)
- File uploaded (Google Drive, Dropbox, S3)
- Database record created or updated
- Form submission (Typeform, Google Forms)
- Chat message (Slack, Discord)

**Manual Triggers**
Start workflows on-demand from the dashboard or via API

## Available Actions

CloudFlow provides a comprehensive library of actions to build powerful automations.

### HTTP Requests

Make HTTP requests to any API endpoint:

**Configuration:**
- **Method**: GET, POST, PUT, PATCH, DELETE, HEAD
- **URL**: Full endpoint URL (supports variable interpolation)
- **Headers**: Custom headers as key-value pairs
- **Query Parameters**: URL parameters
- **Body**: JSON, form data, or raw text
- **Authentication**: Basic Auth, Bearer Token, API Key, OAuth 2.0

**Example:**
```yaml
- id: fetch_user
  action: http_request
  config:
    method: GET
    url: "https://api.example.com/users/{{user_id}}"
    headers:
      Authorization: "Bearer {{secrets.API_TOKEN}}"
      Content-Type: "application/json"
    timeout: 30
```

**Response Handling:**
Access the response in subsequent steps:
- `{{steps.fetch_user.status}}` - HTTP status code
- `{{steps.fetch_user.body}}` - Response body
- `{{steps.fetch_user.headers}}` - Response headers

### Database Queries

Execute queries against supported databases (PostgreSQL, MySQL, MongoDB, Redis):

**SQL Databases (PostgreSQL, MySQL):**
```yaml
- id: get_orders
  action: database_query
  config:
    connection: "{{secrets.DB_CONNECTION_STRING}}"
    query: |
      SELECT * FROM orders 
      WHERE customer_id = $1 
      AND status = $2
      ORDER BY created_at DESC
      LIMIT 10
    parameters:
      - "{{trigger.customer_id}}"
      - "pending"
```

**MongoDB:**
```yaml
- id: find_documents
  action: mongodb_query
  config:
    connection: "{{secrets.MONGO_URI}}"
    database: "production"
    collection: "users"
    operation: find
    filter:
      email: "{{trigger.email}}"
    options:
      limit: 1
```

**Important Notes:**
- Always use parameterized queries to prevent SQL injection
- Connection strings should be stored in secrets
- Query timeout is 30 seconds by default
- Maximum result set size is 10MB

### Email Notifications

Send emails via SMTP or integrated email providers:

**Configuration:**
```yaml
- id: send_notification
  action: email
  config:
    provider: "smtp"  # or "sendgrid", "mailgun", "ses"
    from: "notifications@cloudflow.io"
    to: "{{trigger.recipient}}"
    cc: "manager@company.com"
    subject: "Alert: {{alert_type}}"
    body: |
      Hello {{user.name}},
      
      This is an automated notification about {{event_description}}.
      
      Details:
      - Time: {{timestamp}}
      - Type: {{alert_type}}
      - Priority: {{priority}}
      
      Best regards,
      CloudFlow Automation
    html: true
    attachments:
      - name: "report.pdf"
        content: "{{steps.generate_report.output}}"
        encoding: "base64"
```

**Email Templates:**
CloudFlow supports dynamic templates with conditional content:
```
{{#if priority == 'high'}}
⚠️ URGENT: Immediate attention required
{{else}}
ℹ️ Informational notification
{{/if}}
```

### Slack Messages

Send messages to Slack channels or users:

**Channel Messages:**
```yaml
- id: notify_team
  action: slack_message
  config:
    channel: "#deployments"
    text: "Deployment completed successfully!"
    blocks:
      - type: "section"
        text:
          type: "mrkdwn"
          text: "*Deployment Summary*\nEnvironment: {{environment}}\nVersion: {{version}}\nDuration: {{duration}}s"
      - type: "actions"
        elements:
          - type: "button"
            text: "View Logs"
            url: "{{logs_url}}"
    thread_ts: "{{trigger.thread_ts}}"  # Reply to thread
```

**Direct Messages:**
```yaml
- id: dm_user
  action: slack_message
  config:
    user: "{{user_slack_id}}"
    text: "Your report is ready for review."
```

**Slack App Integration:**
- Install the CloudFlow Slack app in your workspace
- Authenticate once per workspace
- Use @mentions, emojis, and rich formatting
- Send messages as a bot or as yourself

## Variables and Expressions

CloudFlow uses a powerful templating system to work with dynamic data in your workflows.

### Variable Syntax

Variables are enclosed in double curly braces: `{{variable_name}}`

**Accessing Trigger Data:**
```
{{trigger.body.order_id}}
{{trigger.headers.user_agent}}
{{trigger.query.page}}
```

**Accessing Step Outputs:**
```
{{steps.step_id.output}}
{{steps.fetch_user.body.email}}
{{steps.query_db.rows[0].name}}
```

**System Variables:**
```
{{workflow.id}}          # Current workflow ID
{{workflow.name}}        # Workflow name
{{execution.id}}         # Current execution ID
{{execution.started_at}} # Execution start timestamp
{{env.ENVIRONMENT}}      # Environment variable
{{secrets.API_KEY}}      # Secret value (encrypted at rest)
```

### Built-in Functions

CloudFlow provides built-in functions for common data transformations:

**String Functions:**
```
{{upper(user.name)}}                    # Convert to uppercase
{{lower(email)}}                        # Convert to lowercase
{{trim(input)}}                         # Remove whitespace
{{substring(text, 0, 10)}}              # Extract substring
{{replace(text, "old", "new")}}         # Replace text
{{split(csv_string, ",")}}              # Split string into array
{{join(array, ", ")}}                   # Join array into string
```

**Date/Time Functions:**
```
{{now()}}                               # Current timestamp (ISO 8601)
{{now("America/New_York")}}             # Current time in timezone
{{format_date(timestamp, "YYYY-MM-DD")}} # Format date
{{add_days(date, 7)}}                   # Add days to date
{{diff_hours(date1, date2)}}            # Difference in hours
```

**Math Functions:**
```
{{add(num1, num2)}}                     # Addition
{{subtract(num1, num2)}}                # Subtraction
{{multiply(num1, num2)}}                # Multiplication
{{divide(num1, num2)}}                  # Division
{{round(number, 2)}}                    # Round to decimal places
{{max(array)}}                          # Maximum value
{{min(array)}}                          # Minimum value
```

**Array Functions:**
```
{{length(array)}}                       # Array length
{{first(array)}}                        # First element
{{last(array)}}                         # Last element
{{filter(array, "status", "active")}}   # Filter array
{{map(array, "id")}}                    # Extract property from objects
{{unique(array)}}                       # Remove duplicates
```

**Conditional Functions:**
```
{{if(condition, "true_value", "false_value")}}
{{default(value, "fallback")}}          # Use fallback if value is null/undefined
```

### Expression Examples

**Complex nested expressions:**
```
{{upper(trim(steps.get_user.body.name))}}
```

**Conditional formatting:**
```
Order Status: {{if(steps.check_inventory.in_stock, "✓ Available", "✗ Out of Stock")}}
```

**Date calculations:**
```
Due Date: {{format_date(add_days(now(), 7), "MMMM DD, YYYY")}}
```

**Dynamic URLs:**
```
https://api.example.com/v1/users/{{user_id}}/orders?status={{status}}&limit={{default(limit, 10)}}
```

## Scheduling

CloudFlow supports powerful scheduling options for recurring workflows.

### Cron Syntax

Use standard cron expressions to define schedules:

```
*    *    *    *    *
┬    ┬    ┬    ┬    ┬
│    │    │    │    │
│    │    │    │    └─── Day of Week (0-6, Sunday=0)
│    │    │    └──────── Month (1-12)
│    │    └───────────── Day of Month (1-31)
│    └────────────────── Hour (0-23)
└─────────────────────── Minute (0-59)
```

**Common Cron Patterns:**

| Pattern | Description |
|---------|-------------|
| `*/5 * * * *` | Every 5 minutes |
| `0 * * * *` | Every hour at minute 0 |
| `0 9 * * *` | Daily at 9:00 AM |
| `0 9 * * 1` | Every Monday at 9:00 AM |
| `0 0 1 * *` | First day of every month at midnight |
| `0 0 * * 0` | Every Sunday at midnight |
| `0 9-17 * * 1-5` | Every hour from 9 AM to 5 PM, Monday-Friday |
| `*/15 9-17 * * 1-5` | Every 15 minutes during business hours |

**Important:** The minimum scheduling interval is **1 minute**. Expressions that evaluate to more frequent executions will be rejected.

### Timezone Handling

All scheduled workflows run in **UTC by default**. To account for your local timezone:

**Option 1: Convert to UTC**
If you want a workflow to run at 9:00 AM EST (UTC-5), schedule it for 14:00 UTC:
```
0 14 * * *  # 9:00 AM EST = 14:00 UTC
```

**Option 2: Use Timezone Configuration**
Specify a timezone in your workflow configuration:
```yaml
schedule:
  cron: "0 9 * * *"
  timezone: "America/New_York"  # IANA timezone identifier
```

**Supported Timezones:**
CloudFlow supports all IANA timezone identifiers, including:
- `America/New_York` (Eastern Time)
- `America/Chicago` (Central Time)
- `America/Los_Angeles` (Pacific Time)
- `Europe/London` (GMT/BST)
- `Asia/Tokyo` (Japan Standard Time)
- `Australia/Sydney` (Australian Eastern Time)

**Daylight Saving Time:**
When using timezone configuration, CloudFlow automatically handles DST transitions. A workflow scheduled for 9:00 AM will run at 9:00 AM local time regardless of DST changes.

### Schedule Management

**Creating a Schedule:**
1. Open your workflow in the editor
2. Click the **"Trigger"** section
3. Select **"Schedule"** as the trigger type
4. Enter your cron expression or use the visual schedule builder
5. Select your timezone
6. Save and activate

**Testing Schedules:**
Use the built-in schedule calculator to preview upcoming executions:
```
Next 5 executions:
1. 2026-01-24 14:00:00 UTC
2. 2026-01-25 14:00:00 UTC
3. 2026-01-26 14:00:00 UTC
4. 2026-01-27 14:00:00 UTC
5. 2026-01-28 14:00:00 UTC
```

## Error Handling

Robust error handling ensures your workflows are resilient and reliable.

### Retry Policies

Configure automatic retries for failed actions:

```yaml
- id: api_call
  action: http_request
  config:
    url: "https://api.example.com/data"
  retry:
    max_attempts: 3
    backoff_type: "exponential"  # or "fixed", "linear"
    initial_interval: 1000       # milliseconds
    max_interval: 30000
    multiplier: 2.0
    retry_on:
      - timeout
      - network_error
      - status: [500, 502, 503, 504]
```

**Backoff Strategies:**

- **Fixed**: Wait the same amount of time between retries
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 1s
  - Attempt 3: Wait 1s

- **Linear**: Increase wait time by a fixed amount
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 2s
  - Attempt 3: Wait 3s

- **Exponential**: Double the wait time with each retry (recommended)
  - Attempt 1: Wait 1s
  - Attempt 2: Wait 2s
  - Attempt 3: Wait 4s

**Retry Conditions:**
Control which errors trigger retries:
- `timeout`: Request timeout
- `network_error`: Connection failures
- `status`: Specific HTTP status codes
- `error_code`: Application-specific error codes

### Fallback Actions

Execute alternative actions when the primary action fails:

```yaml
- id: primary_payment
  action: http_request
  config:
    url: "https://primary-payment-gateway.com/charge"
    method: POST
    body:
      amount: "{{amount}}"
  on_error:
    - id: fallback_payment
      action: http_request
      config:
        url: "https://backup-payment-gateway.com/charge"
        method: POST
        body:
          amount: "{{amount}}"
    - id: notify_admin
      action: email
      config:
        to: "admin@company.com"
        subject: "Payment Gateway Failure"
        body: "Primary gateway failed, switched to backup"
```

**Error Object:**
Access error details in fallback actions:
```
{{error.message}}        # Error message
{{error.code}}          # Error code
{{error.step_id}}       # Failed step ID
{{error.timestamp}}     # When the error occurred
{{error.attempts}}      # Number of retry attempts
```

### Dead Letter Queue

When all retries and fallbacks fail, CloudFlow can route failed executions to a Dead Letter Queue (DLQ) for manual review:

**Enable DLQ:**
```yaml
workflow:
  error_handling:
    dead_letter_queue:
      enabled: true
      retain_days: 30
```

**DLQ Features:**
- View failed executions in the dashboard
- Inspect complete execution context and error details
- Retry individual executions after fixing issues
- Export failed executions for analysis
- Set up alerts for DLQ threshold breaches

**Accessing the DLQ:**
1. Navigate to **Workflows** > **[Your Workflow]** > **Dead Letter Queue**
2. Filter by error type, date range, or execution ID
3. Click an execution to view full details
4. Click **"Retry"** to reprocess with the same input data

### Error Notifications

Get notified when workflows fail:

```yaml
workflow:
  notifications:
    on_failure:
      - type: email
        to: "ops-team@company.com"
      - type: slack
        channel: "#alerts"
        message: "Workflow {{workflow.name}} failed: {{error.message}}"
    on_success_after_retry:
      - type: slack
        channel: "#monitoring"
        message: "Workflow recovered after {{error.attempts}} attempts"
```

## Workflow Limits

CloudFlow enforces the following limits to ensure platform stability and performance:

### Steps Per Workflow
- **Maximum**: 50 steps per workflow
- **Recommendation**: Keep workflows focused and modular. If you need more steps, consider splitting into multiple workflows connected via webhooks.

### Execution Timeout
- **Default**: 3600 seconds (60 minutes)
- **Behavior**: Workflows exceeding this timeout are automatically terminated
- **Custom Timeouts**: Enterprise plans can request custom timeout limits

**Setting Step-Level Timeouts:**
```yaml
- id: long_running_task
  action: http_request
  config:
    url: "https://api.example.com/process"
    timeout: 300  # 5 minutes for this specific step
```

### Execution Limits
- **Maximum**: 1000 executions per day (per workflow)
- **Rate Limiting**: 100 concurrent executions per workflow
- **Burst Limit**: 10 executions per second

**What happens when limits are reached:**
- New executions are queued automatically
- Webhook triggers return HTTP 429 (Too Many Requests)
- Scheduled executions are skipped (logged in audit trail)
- Email notifications sent to workflow owner

**Monitoring Usage:**
View real-time metrics in your workflow dashboard:
- Executions today: 847 / 1000
- Average duration: 12.3s
- Success rate: 98.2%
- Current queue depth: 3

### Data Limits
- **Maximum request/response size**: 10MB per action
- **Maximum execution payload**: 50MB total
- **Variable value size**: 1MB per variable

### Enterprise Plan Limits
Enterprise customers can request increased limits:
- Up to 100 steps per workflow
- Up to 10,000 executions per day
- Up to 7200 second timeout (2 hours)
- Priority execution queue
- Dedicated capacity allocation

Contact sales@cloudflow.io for Enterprise pricing and custom limits.

## Best Practices

Follow these best practices to build reliable, maintainable workflows:

### 1. Use Descriptive Names

**Good:**
- Workflow: "Sync Customer Data from Salesforce to Database"
- Step: "validate_customer_email"

**Bad:**
- Workflow: "Workflow 1"
- Step: "step3"

### 2. Handle Errors Gracefully

Always implement error handling for external API calls and database operations:

```yaml
- id: fetch_data
  action: http_request
  config:
    url: "https://api.example.com/data"
  retry:
    max_attempts: 3
    backoff_type: exponential
  on_error:
    - id: log_error
      action: database_query
      config:
        query: "INSERT INTO error_log (workflow_id, error) VALUES ($1, $2)"
        parameters:
          - "{{workflow.id}}"
          - "{{error.message}}"
```

### 3. Use Secrets for Sensitive Data

Never hardcode API keys, passwords, or tokens in workflows:

**Bad:**
```yaml
headers:
  Authorization: "Bearer sk_live_abc123xyz789"
```

**Good:**
```yaml
headers:
  Authorization: "Bearer {{secrets.API_TOKEN}}"
```

Store secrets in **Settings** > **Secrets** with encryption at rest.

### 4. Validate Input Data

Always validate trigger data before processing:

```yaml
- id: validate_input
  action: javascript
  code: |
    const required_fields = ['email', 'name', 'order_id'];
    for (const field of required_fields) {
      if (!input[field]) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
    
    // Validate email format
    const email_regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email_regex.test(input.email)) {
      throw new Error('Invalid email format');
    }
    
    return { validated: true };
```

### 5. Use Idempotency Keys

For operations that shouldn't be repeated (payments, record creation), use idempotency keys:

```yaml
- id: create_charge
  action: http_request
  config:
    url: "https://api.stripe.com/v1/charges"
    method: POST
    headers:
      Idempotency-Key: "{{workflow.id}}-{{execution.id}}"
    body:
      amount: "{{amount}}"
```

### 6. Monitor and Log

Add logging steps for important workflow milestones:

```yaml
- id: log_start
  action: database_query
  config:
    query: "INSERT INTO workflow_audit (execution_id, step, timestamp) VALUES ($1, $2, $3)"
    parameters:
      - "{{execution.id}}"
      - "workflow_started"
      - "{{now()}}"
```

### 7. Keep Workflows Modular

Break complex workflows into smaller, reusable components:

- Use sub-workflows for repeated logic
- Trigger child workflows via webhooks
- Share common configurations via templates

### 8. Test Thoroughly

Before activating a workflow:
1. Use test mode with sample data
2. Verify all actions execute correctly
3. Test error handling paths
4. Review execution logs
5. Start with a limited scope (e.g., test channel, small dataset)

### 9. Document Your Workflows

Add descriptions to workflows and steps:

```yaml
workflow:
  name: "Daily Sales Report"
  description: |
    Generates a daily sales report and distributes it to the sales team.
    Runs at 8:00 AM EST Monday-Friday.
    Data source: PostgreSQL sales database
    Recipients: sales-team@company.com

steps:
  - id: fetch_sales
    name: "Fetch Yesterday's Sales"
    description: "Query database for all completed orders from previous day"
```

### 10. Version Control

For critical workflows:
- Export YAML definitions regularly
- Store in version control (Git)
- Use pull requests for changes
- Tag releases with semantic versioning

## Common Workflow Patterns

Here are proven patterns for common automation scenarios:

### Pattern 1: Form to Database

Capture form submissions and store in database:

```yaml
name: "Contact Form to Database"
trigger:
  type: webhook
  method: POST

steps:
  - id: validate_submission
    action: javascript
    code: |
      if (!input.email || !input.message) {
        throw new Error("Email and message required");
      }
      return input;
  
  - id: check_duplicate
    action: database_query
    config:
      connection: "{{secrets.DB_CONNECTION}}"
      query: "SELECT id FROM contacts WHERE email = $1 AND created_at > NOW() - INTERVAL '1 hour'"
      parameters:
        - "{{trigger.body.email}}"
  
  - id: insert_contact
    action: database_query
    condition: "{{length(steps.check_duplicate.rows) == 0}}"
    config:
      connection: "{{secrets.DB_CONNECTION}}"
      query: |
        INSERT INTO contacts (name, email, message, source, created_at)
        VALUES ($1, $2, $3, $4, NOW())
      parameters:
        - "{{trigger.body.name}}"
        - "{{trigger.body.email}}"
        - "{{trigger.body.message}}"
        - "website_form"
  
  - id: notify_sales
    action: slack_message
    condition: "{{steps.insert_contact.affected_rows > 0}}"
    config:
      channel: "#leads"
      text: "New contact form submission from {{trigger.body.email}}"
```

### Pattern 2: API Polling

Periodically check an API and take action on new items:

```yaml
name: "Poll GitHub Issues"
schedule:
  cron: "*/15 * * * *"  # Every 15 minutes
  timezone: "UTC"

steps:
  - id: get_last_check
    action: database_query
    config:
      query: "SELECT last_checked_at FROM workflow_state WHERE workflow_id = $1"
      parameters:
        - "{{workflow.id}}"
  
  - id: fetch_issues
    action: http_request
    config:
      method: GET
      url: "https://api.github.com/repos/company/project/issues"
      params:
        since: "{{steps.get_last_check.rows[0].last_checked_at}}"
        state: "open"
      headers:
        Authorization: "token {{secrets.GITHUB_TOKEN}}"
  
  - id: process_new_issues
    action: javascript
    code: |
      const issues = input.body;
      return {
        count: issues.length,
        issues: issues.map(i => ({
          number: i.number,
          title: i.title,
          url: i.html_url
        }))
      };
  
  - id: notify_team
    action: slack_message
    condition: "{{steps.process_new_issues.output.count > 0}}"
    config:
      channel: "#engineering"
      text: "{{steps.process_new_issues.output.count}} new GitHub issues"
  
  - id: update_last_check
    action: database_query
    config:
      query: |
        UPDATE workflow_state 
        SET last_checked_at = $1 
        WHERE workflow_id = $2
      parameters:
        - "{{now()}}"
        - "{{workflow.id}}"
```

### Pattern 3: Multi-Step Approval

Implement approval workflows with timeouts:

```yaml
name: "Expense Approval Workflow"
trigger:
  type: webhook
  method: POST

steps:
  - id: create_approval_request
    action: database_query
    config:
      query: |
        INSERT INTO approvals (expense_id, amount, requester, status, created_at)
        VALUES ($1, $2, $3, 'pending', NOW())
        RETURNING id
      parameters:
        - "{{trigger.body.expense_id}}"
        - "{{trigger.body.amount}}"
        - "{{trigger.body.requester}}"
  
  - id: notify_manager
    action: email
    config:
      to: "{{trigger.body.manager_email}}"
      subject: "Expense Approval Required: ${{trigger.body.amount}}"
      body: |
        An expense requires your approval:
        
        Amount: ${{trigger.body.amount}}
        Requester: {{trigger.body.requester}}
        Description: {{trigger.body.description}}
        
        Approve: https://app.company.com/approve/{{steps.create_approval_request.rows[0].id}}
        Reject: https://app.company.com/reject/{{steps.create_approval_request.rows[0].id}}
  
  - id: wait_for_approval
    action: wait_for_webhook
    config:
      timeout: 86400  # 24 hours
      webhook_path: "/approval/{{steps.create_approval_request.rows[0].id}}"
  
  - id: process_approval
    action: javascript
    code: |
      if (input.status === 'approved') {
        return { approved: true };
      } else {
        throw new Error('Expense rejected');
      }
    on_error:
      - id: notify_rejection
        action: email
        config:
          to: "{{trigger.body.requester}}"
          subject: "Expense Rejected"
          body: "Your expense request has been rejected."
  
  - id: process_payment
    action: http_request
    config:
      method: POST
      url: "https://api.accounting.com/payments"
      body:
        amount: "{{trigger.body.amount}}"
        recipient: "{{trigger.body.requester}}"
```

### Pattern 4: Data Synchronization

Keep two systems in sync bidirectionally:

```yaml
name: "Sync Customers: CRM to Database"
trigger:
  type: event
  source: "salesforce"
  event: "customer.updated"

steps:
  - id: fetch_customer
    action: http_request
    config:
      url: "https://api.salesforce.com/customers/{{trigger.customer_id}}"
      headers:
        Authorization: "Bearer {{secrets.SALESFORCE_TOKEN}}"
  
  - id: check_existing
    action: database_query
    config:
      query: "SELECT id, last_updated FROM customers WHERE salesforce_id = $1"
      parameters:
        - "{{trigger.customer_id}}"
  
  - id: update_or_insert
    action: database_query
    config:
      query: |
        INSERT INTO customers (salesforce_id, name, email, phone, last_updated)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (salesforce_id) 
        DO UPDATE SET 
          name = $2, 
          email = $3, 
          phone = $4, 
          last_updated = NOW()
      parameters:
        - "{{trigger.customer_id}}"
        - "{{steps.fetch_customer.body.name}}"
        - "{{steps.fetch_customer.body.email}}"
        - "{{steps.fetch_customer.body.phone}}"
  
  - id: log_sync
    action: database_query
    config:
      query: |
        INSERT INTO sync_log (source, target, record_id, synced_at)
        VALUES ('salesforce', 'postgresql', $1, NOW())
      parameters:
        - "{{trigger.customer_id}}"
```

### Pattern 5: Error Aggregation and Alerting

Aggregate errors and send smart alerts:

```yaml
name: "Application Error Monitor"
schedule:
  cron: "*/5 * * * *"
  timezone: "UTC"

steps:
  - id: fetch_recent_errors
    action: database_query
    config:
      query: |
        SELECT error_type, COUNT(*) as count, MAX(created_at) as last_seen
        FROM error_logs
        WHERE created_at > NOW() - INTERVAL '5 minutes'
        AND alerted = false
        GROUP BY error_type
        HAVING COUNT(*) > 5
  
  - id: format_alert
    action: javascript
    condition: "{{length(steps.fetch_recent_errors.rows) > 0}}"
    code: |
      const errors = input.rows;
      let message = "⚠️ Error Alert\n\n";
      
      for (const error of errors) {
        message += `• ${error.error_type}: ${error.count} occurrences (last: ${error.last_seen})\n`;
      }
      
      return { message, total_errors: errors.reduce((sum, e) => sum + e.count, 0) };
  
  - id: send_alert
    action: slack_message
    condition: "{{steps.format_alert.output.total_errors > 10}}"
    config:
      channel: "#incidents"
      text: "{{steps.format_alert.output.message}}"
      priority: "high"
  
  - id: mark_alerted
    action: database_query
    config:
      query: |
        UPDATE error_logs
        SET alerted = true
        WHERE created_at > NOW() - INTERVAL '5 minutes'
```

---

## Getting Help

Need assistance with CloudFlow?

- **Documentation**: https://docs.cloudflow.io
- **Community Forum**: https://community.cloudflow.io
- **Support Email**: support@cloudflow.io
- **Status Page**: https://status.cloudflow.io
- **API Reference**: https://api.cloudflow.io/docs

**Enterprise Support:**
- 24/7 phone support
- Dedicated Slack channel
- Custom onboarding and training
- SLA guarantees

Contact sales@cloudflow.io to learn more.

---

**Version**: 2.1.0  
**Last Updated**: January 2026  
**License**: © 2026 CloudFlow Technologies Inc.