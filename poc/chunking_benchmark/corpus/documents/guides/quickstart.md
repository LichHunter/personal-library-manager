# Quick Start Guide

Get started with CloudFlow in 5 minutes.

## Prerequisites

- CloudFlow account (sign up at cloudflow.io)
- API key (generate in Settings > API Keys)

## Step 1: Install the CLI

```bash
# macOS
brew install cloudflow-cli

# Linux
curl -fsSL https://get.cloudflow.io | bash

# Windows
choco install cloudflow-cli
```

## Step 2: Configure Authentication

```bash
cloudflow auth login
# Follow the browser prompt to authenticate
```

Or use an API key:

```bash
export CLOUDFLOW_API_KEY=your-api-key
```

## Step 3: Create Your First Workflow

Create a file `hello-world.yaml`:

```yaml
name: Hello World
trigger:
  type: manual
steps:
  - id: greet
    type: log
    message: "Hello, CloudFlow!"
```

Deploy it:

```bash
cloudflow workflow create -f hello-world.yaml
```

## Step 4: Execute the Workflow

```bash
cloudflow workflow execute hello-world
```

## Step 5: Check the Results

```bash
cloudflow execution list --workflow hello-world
cloudflow execution logs <execution-id>
```

## Next Steps

- Read the [Workflow Syntax Guide](workflow-syntax.md)
- Explore [Built-in Actions](actions.md)
- Set up [Scheduled Triggers](triggers.md)
