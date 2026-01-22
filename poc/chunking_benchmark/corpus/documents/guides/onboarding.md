# Engineering Onboarding

Welcome to CloudFlow! This guide will help you get set up.

## Day 1

### Accounts Setup

1. **GitHub**: Accept org invite, enable 2FA
2. **Slack**: Join #engineering, #incidents, #random
3. **AWS**: Request access via IT ticket
4. **Datadog**: Get invite from your manager

### Local Development

1. Clone the monorepo:
   ```bash
   git clone git@github.com:cloudflow/cloudflow.git
   cd cloudflow
   ```

2. Install dependencies:
   ```bash
   make setup
   ```

3. Start local environment:
   ```bash
   docker-compose up -d
   make dev
   ```

4. Run tests:
   ```bash
   make test
   ```

### Access the Local App

- API: http://localhost:8080
- UI: http://localhost:3000
- Postgres: localhost:5432
- Redis: localhost:6379

## Week 1

### Codebase Tour

- `api/` - REST API service (Go)
- `web/` - Frontend (React)
- `engine/` - Workflow engine (Go)
- `infra/` - Terraform & Kubernetes configs
- `docs/` - Documentation

### Key Concepts

Read these docs:
1. [Architecture Overview](../architecture/overview.md)
2. [API Architecture](../architecture/api.md)
3. [Database Design](../architecture/database.md)

### First Task

Your manager will assign a "good first issue". These are scoped tasks to help you learn the codebase.

## Week 2-4

### Deep Dives

Schedule 1:1s with:
- Product Manager - understand roadmap
- DevOps - deployment process
- Senior Engineer - code review practices

### Shadowing

- Shadow on-call engineer for a day
- Join incident response call (if one happens)
- Attend sprint planning and retro

## Resources

- Internal wiki: wiki.cloudflow.io
- Runbooks: [runbooks/](../runbooks/)
- ADRs: [decisions/](../decisions/)
- Team calendar: calendar.cloudflow.io/engineering

## Questions?

Ask in #engineering or your onboarding buddy!
