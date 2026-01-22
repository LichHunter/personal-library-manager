#!/usr/bin/env python3
"""Generate expanded test corpus for RAG benchmarking.

Creates 50 documents with 150+ queries for more rigorous testing.
All key facts are exact substrings of document content.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field

random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "expanded_documents"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class Document:
    id: str
    title: str
    content: str
    filename: str


@dataclass 
class Query:
    id: str
    category: str
    query: str
    expected_docs: list[str]
    key_facts: list[str]
    ground_truth_answer: str


# ============================================================================
# DOCUMENT TEMPLATES
# ============================================================================

def gen_api_endpoint_doc(service_name: str, endpoints: list[dict]) -> str:
    """Generate API reference documentation."""
    lines = [
        f"# {service_name} API Reference",
        "",
        "## Overview",
        "",
        f"The {service_name} API provides programmatic access to {service_name.lower()} functionality.",
        f"Base URL: `https://api.cloudflow.io/v1/{service_name.lower()}`",
        "",
        "## Authentication",
        "",
        "All requests require a valid API key in the `Authorization` header:",
        "```",
        "Authorization: Bearer <your-api-key>",
        "```",
        "",
        "## Endpoints",
        "",
    ]
    
    for ep in endpoints:
        lines.extend([
            f"### {ep['method']} {ep['path']}",
            "",
            ep['description'],
            "",
            "**Parameters:**",
            "",
        ])
        for param in ep.get('params', []):
            lines.append(f"- `{param['name']}` ({param['type']}): {param['desc']}")
        
        lines.extend([
            "",
            "**Response:**",
            f"- Success: `{ep.get('success_code', 200)}` - {ep.get('success_desc', 'Success')}",
            f"- Error: `{ep.get('error_code', 400)}` - {ep.get('error_desc', 'Bad Request')}",
            "",
            "**Example:**",
            "```json",
            ep.get('example', '{}'),
            "```",
            "",
        ])
    
    return "\n".join(lines)


def gen_architecture_doc(component: str, details: dict) -> str:
    """Generate architecture documentation."""
    lines = [
        f"# {component} Architecture",
        "",
        "## Overview",
        "",
        details['overview'],
        "",
        "## Design Principles",
        "",
    ]
    
    for principle in details.get('principles', []):
        lines.append(f"- **{principle['name']}**: {principle['desc']}")
    
    lines.extend([
        "",
        "## Components",
        "",
    ])
    
    for comp in details.get('components', []):
        lines.extend([
            f"### {comp['name']}",
            "",
            comp['desc'],
            "",
            f"**Technology:** {comp['tech']}",
            f"**Scaling:** {comp['scaling']}",
            "",
        ])
    
    lines.extend([
        "## Data Flow",
        "",
        details.get('data_flow', 'Data flows through the system sequentially.'),
        "",
        "## Performance Characteristics",
        "",
        f"- **Latency P50:** {details.get('latency_p50', '50ms')}",
        f"- **Latency P99:** {details.get('latency_p99', '200ms')}",
        f"- **Throughput:** {details.get('throughput', '1000 req/s')}",
        "",
    ])
    
    return "\n".join(lines)


def gen_howto_doc(title: str, steps: list[dict], prereqs: list[str] = None) -> str:
    """Generate how-to guide."""
    lines = [
        f"# How to {title}",
        "",
        "## Prerequisites",
        "",
    ]
    
    for prereq in (prereqs or ["CloudFlow CLI installed", "Valid API credentials"]):
        lines.append(f"- {prereq}")
    
    lines.extend([
        "",
        "## Steps",
        "",
    ])
    
    for i, step in enumerate(steps, 1):
        lines.extend([
            f"### Step {i}: {step['title']}",
            "",
            step['desc'],
            "",
        ])
        if step.get('command'):
            lines.extend([
                "```bash",
                step['command'],
                "```",
                "",
            ])
        if step.get('note'):
            lines.extend([
                f"> **Note:** {step['note']}",
                "",
            ])
    
    lines.extend([
        "## Verification",
        "",
        "To verify the setup completed successfully:",
        "",
        "```bash",
        "cloudflow status",
        "```",
        "",
        "Expected output: `Status: OK`",
        "",
    ])
    
    return "\n".join(lines)


def gen_adr_doc(number: int, title: str, context: str, options: list[dict], decision: str, consequences: list[str]) -> str:
    """Generate Architecture Decision Record."""
    lines = [
        f"# ADR-{number:03d}: {title}",
        "",
        f"**Status:** Accepted",
        f"**Date:** 2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "",
        "## Context",
        "",
        context,
        "",
        "## Options Considered",
        "",
    ]
    
    for i, opt in enumerate(options, 1):
        lines.extend([
            f"### Option {i}: {opt['name']}",
            "",
            opt['desc'],
            "",
            "**Pros:**",
        ])
        for pro in opt.get('pros', []):
            lines.append(f"- {pro}")
        lines.extend([
            "",
            "**Cons:**",
        ])
        for con in opt.get('cons', []):
            lines.append(f"- {con}")
        lines.append("")
    
    lines.extend([
        "## Decision",
        "",
        decision,
        "",
        "## Consequences",
        "",
    ])
    
    for cons in consequences:
        lines.append(f"- {cons}")
    
    return "\n".join(lines)


def gen_runbook_doc(incident_type: str, symptoms: list[str], diagnosis_steps: list[str], resolution_steps: list[str], escalation: str) -> str:
    """Generate incident runbook."""
    lines = [
        f"# Runbook: {incident_type}",
        "",
        "## Severity",
        "",
        f"**Level:** {random.choice(['P1 - Critical', 'P2 - High', 'P3 - Medium'])}",
        f"**Response Time:** {random.choice(['15 minutes', '30 minutes', '1 hour'])}",
        "",
        "## Symptoms",
        "",
    ]
    
    for symptom in symptoms:
        lines.append(f"- {symptom}")
    
    lines.extend([
        "",
        "## Diagnosis",
        "",
    ])
    
    for i, step in enumerate(diagnosis_steps, 1):
        lines.extend([
            f"### Step {i}",
            "",
            step,
            "",
        ])
    
    lines.extend([
        "## Resolution",
        "",
    ])
    
    for i, step in enumerate(resolution_steps, 1):
        lines.extend([
            f"### Step {i}",
            "",
            step,
            "",
        ])
    
    lines.extend([
        "## Escalation",
        "",
        escalation,
        "",
        "## Post-Incident",
        "",
        "- Document timeline in incident log",
        "- Schedule post-mortem within 48 hours",
        "- Update runbook if new patterns discovered",
        "",
    ])
    
    return "\n".join(lines)


def gen_config_reference(component: str, configs: list[dict]) -> str:
    """Generate configuration reference."""
    lines = [
        f"# {component} Configuration Reference",
        "",
        "## Overview",
        "",
        f"This document describes all configuration options for {component}.",
        "",
        "## Environment Variables",
        "",
        "| Variable | Type | Default | Description |",
        "|----------|------|---------|-------------|",
    ]
    
    for cfg in configs:
        lines.append(f"| `{cfg['name']}` | {cfg['type']} | {cfg.get('default', 'required')} | {cfg['desc']} |")
    
    lines.extend([
        "",
        "## Configuration File",
        "",
        f"Configuration can also be provided via `{component.lower()}.yaml`:",
        "",
        "```yaml",
    ])
    
    for cfg in configs[:5]:  # First 5 as example
        lines.append(f"{cfg['name'].lower()}: {cfg.get('default', '<value>')}")
    
    lines.extend([
        "```",
        "",
        "## Validation",
        "",
        "Configuration is validated at startup. Invalid configuration will prevent the service from starting.",
        "",
    ])
    
    return "\n".join(lines)


def gen_release_notes(version: str, date: str, features: list[str], fixes: list[str], breaking: list[str] = None) -> str:
    """Generate release notes."""
    lines = [
        f"# Release Notes - v{version}",
        "",
        f"**Release Date:** {date}",
        "",
        "## New Features",
        "",
    ]
    
    for feat in features:
        lines.append(f"- {feat}")
    
    lines.extend([
        "",
        "## Bug Fixes",
        "",
    ])
    
    for fix in fixes:
        lines.append(f"- {fix}")
    
    if breaking:
        lines.extend([
            "",
            "## Breaking Changes",
            "",
        ])
        for brk in breaking:
            lines.append(f"- **BREAKING:** {brk}")
    
    lines.extend([
        "",
        "## Upgrade Instructions",
        "",
        "```bash",
        f"cloudflow upgrade --version {version}",
        "```",
        "",
        "## Known Issues",
        "",
        "None at this time.",
        "",
    ])
    
    return "\n".join(lines)


# ============================================================================
# DOCUMENT GENERATION
# ============================================================================

def generate_documents() -> list[Document]:
    """Generate all documents."""
    docs = []
    
    # API References (10 docs)
    api_services = [
        ("Workflows", [
            {"method": "GET", "path": "/workflows", "description": "List all workflows for the authenticated user.", 
             "params": [{"name": "limit", "type": "integer", "desc": "Maximum number of results (default: 50, max: 200)"},
                       {"name": "status", "type": "string", "desc": "Filter by status: active, paused, archived"}],
             "success_code": 200, "success_desc": "Returns array of workflow objects",
             "example": '{"workflows": [{"id": "wf_123", "name": "Daily Report", "status": "active"}]}'},
            {"method": "POST", "path": "/workflows", "description": "Create a new workflow.",
             "params": [{"name": "name", "type": "string", "desc": "Workflow name (required)"},
                       {"name": "definition", "type": "object", "desc": "Workflow definition in JSON format"}],
             "success_code": 201, "success_desc": "Workflow created successfully",
             "example": '{"id": "wf_456", "name": "New Workflow", "created_at": "2024-01-15T10:00:00Z"}'},
            {"method": "DELETE", "path": "/workflows/{id}", "description": "Delete a workflow permanently.",
             "params": [{"name": "id", "type": "string", "desc": "Workflow ID (required)"}],
             "success_code": 204, "success_desc": "Workflow deleted",
             "error_code": 404, "error_desc": "Workflow not found",
             "example": '{}'},
        ]),
        ("Executions", [
            {"method": "GET", "path": "/executions", "description": "List workflow executions.",
             "params": [{"name": "workflow_id", "type": "string", "desc": "Filter by workflow ID"},
                       {"name": "since", "type": "datetime", "desc": "Filter executions after this timestamp"}],
             "success_code": 200, "success_desc": "Returns array of execution objects",
             "example": '{"executions": [{"id": "ex_789", "status": "completed", "duration_ms": 1523}]}'},
            {"method": "POST", "path": "/executions/{id}/cancel", "description": "Cancel a running execution.",
             "params": [{"name": "id", "type": "string", "desc": "Execution ID"},
                       {"name": "reason", "type": "string", "desc": "Cancellation reason (optional)"}],
             "success_code": 200, "success_desc": "Execution cancelled",
             "example": '{"id": "ex_789", "status": "cancelled", "cancelled_at": "2024-01-15T10:05:00Z"}'},
        ]),
        ("Users", [
            {"method": "GET", "path": "/users/me", "description": "Get current user profile.",
             "params": [],
             "success_code": 200, "success_desc": "Returns user object",
             "example": '{"id": "usr_001", "email": "user@example.com", "role": "admin", "created_at": "2023-06-01"}'},
            {"method": "PATCH", "path": "/users/me", "description": "Update current user profile.",
             "params": [{"name": "name", "type": "string", "desc": "Display name"},
                       {"name": "timezone", "type": "string", "desc": "Timezone (e.g., America/New_York)"}],
             "success_code": 200, "success_desc": "User updated",
             "example": '{"id": "usr_001", "name": "John Doe", "timezone": "America/New_York"}'},
        ]),
        ("Teams", [
            {"method": "GET", "path": "/teams", "description": "List teams the user belongs to.",
             "params": [],
             "success_code": 200, "success_desc": "Returns array of team objects",
             "example": '{"teams": [{"id": "team_01", "name": "Engineering", "member_count": 15}]}'},
            {"method": "POST", "path": "/teams/{id}/members", "description": "Add a member to a team.",
             "params": [{"name": "user_id", "type": "string", "desc": "User ID to add"},
                       {"name": "role", "type": "string", "desc": "Role: owner, admin, member, viewer"}],
             "success_code": 201, "success_desc": "Member added",
             "example": '{"team_id": "team_01", "user_id": "usr_002", "role": "member"}'},
        ]),
        ("Webhooks", [
            {"method": "GET", "path": "/webhooks", "description": "List configured webhooks.",
             "params": [{"name": "active", "type": "boolean", "desc": "Filter by active status"}],
             "success_code": 200, "success_desc": "Returns array of webhook objects",
             "example": '{"webhooks": [{"id": "wh_01", "url": "https://example.com/hook", "events": ["workflow.completed"]}]}'},
            {"method": "POST", "path": "/webhooks", "description": "Create a new webhook endpoint.",
             "params": [{"name": "url", "type": "string", "desc": "Webhook URL (must be HTTPS)"},
                       {"name": "events", "type": "array", "desc": "Events to subscribe to"},
                       {"name": "secret", "type": "string", "desc": "Signing secret for verification"}],
             "success_code": 201, "success_desc": "Webhook created",
             "example": '{"id": "wh_02", "url": "https://example.com/new-hook", "secret": "whsec_..."}'},
        ]),
        ("Integrations", [
            {"method": "GET", "path": "/integrations", "description": "List available integrations.",
             "params": [{"name": "category", "type": "string", "desc": "Filter by category: storage, notification, database"}],
             "success_code": 200, "success_desc": "Returns array of integration objects",
             "example": '{"integrations": [{"id": "int_s3", "name": "Amazon S3", "category": "storage", "status": "connected"}]}'},
            {"method": "POST", "path": "/integrations/{id}/connect", "description": "Connect an integration.",
             "params": [{"name": "credentials", "type": "object", "desc": "Integration-specific credentials"}],
             "success_code": 200, "success_desc": "Integration connected",
             "example": '{"id": "int_s3", "status": "connected", "connected_at": "2024-01-15T10:00:00Z"}'},
        ]),
        ("Audit", [
            {"method": "GET", "path": "/audit/logs", "description": "Retrieve audit logs for compliance.",
             "params": [{"name": "start_date", "type": "date", "desc": "Start date for log range"},
                       {"name": "end_date", "type": "date", "desc": "End date for log range"},
                       {"name": "actor", "type": "string", "desc": "Filter by user ID"}],
             "success_code": 200, "success_desc": "Returns array of audit log entries",
             "example": '{"logs": [{"timestamp": "2024-01-15T10:00:00Z", "actor": "usr_001", "action": "workflow.created", "resource": "wf_123"}]}'},
        ]),
        ("Billing", [
            {"method": "GET", "path": "/billing/usage", "description": "Get current billing period usage.",
             "params": [{"name": "breakdown", "type": "boolean", "desc": "Include detailed breakdown by resource"}],
             "success_code": 200, "success_desc": "Returns usage summary",
             "example": '{"period": "2024-01", "executions": 15000, "compute_minutes": 2500, "storage_gb": 50}'},
            {"method": "GET", "path": "/billing/invoices", "description": "List past invoices.",
             "params": [{"name": "year", "type": "integer", "desc": "Filter by year"}],
             "success_code": 200, "success_desc": "Returns array of invoice objects",
             "example": '{"invoices": [{"id": "inv_202401", "amount": 299.00, "status": "paid", "date": "2024-01-01"}]}'},
        ]),
        ("Secrets", [
            {"method": "GET", "path": "/secrets", "description": "List secret names (values not returned).",
             "params": [],
             "success_code": 200, "success_desc": "Returns array of secret metadata",
             "example": '{"secrets": [{"name": "DATABASE_URL", "created_at": "2024-01-01", "updated_at": "2024-01-10"}]}'},
            {"method": "PUT", "path": "/secrets/{name}", "description": "Create or update a secret.",
             "params": [{"name": "value", "type": "string", "desc": "Secret value (encrypted at rest)"}],
             "success_code": 200, "success_desc": "Secret saved",
             "example": '{"name": "DATABASE_URL", "updated_at": "2024-01-15T10:00:00Z"}'},
            {"method": "DELETE", "path": "/secrets/{name}", "description": "Delete a secret.",
             "params": [],
             "success_code": 204, "success_desc": "Secret deleted",
             "example": '{}'},
        ]),
        ("Metrics", [
            {"method": "GET", "path": "/metrics/workflows", "description": "Get workflow performance metrics.",
             "params": [{"name": "period", "type": "string", "desc": "Time period: hour, day, week, month"},
                       {"name": "workflow_id", "type": "string", "desc": "Filter by workflow ID"}],
             "success_code": 200, "success_desc": "Returns metrics data",
             "example": '{"success_rate": 0.985, "avg_duration_ms": 1250, "p99_duration_ms": 5000, "total_executions": 10000}'},
        ]),
    ]
    
    for service, endpoints in api_services:
        doc_id = f"api_{service.lower()}"
        content = gen_api_endpoint_doc(service, endpoints)
        docs.append(Document(
            id=doc_id,
            title=f"{service} API Reference",
            content=content,
            filename=f"{doc_id}.md"
        ))
    
    # Architecture docs (8 docs)
    arch_components = [
        ("Workflow Engine", {
            "overview": "The Workflow Engine is the core component responsible for executing user-defined workflows. It uses an event-driven architecture with support for parallel execution, conditional branching, and automatic retries.",
            "principles": [
                {"name": "Idempotency", "desc": "All operations can be safely retried without side effects"},
                {"name": "Eventual Consistency", "desc": "State converges to consistent view within 5 seconds"},
                {"name": "Fault Tolerance", "desc": "System continues operating despite component failures"},
            ],
            "components": [
                {"name": "Scheduler", "tech": "Custom Go service", "scaling": "Horizontal with leader election", "desc": "Manages workflow scheduling and triggers. Supports cron expressions, webhooks, and event-based triggers."},
                {"name": "Executor", "tech": "Kubernetes Jobs", "scaling": "Auto-scaling based on queue depth", "desc": "Runs individual workflow steps in isolated containers. Maximum execution time is 30 minutes per step."},
                {"name": "State Store", "tech": "PostgreSQL with JSONB", "scaling": "Primary-replica with read replicas", "desc": "Persists workflow state, execution history, and step results. Uses optimistic locking for concurrent updates."},
            ],
            "data_flow": "Triggers → Scheduler → Job Queue → Executor → State Store → Webhooks/Notifications",
            "latency_p50": "45ms",
            "latency_p99": "250ms",
            "throughput": "5000 executions/minute",
        }),
        ("API Gateway", {
            "overview": "The API Gateway handles all incoming HTTP requests, providing authentication, rate limiting, and request routing. Built on Kong with custom plugins for CloudFlow-specific functionality.",
            "principles": [
                {"name": "Zero Trust", "desc": "Every request is authenticated and authorized"},
                {"name": "Defense in Depth", "desc": "Multiple layers of security validation"},
            ],
            "components": [
                {"name": "Kong Gateway", "tech": "Kong 3.4 on Kubernetes", "scaling": "Horizontal with HPA", "desc": "Primary ingress controller. Handles TLS termination, request routing, and plugin execution."},
                {"name": "Auth Plugin", "tech": "Custom Lua plugin", "scaling": "Stateless", "desc": "Validates JWT tokens, API keys, and OAuth2 flows. Token validation latency under 5ms."},
                {"name": "Rate Limiter", "tech": "Redis-backed sliding window", "scaling": "Shared Redis cluster", "desc": "Enforces per-user and per-IP rate limits. Default limit is 100 requests per minute."},
            ],
            "data_flow": "Client → TLS → Kong → Auth → Rate Limit → Backend Services",
            "latency_p50": "12ms",
            "latency_p99": "45ms",
            "throughput": "50000 requests/second",
        }),
        ("Data Pipeline", {
            "overview": "The Data Pipeline processes and transforms data between workflow steps. It supports batch processing up to 10GB and streaming for real-time use cases.",
            "principles": [
                {"name": "Exactly-Once Processing", "desc": "Each record is processed exactly once"},
                {"name": "Backpressure Handling", "desc": "System gracefully handles varying load"},
            ],
            "components": [
                {"name": "Stream Processor", "tech": "Apache Kafka with Flink", "scaling": "Partition-based scaling", "desc": "Handles real-time data streams. Maximum throughput of 100000 messages per second."},
                {"name": "Batch Processor", "tech": "Apache Spark on Kubernetes", "scaling": "Dynamic resource allocation", "desc": "Processes large batch jobs. Supports data formats: JSON, CSV, Parquet, Avro."},
                {"name": "Data Lake", "tech": "S3 with Iceberg tables", "scaling": "Unlimited with tiered storage", "desc": "Long-term data storage with time-travel queries. Retention configurable up to 7 years."},
            ],
            "data_flow": "Input → Validation → Transform → Enrich → Output",
            "latency_p50": "100ms (streaming), 5min (batch)",
            "latency_p99": "500ms (streaming), 30min (batch)",
            "throughput": "100000 messages/second (streaming)",
        }),
        ("Notification Service", {
            "overview": "The Notification Service delivers alerts and updates through multiple channels including email, Slack, webhooks, and mobile push notifications.",
            "principles": [
                {"name": "Delivery Guarantee", "desc": "At-least-once delivery with deduplication"},
                {"name": "Channel Abstraction", "desc": "Unified API regardless of delivery channel"},
            ],
            "components": [
                {"name": "Dispatcher", "tech": "Go microservice", "scaling": "Horizontal with queue partitioning", "desc": "Routes notifications to appropriate channel handlers. Supports priority queues for urgent notifications."},
                {"name": "Email Provider", "tech": "SendGrid integration", "scaling": "API rate limited", "desc": "Sends transactional emails. Maximum 10000 emails per hour per account."},
                {"name": "Slack Integration", "tech": "Slack Bolt SDK", "scaling": "Per-workspace rate limits", "desc": "Posts messages to Slack channels. Supports message formatting and interactive components."},
            ],
            "data_flow": "Event → Template → Dispatcher → Channel → Delivery Confirmation",
            "latency_p50": "200ms",
            "latency_p99": "2000ms",
            "throughput": "10000 notifications/minute",
        }),
        ("Search Service", {
            "overview": "The Search Service provides full-text and semantic search across workflows, executions, and documentation. Built on Elasticsearch with custom analyzers for technical content.",
            "principles": [
                {"name": "Relevance First", "desc": "Results ranked by relevance, not recency"},
                {"name": "Typo Tolerance", "desc": "Handles misspellings and variations"},
            ],
            "components": [
                {"name": "Indexer", "tech": "Custom Go service", "scaling": "Queue-based with backpressure", "desc": "Indexes documents in near real-time. Typical indexing latency is 500ms."},
                {"name": "Query Engine", "tech": "Elasticsearch 8.x", "scaling": "3-node cluster with replicas", "desc": "Executes search queries. Supports filters, facets, and aggregations."},
                {"name": "Suggestion Service", "tech": "Completion suggester", "scaling": "In-memory FST", "desc": "Provides autocomplete suggestions. Returns suggestions within 20ms."},
            ],
            "data_flow": "Document → Analyzer → Index → Query → Rank → Results",
            "latency_p50": "25ms",
            "latency_p99": "100ms",
            "throughput": "2000 queries/second",
        }),
        ("Caching Layer", {
            "overview": "The Caching Layer reduces latency and database load by caching frequently accessed data. Uses a multi-tier strategy with local and distributed caches.",
            "principles": [
                {"name": "Cache Invalidation", "desc": "Explicit invalidation on writes"},
                {"name": "Graceful Degradation", "desc": "System works without cache"},
            ],
            "components": [
                {"name": "Local Cache", "tech": "Ristretto in-memory cache", "scaling": "Per-pod memory allocation", "desc": "First-level cache with 100ms TTL. Reduces Redis round-trips for hot data."},
                {"name": "Distributed Cache", "tech": "Redis Cluster 7.x", "scaling": "6-node cluster with sharding", "desc": "Second-level cache with configurable TTL. Maximum key size 512MB."},
                {"name": "CDN", "tech": "CloudFront with Lambda@Edge", "scaling": "Global edge locations", "desc": "Caches static assets and API responses. Cache hit ratio target 95%."},
            ],
            "data_flow": "Request → Local Cache → Redis → Database → Cache Population",
            "latency_p50": "1ms (hit), 50ms (miss)",
            "latency_p99": "5ms (hit), 200ms (miss)",
            "throughput": "100000 reads/second",
        }),
        ("Authentication Service", {
            "overview": "The Authentication Service handles user identity, session management, and token issuance. Supports OAuth 2.0, SAML, and API key authentication.",
            "principles": [
                {"name": "Zero Knowledge", "desc": "Passwords never stored in plaintext"},
                {"name": "Token Rotation", "desc": "Refresh tokens rotate on each use"},
            ],
            "components": [
                {"name": "Identity Provider", "tech": "Custom Go service", "scaling": "Horizontal stateless", "desc": "Manages user accounts and credentials. Password hashing uses bcrypt with cost factor 12."},
                {"name": "Token Service", "tech": "JWT with RS256", "scaling": "Horizontal stateless", "desc": "Issues and validates access tokens. Token lifetime is 1 hour with 7-day refresh tokens."},
                {"name": "SSO Gateway", "tech": "Dex OIDC proxy", "scaling": "Active-passive", "desc": "Integrates with external identity providers. Supports Google, Microsoft, Okta, and custom SAML."},
            ],
            "data_flow": "Credentials → Validation → Token Issue → Token Refresh → Logout",
            "latency_p50": "50ms",
            "latency_p99": "200ms",
            "throughput": "5000 auth/second",
        }),
        ("Monitoring Stack", {
            "overview": "The Monitoring Stack provides observability into system health, performance, and errors. Includes metrics, logs, traces, and alerting.",
            "principles": [
                {"name": "Observability", "desc": "Every component emits metrics and traces"},
                {"name": "Actionable Alerts", "desc": "Alerts include runbook links"},
            ],
            "components": [
                {"name": "Metrics", "tech": "Prometheus with Thanos", "scaling": "Federated with long-term storage", "desc": "Collects time-series metrics. Retention is 15 days local, 1 year in object storage."},
                {"name": "Logging", "tech": "Loki with Grafana", "scaling": "Distributed with sharding", "desc": "Centralized log aggregation. Supports structured JSON logs with label indexing."},
                {"name": "Tracing", "tech": "Jaeger with OpenTelemetry", "scaling": "Sampling at 10%", "desc": "Distributed request tracing. Trace retention is 7 days."},
                {"name": "Alerting", "tech": "Alertmanager with PagerDuty", "scaling": "HA pair", "desc": "Routes alerts to on-call engineers. Supports escalation policies and silencing."},
            ],
            "data_flow": "Application → Collector → Storage → Query → Dashboard/Alert",
            "latency_p50": "N/A",
            "latency_p99": "N/A",
            "throughput": "1M metrics/second, 100GB logs/day",
        }),
    ]
    
    for component, details in arch_components:
        doc_id = f"arch_{component.lower().replace(' ', '_')}"
        content = gen_architecture_doc(component, details)
        docs.append(Document(
            id=doc_id,
            title=f"{component} Architecture",
            content=content,
            filename=f"{doc_id}.md"
        ))
    
    # How-to guides (12 docs)
    howto_guides = [
        ("Set Up CI/CD Pipeline", [
            {"title": "Create GitHub Actions workflow", "desc": "Create a new file at `.github/workflows/cloudflow.yml` with the deployment configuration.", "command": "mkdir -p .github/workflows && touch .github/workflows/cloudflow.yml"},
            {"title": "Configure secrets", "desc": "Add your CloudFlow API key to GitHub secrets.", "command": "gh secret set CLOUDFLOW_API_KEY"},
            {"title": "Add deployment step", "desc": "Add the CloudFlow deployment action to your workflow.", "note": "Use the official cloudflow/deploy-action@v2"},
            {"title": "Test the pipeline", "desc": "Push a commit to trigger the workflow.", "command": "git push origin main"},
        ], ["GitHub account", "CloudFlow API key", "Repository admin access"]),
        
        ("Configure Custom Domain", [
            {"title": "Verify domain ownership", "desc": "Add a TXT record to verify domain ownership.", "command": "cloudflow domains verify --domain app.example.com"},
            {"title": "Configure DNS", "desc": "Add a CNAME record pointing to your CloudFlow endpoint.", "note": "CNAME should point to ingress.cloudflow.io"},
            {"title": "Enable SSL", "desc": "CloudFlow automatically provisions SSL certificates via Let's Encrypt.", "command": "cloudflow domains ssl --domain app.example.com"},
            {"title": "Set as primary", "desc": "Make this the primary domain for your application.", "command": "cloudflow domains set-primary --domain app.example.com"},
        ], ["Domain with DNS access", "CloudFlow Pro plan or higher"]),
        
        ("Implement Retry Logic", [
            {"title": "Define retry policy", "desc": "Add retry configuration to your workflow step.", "note": "Maximum 5 retry attempts allowed"},
            {"title": "Configure backoff", "desc": "Set exponential backoff with base delay of 1 second. The formula is: delay = base * 2^attempt.", "command": None},
            {"title": "Add timeout", "desc": "Set maximum execution time to prevent infinite loops. Default timeout is 30 seconds.", "command": None},
            {"title": "Handle final failure", "desc": "Configure on_failure handler to notify or escalate when all retries are exhausted.", "command": None},
        ], ["Existing workflow"]),
        
        ("Set Up Database Connection", [
            {"title": "Add connection secret", "desc": "Store your database connection string as a secret.", "command": "cloudflow secrets set DATABASE_URL postgresql://user:pass@host:5432/db"},
            {"title": "Configure connection pool", "desc": "Set pool size based on your plan limits. Free tier allows 5 connections, Pro allows 20.", "note": "Pool exhaustion will queue requests"},
            {"title": "Test connection", "desc": "Verify the database is accessible from CloudFlow.", "command": "cloudflow db test --secret DATABASE_URL"},
            {"title": "Enable SSL", "desc": "For production, always require SSL connections.", "command": "cloudflow db configure --require-ssl"},
        ], ["Database with network access to CloudFlow", "Connection credentials"]),
        
        ("Create Scheduled Workflow", [
            {"title": "Define schedule", "desc": "Use cron expression to set the schedule. Example: '0 9 * * MON-FRI' runs at 9 AM on weekdays.", "command": None},
            {"title": "Set timezone", "desc": "Specify timezone for the schedule. Default is UTC.", "note": "Use IANA timezone names like America/New_York"},
            {"title": "Configure overlap handling", "desc": "Choose behavior when previous execution is still running: skip, queue, or cancel.", "command": None},
            {"title": "Enable notifications", "desc": "Get notified on schedule failures.", "command": "cloudflow workflows notify --on-failure"},
        ], ["Workflow definition"]),
        
        ("Migrate from v1 to v2 API", [
            {"title": "Review breaking changes", "desc": "The v2 API changes authentication from API keys to OAuth 2.0. Review the migration guide.", "command": None},
            {"title": "Update client library", "desc": "Upgrade to the latest SDK version.", "command": "npm install @cloudflow/sdk@latest"},
            {"title": "Update authentication", "desc": "Replace API key authentication with OAuth 2.0 client credentials flow.", "note": "Generate new credentials in the dashboard"},
            {"title": "Test in staging", "desc": "Run your integration tests against the v2 staging endpoint.", "command": "CLOUDFLOW_API_URL=https://api-v2-staging.cloudflow.io npm test"},
            {"title": "Switch production", "desc": "Update production configuration to use v2 API.", "command": None},
        ], ["Existing v1 integration", "v2 API access enabled"]),
        
        ("Set Up Monitoring Alerts", [
            {"title": "Define alert rules", "desc": "Create alert conditions based on metrics thresholds.", "command": "cloudflow alerts create --name 'High Error Rate' --condition 'error_rate > 0.05'"},
            {"title": "Configure channels", "desc": "Set up notification channels: email, Slack, or PagerDuty.", "command": "cloudflow alerts channel add --type slack --webhook https://hooks.slack.com/..."},
            {"title": "Set severity levels", "desc": "Assign severity to each alert: critical, warning, or info.", "note": "Critical alerts bypass quiet hours"},
            {"title": "Test alerts", "desc": "Trigger a test alert to verify routing.", "command": "cloudflow alerts test --name 'High Error Rate'"},
        ], ["CloudFlow Pro plan or higher"]),
        
        ("Implement Workflow Versioning", [
            {"title": "Enable versioning", "desc": "Turn on versioning for your workflow. This is required for the feature.", "command": "cloudflow workflows update wf_123 --enable-versioning"},
            {"title": "Create new version", "desc": "Save changes as a new version instead of overwriting.", "command": "cloudflow workflows publish wf_123 --version 2.0.0"},
            {"title": "Set active version", "desc": "Choose which version handles new executions.", "command": "cloudflow workflows activate wf_123 --version 2.0.0"},
            {"title": "Roll back if needed", "desc": "Quickly revert to a previous version.", "command": "cloudflow workflows rollback wf_123 --to-version 1.0.0"},
        ], ["Existing workflow"]),
        
        ("Configure Single Sign-On", [
            {"title": "Choose provider", "desc": "CloudFlow supports SAML 2.0, OpenID Connect, and OAuth 2.0 providers.", "command": None},
            {"title": "Get metadata", "desc": "Download CloudFlow's SAML metadata for your IdP configuration.", "command": "cloudflow sso metadata --format xml > cloudflow-metadata.xml"},
            {"title": "Configure IdP", "desc": "Add CloudFlow as a service provider in your identity provider.", "note": "ACS URL: https://auth.cloudflow.io/saml/callback"},
            {"title": "Upload IdP metadata", "desc": "Provide your IdP's metadata to CloudFlow.", "command": "cloudflow sso configure --idp-metadata ./idp-metadata.xml"},
            {"title": "Test login", "desc": "Verify SSO login works correctly.", "command": "cloudflow sso test"},
        ], ["Admin access", "Enterprise plan", "Identity provider access"]),
        
        ("Set Up Data Encryption", [
            {"title": "Enable encryption at rest", "desc": "All data is encrypted at rest using AES-256 by default.", "note": "Encryption is automatic for all plans"},
            {"title": "Configure KMS", "desc": "Bring your own encryption keys using AWS KMS or GCP KMS.", "command": "cloudflow encryption configure --kms-key arn:aws:kms:..."},
            {"title": "Enable field-level encryption", "desc": "Encrypt sensitive fields within your workflow data.", "command": "cloudflow encryption add-field --path $.user.ssn --key-alias sensitive-data"},
            {"title": "Verify encryption", "desc": "Confirm encryption is active for your workspace.", "command": "cloudflow encryption status"},
        ], ["CloudFlow Enterprise plan"]),
        
        ("Create API Integration", [
            {"title": "Generate API key", "desc": "Create an API key for programmatic access.", "command": "cloudflow apikeys create --name 'CI Integration' --scopes workflows:read,workflows:execute"},
            {"title": "Store key securely", "desc": "Never commit API keys to source control. Use environment variables or secret management.", "note": "Rotate keys every 90 days"},
            {"title": "Test authentication", "desc": "Verify the API key works correctly.", "command": "curl -H 'Authorization: Bearer cf_...' https://api.cloudflow.io/v1/me"},
            {"title": "Implement rate limiting", "desc": "Handle rate limit responses (HTTP 429) with exponential backoff.", "note": "Default limit is 100 requests per minute"},
        ], ["CloudFlow account"]),
        
        ("Debug Failed Execution", [
            {"title": "Get execution details", "desc": "Retrieve the full execution record including step outputs.", "command": "cloudflow executions get ex_123 --include-steps"},
            {"title": "View step logs", "desc": "Check logs for the failed step.", "command": "cloudflow executions logs ex_123 --step 3"},
            {"title": "Inspect input/output", "desc": "Review the data passed between steps.", "command": "cloudflow executions inspect ex_123 --step 3"},
            {"title": "Enable debug mode", "desc": "Re-run with verbose logging enabled.", "command": "cloudflow workflows execute wf_123 --debug"},
            {"title": "Check error patterns", "desc": "Look for similar failures in recent executions.", "command": "cloudflow executions list --status failed --since 24h"},
        ], ["Failed execution ID"]),
    ]
    
    for title, steps, prereqs in howto_guides:
        doc_id = f"howto_{title.lower().replace(' ', '_').replace('/', '_').replace('-', '_')[:30]}"
        content = gen_howto_doc(title, steps, prereqs)
        docs.append(Document(
            id=doc_id,
            title=f"How to {title}",
            content=content,
            filename=f"{doc_id}.md"
        ))
    
    # ADRs (8 docs)
    adrs = [
        (1, "Use PostgreSQL for Primary Database",
         "We need a database to store workflow definitions, execution state, and user data. The database must support ACID transactions, JSON queries, and scale to millions of records.",
         [
             {"name": "PostgreSQL", "desc": "Mature relational database with JSONB support.",
              "pros": ["ACID compliance", "Rich JSONB queries", "Team expertise", "Proven at scale"],
              "cons": ["Manual sharding for extreme scale", "Requires connection pooling"]},
             {"name": "MongoDB", "desc": "Document database with native JSON.",
              "pros": ["Flexible schema", "Built-in sharding", "Good for documents"],
              "cons": ["Weaker consistency guarantees", "Complex transactions", "Higher operational cost"]},
             {"name": "CockroachDB", "desc": "Distributed SQL database.",
              "pros": ["Automatic sharding", "Strong consistency", "PostgreSQL compatible"],
              "cons": ["Higher latency", "More expensive", "Less mature ecosystem"]},
         ],
         "We will use PostgreSQL as our primary database with JSONB for flexible workflow definitions.",
         ["Need PgBouncer for connection pooling", "Must plan for sharding at 100M+ records", "Requires regular VACUUM maintenance"]),
        
        (2, "Adopt Kubernetes for Container Orchestration",
         "We need a platform to deploy and scale our microservices. The platform must support auto-scaling, rolling deployments, and multi-region deployment.",
         [
             {"name": "Kubernetes (EKS)", "desc": "Industry-standard container orchestration.",
              "pros": ["Industry standard", "Rich ecosystem", "Cloud agnostic", "Auto-scaling"],
              "cons": ["Operational complexity", "Learning curve", "Resource overhead"]},
             {"name": "AWS ECS", "desc": "Amazon's native container service.",
              "pros": ["Simple operation", "Native AWS integration", "Lower overhead"],
              "cons": ["AWS lock-in", "Less flexible", "Smaller community"]},
             {"name": "Serverless (Lambda)", "desc": "Function-as-a-service approach.",
              "pros": ["Zero infrastructure", "Pay per use", "Auto-scaling"],
              "cons": ["Cold starts", "15-minute limit", "Vendor lock-in"]},
         ],
         "We will use Kubernetes on EKS for container orchestration with Helm for deployments.",
         ["Need platform team expertise", "Implement GitOps with ArgoCD", "Use managed node groups for easier operations"]),
        
        (3, "Implement OAuth 2.0 with JWT for Authentication",
         "We need a secure authentication system that supports API keys, user sessions, and third-party integrations.",
         [
             {"name": "OAuth 2.0 with JWT", "desc": "Industry standard token-based auth.",
              "pros": ["Industry standard", "Stateless validation", "Supports refresh tokens", "SSO compatible"],
              "cons": ["Token size overhead", "Cannot revoke individual tokens", "Complexity"]},
             {"name": "Session-based auth", "desc": "Traditional server-side sessions.",
              "pros": ["Simple implementation", "Easy revocation", "Smaller payloads"],
              "cons": ["Requires session store", "Not suitable for APIs", "Scaling challenges"]},
             {"name": "API keys only", "desc": "Simple key-based authentication.",
              "pros": ["Very simple", "Low overhead", "Easy to implement"],
              "cons": ["No user context", "Hard to rotate", "No SSO support"]},
         ],
         "We will implement OAuth 2.0 with JWT tokens. Access tokens expire after 1 hour, refresh tokens after 7 days.",
         ["Implement token rotation on refresh", "Support API keys for machine-to-machine auth", "Add MFA for admin accounts"]),
        
        (4, "Use Redis for Caching and Session Storage",
         "We need a fast data store for caching API responses, session data, and rate limiting counters.",
         [
             {"name": "Redis Cluster", "desc": "In-memory data structure store.",
              "pros": ["Sub-millisecond latency", "Rich data structures", "Pub/sub support", "Clustering"],
              "cons": ["Memory cost", "Persistence complexity", "Cluster management"]},
             {"name": "Memcached", "desc": "Simple key-value cache.",
              "pros": ["Simple operation", "Multi-threaded", "Lower memory overhead"],
              "cons": ["No persistence", "Limited data types", "No clustering"]},
             {"name": "Application-level cache", "desc": "In-process caching only.",
              "pros": ["No external dependency", "Fastest possible", "Simple"],
              "cons": ["Not shared across pods", "Memory pressure", "Cold starts"]},
         ],
         "We will use Redis Cluster for caching with a 6-node cluster for high availability.",
         ["Implement cache-aside pattern", "Set appropriate TTLs", "Monitor memory usage", "Plan for cache stampede prevention"]),
        
        (5, "Adopt Event-Driven Architecture for Workflow Execution",
         "We need an architecture that supports asynchronous workflow execution, step dependencies, and failure recovery.",
         [
             {"name": "Event-driven with Kafka", "desc": "Publish-subscribe messaging.",
              "pros": ["Loose coupling", "Scalability", "Event replay", "Audit trail"],
              "cons": ["Eventual consistency", "Operational complexity", "Message ordering challenges"]},
             {"name": "Request-response", "desc": "Synchronous API calls.",
              "pros": ["Simple mental model", "Immediate feedback", "Easy debugging"],
              "cons": ["Tight coupling", "Cascading failures", "Timeout issues"]},
             {"name": "Polling-based", "desc": "Workers poll for tasks.",
              "pros": ["Simple implementation", "Easy rate limiting", "No message broker"],
              "cons": ["Polling overhead", "Higher latency", "Inefficient at scale"]},
         ],
         "We will use an event-driven architecture with Kafka for workflow step execution.",
         ["Implement idempotent consumers", "Use dead letter queues for failed messages", "Monitor consumer lag"]),
        
        (6, "Use Terraform for Infrastructure as Code",
         "We need a way to manage cloud infrastructure consistently across environments.",
         [
             {"name": "Terraform", "desc": "Multi-cloud IaC tool.",
              "pros": ["Multi-cloud support", "Large community", "State management", "Plan before apply"],
              "cons": ["State file management", "Learning curve", "HCL limitations"]},
             {"name": "AWS CloudFormation", "desc": "AWS-native IaC.",
              "pros": ["Native AWS support", "No state file", "Drift detection"],
              "cons": ["AWS only", "Verbose YAML/JSON", "Slower development"]},
             {"name": "Pulumi", "desc": "IaC with real programming languages.",
              "pros": ["Real languages", "Type safety", "Reusable components"],
              "cons": ["Smaller community", "Vendor lock-in", "Commercial features"]},
         ],
         "We will use Terraform with S3 backend for state and DynamoDB for locking.",
         ["Implement module structure", "Use workspaces for environments", "Automate with Atlantis"]),
        
        (7, "Implement Rate Limiting at API Gateway",
         "We need to protect our services from abuse and ensure fair resource usage across customers.",
         [
             {"name": "Sliding window at gateway", "desc": "Rate limit at API gateway level.",
              "pros": ["Centralized enforcement", "Low latency", "Easy configuration"],
              "cons": ["Gateway becomes bottleneck", "Coarse granularity", "Redis dependency"]},
             {"name": "Per-service limiting", "desc": "Each service implements its own limits.",
              "pros": ["Fine-grained control", "Service autonomy", "No single point of failure"],
              "cons": ["Inconsistent implementation", "Higher complexity", "Distributed state"]},
             {"name": "Token bucket", "desc": "Token-based rate limiting.",
              "pros": ["Allows bursts", "Smooth rate limiting", "Well understood"],
              "cons": ["More complex", "Harder to explain to users", "State management"]},
         ],
         "We will implement sliding window rate limiting at the Kong API Gateway using Redis.",
         ["Default limit: 100 requests per minute", "Implement per-plan limits", "Return rate limit headers", "Add retry-after on 429"]),
        
        (8, "Use S3 for Object Storage",
         "We need storage for workflow artifacts, execution outputs, and user uploads.",
         [
             {"name": "Amazon S3", "desc": "Object storage service.",
              "pros": ["99.999999999% durability", "Lifecycle policies", "Event notifications", "CDN integration"],
              "cons": ["AWS lock-in", "Eventual consistency for overwrites", "Cost at scale"]},
             {"name": "Self-hosted MinIO", "desc": "S3-compatible object storage.",
              "pros": ["No vendor lock-in", "S3 compatible", "On-premises option"],
              "cons": ["Operational burden", "Lower durability guarantees", "Scaling complexity"]},
             {"name": "Google Cloud Storage", "desc": "GCP object storage.",
              "pros": ["Strong consistency", "Good performance", "Unified billing"],
              "cons": ["GCP lock-in", "Fewer features than S3", "Smaller ecosystem"]},
         ],
         "We will use Amazon S3 for object storage with intelligent tiering for cost optimization.",
         ["Enable versioning for critical buckets", "Implement lifecycle policies", "Use pre-signed URLs for uploads", "Enable server-side encryption"]),
    ]
    
    for num, title, context, options, decision, consequences in adrs:
        doc_id = f"adr_{num:03d}"
        content = gen_adr_doc(num, title, context, options, decision, consequences)
        docs.append(Document(
            id=doc_id,
            title=f"ADR-{num:03d}: {title}",
            content=content,
            filename=f"{doc_id}.md"
        ))
    
    # Runbooks (6 docs)
    runbooks = [
        ("High CPU Usage",
         ["CPU utilization > 80% sustained", "Increased response latency", "Autoscaler at maximum replicas"],
         ["Check current CPU usage: `kubectl top pods -n production`",
          "Identify hot pods: `kubectl top pods --sort-by=cpu -n production | head -10`",
          "Check for recent deployments: `kubectl rollout history deployment/api -n production`",
          "Profile the application: `cloudflow debug profile --pod api-xyz --duration 60s`"],
         ["Scale up if traffic-related: `kubectl scale deployment/api --replicas=10 -n production`",
          "Restart pods if memory leak suspected: `kubectl rollout restart deployment/api -n production`",
          "Roll back recent deployment if regression: `kubectl rollout undo deployment/api -n production`",
          "Add resource limits if unbounded: update deployment with CPU limits"],
         "If CPU remains high after scaling, escalate to platform team. Page on-call if affecting users."),
        
        ("Database Connection Exhaustion",
         ["Connection pool at 100%", "Queries timing out", "'too many connections' errors in logs"],
         ["Check active connections: `SELECT count(*) FROM pg_stat_activity`",
          "Identify long-running queries: `SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE state != 'idle' ORDER BY duration DESC`",
          "Check PgBouncer stats: `SHOW POOLS`",
          "Verify connection limits: `SHOW max_connections`"],
         ["Kill long-running queries: `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE duration > interval '5 minutes'`",
          "Increase PgBouncer pool size temporarily: update pgbouncer.ini",
          "Restart affected services to release connections: `kubectl rollout restart deployment/api`",
          "Scale down non-critical services to free connections"],
         "If connections remain exhausted, escalate to DBA. Consider enabling connection queueing."),
        
        ("Kafka Consumer Lag",
         ["Consumer lag > 10000 messages", "Processing delays reported", "Alerts from lag monitor"],
         ["Check consumer group lag: `kafka-consumer-groups.sh --bootstrap-server kafka:9092 --describe --group workflow-executor`",
          "Verify consumer health: `kubectl get pods -l app=workflow-executor -n production`",
          "Check for processing errors: `kubectl logs -l app=workflow-executor --tail=100 -n production | grep ERROR`",
          "Monitor throughput: `cloudflow metrics get kafka.consumer.records_per_second`"],
         ["Scale up consumers: `kubectl scale deployment/workflow-executor --replicas=10 -n production`",
          "Reset offset if messages are stale: `kafka-consumer-groups.sh --bootstrap-server kafka:9092 --group workflow-executor --reset-offsets --to-latest --execute --topic workflows`",
          "Pause producer if backpressure needed: `cloudflow workflows pause-triggers`",
          "Increase partition count for parallelism (requires coordination)"],
         "If lag persists > 30 minutes, page platform team. May need Kafka cluster scaling."),
        
        ("SSL Certificate Expiry",
         ["Certificate expiring within 7 days", "SSL handshake failures", "Browser security warnings"],
         ["Check certificate expiry: `echo | openssl s_client -connect api.cloudflow.io:443 2>/dev/null | openssl x509 -noout -dates`",
          "Verify cert-manager status: `kubectl get certificates -n production`",
          "Check cert-manager logs: `kubectl logs -l app=cert-manager -n cert-manager --tail=50`",
          "Verify DNS is correct: `dig +short api.cloudflow.io`"],
         ["Trigger manual renewal: `kubectl delete certificate api-cert -n production && kubectl apply -f certificate.yaml`",
          "Check rate limits: Let's Encrypt has 50 certs/domain/week limit",
          "Use staging issuer for testing: switch to letsencrypt-staging ClusterIssuer",
          "If Let's Encrypt fails, manually upload certificate"],
         "If certificate cannot be renewed, escalate to security team immediately. Consider using backup certificate."),
        
        ("Out of Memory (OOM) Kills",
         ["Pods restarting with OOMKilled status", "Memory usage at limit", "Application slowdown before restart"],
         ["Check OOM events: `kubectl get events -n production --field-selector reason=OOMKilled`",
          "Review memory usage: `kubectl top pods -n production --sort-by=memory`",
          "Check container limits: `kubectl describe pod <pod-name> -n production | grep -A5 Limits`",
          "Analyze heap dumps if available: `cloudflow debug heapdump --pod <pod-name>`"],
         ["Increase memory limit: update deployment resources.limits.memory",
          "Add memory request equal to limit to guarantee allocation",
          "Enable JVM heap dump on OOM: add -XX:+HeapDumpOnOutOfMemoryError",
          "Investigate memory leaks with profiling tools",
          "Consider horizontal scaling instead of vertical"],
         "If OOMs continue after limit increase, escalate to development team for memory leak investigation."),
        
        ("API Gateway 5xx Errors",
         ["Error rate > 1%", "5xx responses in access logs", "Downstream service failures"],
         ["Check Kong error logs: `kubectl logs -l app=kong -n kong --tail=100 | grep -E '5[0-9]{2}'`",
          "Verify backend health: `kubectl get pods -n production -o wide`",
          "Check service endpoints: `kubectl get endpoints -n production`",
          "Test backend directly: `kubectl exec -it kong-xxx -n kong -- curl http://api.production.svc:8080/health`"],
         ["Restart unhealthy backends: `kubectl rollout restart deployment/api -n production`",
          "Enable circuit breaker if cascading: update Kong circuit-breaker plugin",
          "Scale up backends: `kubectl scale deployment/api --replicas=5 -n production`",
          "Check for resource exhaustion on backend pods",
          "Verify network policies allow traffic"],
         "If 5xx persists > 15 minutes, page on-call. May indicate infrastructure issue."),
    ]
    
    for incident_type, symptoms, diagnosis, resolution, escalation in runbooks:
        doc_id = f"runbook_{incident_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        content = gen_runbook_doc(incident_type, symptoms, diagnosis, resolution, escalation)
        docs.append(Document(
            id=doc_id,
            title=f"Runbook: {incident_type}",
            content=content,
            filename=f"{doc_id}.md"
        ))
    
    # Config references (4 docs)
    config_refs = [
        ("Workflow Engine", [
            {"name": "WORKFLOW_MAX_STEPS", "type": "integer", "default": "100", "desc": "Maximum steps per workflow"},
            {"name": "WORKFLOW_TIMEOUT_SECONDS", "type": "integer", "default": "3600", "desc": "Maximum workflow execution time"},
            {"name": "WORKFLOW_RETRY_LIMIT", "type": "integer", "default": "3", "desc": "Default retry attempts for failed steps"},
            {"name": "WORKFLOW_CONCURRENT_LIMIT", "type": "integer", "default": "10", "desc": "Maximum concurrent executions per workflow"},
            {"name": "EXECUTOR_POOL_SIZE", "type": "integer", "default": "50", "desc": "Number of executor workers"},
            {"name": "EXECUTOR_MEMORY_LIMIT", "type": "string", "default": "512Mi", "desc": "Memory limit per executor"},
        ]),
        ("API Gateway", [
            {"name": "GATEWAY_PORT", "type": "integer", "default": "8000", "desc": "Gateway listen port"},
            {"name": "GATEWAY_RATE_LIMIT", "type": "integer", "default": "100", "desc": "Requests per minute per user"},
            {"name": "GATEWAY_TIMEOUT_MS", "type": "integer", "default": "30000", "desc": "Request timeout in milliseconds"},
            {"name": "GATEWAY_MAX_BODY_SIZE", "type": "string", "default": "10MB", "desc": "Maximum request body size"},
            {"name": "GATEWAY_CORS_ORIGINS", "type": "string", "default": "*", "desc": "Allowed CORS origins"},
            {"name": "GATEWAY_SSL_ENABLED", "type": "boolean", "default": "true", "desc": "Enable TLS termination"},
        ]),
        ("Database", [
            {"name": "DATABASE_URL", "type": "string", "default": "required", "desc": "PostgreSQL connection string"},
            {"name": "DATABASE_POOL_MIN", "type": "integer", "default": "5", "desc": "Minimum connection pool size"},
            {"name": "DATABASE_POOL_MAX", "type": "integer", "default": "20", "desc": "Maximum connection pool size"},
            {"name": "DATABASE_STATEMENT_TIMEOUT", "type": "integer", "default": "30000", "desc": "Query timeout in milliseconds"},
            {"name": "DATABASE_SSL_MODE", "type": "string", "default": "require", "desc": "SSL mode: disable, require, verify-full"},
            {"name": "DATABASE_MIGRATION_AUTO", "type": "boolean", "default": "false", "desc": "Run migrations on startup"},
        ]),
        ("Cache", [
            {"name": "REDIS_URL", "type": "string", "default": "required", "desc": "Redis connection string"},
            {"name": "REDIS_POOL_SIZE", "type": "integer", "default": "10", "desc": "Connection pool size"},
            {"name": "CACHE_TTL_SECONDS", "type": "integer", "default": "300", "desc": "Default cache TTL"},
            {"name": "CACHE_PREFIX", "type": "string", "default": "cf:", "desc": "Key prefix for namespacing"},
            {"name": "CACHE_COMPRESSION", "type": "boolean", "default": "true", "desc": "Enable value compression"},
            {"name": "CACHE_MAX_SIZE_MB", "type": "integer", "default": "100", "desc": "Maximum cache size per pod"},
        ]),
    ]
    
    for component, configs in config_refs:
        doc_id = f"config_{component.lower().replace(' ', '_')}"
        content = gen_config_reference(component, configs)
        docs.append(Document(
            id=doc_id,
            title=f"{component} Configuration Reference",
            content=content,
            filename=f"{doc_id}.md"
        ))
    
    # Release notes (4 docs)
    releases = [
        ("3.2.0", "2024-01-15", 
         ["Workflow versioning with rollback support", "New Python SDK with async support", "Bulk operations API for workflows", "Custom retry policies per step"],
         ["Fixed race condition in concurrent execution handler", "Resolved memory leak in long-running workflows", "Fixed timezone handling in scheduled triggers"],
         ["API v1 endpoints deprecated, will be removed in v4.0", "Minimum Python version now 3.9"]),
        ("3.1.0", "2023-12-01",
         ["GraphQL API beta release", "Workflow templates marketplace", "Enhanced audit logging", "SSO support for Okta"],
         ["Fixed webhook delivery retry logic", "Resolved issue with large payload handling", "Fixed dashboard loading performance"],
         None),
        ("3.0.0", "2023-10-15",
         ["Complete UI redesign", "New workflow editor with visual builder", "Real-time execution monitoring", "Team workspaces"],
         ["Fixed critical security issue in token validation", "Resolved data loss bug in failed migrations"],
         ["Authentication API completely rewritten", "Legacy webhook format no longer supported", "Minimum Node.js version now 18"]),
        ("2.9.0", "2023-08-01",
         ["Parallel step execution", "Conditional branching in workflows", "Slack integration improvements", "Performance dashboard"],
         ["Fixed issue with special characters in secrets", "Resolved timeout in large file uploads"],
         None),
    ]
    
    for version, date, features, fixes, breaking in releases:
        doc_id = f"release_{version.replace('.', '_')}"
        content = gen_release_notes(version, date, features, fixes, breaking)
        docs.append(Document(
            id=doc_id,
            title=f"Release Notes - v{version}",
            content=content,
            filename=f"{doc_id}.md"
        ))
    
    return docs


# ============================================================================
# QUERY GENERATION
# ============================================================================

def generate_queries(docs: list[Document]) -> list[Query]:
    """Generate test queries from documents."""
    queries = []
    query_id = 1
    
    # Simple lookup queries
    simple_lookups = [
        ("What is the default rate limit for API requests?", ["api_workflows", "arch_api_gateway"], 
         ["100 requests per minute", "rate limit"], "The default rate limit is 100 requests per minute per user."),
        ("What database does CloudFlow use?", ["adr_001", "config_database"],
         ["PostgreSQL", "JSONB"], "CloudFlow uses PostgreSQL as the primary database with JSONB for flexible data."),
        ("How long do access tokens last?", ["adr_003", "arch_authentication_service"],
         ["1 hour", "7 days", "refresh tokens"], "Access tokens expire after 1 hour. Refresh tokens last 7 days."),
        ("What is the maximum workflow execution time?", ["config_workflow_engine"],
         ["3600", "WORKFLOW_TIMEOUT_SECONDS"], "The maximum workflow execution time is 3600 seconds (1 hour)."),
        ("What container orchestration does CloudFlow use?", ["adr_002"],
         ["Kubernetes", "EKS", "Helm"], "CloudFlow uses Kubernetes on EKS with Helm for deployments."),
        ("What is the cache TTL?", ["config_cache"],
         ["300", "CACHE_TTL_SECONDS"], "The default cache TTL is 300 seconds."),
        ("What encryption is used for data at rest?", ["howto_set_up_data_encryption"],
         ["AES-256", "encryption at rest"], "All data is encrypted at rest using AES-256."),
        ("What is the maximum number of steps in a workflow?", ["config_workflow_engine"],
         ["100", "WORKFLOW_MAX_STEPS"], "The maximum number of steps per workflow is 100."),
        ("What is the API gateway timeout?", ["config_api_gateway"],
         ["30000", "GATEWAY_TIMEOUT_MS"], "The API gateway timeout is 30000 milliseconds."),
        ("What message broker is used for workflows?", ["adr_005"],
         ["Kafka", "event-driven"], "CloudFlow uses Kafka for event-driven workflow execution."),
    ]
    
    for question, doc_ids, facts, answer in simple_lookups:
        queries.append(Query(
            id=f"simple_{query_id:03d}",
            category="simple_lookup",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    # Cross-document queries
    cross_doc = [
        ("How is authentication implemented across the system?",
         ["adr_003", "arch_authentication_service", "arch_api_gateway"],
         ["OAuth 2.0", "JWT", "bcrypt", "RS256", "cost factor 12"],
         "Authentication uses OAuth 2.0 with JWT tokens signed with RS256. Passwords are hashed with bcrypt cost factor 12."),
        ("What monitoring and observability tools does CloudFlow use?",
         ["arch_monitoring_stack"],
         ["Prometheus", "Thanos", "Loki", "Grafana", "Jaeger", "OpenTelemetry"],
         "CloudFlow uses Prometheus with Thanos for metrics, Loki with Grafana for logs, and Jaeger with OpenTelemetry for tracing."),
        ("What caching strategy is used across the system?",
         ["arch_caching_layer", "adr_004", "config_cache"],
         ["Redis Cluster", "Ristretto", "CloudFront", "cache-aside"],
         "Multi-tier caching with Ristretto local cache, Redis Cluster distributed cache, and CloudFront CDN."),
        ("How does CloudFlow handle data storage?",
         ["adr_001", "adr_008", "arch_data_pipeline"],
         ["PostgreSQL", "S3", "Iceberg", "Parquet"],
         "PostgreSQL for transactional data, S3 for object storage, and Iceberg tables for data lake."),
        ("What are all the API authentication methods?",
         ["api_users", "api_integrations", "adr_003"],
         ["Bearer", "API key", "OAuth 2.0", "Authorization"],
         "API authentication supports Bearer tokens (JWT), API keys, and OAuth 2.0."),
    ]
    
    for question, doc_ids, facts, answer in cross_doc:
        queries.append(Query(
            id=f"cross_{query_id:03d}",
            category="cross_document",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    # How-to queries
    howto = [
        ("How do I set up CI/CD for CloudFlow?",
         ["howto_set_up_ci/cd_pipeline"],
         ["GitHub Actions", "cloudflow/deploy-action", "CLOUDFLOW_API_KEY"],
         "Create GitHub Actions workflow, add CLOUDFLOW_API_KEY secret, use cloudflow/deploy-action@v2."),
        ("How do I configure a custom domain?",
         ["howto_configure_custom_domain"],
         ["CNAME", "ingress.cloudflow.io", "Let's Encrypt"],
         "Add CNAME pointing to ingress.cloudflow.io, SSL is automatically provisioned via Let's Encrypt."),
        ("How do I implement retry logic?",
         ["howto_implement_retry_logic"],
         ["retry", "exponential backoff", "on_failure"],
         "Define retry policy with exponential backoff. Configure on_failure handler for exhausted retries."),
        ("How do I set up database connection?",
         ["howto_set_up_database_connecti"],
         ["DATABASE_URL", "cloudflow secrets set", "require-ssl"],
         "Store DATABASE_URL as secret, configure pool size, enable SSL for production."),
        ("How do I create a scheduled workflow?",
         ["howto_create_scheduled_workflo"],
         ["cron", "0 9 * * MON-FRI", "timezone"],
         "Use cron expression like '0 9 * * MON-FRI' for weekdays at 9 AM, specify timezone."),
        ("How do I configure SSO?",
         ["howto_configure_single_sign-on"],
         ["SAML 2.0", "OpenID Connect", "ACS URL"],
         "CloudFlow supports SAML 2.0 and OpenID Connect. ACS URL is https://auth.cloudflow.io/saml/callback."),
        ("How do I debug a failed execution?",
         ["howto_debug_failed_execution"],
         ["cloudflow executions get", "cloudflow executions logs", "--debug"],
         "Use cloudflow executions get for details, logs for output, and --debug flag for verbose mode."),
    ]
    
    for question, doc_ids, facts, answer in howto:
        queries.append(Query(
            id=f"howto_{query_id:03d}",
            category="how_to",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    # Comparison queries
    comparisons = [
        ("What database options were considered and why was PostgreSQL chosen?",
         ["adr_001"],
         ["PostgreSQL", "MongoDB", "CockroachDB", "JSONB", "Team expertise"],
         "PostgreSQL, MongoDB, and CockroachDB were considered. PostgreSQL chosen for JSONB support and team expertise."),
        ("Compare Kubernetes vs ECS for container orchestration",
         ["adr_002"],
         ["Kubernetes", "ECS", "industry standard", "AWS lock-in", "ecosystem"],
         "Kubernetes offers industry standard and rich ecosystem. ECS has AWS lock-in but simpler operations."),
        ("What authentication options were evaluated?",
         ["adr_003"],
         ["OAuth 2.0", "Session-based", "API keys", "stateless validation"],
         "OAuth 2.0 with JWT, session-based auth, and API keys only were evaluated. JWT chosen for stateless validation."),
        ("Compare caching options: Redis vs Memcached",
         ["adr_004"],
         ["Redis Cluster", "Memcached", "data structures", "persistence"],
         "Redis offers rich data structures and persistence. Memcached is simpler but lacks clustering."),
    ]
    
    for question, doc_ids, facts, answer in comparisons:
        queries.append(Query(
            id=f"compare_{query_id:03d}",
            category="comparison",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    # Troubleshooting queries
    troubleshooting = [
        ("How do I diagnose high CPU usage?",
         ["runbook_high_cpu_usage"],
         ["kubectl top pods", "cloudflow debug profile", "rollout restart"],
         "Check CPU with kubectl top pods, profile with cloudflow debug profile, restart or scale as needed."),
        ("What to do when database connections are exhausted?",
         ["runbook_database_connection_ex"],
         ["pg_stat_activity", "SHOW POOLS", "pg_terminate_backend"],
         "Check connections with pg_stat_activity, verify PgBouncer with SHOW POOLS, kill long queries."),
        ("How to handle Kafka consumer lag?",
         ["runbook_kafka_consumer_lag"],
         ["kafka-consumer-groups.sh", "consumer lag", "reset-offsets"],
         "Check lag with kafka-consumer-groups.sh, scale consumers or reset offsets if needed."),
        ("What to do when SSL certificate is expiring?",
         ["runbook_ssl_certificate_expir"],
         ["openssl s_client", "cert-manager", "Let's Encrypt"],
         "Check expiry with openssl, verify cert-manager status, trigger renewal or manually upload."),
        ("How to fix OOM kills?",
         ["runbook_out_of_memory_oom_kill"],
         ["OOMKilled", "resources.limits.memory", "HeapDumpOnOutOfMemoryError"],
         "Check OOM events, increase memory limits, enable heap dumps for leak investigation."),
        ("How to troubleshoot API Gateway 5xx errors?",
         ["runbook_api_gateway_5xx_error"],
         ["kubectl logs", "circuit breaker", "kubectl get endpoints"],
         "Check Kong logs, verify backend health, enable circuit breaker if cascading failures."),
    ]
    
    for question, doc_ids, facts, answer in troubleshooting:
        queries.append(Query(
            id=f"troubleshoot_{query_id:03d}",
            category="troubleshooting",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    # Architecture queries
    architecture = [
        ("Explain the Workflow Engine architecture",
         ["arch_workflow_engine"],
         ["Scheduler", "Executor", "State Store", "PostgreSQL with JSONB", "5000 executions/minute"],
         "Workflow Engine has Scheduler, Executor (K8s Jobs), and State Store (PostgreSQL). Throughput: 5000 executions/minute."),
        ("How does the API Gateway work?",
         ["arch_api_gateway"],
         ["Kong", "TLS termination", "50000 requests/second", "5ms"],
         "Kong Gateway handles TLS, auth, rate limiting. 50000 req/s throughput, token validation under 5ms."),
        ("Describe the Data Pipeline architecture",
         ["arch_data_pipeline"],
         ["Kafka", "Flink", "Spark", "100000 messages per second", "Iceberg"],
         "Stream processing with Kafka/Flink (100k msg/s), batch with Spark, storage in S3 with Iceberg."),
        ("How does the Search Service work?",
         ["arch_search_service"],
         ["Elasticsearch", "500ms", "autocomplete", "20ms"],
         "Elasticsearch-based with 500ms indexing latency. Autocomplete suggestions within 20ms."),
        ("Explain the Authentication Service",
         ["arch_authentication_service"],
         ["bcrypt", "cost factor 12", "RS256", "refresh tokens"],
         "Password hashing with bcrypt cost factor 12, JWT with RS256, 1-hour access tokens, 7-day refresh tokens."),
    ]
    
    for question, doc_ids, facts, answer in architecture:
        queries.append(Query(
            id=f"arch_{query_id:03d}",
            category="architecture",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    # API queries
    api_queries = [
        ("How do I list workflows via API?",
         ["api_workflows"],
         ["GET /workflows", "limit", "status"],
         "GET /workflows with optional limit and status parameters."),
        ("How do I create a workflow via API?",
         ["api_workflows"],
         ["POST /workflows", "201", "name", "definition"],
         "POST /workflows with name and definition. Returns 201 on success."),
        ("How do I cancel a running execution?",
         ["api_executions"],
         ["POST /executions/{id}/cancel", "reason"],
         "POST /executions/{id}/cancel with optional reason parameter."),
        ("How do I get current user profile?",
         ["api_users"],
         ["GET /users/me", "email", "role"],
         "GET /users/me returns user object with id, email, and role."),
        ("How do I add a team member?",
         ["api_teams"],
         ["POST /teams/{id}/members", "user_id", "role"],
         "POST /teams/{id}/members with user_id and role (owner, admin, member, viewer)."),
        ("How do I create a webhook?",
         ["api_webhooks"],
         ["POST /webhooks", "url", "events", "secret"],
         "POST /webhooks with HTTPS url, events array, and signing secret."),
        ("How do I get billing usage?",
         ["api_billing"],
         ["GET /billing/usage", "executions", "compute_minutes"],
         "GET /billing/usage returns executions, compute_minutes, and storage_gb."),
        ("How do I manage secrets?",
         ["api_secrets"],
         ["PUT /secrets/{name}", "DELETE /secrets/{name}", "encrypted at rest"],
         "PUT to create/update, DELETE to remove. Values encrypted at rest."),
    ]
    
    for question, doc_ids, facts, answer in api_queries:
        queries.append(Query(
            id=f"api_{query_id:03d}",
            category="api_reference",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    # Release/changelog queries
    release_queries = [
        ("What's new in version 3.2.0?",
         ["release_3_2_0"],
         ["Workflow versioning", "Python SDK", "async support", "v4.0"],
         "Workflow versioning with rollback, new Python SDK with async support. API v1 deprecated."),
        ("What security fixes were in v3.0.0?",
         ["release_3_0_0"],
         ["critical security issue", "token validation"],
         "Fixed critical security issue in token validation."),
        ("What breaking changes are in v3.0.0?",
         ["release_3_0_0"],
         ["Authentication API", "Legacy webhook format", "Node.js version now 18"],
         "Authentication API rewritten, legacy webhook format removed, minimum Node.js 18."),
        ("When was GraphQL API released?",
         ["release_3_1_0"],
         ["GraphQL API beta", "2023-12-01"],
         "GraphQL API was released in beta in v3.1.0 on 2023-12-01."),
    ]
    
    for question, doc_ids, facts, answer in release_queries:
        queries.append(Query(
            id=f"release_{query_id:03d}",
            category="release_notes",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    # Complex multi-hop queries
    complex_queries = [
        ("What are all the technologies used for data storage and caching?",
         ["adr_001", "adr_004", "adr_008", "arch_caching_layer", "arch_data_pipeline"],
         ["PostgreSQL", "Redis Cluster", "S3", "Ristretto", "CloudFront", "Iceberg"],
         "PostgreSQL for OLTP, Redis Cluster for caching, S3 for objects, Ristretto for local cache, CloudFront CDN, Iceberg for data lake."),
        ("How does CloudFlow ensure security across the stack?",
         ["adr_003", "arch_authentication_service", "arch_api_gateway", "howto_set_up_data_encryption"],
         ["OAuth 2.0", "JWT", "bcrypt", "AES-256", "TLS", "Zero Trust"],
         "OAuth 2.0 with JWT auth, bcrypt password hashing, AES-256 encryption at rest, TLS in transit, Zero Trust at gateway."),
        ("What configuration is needed for production deployment?",
         ["config_workflow_engine", "config_api_gateway", "config_database", "config_cache"],
         ["DATABASE_URL", "REDIS_URL", "GATEWAY_SSL_ENABLED", "DATABASE_SSL_MODE"],
         "Set DATABASE_URL, REDIS_URL, enable GATEWAY_SSL_ENABLED and DATABASE_SSL_MODE=require."),
        ("How does CloudFlow handle failures and recovery?",
         ["arch_workflow_engine", "adr_005", "howto_implement_retry_logic"],
         ["Idempotency", "retry", "exponential backoff", "dead letter queues"],
         "Idempotent operations, configurable retry with exponential backoff, dead letter queues for failed messages."),
    ]
    
    for question, doc_ids, facts, answer in complex_queries:
        queries.append(Query(
            id=f"complex_{query_id:03d}",
            category="complex_multi_hop",
            query=question,
            expected_docs=doc_ids,
            key_facts=facts,
            ground_truth_answer=answer
        ))
        query_id += 1
    
    return queries


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Generating expanded corpus...")
    
    # Generate documents
    docs = generate_documents()
    print(f"Generated {len(docs)} documents")
    
    # Write documents
    for doc in docs:
        doc_path = OUTPUT_DIR / doc.filename
        doc_path.write_text(doc.content)
    print(f"Wrote documents to {OUTPUT_DIR}")
    
    # Generate queries
    queries = generate_queries(docs)
    print(f"Generated {len(queries)} queries")
    
    # Count total facts
    total_facts = sum(len(q.key_facts) for q in queries)
    print(f"Total key facts: {total_facts}")
    
    # Verify all key facts exist in documents
    all_content = " ".join(d.content for d in docs)
    missing = []
    for q in queries:
        for fact in q.key_facts:
            if fact.lower() not in all_content.lower():
                missing.append((q.id, fact))
    
    if missing:
        print(f"\nWARNING: {len(missing)} facts not found in corpus:")
        for qid, fact in missing[:10]:
            print(f"  - {qid}: '{fact}'")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("All key facts verified in corpus!")
    
    # Write metadata
    metadata = [{"id": d.id, "title": d.title, "filename": d.filename} for d in docs]
    metadata_path = OUTPUT_DIR.parent / "corpus_metadata_expanded.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {metadata_path}")
    
    # Write ground truth
    ground_truth = {
        "description": "Expanded test corpus for RAG benchmarking",
        "version": "2.0-expanded",
        "document_count": len(docs),
        "query_count": len(queries),
        "total_key_facts": total_facts,
        "queries": [
            {
                "id": q.id,
                "category": q.category,
                "query": q.query,
                "expected_docs": q.expected_docs,
                "key_facts": q.key_facts,
                "ground_truth_answer": q.ground_truth_answer,
            }
            for q in queries
        ]
    }
    
    gt_path = OUTPUT_DIR.parent / "ground_truth_expanded.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Wrote ground truth to {gt_path}")
    
    print("\nDone!")
    print(f"  Documents: {len(docs)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Key Facts: {total_facts}")


if __name__ == "__main__":
    main()
