# ADR-002: Choose Kubernetes for Container Orchestration

## Status

Accepted

## Context

CloudFlow needs a container orchestration platform. Requirements:
- Auto-scaling based on load
- Service discovery
- Rolling deployments
- Secret management
- Multi-environment support

## Options Considered

### Option 1: Kubernetes (EKS)

Pros:
- Industry standard
- Rich ecosystem
- Auto-scaling (HPA, VPA)
- Strong community

Cons:
- Complex to operate
- Learning curve
- Can be expensive at small scale

### Option 2: AWS ECS

Pros:
- Simpler than Kubernetes
- Native AWS integration
- Lower operational overhead

Cons:
- AWS lock-in
- Less flexible
- Smaller ecosystem

### Option 3: Docker Swarm

Pros:
- Simple to set up
- Built into Docker

Cons:
- Limited features
- Smaller community
- Less active development

## Decision

We will use **Kubernetes (EKS)** for container orchestration.

## Rationale

1. Industry standard - easy to hire engineers with experience
2. Rich ecosystem of tools (Helm, Istio, Prometheus)
3. Cloud-agnostic - can migrate to GKE/AKS if needed
4. Powerful auto-scaling capabilities
5. Strong secret management with external-secrets operator

## Consequences

- Need to invest in Kubernetes training
- Higher initial setup complexity
- Need dedicated DevOps capacity
- Must establish GitOps practices
- Consider managed add-ons to reduce operational burden
