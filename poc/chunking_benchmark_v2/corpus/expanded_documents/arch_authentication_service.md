# Authentication Service Architecture

## Overview

The Authentication Service handles user identity, session management, and token issuance. Supports OAuth 2.0, SAML, and API key authentication.

## Design Principles

- **Zero Knowledge**: Passwords never stored in plaintext
- **Token Rotation**: Refresh tokens rotate on each use

## Components

### Identity Provider

Manages user accounts and credentials. Password hashing uses bcrypt with cost factor 12.

**Technology:** Custom Go service
**Scaling:** Horizontal stateless

### Token Service

Issues and validates access tokens. Token lifetime is 1 hour with 7-day refresh tokens.

**Technology:** JWT with RS256
**Scaling:** Horizontal stateless

### SSO Gateway

Integrates with external identity providers. Supports Google, Microsoft, Okta, and custom SAML.

**Technology:** Dex OIDC proxy
**Scaling:** Active-passive

## Data Flow

Credentials → Validation → Token Issue → Token Refresh → Logout

## Performance Characteristics

- **Latency P50:** 50ms
- **Latency P99:** 200ms
- **Throughput:** 5000 auth/second
