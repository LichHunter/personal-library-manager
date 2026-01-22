# ADR-003: API Authentication Strategy

## Status

Accepted

## Context

We need to implement authentication for the CloudFlow API. Requirements:
- Secure token-based authentication
- Support for API keys (machine-to-machine)
- Single sign-on capability
- Token refresh mechanism

## Options Considered

### Option 1: JWT with OAuth 2.0

Pros:
- Industry standard
- Stateless verification
- Works with SSO providers
- Supports refresh tokens

Cons:
- Token revocation is complex
- Larger token size

### Option 2: Session-based with Redis

Pros:
- Easy revocation
- Small token size

Cons:
- Requires session storage
- Not as scalable
- Doesn't work well with microservices

### Option 3: API Keys only

Pros:
- Simple implementation
- Easy to understand

Cons:
- No user identity
- Hard to rotate
- No SSO support

## Decision

We will use **JWT with OAuth 2.0** for authentication.

## Implementation Details

### Token Structure

```json
{
  "sub": "user_id",
  "email": "user@example.com",
  "role": "editor",
  "iat": 1234567890,
  "exp": 1234571490,
  "iss": "cloudflow"
}
```

### Token Lifetimes

- Access token: 1 hour
- Refresh token: 7 days

### API Keys

For machine-to-machine auth, we'll also support API keys:
- Stored hashed in database
- Scoped to specific permissions
- Can be rotated without downtime

## Consequences

- Need to implement token refresh flow
- Must handle token revocation via short expiry + deny list
- Need to integrate with identity provider (Auth0)
- API keys require separate management interface
