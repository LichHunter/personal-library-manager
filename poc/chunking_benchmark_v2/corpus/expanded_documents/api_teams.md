# Teams API Reference

## Overview

The Teams API provides programmatic access to teams functionality.
Base URL: `https://api.cloudflow.io/v1/teams`

## Authentication

All requests require a valid API key in the `Authorization` header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### GET /teams

List teams the user belongs to.

**Parameters:**


**Response:**
- Success: `200` - Returns array of team objects
- Error: `400` - Bad Request

**Example:**
```json
{"teams": [{"id": "team_01", "name": "Engineering", "member_count": 15}]}
```

### POST /teams/{id}/members

Add a member to a team.

**Parameters:**

- `user_id` (string): User ID to add
- `role` (string): Role: owner, admin, member, viewer

**Response:**
- Success: `201` - Member added
- Error: `400` - Bad Request

**Example:**
```json
{"team_id": "team_01", "user_id": "usr_002", "role": "member"}
```
