#!/usr/bin/env python3
"""Test the magic system prompt solution."""

import json
from pathlib import Path
import httpx

# Load auth
auth_path = Path.home() / ".local/share/opencode/auth.json"
with open(auth_path) as f:
    auth_data = json.load(f)["anthropic"]

# Test WITH the magic system prompt
print("Testing Sonnet WITH magic system prompt...")
response = httpx.post(
    "https://api.anthropic.com/v1/messages",
    json={
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "system": [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            }
        ],
        "messages": [{"role": "user", "content": "Say 'SUCCESS' and nothing else."}],
    },
    headers={
        "Authorization": f"Bearer {auth_data['access']}",
        "anthropic-beta": "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    },
    timeout=30,
)

print(f"Status: {response.status_code}")
if response.is_success:
    data = response.json()
    content = data.get("content", [{}])[0].get("text", "")
    print(f"✅ SUCCESS! Sonnet responded: {content}")
    print(f"\nFull response: {json.dumps(data, indent=2)}")
else:
    print(f"❌ FAILED: {response.json()}")
