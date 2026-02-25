#!/usr/bin/env python3
"""Detailed test of Anthropic API to understand model restrictions."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx
from enrichment.provider import AnthropicProvider
from logger import get_logger


def test_raw_api_call(model_id: str, access_token: str) -> dict:
    """Make a raw API call to test model directly."""
    log = get_logger()

    print(f"\n{'=' * 60}")
    print(f"RAW API TEST: {model_id}")
    print(f"{'=' * 60}")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say 'test' and nothing else."}],
    }

    print(f"URL: {url}")
    print(f"Model: {model_id}")
    print(
        f"Headers: {json.dumps({k: v[:20] + '...' if k == 'Authorization' else v for k, v in headers.items()}, indent=2)}"
    )

    try:
        response = httpx.post(url, json=body, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")

        if response.is_success:
            data = response.json()
            content = data.get("content", [{}])[0].get("text", "")
            print(f"✅ SUCCESS")
            print(f"Response: {content[:100]}")
            return {"success": True, "response": content}
        else:
            error_data = response.json()
            print(f"❌ FAILED")
            print(f"Error: {json.dumps(error_data, indent=2)}")
            return {"success": False, "error": error_data}

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return {"success": False, "error": str(e)}


def test_with_different_beta_headers(model_id: str, access_token: str):
    """Test different anthropic-beta header combinations."""
    log = get_logger()

    print(f"\n{'=' * 60}")
    print(f"BETA HEADER TEST: {model_id}")
    print(f"{'=' * 60}")

    beta_variants = [
        "oauth-2025-04-20",
        "claude-code-20250219,oauth-2025-04-20",
        "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14",
        "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
        "",  # No beta header
    ]

    url = "https://api.anthropic.com/v1/messages"
    body = {
        "model": model_id,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}],
    }

    results = {}
    for beta in beta_variants:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if beta:
            headers["anthropic-beta"] = beta

        label = beta if beta else "(no beta header)"
        print(f"\nTesting: {label}")

        try:
            response = httpx.post(url, json=body, headers=headers, timeout=30)
            if response.is_success:
                print(f"  ✅ SUCCESS")
                results[label] = "SUCCESS"
            else:
                error = response.json().get("error", {})
                error_type = error.get("type", "unknown")
                error_msg = error.get("message", "")[:80]
                print(f"  ❌ {error_type}: {error_msg}")
                results[label] = f"FAILED: {error_type}"
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)[:80]}")
            results[label] = f"EXCEPTION: {str(e)[:80]}"

    return results


def main():
    """Run comprehensive Anthropic API tests."""
    log = get_logger()

    # Load auth
    auth_path = Path.home() / ".local/share/opencode/auth.json"
    if not auth_path.exists():
        print(f"❌ Auth file not found: {auth_path}")
        sys.exit(1)

    with open(auth_path) as f:
        auth_data = json.load(f)

    if "anthropic" not in auth_data:
        print("❌ No Anthropic credentials in auth file")
        sys.exit(1)

    anthropic_auth = auth_data["anthropic"]
    access_token = anthropic_auth["access"]

    print("=" * 60)
    print("ANTHROPIC API COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"Auth type: {anthropic_auth.get('type')}")
    print(f"Token expires: {anthropic_auth.get('expires')}")

    # Models to test
    models = [
        ("Haiku", "claude-3-5-haiku-latest"),
        ("Sonnet 4", "claude-sonnet-4-20250514"),
        ("Sonnet 4.5", "claude-sonnet-4-5-20250929"),
        ("Opus 4", "claude-opus-4-20250514"),
    ]

    # Test 1: Raw API calls
    print("\n" + "=" * 60)
    print("TEST 1: RAW API CALLS")
    print("=" * 60)

    raw_results = {}
    for name, model_id in models:
        result = test_raw_api_call(model_id, access_token)
        raw_results[name] = result

    # Test 2: Beta header variations (only for failed models)
    print("\n" + "=" * 60)
    print("TEST 2: BETA HEADER VARIATIONS")
    print("=" * 60)

    beta_results = {}
    for name, model_id in models:
        if not raw_results[name]["success"]:
            print(f"\nTesting {name} with different beta headers...")
            beta_results[name] = test_with_different_beta_headers(
                model_id, access_token
            )

    # Test 3: Provider class
    print("\n" + "=" * 60)
    print("TEST 3: PROVIDER CLASS")
    print("=" * 60)

    provider = AnthropicProvider()
    provider_results = {}

    for name, model_id in models:
        print(f"\nTesting {name} via provider...")
        response = provider.generate("Say 'test'", model_id, timeout=30)
        if response:
            print(f"  ✅ SUCCESS: {response[:50]}")
            provider_results[name] = "SUCCESS"
        else:
            print(f"  ❌ FAILED (empty response)")
            provider_results[name] = "FAILED"

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nRAW API RESULTS:")
    for name, result in raw_results.items():
        status = "✅ WORKS" if result["success"] else "❌ FAILED"
        print(f"  {status} - {name}")
        if not result["success"] and "error" in result:
            error = result["error"]
            if isinstance(error, dict):
                error_msg = error.get("error", {}).get("message", "")
                print(f"    Error: {error_msg[:100]}")

    if beta_results:
        print("\nBETA HEADER RESULTS:")
        for name, results in beta_results.items():
            print(f"\n  {name}:")
            for beta, status in results.items():
                print(f"    {beta[:50]}: {status}")

    print("\nPROVIDER CLASS RESULTS:")
    for name, status in provider_results.items():
        symbol = "✅" if status == "SUCCESS" else "❌"
        print(f"  {symbol} {name}: {status}")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    working_models = [name for name, result in raw_results.items() if result["success"]]
    failed_models = [
        name for name, result in raw_results.items() if not result["success"]
    ]

    if working_models:
        print(f"\n✅ Working models: {', '.join(working_models)}")
    if failed_models:
        print(f"\n❌ Failed models: {', '.join(failed_models)}")

        # Check if all failures are due to Claude Code restriction
        all_claude_code_restricted = all(
            "Claude Code" in str(raw_results[name].get("error", ""))
            for name in failed_models
        )

        if all_claude_code_restricted:
            print("\n⚠️  All failures are due to 'Claude Code' credential restriction")
            print("    This OAuth token is restricted to Haiku models only")
            print(
                "    To use Sonnet/Opus: Need full Claude Pro/Max subscription OAuth token"
            )


if __name__ == "__main__":
    main()
