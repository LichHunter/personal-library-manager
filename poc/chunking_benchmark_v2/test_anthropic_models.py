#!/usr/bin/env python3
"""Test Anthropic model connections using chunking v2 provider."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from enrichment.provider import AnthropicProvider
from logger import get_logger


def test_model(provider: AnthropicProvider, model_name: str, model_id: str) -> bool:
    """Test a specific Anthropic model."""
    log = get_logger()

    print(f"\n{'=' * 60}")
    print(f"Testing {model_name}: {model_id}")
    print(f"{'=' * 60}")

    test_prompt = "Say 'Hello! I am working.' and nothing else."

    try:
        response = provider.generate(test_prompt, model_id, timeout=30)

        if response:
            print(f"✅ SUCCESS - {model_name} connected")
            print(f"Response: {response[:100]}")
            return True
        else:
            print(f"❌ FAILED - {model_name} returned empty response")
            return False

    except Exception as e:
        print(f"❌ ERROR - {model_name} failed with exception: {e}")
        return False


def main():
    """Test all Anthropic models."""
    log = get_logger()

    print("Initializing Anthropic Provider...")
    try:
        provider = AnthropicProvider()
        print("✅ Provider initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize provider: {e}")
        sys.exit(1)

    # Test models
    models_to_test = [
        ("Haiku", "claude-haiku"),
        ("Haiku (full ID)", "claude-3-5-haiku-latest"),
        ("Sonnet", "claude-sonnet"),
        ("Sonnet (full ID)", "claude-sonnet-4-20250514"),
        ("Opus", "claude-opus"),
        ("Opus (full ID)", "claude-opus-4-20250514"),
    ]

    results = {}
    for model_name, model_id in models_to_test:
        results[model_name] = test_model(provider, model_name, model_id)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for model_name, success in results.items():
        status = "✅ WORKS" if success else "❌ FAILED"
        print(f"{status} - {model_name}")

    total_tests = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total_tests} passed")


if __name__ == "__main__":
    main()
