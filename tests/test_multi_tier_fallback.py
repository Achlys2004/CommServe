#!/usr/bin/env python3
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from backend.llm_client import (
        call_llm,
        get_tier_status,
        reset_tier_failures,
        force_tier_cooldown,
    )
except ImportError as e:
    pytest.skip(
        f"Skipping test due to missing dependencies: {e}", allow_module_level=True
    )

import json
from datetime import datetime


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print("=" * 80)
    else:
        print("-" * 80)


def print_tier_status():
    """Print the current status of all tiers"""
    print_separator("🔍 Current Tier Status")
    status = get_tier_status()

    print("\n📊 Multi-Tier Configuration:")
    for tier in ["tier1", "tier2", "tier3"]:
        info = status[tier]
        provider = info.get("provider", "unknown")
        print(f"\n  {tier.upper()} ({provider.upper()}): {info['status']}")
        print(f"    ├─ Configured: {'✓' if info['configured'] else '✗'}")
        print(f"    ├─ Available: {'✓' if info['available'] else '✗'}")
        print(f"    ├─ In Cooldown: {'Yes' if info['in_cooldown'] else 'No'}")
        if info["in_cooldown"]:
            print(f"    ├─ Cooldown Remaining: {info['cooldown_remaining_seconds']}s")
        print(f"    └─ Failure Count: {info['failure_count']}")


def test_normal_call():
    """Test a normal LLM call with tier1"""
    print_separator("✅ Test 1: Normal Call (Should use Tier 1)")

    result = call_llm("Say 'Hello from tier1' in one sentence", max_tokens=50)
    print(f"\n📝 Response: {result}\n")

    print_tier_status()


def test_tier_failover():
    """Test failover when tier1 is in cooldown"""
    print_separator("🔄 Test 2: Tier Failover (Simulate Tier 1 Failure)")

    # Simulate tier1 being rate limited
    print("\n⚠️  Simulating tier1 rate limit (60s cooldown)...")
    force_tier_cooldown("tier1", 60)

    print_tier_status()

    print("\n📞 Making LLM call (should fallback to tier2 or alternative providers)...")
    result = call_llm("Say 'Hello from fallback' in one sentence", max_tokens=50)
    print(f"\n📝 Response: {result}\n")

    print_tier_status()


def test_recovery():
    """Test recovery by resetting failures"""
    print_separator("🔄 Test 3: Recovery (Reset All Failures)")

    print("\n🔧 Resetting all tier failures and cooldowns...")
    reset_tier_failures()

    print_tier_status()

    print("\n📞 Making LLM call (should use tier1 again)...")
    result = call_llm("Say 'Hello after recovery' in one sentence", max_tokens=50)
    print(f"\n📝 Response: {result}\n")


def test_multiple_calls_rapid():
    """Test multiple rapid calls to trigger rate limits"""
    print_separator("⚡ Test 4: Rapid Fire (Test Rate Limit Handling)")

    print("\n📞 Making 5 rapid LLM calls...")
    for i in range(5):
        print(f"\n  Call {i+1}/5...")
        result = call_llm(f"Count to {i+1}", max_tokens=30)
        result_str = str(result) if result is not None else "None"
        print(f"    Response: {result_str[:100]}...")

    print_tier_status()


def test_status_export():
    """Export tier status as JSON"""
    print_separator("📤 Test 5: Export Status as JSON")

    status = get_tier_status()
    print("\n" + json.dumps(status, indent=2))


def main():
    """Run all tests"""
    print_separator("🚀 Multi-Tier API Key Fallback System Test")
    print(f"\n⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Test 1: Normal operation
        test_normal_call()
        input("\nPress Enter to continue to Test 2...")

        # Test 2: Failover behavior
        test_tier_failover()
        input("\nPress Enter to continue to Test 3...")

        # Test 3: Recovery
        test_recovery()
        input("\nPress Enter to continue to Test 4...")

        # Test 4: Rapid calls
        test_multiple_calls_rapid()
        input("\nPress Enter to continue to Test 5...")

        # Test 5: Status export
        test_status_export()

        print_separator("✅ All Tests Complete")
        print(f"\n⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during tests: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
