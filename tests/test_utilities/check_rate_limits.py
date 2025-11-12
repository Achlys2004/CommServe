"""
Quick script to check API rate limit status
"""

import time
from backend.llm_client import _key_status, TIER_KEYS


def check_rate_limits():
    print("=" * 60)
    print("API KEY RATE LIMIT STATUS")
    print("=" * 60)

    current_time = time.time()

    for tier, status in _key_status.items():
        tier_info = TIER_KEYS.get(tier, {})
        provider = tier_info.get("provider", "unknown")
        has_key = bool(tier_info.get("key"))

        print(f"\n{tier.upper()} ({provider}):")
        print(f"  ├─ Key Configured: {'✓ Yes' if has_key else '✗ No'}")
        print(f"  ├─ Available: {'✓ Yes' if status['available'] else '✗ No'}")
        print(f"  ├─ Failure Count: {status['failure_count']}")

        cooldown = status["cooldown_until"]
        if cooldown > current_time:
            remaining = int(cooldown - current_time)
            print(f"  └─ Cooldown: ⏳ {remaining}s remaining")
        else:
            print(f"  └─ Cooldown: ✓ None")

    print("\n" + "=" * 60)
    print("\nRate Limit Indicators:")
    print("  • If 'Available' = No → That tier is in cooldown")
    print("  • If 'Failure Count' > 0 → Recent errors occurred")
    print("  • If 'Cooldown' shows time → Wait before retrying")
    print("\nTips:")
    print("  • Tier 1 (OpenAI) has strict rate limits on free tier")
    print("  • System auto-falls back to Tier 2 (Gemini) → Tier 3 (OpenRouter)")
    print("  • Check logs for detailed error messages")
    print("=" * 60)


if __name__ == "__main__":
    check_rate_limits()
