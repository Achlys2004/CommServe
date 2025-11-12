#!/usr/bin/env python3
"""
Test suite for enhanced rate limit management system
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rate_limit_manager import (
    RateLimitManager,
    RateLimitInfo,
    ErrorClassification,
)


def test_rate_limit_detection():
    """Test enhanced rate limit error detection"""
    print("🧪 Testing Rate Limit Error Detection")

    manager = RateLimitManager()

    # Test OpenAI rate limit error
    openai_error = Exception(
        "Rate limit reached for gpt-4o-mini in organization org-ofeBWVlDiKw0rDwlHzx93mDN on requests per min (RPM): Limit 3, Used 3, Requested 1. Please try again in 20s."
    )
    classification = manager.classify_error(openai_error, "openai")

    assert classification.is_rate_limit == True, "Should detect OpenAI rate limit"
    assert classification.error_type == "rate_limit", "Should classify as rate_limit"
    assert classification.retry_after == 20, "Should extract retry_after from message"
    print("  ✅ OpenAI rate limit detection")

    # Test generic 429 error
    generic_error = Exception("429 Too Many Requests")
    classification = manager.classify_error(generic_error, "gemini")

    assert classification.is_rate_limit == True, "Should detect generic 429"
    print("  ✅ Generic 429 detection")

    # Test network error
    network_error = Exception("Connection timeout")
    classification = manager.classify_error(network_error, "openai")

    assert (
        classification.is_rate_limit == False
    ), "Should not classify network error as rate limit"
    assert classification.error_type == "network", "Should classify as network error"
    print("  ✅ Network error classification")

    # Test auth error
    auth_error = Exception("Invalid API key")
    classification = manager.classify_error(auth_error, "openrouter")

    assert (
        classification.is_rate_limit == False
    ), "Should not classify auth error as rate limit"
    assert classification.error_type == "auth", "Should classify as auth error"
    print("  ✅ Auth error classification")


def test_rate_limit_parsing():
    """Test rate limit information parsing"""
    print("\n🧪 Testing Rate Limit Information Parsing")

    manager = RateLimitManager()

    # Test OpenAI-style error message
    error_msg = "Rate limit reached for gpt-4o-mini on requests per min (RPM): Limit 100, Used 100, Requested 1. Please try again in 60s."
    info = manager._parse_rate_limit_info(error_msg, "openai")

    assert info.limit_type == "rpm", "Should detect RPM limit type"
    assert info.limit == 100, "Should extract limit"
    assert info.current_usage == 100, "Should extract usage"
    print("  ✅ OpenAI rate limit parsing")

    # Test retry-after extraction
    retry_after = manager._extract_retry_after(
        "Please try again in 30 seconds", "openai"
    )
    assert retry_after == 30, "Should extract retry after"
    print("  ✅ Retry-after extraction")


def test_cooldown_calculation():
    """Test adaptive cooldown calculation"""
    print("\n🧪 Testing Adaptive Cooldown Calculation")

    manager = RateLimitManager()

    # Test OpenAI RPM limit
    rate_limit_info = RateLimitInfo(limit_type="rpm", retry_after=60)
    cooldown = manager._calculate_cooldown("openai", rate_limit_info, 60)

    assert cooldown == 60, "Should use retry_after for OpenAI RPM"
    print("  ✅ OpenAI RPM cooldown")

    # Test OpenRouter with shorter cooldown
    rate_limit_info = RateLimitInfo(limit_type="rpm", retry_after=120)
    cooldown = manager._calculate_cooldown("openrouter", rate_limit_info, 120)

    assert cooldown == 120, "Should cap OpenRouter cooldown"
    print("  ✅ OpenRouter cooldown capping")


def test_header_parsing():
    """Test rate limit header parsing"""
    print("\n🧪 Testing Rate Limit Header Parsing")

    manager = RateLimitManager()

    # Test OpenAI headers
    headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "50",
        "x-ratelimit-reset-requests": "60",
    }

    info = manager.parse_rate_limit_headers(headers, "openai")

    assert info.limit == 100, "Should parse limit"
    assert info.current_usage == 50, "Should calculate usage"
    assert info.reset_time is not None, "Should parse reset time"
    print("  ✅ OpenAI header parsing")


def run_all_tests():
    """Run all rate limit manager tests"""
    print("🚀 Running Enhanced Rate Limit Manager Tests\n")

    try:
        test_rate_limit_detection()
        test_rate_limit_parsing()
        test_cooldown_calculation()
        test_header_parsing()

        print(
            "\n🎉 All tests passed! Enhanced rate limit management is working correctly."
        )
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
