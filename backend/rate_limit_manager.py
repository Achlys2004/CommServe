"""
Enhanced Rate Limit Management System

This module provides sophisticated rate limit detection, parsing, and management
for multiple LLM providers with adaptive cooldown strategies.
"""

import time
import re
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Suppress INFO logs about fallback attempts


@dataclass
class RateLimitInfo:
    """Structured rate limit information"""

    limit_type: str  # 'rpm', 'tpm', 'daily', 'unknown'
    current_usage: Optional[int] = None
    limit: Optional[int] = None
    reset_time: Optional[float] = None
    retry_after: Optional[int] = None


@dataclass
class ErrorClassification:
    """Classified error information"""

    is_rate_limit: bool
    error_type: str  # 'rate_limit', 'quota_exceeded', 'network', 'auth', 'other'
    provider: str
    retry_after: Optional[int] = None
    cooldown_seconds: int = 30


class RateLimitManager:
    """
    Advanced rate limit manager with provider-specific handling and adaptive cooldowns
    """

    def __init__(self):
        # Rate limit patterns for different providers
        self.rate_limit_patterns = {
            "openai": [
                re.compile(r"rate.?limit", re.IGNORECASE),
                re.compile(r"429", re.IGNORECASE),
                re.compile(r"RateLimitError", re.IGNORECASE),
                re.compile(r"requests per (min|hour|day)", re.IGNORECASE),
                re.compile(r"tokens per (min|hour|day)", re.IGNORECASE),
            ],
            "gemini": [
                re.compile(r"rate.?limit", re.IGNORECASE),
                re.compile(r"429", re.IGNORECASE),
                re.compile(r"quota.?exceeded", re.IGNORECASE),
                re.compile(r"resource.exhausted", re.IGNORECASE),
            ],
            "openrouter": [
                re.compile(r"rate.?limit", re.IGNORECASE),
                re.compile(r"429", re.IGNORECASE),
                re.compile(r"insufficient.?balance", re.IGNORECASE),
            ],
        }

        # Provider-specific cooldown strategies
        self.cooldown_strategies = {
            "openai": {
                "rpm": lambda info: min(info.retry_after or 60, 300),  # Max 5 minutes
                "tpm": lambda info: min(info.retry_after or 60, 300),
                "daily": lambda info: min(
                    info.retry_after or 3600, 86400
                ),  # Max 24 hours
                "default": 60,
            },
            "gemini": {
                "rpm": lambda info: min(info.retry_after or 60, 300),
                "daily": lambda info: min(info.retry_after or 3600, 86400),
                "default": 60,
            },
            "openrouter": {
                "rpm": lambda info: min(info.retry_after or 30, 180),  # More aggressive
                "default": 30,
            },
        }

    def classify_error(self, error: Exception, provider: str) -> ErrorClassification:
        """
        Classify an error and determine if it's a rate limit with detailed information

        Args:
            error: The exception that occurred
            provider: The provider that generated the error ('openai', 'gemini', 'openrouter')

        Returns:
            ErrorClassification with detailed error information
        """
        error_name = type(error).__name__
        error_msg = str(error).lower()

        # Check for rate limit indicators
        is_rate_limit = self._is_rate_limit_error(error_name, error_msg, provider)

        if not is_rate_limit:
            # Check for other error types
            if any(term in error_msg for term in ["timeout", "connection", "network"]):
                return ErrorClassification(
                    is_rate_limit=False,
                    error_type="network",
                    provider=provider,
                    cooldown_seconds=20,
                )
            elif any(
                term in error_msg
                for term in [
                    "auth",
                    "unauthorized",
                    "invalid.api.key",
                    "invalid api key",
                    "authentication",
                    "api key",
                ]
            ):
                return ErrorClassification(
                    is_rate_limit=False,
                    error_type="auth",
                    provider=provider,
                    cooldown_seconds=300,  # Don't retry auth errors quickly
                )
            else:
                return ErrorClassification(
                    is_rate_limit=False,
                    error_type="other",
                    provider=provider,
                    cooldown_seconds=30,
                )

        # It's a rate limit error - extract detailed information
        rate_limit_info = self._parse_rate_limit_info(error_msg, provider)
        retry_after = self._extract_retry_after(error_msg, provider)

        # Determine cooldown based on provider and limit type
        cooldown = self._calculate_cooldown(provider, rate_limit_info, retry_after)

        return ErrorClassification(
            is_rate_limit=True,
            error_type="rate_limit",
            provider=provider,
            retry_after=retry_after,
            cooldown_seconds=cooldown,
        )

    def _is_rate_limit_error(
        self, error_name: str, error_msg: str, provider: str
    ) -> bool:
        """Check if an error is a rate limit error"""
        # Check error name
        if "ratelimit" in error_name.lower() or "rate_limit" in error_name.lower():
            return True

        # Check error message against provider-specific patterns
        if provider in self.rate_limit_patterns:
            for pattern in self.rate_limit_patterns[provider]:
                if pattern.search(error_msg):
                    return True

        # Generic rate limit indicators
        generic_indicators = [
            "429",
            "too many requests",
            "rate limit exceeded",
            "quota exceeded",
            "resource exhausted",
        ]

        return any(indicator in error_msg for indicator in generic_indicators)

    def _parse_rate_limit_info(self, error_msg: str, provider: str) -> RateLimitInfo:
        """Parse detailed rate limit information from error message"""
        info = RateLimitInfo(limit_type="unknown")

        # Extract limit type
        if "requests per min" in error_msg or "rpm" in error_msg:
            info.limit_type = "rpm"
        elif "tokens per min" in error_msg or "tpm" in error_msg:
            info.limit_type = "tpm"
        elif "per day" in error_msg or "daily" in error_msg:
            info.limit_type = "daily"

        # Extract usage and limits using regex
        usage_match = re.search(r"used[:\s]+(\d+)", error_msg, re.IGNORECASE)
        if usage_match:
            info.current_usage = int(usage_match.group(1))

        limit_match = re.search(r"limit[:\s]+(\d+)", error_msg, re.IGNORECASE)
        if limit_match:
            info.limit = int(limit_match.group(1))

        return info

    def _extract_retry_after(self, error_msg: str, provider: str) -> Optional[int]:
        """Extract retry-after time from error message"""
        # Provider-specific retry extraction first (most reliable)
        if provider == "openai":
            # OpenAI often says "try again in X seconds" or "in Xs"
            openai_match = re.search(
                r"(?:try again in|in)\s+(\d+)\s*s(?:econds?)?", error_msg, re.IGNORECASE
            )
            if openai_match:
                return int(openai_match.group(1))

        # Look for explicit retry-after in seconds
        retry_match = re.search(
            r"retry.?after:?\s*(\d+)\s*seconds?", error_msg, re.IGNORECASE
        )
        if retry_match:
            return int(retry_match.group(1))

        # Look for time patterns like "20s", "1m", "1h" but be more specific
        # Only match if preceded by "in" or "after" or similar time-related words
        time_match = re.search(
            r"(?:in|after|wait)\s+(\d+)\s*([smhd])(?:\s|$)", error_msg, re.IGNORECASE
        )
        if time_match:
            value, unit = int(time_match.group(1)), time_match.group(2).lower()
            multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
            return value * multipliers.get(unit, 1)

        return None

    def _calculate_cooldown(
        self, provider: str, rate_limit_info: RateLimitInfo, retry_after: Optional[int]
    ) -> int:
        """Calculate appropriate cooldown period"""
        if provider not in self.cooldown_strategies:
            return retry_after or 60

        strategy = self.cooldown_strategies[provider]

        # Use retry_after if available
        if retry_after:
            return min(retry_after, 300)  # Cap at 5 minutes

        # Use limit-type specific strategy
        if rate_limit_info.limit_type in strategy:
            return strategy[rate_limit_info.limit_type](rate_limit_info)

        # Default strategy
        return strategy.get("default", 60)

    def parse_rate_limit_headers(
        self, response_headers: Dict[str, Any], provider: str
    ) -> RateLimitInfo:
        """
        Parse rate limit information from HTTP response headers

        Args:
            response_headers: HTTP response headers
            provider: The provider ('openai', 'gemini', 'openrouter')

        Returns:
            RateLimitInfo with parsed header data
        """
        info = RateLimitInfo(limit_type="unknown")

        if provider == "openai":
            # OpenAI headers: x-ratelimit-limit-requests, x-ratelimit-remaining-requests, etc.
            info.limit = self._extract_header_int(
                response_headers, "x-ratelimit-limit-requests"
            )
            remaining = self._extract_header_int(
                response_headers, "x-ratelimit-remaining-requests"
            )
            if info.limit and remaining is not None:
                info.current_usage = info.limit - remaining

            # Reset time
            reset_time = self._extract_header_int(
                response_headers, "x-ratelimit-reset-requests"
            )
            if reset_time:
                info.reset_time = time.time() + reset_time

        elif provider == "openrouter":
            # OpenRouter headers
            info.limit = self._extract_header_int(
                response_headers, "x-ratelimit-limit-requests"
            )
            remaining = self._extract_header_int(
                response_headers, "x-ratelimit-remaining-requests"
            )
            if info.limit and remaining is not None:
                info.current_usage = info.limit - remaining

        # Add retry-after header if present
        retry_after = self._extract_header_int(response_headers, "retry-after")
        if retry_after:
            info.retry_after = retry_after

        return info

    def _extract_header_int(
        self, headers: Dict[str, Any], header_name: str
    ) -> Optional[int]:
        """Extract integer value from header"""
        value = headers.get(header_name)
        if value is not None:
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
        return None

    def should_prevent_request(
        self, provider: str, current_tier_status: Dict
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if a request should be prevented based on current rate limit status

        Args:
            provider: The provider to check
            current_tier_status: Current status of the tier

        Returns:
            Tuple of (should_prevent, cooldown_remaining_seconds)
        """
        # Check if tier is in cooldown
        cooldown_until = current_tier_status.get("cooldown_until", 0)
        current_time = time.time()

        if current_time < cooldown_until:
            remaining = int(cooldown_until - current_time)
            return True, remaining

        return False, None


# Global rate limit manager instance

# Global rate limit manager instance
rate_limit_manager = RateLimitManager()
