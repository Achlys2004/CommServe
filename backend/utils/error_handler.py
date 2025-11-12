"""
Error handling utilities for consistent exception management across the codebase.
"""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def handle_errors(
    default_return=None,
    log_level="exception",
    error_message=None,
):
    """
    Decorator for consistent error handling across functions.

    Args:
        default_return: Value to return on error (default: None)
        log_level: Logging level - 'exception', 'warning', or 'error' (default: 'exception')
        error_message: Custom error message prefix (default: function name)

    Usage:
        @handle_errors(default_return="No data", log_level="warning")
        def my_function():
            # function code
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Determine log level
                log_func = getattr(logger, log_level, logger.exception)

                # Build error message
                msg = error_message or f"{func.__name__} failed"
                log_func(f"{msg}: {e}")

                return default_return

        return wrapper

    return decorator
