"""
LLM configuration constants for consistent token limits across the codebase.
"""


# Token limits for different use cases
class LLMTokenLimits:
    """Standardized token limits for LLM calls based on use case."""

    # Short responses (summaries, classifications, simple queries)
    SHORT = 150

    # Default for general purpose calls
    DEFAULT = 256

    # SQL query generation
    SQL = 400

    # Medium responses (detailed summaries, contextual answers)
    MEDIUM = 500

    # Long responses (chunked answers, detailed explanations)
    LONG = 600

    # Code generation (Python scripts, complex logic)
    CODE = 800


# Token limit mapping by function type
TOKEN_LIMITS = {
    "summary": LLMTokenLimits.SHORT,  # Natural language summaries
    "sql": LLMTokenLimits.SQL,  # SQL generation
    "context_summary": LLMTokenLimits.MEDIUM,  # Context summarization
    "answer": LLMTokenLimits.LONG,  # Full answers
    "code": LLMTokenLimits.CODE,  # Code generation
    "default": LLMTokenLimits.DEFAULT,  # Fallback
}


def get_token_limit(purpose: str) -> int:
    """
    Get appropriate token limit for a given purpose.

    Args:
        purpose: Type of LLM call (summary, sql, code, etc.)

    Returns:
        Appropriate token limit
    """
    return TOKEN_LIMITS.get(purpose.lower(), LLMTokenLimits.DEFAULT)
