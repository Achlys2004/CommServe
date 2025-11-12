import os, re, logging, time
import threading
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from .rate_limit_manager import rate_limit_manager, ErrorClassification

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Multi-tier API key configuration
# Tier 1: OpenRouter (Primary)
# Tier 2: OpenAI (Fallback)
# Tier 3: Gemini (Emergency)
TIER_KEYS = {
    "tier1": {"key": os.getenv("OPENROUTER_API_KEY"), "provider": "openrouter"},
    "tier2": {"key": os.getenv("OPENAI_API_KEY"), "provider": "openai"},
    "tier3": {
        "key": os.getenv("GEMINI_API_KEY"),
        "provider": "gemini",
    },
}

OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.0-flash"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

# OpenAI text-embedding-3-small produces 1536-dimensional embeddings
FALLBACK_EMBED_DIM = int(os.getenv("FALLBACK_EMBED_DIM", "1536"))

_tier_lock = threading.Lock()

# Track which tiers are working
_key_status = {
    "tier1": {
        "available": bool(TIER_KEYS["tier1"]["key"]),
        "cooldown_until": 0,
        "failure_count": 0,
    },
    "tier2": {
        "available": bool(TIER_KEYS["tier2"]["key"]),
        "cooldown_until": 0,
        "failure_count": 0,
    },
    "tier3": {
        "available": bool(TIER_KEYS["tier3"]["key"]),
        "cooldown_until": 0,
        "failure_count": 0,
    },
}

# Create OpenRouter client for tier1 (Primary)
_openrouter_client = None
if TIER_KEYS["tier1"]["key"]:
    _openrouter_client = OpenAI(
        api_key=TIER_KEYS["tier1"]["key"],
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
        max_retries=0,  # Disable internal retries to allow our fallback logic
    )

# Create OpenAI client for tier2 (Fallback)
_openai_client = None
if TIER_KEYS["tier2"]["key"]:
    _openai_client = OpenAI(
        api_key=TIER_KEYS["tier2"]["key"],
        timeout=60.0,
        max_retries=0,  # Disable internal retries to allow our fallback logic
    )

# Create Gemini client for tier3 (Emergency)
_gemini_configured = False
if TIER_KEYS["tier3"]["key"]:
    genai.configure(api_key=TIER_KEYS["tier3"]["key"])  # type: ignore
    _gemini_configured = True


def _get_next_available_tier():
    current_time = time.time()

    with _tier_lock:
        for tier in ["tier1", "tier2", "tier3"]:
            status = _key_status[tier]
            tier_config = TIER_KEYS[tier]
            provider = tier_config["provider"]

            # Check if tier is available and not in cooldown
            if (
                status["available"]
                and tier_config["key"]
                and current_time >= status["cooldown_until"]
            ):
                # Additional proactive rate limit check
                should_prevent, cooldown_remaining = (
                    rate_limit_manager.should_prevent_request(provider, status)
                )
                if should_prevent:
                    logger.warning(
                        f"Tier {tier} proactively blocked due to rate limit. "
                        f"Cooldown remaining: {cooldown_remaining}s"
                    )
                    continue

                return tier

        return None


def _mark_tier_failed(tier, cooldown_seconds=30):
    with _tier_lock:
        if tier in _key_status:
            _key_status[tier]["cooldown_until"] = time.time() + cooldown_seconds
            _key_status[tier]["failure_count"] += 1
            logger.debug(  # Changed from warning to debug
                f"Tier {tier} marked as failed. Cooldown: {cooldown_seconds}s. Total failures: {_key_status[tier]['failure_count']}"
            )


def _mark_tier_success(tier):
    with _tier_lock:
        if tier in _key_status:
            _key_status[tier]["failure_count"] = 0
            _key_status[tier]["cooldown_until"] = 0


def _update_tier_rate_limit_info(tier, provider, rate_limit_data):
    try:
        if provider == "openai" and hasattr(rate_limit_data, "usage"):
            usage = rate_limit_data.usage
            if usage:
                logger.debug(f"OpenAI usage for {tier}: {usage.total_tokens} tokens")
        elif hasattr(rate_limit_data, "limit_type"):  # RateLimitInfo object
            if rate_limit_data.reset_time:
                current_time = time.time()
                time_to_reset = rate_limit_data.reset_time - current_time
                if time_to_reset > 0 and time_to_reset < 60:  # Less than 1 minute
                    logger.info(
                        f"Proactive cooldown for {tier}: {time_to_reset:.1f}s until reset"
                    )
                    _mark_tier_failed(tier, int(time_to_reset))
    except Exception as e:
        logger.debug(f"Failed to update rate limit info for {tier}: {e}")


if PROVIDER == "google":
    if not TIER_KEYS["tier2"]["key"]:
        raise ValueError("Missing GEMINI_API_KEY in environment.")
elif PROVIDER == "openai":
    if not any(tier["key"] for tier in TIER_KEYS.values()):
        raise ValueError(
            "No API keys found. Please provide at least one of: OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY"
        )
    configured_tiers = [tier for tier, config in TIER_KEYS.items() if config["key"]]
    logger.info(f"Configured tiers: {configured_tiers}")
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {PROVIDER}")


def _call_openai_with_fallback(messages, max_tokens=256, retry_count=3):
    last_error = None
    attempts_log = []
    successful_provider = None

    # Try each tier with smart retry logic
    for attempt in range(retry_count):
        tier = _get_next_available_tier()

        if tier:
            tier_config = TIER_KEYS[tier]
            provider = tier_config["provider"]

            try:
                logger.debug(
                    f"Attempting {provider} call with {tier} (attempt {attempt + 1}/{retry_count})"
                )

                if provider == "openai":
                    if _openai_client is None:
                        raise ValueError(f"OpenAI client not configured for {tier}")
                    completion = _openai_client.chat.completions.create(
                        model=OPENAI_MODEL, messages=messages, max_tokens=max_tokens
                    )
                    content = completion.choices[0].message.content

                    # Parse rate limit headers if available
                    if hasattr(completion, "usage") and completion.usage:
                        # Store rate limit info for proactive management
                        _update_tier_rate_limit_info(tier, provider, completion)

                elif provider == "gemini":
                    model = genai.GenerativeModel(GEMINI_MODEL)  # type: ignore
                    prompt_parts = []
                    for msg in messages:
                        if msg["role"] == "system":
                            prompt_parts.append(f"System: {msg['content']}")
                        elif msg["role"] == "user":
                            prompt_parts.append(f"User: {msg['content']}")
                    full_prompt = "\n\n".join(prompt_parts)
                    response = model.generate_content(full_prompt)
                    content = (
                        response.text
                        if response and hasattr(response, "text")
                        else None
                    )
                    if not content:
                        raise ValueError("Empty response from Gemini")

                elif provider == "openrouter":
                    if _openrouter_client is None:
                        raise ValueError(f"OpenRouter client not configured for {tier}")
                    completion = _openrouter_client.chat.completions.create(
                        model=OPENROUTER_MODEL, messages=messages, max_tokens=max_tokens
                    )
                    content = completion.choices[0].message.content

                    # Note: OpenRouter headers not directly accessible via OpenAI client
                    # Rate limit parsing would need custom HTTP client implementation
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                _mark_tier_success(tier)
                successful_provider = f"{tier} ({provider})"
                logger.info(f"✓ Success with {successful_provider}")
                content = content.strip() if content is not None else "Empty content"

                # Additional safeguard: Clean any invalid starting text from LLM responses
                if content and content != "Empty content":
                    lines = content.split("\n")
                    if lines:
                        first_line = lines[0].strip()
                        # If first line doesn't start with valid content, try to find the first valid line
                        if first_line and not any(
                            first_line.startswith(prefix)
                            for prefix in [
                                "import ",
                                "from ",
                                "def ",
                                "class ",
                                "if ",
                                "try:",
                                "with ",
                                "@",
                                "async ",
                                "for ",
                                "while ",
                                "print(",
                                "raise ",
                                "return ",
                                "yield ",
                                "global ",
                                "nonlocal ",
                                "assert ",
                                "break",
                                "continue",
                                "pass",
                                "#",
                            ]
                        ):
                            # Look for the first line that starts with valid content
                            for i, line in enumerate(lines):
                                line_stripped = line.strip()
                                if line_stripped and any(
                                    line_stripped.startswith(prefix)
                                    for prefix in [
                                        "import ",
                                        "from ",
                                        "def ",
                                        "class ",
                                        "if ",
                                        "try:",
                                        "with ",
                                        "@",
                                        "async ",
                                        "for ",
                                        "while ",
                                        "print(",
                                        "raise ",
                                        "return ",
                                        "yield ",
                                        "global ",
                                        "nonlocal ",
                                        "assert ",
                                        "break",
                                        "continue",
                                        "pass",
                                        "#",
                                    ]
                                ):
                                    content = "\n".join(lines[i:])
                                    logger.warning(
                                        f"Cleaned invalid starting text from {provider} response"
                                    )
                                    break

                return content

            except Exception as e:
                error_name = type(e).__name__
                error_msg = str(e)
                attempts_log.append(f"{tier}({provider}): {error_name}")
                last_error = e

                # Enhanced logging for debugging
                logger.warning(
                    f"✗ {tier} ({provider}) failed: {error_name} - {error_msg[:100]}"
                )

                error_classification = rate_limit_manager.classify_error(e, provider)

                if error_classification.is_rate_limit:
                    cooldown = error_classification.cooldown_seconds
                    logger.debug(
                        f"X {tier} hit rate limit ({error_classification.error_type}). "
                        f"Cooldown: {cooldown}s"
                    )
                    # For rate limits, don't retry the same tier - switch immediately
                    _mark_tier_failed(tier, cooldown)
                    continue
                elif error_classification.error_type == "network":
                    cooldown = error_classification.cooldown_seconds
                    logger.debug(f"X {tier} network error. Cooldown: {cooldown}s")
                elif error_classification.error_type == "auth":
                    # Shorter cooldown for auth errors - they might be temporary API issues
                    cooldown = 60  # 1 minute instead of 5 minutes
                    logger.warning(f"X {tier} auth error. Cooldown: {cooldown}s")
                    _mark_tier_failed(tier, cooldown)
                    continue
                else:
                    cooldown = error_classification.cooldown_seconds
                    logger.debug(
                        f"X {tier} error ({error_classification.error_type}). Cooldown: {cooldown}s"
                    )

                _mark_tier_failed(tier, cooldown)

                if attempt < retry_count - 1 and not error_classification.is_rate_limit:
                    backoff_seconds = 2**attempt  # 1s, 2s, 4s...
                    logger.info(
                        f" Backing off for {backoff_seconds}s before next attempt"
                    )
                    time.sleep(backoff_seconds)
        else:
            logger.warning(f"No tier available for attempt {attempt + 1}")

            # If no tiers available, force-clear cooldowns and try again once
            if attempt == 0:
                logger.info("All tiers in cooldown - force clearing to retry")
                with _tier_lock:
                    for tier_key in _key_status:
                        if TIER_KEYS[tier_key]["key"]:  # Only reset if key exists
                            _key_status[tier_key]["cooldown_until"] = 0
                continue  # Retry with cleared cooldowns

            if attempt < retry_count - 1:
                backoff_seconds = 2**attempt
                logger.info(f" No tiers available, backing off for {backoff_seconds}s")
                time.sleep(backoff_seconds)

    # All options exhausted
    logger.error(f"All LLM providers failed. Attempts: {', '.join(attempts_log)}")
    if successful_provider:
        logger.info(f"Last successful provider: {successful_provider}")

    # Provide user-friendly conversational error message based on error type
    error_str = str(last_error).lower()

    if (
        "resourceexhausted" in str(last_error)
        or "quota" in error_str
        or "402" in str(last_error)
    ):
        return """I apologize, but I'm experiencing some technical difficulties right now. It looks like the AI services have reached their usage limits. 

Here's what I can still help you with:
- Answer questions about the dataset structure
- Explain what data is available
- Guide you on how to explore the data

Please check your API credentials and billing status, or try again in a few minutes. If you need immediate help, let me know what you'd like to know about the data!"""

    elif "rate limit" in error_str or "429" in str(last_error):
        return """Hey there! I'm getting a lot of requests right now and need to slow down a bit. Give me about 30 seconds and I'll be ready to help you again.

In the meantime, you can:
- Browse the example queries above
- Think about what insights you'd like from the data
- Check out the dataset info in the sidebar

Thanks for your patience! 😊"""

    elif "auth" in error_str or "401" in str(last_error) or "403" in str(last_error):
        return """Hmm, I'm having trouble connecting to my AI backend. It looks like there might be an authentication issue with the API keys.

Please check that:
- Your API keys are correctly set in the .env file
- The keys are active and have the necessary permissions
- You're using valid credentials for at least one provider (OpenAI, Gemini, or OpenRouter)

Once that's fixed, I'll be ready to chat! 🔑"""

    else:
        return f"""Oops! I ran into an unexpected issue while processing your request. 

Technical details: {type(last_error).__name__}

Let me try to help you differently:
- I can still answer general questions about the dataset
- You can try rephrasing your question
- Or ask something simpler to start with

What would you like to know about the e-commerce data? 🤔"""


def call_llm(prompt, system=None, max_tokens=256):
    try:
        if PROVIDER == "google":
            model = genai.GenerativeModel(GEMINI_MODEL)  # type: ignore
            full_prompt = (system + "\n\n" if system else "") + prompt
            response = model.generate_content(full_prompt)
            return response.text.strip()

        elif PROVIDER == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Use multi-tier fallback system
            return _call_openai_with_fallback(messages, max_tokens)

    except Exception as e:
        logger.exception("LLM call failed")
        return f"[ERROR: {type(e).__name__}] {str(e)}"


def embed_texts(texts):
    cleaned_texts = [
        re.sub(r"\s+", " ", t.strip())
        for t in texts
        if isinstance(t, str) and t.strip()
    ]
    if not cleaned_texts:
        raise ValueError("No valid text input available for embedding")

    # Try each tier with fallback logic
    last_error = None

    for tier in ["tier1", "tier2", "tier3"]:
        if not TIER_KEYS[tier]["key"]:
            continue

        provider = TIER_KEYS[tier]["provider"]

        try:
            if provider == "openai" and _openai_client:
                response = _openai_client.embeddings.create(
                    model="text-embedding-3-small", input=cleaned_texts
                )
                embeddings = [d.embedding for d in response.data]
                logger.info(f" Embeddings successful with {tier} ({provider})")
                return embeddings

            elif provider == "gemini":
                embeddings = []
                for text in cleaned_texts:
                    res = genai.embed_content(  # type: ignore
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document",
                    )
                    embeddings.append(res["embedding"])
                logger.info(f" Embeddings successful with {tier} ({provider})")
                return embeddings

            elif provider == "openrouter" and _openrouter_client:
                try:
                    response = _openrouter_client.embeddings.create(
                        model="text-embedding-3-small", input=cleaned_texts
                    )
                    embeddings = [d.embedding for d in response.data]
                    logger.info(f"Embeddings successful with {tier} ({provider})")
                    return embeddings
                except Exception as e:
                    logger.debug(f"OpenRouter embeddings not supported: {e}")
                    continue

        except Exception as e:
            error_msg = f"Embedding failed with {tier} ({provider}): {type(e).__name__}"
            logger.debug(error_msg)  # Changed from warning to debug
            last_error = e

            _mark_tier_failed(tier, 60)  # 1 minute cooldown for embeddings
            continue

    # If all tiers failed, return fallback embeddings
    logger.error(
        f"All embedding tiers failed, using fallback. Last error: {last_error}"
    )
    return [[0.0] * FALLBACK_EMBED_DIM for _ in cleaned_texts]


def get_tier_status():
    current_time = time.time()
    status = {}

    for tier, info in _key_status.items():
        tier_config = TIER_KEYS[tier]
        cooldown_remaining = max(0, info["cooldown_until"] - current_time)
        status[tier] = {
            "configured": bool(tier_config["key"]),
            "provider": tier_config["provider"],
            "available": info["available"],
            "in_cooldown": cooldown_remaining > 0,
            "cooldown_remaining_seconds": round(cooldown_remaining, 1),
            "failure_count": info["failure_count"],
            "status": (
                " Active"
                if info["available"] and cooldown_remaining == 0
                else " Cooldown" if cooldown_remaining > 0 else " Unavailable"
            ),
        }

    return status


def reset_tier_failures():
    for tier in _key_status:
        _key_status[tier]["cooldown_until"] = 0
        _key_status[tier]["failure_count"] = 0
    logger.info("All tier failures and cooldowns have been reset")


def force_tier_cooldown(tier, seconds):
    if tier in _key_status:
        _mark_tier_failed(tier, seconds)
        logger.info(f"Manually set {tier} cooldown for {seconds} seconds")
    else:
        logger.warning(f"Unknown tier: {tier}")
