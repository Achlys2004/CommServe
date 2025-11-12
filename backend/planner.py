from backend.llm_client import call_llm
from backend.llm_config import LLMTokenLimits
import logging
import json
import os

logger = logging.getLogger(__name__)

# Feature flag for intelligent planner
USE_INTELLIGENT_PLANNER = os.getenv("USE_INTELLIGENT_PLANNER", "true").lower() == "true"


def decide_action(user_query, conversation_history=""):
    """
    Intelligent LLM-based query planner that understands context and intent.
    Now enhanced with few-shot learning when USE_INTELLIGENT_PLANNER is enabled.

    Args:
        user_query: The user's query
        conversation_history: Recent conversation context (optional)

    Returns:
        dict: Action plan with reasoning
    """
    # Use intelligent planner if enabled
    if USE_INTELLIGENT_PLANNER:
        try:
            from backend.utils.intelligent_planner import IntelligentPlanner

            planner = IntelligentPlanner()
            return planner.decide_action(user_query, conversation_history)
        except Exception as e:
            logger.warning(f"Intelligent planner failed, using fallback: {e}")
            # Continue with original logic below
    q = user_query.lower().strip()

    # Very short greetings can be handled quickly
    if len(q.split()) <= 2 and any(
        word in q for word in ["hi", "hello", "hey", "thanks", "bye"]
    ):
        return {
            "action": "CONVERSATION",
            "reason": "Simple greeting detected",
            "scores": {
                "SQL": 0,
                "RAG": 0,
                "CODE": 0,
                "SQL+RAG": 0,
                "CONVERSATION": 1.0,
                "METADATA": 0,
            },
            "confidence": 1.0,
            "sql_template": None,
            "retrieval_query": None,
            "visualization": None,
        }

    # Use LLM to intelligently classify the query
    classification_prompt = f"""You are an intelligent query router for an e-commerce data analysis system. Analyze the user's query and determine the best action to take.

**Available Actions:**
1. **SQL** - For queries requiring structured data analysis (counts, aggregations, filtering, top/bottom items, trends, review scores)
   Examples: "Show top 5 products", "How many orders in 2018?", "Average order value by state"
   IMPORTANT: Queries about "best/worst", "most loved/hated", "highest/lowest rated" products should use SQL to analyze review_score numbers
   Examples: "Most hated product" → SQL (analyze review scores), "Products with worst ratings" → SQL

2. **RAG** (Retrieval-Augmented Generation) - For queries needing qualitative insights from review TEXT (comments, feedback, opinions)
   Examples: "Why are customers unhappy?", "What do review COMMENTS say about shipping?", "Common complaints in feedback TEXT"
   NOTE: If asking about sentiment SCORES or ratings (numeric), use SQL instead

3. **CODE** - For complex analysis, visualizations, or custom data transformations explicitly requested
   Examples: "Generate Python code to analyze...", "Create a correlation matrix", "Build a prediction model"

4. **SQL+RAG** (Hybrid) - For queries needing both structured data AND contextual insights
   Examples: "Show top products and explain why they're popular", "Sales trends and customer sentiment"

5. **CONVERSATION** - For greetings, thanks, clarifications, or general chat
   Examples: "Hi", "Thank you", "Can you help me?", "What can you do?"

6. **METADATA** - For queries about the dataset itself (structure, available data, schema, overview)
   Examples: "What data do you have?", "Tell me about this dataset", "What can I analyze?", "Available tables?"

**Dataset Context:**
This is an e-commerce (Olist Brazilian) dataset with:
- Orders, products, customers, sellers, payments, reviews
- Can analyze: sales, revenue, trends, geography, sentiment, customer behavior

**Conversation History:**
{conversation_history if conversation_history else "None - This is a fresh query"}

**User Query:** "{user_query}"

**Your Task:**
Analyze the query intent and classify it. Respond ONLY with a valid JSON object (no markdown, no code blocks):

{{
  "action": "SQL|RAG|CODE|SQL+RAG|CONVERSATION|METADATA",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this action was chosen",
  "requires_context": true/false,
  "suggested_approach": "Brief hint about how to handle this query"
}}

JSON Response:"""

    try:
        # Call LLM for intelligent classification
        llm_response = call_llm(classification_prompt, max_tokens=LLMTokenLimits.SHORT)

        if not llm_response:
            # Fallback to basic keyword matching if LLM fails
            logger.warning("LLM classification failed, using fallback logic")
            return _fallback_classification(user_query)

        # Clean up response - remove markdown code blocks if present
        llm_response = llm_response.strip()
        if llm_response.startswith("```"):
            llm_response = llm_response.split("```")[1]
            if llm_response.startswith("json"):
                llm_response = llm_response[4:]
        llm_response = llm_response.strip()

        # Parse LLM response
        try:
            classification = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse LLM response as JSON: {e}. Response: {llm_response[:200]}"
            )
            return _fallback_classification(user_query)

        action = classification.get("action", "RAG")
        confidence = classification.get("confidence", 0.5)
        reasoning = classification.get("reasoning", "LLM-based classification")

        # Validate action
        valid_actions = ["SQL", "RAG", "CODE", "SQL+RAG", "CONVERSATION", "METADATA"]
        if action not in valid_actions:
            logger.warning(f"Invalid action '{action}' from LLM, using fallback")
            return _fallback_classification(user_query)

        # Build scores dict
        scores = {a: 0.0 for a in valid_actions}
        scores[action] = confidence

        return {
            "action": action,
            "reason": reasoning,
            "scores": scores,
            "confidence": confidence,
            "sql_template": None,
            "retrieval_query": None,
            "visualization": classification.get("suggested_approach"),
            "requires_context": classification.get("requires_context", False),
        }

    except Exception as e:
        logger.exception(f"Error in LLM-based classification: {e}")
        return _fallback_classification(user_query)


def _fallback_classification(user_query):
    """
    Fallback classification using basic keyword matching when LLM fails.
    This is a safety net to ensure the system always returns something.
    """
    q = user_query.lower().strip()

    # Greetings/conversation
    if any(
        word in q
        for word in ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
    ):
        return {
            "action": "CONVERSATION",
            "reason": "Greeting/conversational query (fallback)",
            "scores": {
                "SQL": 0,
                "RAG": 0,
                "CODE": 0,
                "SQL+RAG": 0,
                "CONVERSATION": 1.0,
                "METADATA": 0,
            },
            "confidence": 0.9,
            "sql_template": None,
            "retrieval_query": None,
            "visualization": None,
        }

    # Metadata queries
    if any(
        phrase in q
        for phrase in [
            "tell me about",
            "what data",
            "dataset",
            "available data",
            "what information",
            "schema",
        ]
    ):
        return {
            "action": "METADATA",
            "reason": "Dataset overview request (fallback)",
            "scores": {
                "SQL": 0,
                "RAG": 0,
                "CODE": 0,
                "SQL+RAG": 0,
                "CONVERSATION": 0,
                "METADATA": 1.0,
            },
            "confidence": 0.85,
            "sql_template": None,
            "retrieval_query": None,
            "visualization": None,
        }

    # Code generation
    if any(
        phrase in q
        for phrase in [
            "generate code",
            "create code",
            "python code",
            "write code",
            "script",
        ]
    ):
        return {
            "action": "CODE",
            "reason": "Code generation requested (fallback)",
            "scores": {
                "SQL": 0,
                "RAG": 0,
                "CODE": 1.0,
                "SQL+RAG": 0,
                "CONVERSATION": 0,
                "METADATA": 0,
            },
            "confidence": 0.9,
            "sql_template": None,
            "retrieval_query": None,
            "visualization": None,
        }

    # SQL queries (numeric, counts, aggregations, ratings)
    if any(
        word in q
        for word in [
            "how many",
            "count",
            "total",
            "sum",
            "average",
            "top",
            "bottom",
            "show",
            "list",
            "get",
            "most",
            "least",
            "best",
            "worst",
            "highest",
            "lowest",
            "hated",
            "loved",
            "rated",
            "rating",
            "score",
        ]
    ):
        return {
            "action": "SQL",
            "reason": "Structured data query detected (fallback)",
            "scores": {
                "SQL": 0.8,
                "RAG": 0,
                "CODE": 0,
                "SQL+RAG": 0,
                "CONVERSATION": 0,
                "METADATA": 0,
            },
            "confidence": 0.75,
            "sql_template": None,
            "retrieval_query": None,
            "visualization": None,
        }

    # RAG queries (reviews, sentiment, why/explain)
    if any(
        word in q
        for word in [
            "why",
            "review",
            "sentiment",
            "feedback",
            "opinion",
            "feel",
            "think",
            "explain",
        ]
    ):
        return {
            "action": "RAG",
            "reason": "Qualitative analysis needed (fallback)",
            "scores": {
                "SQL": 0,
                "RAG": 0.8,
                "CODE": 0,
                "SQL+RAG": 0,
                "CONVERSATION": 0,
                "METADATA": 0,
            },
            "confidence": 0.7,
            "sql_template": None,
            "retrieval_query": None,
            "visualization": None,
        }

    # Default to hybrid approach
    return {
        "action": "SQL+RAG",
        "reason": "Uncertain query type, using hybrid approach (fallback)",
        "scores": {
            "SQL": 0.5,
            "RAG": 0.5,
            "CODE": 0,
            "SQL+RAG": 0.6,
            "CONVERSATION": 0,
            "METADATA": 0,
        },
        "confidence": 0.5,
        "sql_template": None,
        "retrieval_query": None,
        "visualization": None,
    }
