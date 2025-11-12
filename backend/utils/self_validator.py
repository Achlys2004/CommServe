"""
Self-Validation Engine - AI validates its own outputs before returning
"""

import sqlite3
import logging
from typing import Dict, Any, Optional, Tuple
from backend.llm_client import call_llm
from backend.llm_config import LLMTokenLimits

logger = logging.getLogger(__name__)


class SelfValidator:
    """
    Intelligent self-validation system that checks AI outputs
    before returning them to users.
    """

    def __init__(self, db_path="data/olist.db"):
        self.db_path = db_path

    def validate_sql(
        self, query: str, user_intent: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate SQL query for correctness and intent alignment.

        Returns:
            (is_valid, corrected_sql, reason)
        """
        # First, test if SQL is syntactically valid
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Try to execute with LIMIT 0 to validate syntax without running
            # But only if the query doesn't already have a LIMIT clause
            query_lower = query.lower().strip()
            if "limit" in query_lower:
                # Query already has LIMIT, use it as-is for syntax check
                test_query = query.rstrip(";")
            else:
                test_query = f"{query.rstrip(';')} LIMIT 0"
            cursor.execute(test_query)
            conn.close()

        except sqlite3.Error as e:
            logger.warning(f"SQL syntax error detected: {e}")
            # Ask AI to fix it
            return self._ask_ai_to_fix_sql(query, str(e), user_intent)

        # Second, check if results match intent
        return self._validate_intent_alignment(query, user_intent)

    def _ask_ai_to_fix_sql(
        self, broken_sql: str, error: str, user_intent: str
    ) -> Tuple[bool, Optional[str], str]:
        """Ask AI to fix the SQL error."""

        prompt = f"""You are a SQL expert fixing errors in Olist e-commerce database queries.

CRITICAL SCHEMA RULES:
1. Table name is 'order_payments' NOT 'payments'
2. City/state columns are in 'customers' table, NOT 'orders'
3. For geographic queries, JOIN orders with customers:
   FROM orders o JOIN customers c ON o.customer_id = c.customer_id
   WHERE c.customer_city = 'sao paulo'

Database Tables:
- orders: order_id, customer_id, order_status, order_purchase_timestamp, ...
- customers: customer_id, customer_city, customer_state, ...
- order_payments: order_id, payment_type, payment_value, ...
- order_items: order_id, product_id, price, freight_value, ...
- products: product_id, product_category_name, ...
- category_translation: product_category_name, product_category_name_english

BROKEN Query:
{broken_sql}

Error:
{error}

User wanted: {user_intent}

Fix the query by:
1. Correcting table names (order_payments NOT payments)
2. Adding proper JOINs if missing (e.g., orders with customers for city)
3. Using correct column names
4. Making it valid SQLite syntax

Return ONLY the corrected SQL query:"""

        try:
            fixed_sql = call_llm(prompt, max_tokens=LLMTokenLimits.SQL)
            if fixed_sql:
                fixed_sql = fixed_sql.strip()
                # Clean markdown if present
                if fixed_sql.startswith("```"):
                    fixed_sql = fixed_sql.split("```")[1]
                    if fixed_sql.startswith("sql"):
                        fixed_sql = fixed_sql[3:]
                fixed_sql = fixed_sql.strip()

                return True, fixed_sql, "Auto-corrected SQL error"
        except Exception as e:
            logger.error(f"Failed to fix SQL: {e}")

        return False, None, f"SQL validation failed: {error}"

    def _validate_intent_alignment(
        self, query: str, user_intent: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Use AI to check if the SQL query actually answers the user's question.
        """

        prompt = f"""You are a data analyst validator. Check if this SQL query correctly answers the user's question.

User Question: {user_intent}

Generated SQL:
{query}

Analysis checklist:
1. Does the query answer what the user asked?
2. Are the columns named appropriately?
3. Is the grouping/aggregation correct?
4. Are there any logic errors?
5. Will this return meaningful results?

Respond in JSON format:
{{
  "is_valid": true/false,
  "issues": ["issue1", "issue2"] or [],
  "suggested_fix": "corrected SQL" or null,
  "confidence": 0.0-1.0
}}

JSON Response:"""

        try:
            response = call_llm(prompt, max_tokens=LLMTokenLimits.MEDIUM)
            if response:
                # Check if response is an error message
                error_indicators = [
                    "I apologize, but I'm experiencing",
                    "Hey there! I'm getting a lot of requests",
                    "Hmm, I'm having trouble connecting",
                    "Oops! I ran into an unexpected issue",
                    "All LLM providers failed",
                    "[SYSTEM NOTICE]",
                    "[ERROR:",
                ]

                is_error_response = any(
                    response.startswith(indicator) for indicator in error_indicators
                )

                if is_error_response:
                    logger.warning(
                        f"LLM returned error message instead of JSON: {response[:100]}..."
                    )
                    return True, None, None  # Default to accepting query

                # Parse JSON
                import json

                response = response.strip()
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                response = response.strip()

                validation = json.loads(response)

                if not validation.get("is_valid", True):
                    issues = ", ".join(validation.get("issues", []))
                    suggested = validation.get("suggested_fix")

                    if suggested:
                        return True, suggested, f"Auto-corrected: {issues}"
                    else:
                        return False, None, f"Query issues: {issues}"

                # Query is valid
                return True, None, None

        except Exception as e:
            logger.warning(f"Intent validation failed: {e}")

        # Default to accepting the query if validation fails
        return True, None, None

    def validate_code_results(
        self, code: str, execution_result: Dict[str, Any], user_intent: str
    ) -> Tuple[bool, str]:
        """
        Validate if code execution results are meaningful.

        Returns:
            (is_valid, feedback_message)
        """

        if execution_result.get("status") != "success":
            return False, "Code execution failed"

        stdout = execution_result.get("stdout", "")
        has_viz = execution_result.get("output_type") == "image"
        images_count = len(execution_result.get("images", []))

        # Check if results are meaningful
        if not stdout and not has_viz:
            return False, "No output generated"

        # Ask AI if the results make sense
        prompt = f"""Validate if this code execution produced meaningful results.

User asked: {user_intent}

Execution output:
{stdout[:1000]}

Visualizations generated: {images_count}

Is this a valid, meaningful response? Answer with JSON:
{{
  "is_meaningful": true/false,
  "reason": "brief explanation"
}}

JSON Response:"""

        try:
            response = call_llm(prompt, max_tokens=LLMTokenLimits.SHORT)
            if response:
                # Check if response is an error message
                error_indicators = [
                    "I apologize, but I'm experiencing",
                    "Hey there! I'm getting a lot of requests",
                    "Hmm, I'm having trouble connecting",
                    "Oops! I ran into an unexpected issue",
                    "All LLM providers failed",
                    "[SYSTEM NOTICE]",
                    "[ERROR:",
                ]

                is_error_response = any(
                    response.startswith(indicator) for indicator in error_indicators
                )

                if is_error_response:
                    logger.warning(
                        f"LLM returned error message instead of JSON: {response[:100]}..."
                    )
                    return (
                        True,
                        "LLM validation unavailable - assuming results are meaningful",
                    )

                import json

                response = response.strip()
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                response = response.strip()

                validation = json.loads(response)

                if not validation.get("is_meaningful", True):
                    return False, validation.get("reason", "Results not meaningful")
        except Exception as e:
            logger.warning(f"Code validation failed: {e}")

        return True, "Results validated"

    def should_show_insights(
        self, query: str, result: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Intelligently decide if AI insights would add value.

        Returns:
            (should_show, reason)
        """

        result_type = result.get("type", "")

        # Never show insights for metadata queries - they're already comprehensive
        if result_type == "metadata":
            return (
                False,
                "Metadata responses are already comprehensive and conversational",
            )

        # Build context about the result
        context_parts = []

        if result_type == "sql":
            sql_result = result.get("sql_result", {})
            row_count = len(sql_result.get("rows", []))
            context_parts.append(f"SQL query returned {row_count} rows")

        elif result_type == "code":
            exec_result = result.get("execution_result", {})
            has_viz = exec_result.get("output_type") == "image"
            images = len(exec_result.get("images", []))
            context_parts.append(
                f"Code generated {images} visualizations"
                if has_viz
                else "Code executed"
            )

        elif result_type == "rag":
            context_parts.append("Retrieved contextual information")

        result_summary = ", ".join(context_parts)

        # Ask AI if insights would be valuable
        prompt = f"""You are an AI assistant deciding whether to show additional insights.

User query: {query}
Result summary: {result_summary}

Should you provide additional AI insights? Only say YES if insights would:
1. Help interpret complex data patterns
2. Explain visualizations or statistics
3. Provide business context to technical results
4. Clarify non-obvious trends

Do NOT show insights if:
- Results are self-explanatory (simple counts, lists)
- User asked a simple factual question
- Response already contains conversational explanation
- It would just repeat what's already shown

Respond with JSON:
{{
  "show_insights": true/false,
  "reason": "brief justification"
}}

JSON Response:"""

        try:
            response = call_llm(prompt, max_tokens=LLMTokenLimits.SHORT)
            if response:
                # Check if response is an error message
                error_indicators = [
                    "I apologize, but I'm experiencing",
                    "Hey there! I'm getting a lot of requests",
                    "Hmm, I'm having trouble connecting",
                    "Oops! I ran into an unexpected issue",
                    "All LLM providers failed",
                    "[SYSTEM NOTICE]",
                    "[ERROR:",
                ]

                is_error_response = any(
                    response.startswith(indicator) for indicator in error_indicators
                )

                if is_error_response:
                    logger.warning(
                        f"LLM returned error message instead of JSON: {response[:100]}..."
                    )
                    return False, "LLM unavailable - skipping insights"

                import json

                response = response.strip()
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                response = response.strip()

                decision = json.loads(response)

                should_show = decision.get("show_insights", False)
                reason = decision.get("reason", "")

                logger.info(f"Insight decision: {should_show} - {reason}")
                return should_show, reason

        except Exception as e:
            logger.warning(f"Insight decision failed: {e}")

        # Default: show insights for visualizations and code, not for simple SQL
        if result_type == "code":
            return True, "Visualization results benefit from insights"
        elif result_type == "sql":
            sql_result = result.get("sql_result", {})
            row_count = len(sql_result.get("rows", []))
            if row_count > 5:
                return True, "Large result set benefits from insights"
            return False, "Simple result is self-explanatory"

        return False, "Default: no insights needed"
