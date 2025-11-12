import os
import logging
import pandas as pd
from datetime import datetime
from backend.executor_engine import ExecutorEngine
from backend.features.summariser import Summariser
from backend.utils.session_memory import SessionMemory, calculate_confidence
from backend.utils.error_handler import handle_errors
from backend.sql_executor import run_sql_safe, validate_sql, log_execution
from backend.retriever import retrieve_by_text
from backend.llm_client import call_llm
from backend.llm_config import LLMTokenLimits
from backend.planner import decide_action
from backend.utils.hybrid_sql_context import fetch_sql_context
from backend.features import (
    generate_code,
    generate_sql,
    analyze_sentiment_emotion,
    detect_language,
    translate_to_english,
)
from config import GENERATED_DIR

logger = logging.getLogger(__name__)


# Helper: Natural Language SQL Summaries
@handle_errors(default_return="No data available for analysis.", log_level="warning")
def summarize_sql_result_natural_language(query, result):
    """Generate natural language summary from SQL results for conversational responses."""
    if not result or isinstance(result, dict) and "error" in result:
        return "No data available for analysis."

    rows = result.get("rows", [])
    if not rows:
        return "The query returned no results."

    # Convert to DataFrame for analysis
    df = pd.DataFrame(rows)

    if df.empty:
        return "The query returned an empty dataset."

    # Get basic stats
    record_count = len(df)

    # Analyze query intent to provide appropriate summary
    query_lower = query.lower()

    # Top/bottom queries
    if "top" in query_lower and ("category" in query_lower or "product" in query_lower):
        return _summarize_top_categories(df, record_count, query)
    elif "bottom" in query_lower and (
        "category" in query_lower or "product" in query_lower
    ):
        return _summarize_bottom_categories(df, record_count, query)

    # Review/rating queries (hated, loved, best, worst rated)
    elif any(
        word in query_lower
        for word in [
            "hated",
            "loved",
            "worst rated",
            "best rated",
            "lowest rated",
            "highest rated",
        ]
    ):
        return _summarize_review_sentiment(df, record_count, query)

    # Sales/revenue queries
    elif "sales" in query_lower or "revenue" in query_lower or "price" in query_lower:
        return _summarize_sales_data(df, record_count, query)

    # Count queries
    elif "count" in query_lower or "number" in query_lower or "how many" in query_lower:
        return _summarize_count_data(df, record_count, query)

    # Generic summary using LLM
    else:
        return _generate_generic_summary(query, df, record_count)


@handle_errors(default_return=None, log_level="warning")
def _summarize_top_categories(df, record_count, query):
    """Summarize top categories/products query with human-friendly language."""
    # Find category/product and metric columns
    cat_col = None
    metric_col = None
    metric_type = "items"  # Default

    for col in df.columns:
        col_lower = col.lower()
        if "category" in col_lower or "product" in col_lower or "name" in col_lower:
            cat_col = col
        elif any(
            keyword in col_lower for keyword in ["sold", "count", "quantity", "items"]
        ):
            metric_col = col
            metric_type = "items"
        elif any(
            keyword in col_lower for keyword in ["sales", "revenue", "price", "total"]
        ):
            metric_col = col
            metric_type = "revenue"

    if cat_col and metric_col and len(df) > 0:
        # Build list of top categories
        categories = []
        for i, row in df.iterrows():
            category = row[cat_col]
            metric_value = row[metric_col]

            # Format based on metric type
            if metric_type == "revenue":
                formatted_value = f"${metric_value:,.2f}"
                metric_label = "in revenue"
            else:
                formatted_value = f"{int(metric_value):,}"
                metric_label = "items sold"

            categories.append(f"**{category}** with {formatted_value} {metric_label}")

        if len(categories) == 1:
            return f"The top category is {categories[0]}."
        elif len(categories) > 1:
            result = f"The top category is {categories[0]}, followed by "
            if len(categories) == 2:
                result += categories[1] + "."
            else:
                result += ", ".join(categories[1:-1]) + ", and " + categories[-1] + "."
            return result

    return f"Found {record_count} categories in the results."


@handle_errors(default_return=None, log_level="warning")
def _summarize_bottom_categories(df, record_count, query):
    """Summarize bottom categories/products query."""
    cat_col = None
    sales_col = None

    for col in df.columns:
        col_lower = col.lower()
        if "category" in col_lower or "product" in col_lower:
            cat_col = col
        elif "sales" in col_lower or "price" in col_lower or "total" in col_lower:
            sales_col = col

    if cat_col and sales_col and len(df) > 0:
        bottom_row = df.iloc[0]
        category = bottom_row[cat_col]
        sales_value = bottom_row[sales_col]

        return f"The lowest-selling category is {category} with {sales_value:,.0f} in total sales."

    return f"Analysis shows {record_count} categories in the results."


@handle_errors(default_return=None, log_level="warning")
def _summarize_review_sentiment(df, record_count, query):
    """Summarize queries about product ratings and customer sentiment (hated/loved products)."""
    query_lower = query.lower()

    # Find relevant columns
    product_col = None
    rating_col = None
    count_col = None
    one_star_col = None

    for col in df.columns:
        col_lower = col.lower()
        if "product" in col_lower or "name" in col_lower:
            product_col = col
        elif (
            "avg" in col_lower
            or "average" in col_lower
            or "score" in col_lower
            or "rating" in col_lower
        ):
            rating_col = col
        elif "count" in col_lower or "review" in col_lower:
            count_col = col
        elif "one_star" in col_lower or "1" in col_lower:
            one_star_col = col

    if not product_col or not rating_col or len(df) == 0:
        return f"Found {record_count} products in the results."

    # Determine if query is about "hated" (worst) or "loved" (best) products
    is_negative = any(
        word in query_lower for word in ["hated", "worst", "lowest", "bad"]
    )
    is_positive = any(
        word in query_lower for word in ["loved", "best", "highest", "good"]
    )

    # Get top result
    top_product = df.iloc[0]
    product_id = top_product[product_col]
    rating = top_product[rating_col]
    review_count = top_product[count_col] if count_col else "several"

    # Build sentiment description
    if is_negative:
        if rating <= 1.5:
            sentiment = "**extremely disliked**"
        elif rating <= 2.5:
            sentiment = "**poorly rated**"
        else:
            sentiment = "**lower-rated**"

        result = f"The most hated product is **{product_id}** with an average rating of **{rating:.2f}/5.0**"

        if isinstance(review_count, (int, float)):
            result += f" based on {int(review_count)} reviews"

        # Add 1-star info if available
        if one_star_col and top_product[one_star_col]:
            one_star_count = int(top_product[one_star_col])
            if isinstance(review_count, (int, float)):
                percentage = (one_star_count / review_count) * 100
                result += f" ({one_star_count} were 1-star reviews, {percentage:.0f}%)"

        result += f". This product is {sentiment} by customers."

    elif is_positive:
        if rating >= 4.5:
            sentiment = "**highly loved**"
        elif rating >= 3.5:
            sentiment = "**well-received**"
        else:
            sentiment = "**positively rated**"

        result = f"The most loved product is **{product_id}** with an average rating of **{rating:.2f}/5.0**"

        if isinstance(review_count, (int, float)):
            result += f" from {int(review_count)} reviews"

        result += f". This product is {sentiment} by customers."
    else:
        # Neutral description
        result = (
            f"Product **{product_id}** has an average rating of **{rating:.2f}/5.0**"
        )
        if isinstance(review_count, (int, float)):
            result += f" based on {int(review_count)} reviews"
        result += "."

    # Add context about other results
    if len(df) > 1:
        second_product = df.iloc[1]
        second_rating = second_product[rating_col]
        result += f" The second {'worst' if is_negative else 'best'} is **{second_product[product_col]}** with {second_rating:.2f}/5.0."

    return result


@handle_errors(default_return=None, log_level="warning")
def _summarize_sales_data(df, record_count, query):
    """Summarize sales/revenue data."""
    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0:
        sales_col = numeric_cols[0]  # Assume first numeric column is sales
        total_sales = df[sales_col].sum()
        avg_sales = df[sales_col].mean()
        max_sales = df[sales_col].max()

        return f"Total sales across {record_count} records: {total_sales:,.0f} (average: {avg_sales:,.0f}, highest: {max_sales:,.0f})."

    return f"Analysis shows {record_count} sales records."


@handle_errors(default_return=None, log_level="warning")
def _summarize_count_data(df, record_count, query):
    """Summarize count queries."""
    if record_count == 1 and len(df.columns) > 0:
        # Single count result
        count_value = df.iloc[0, 0] if len(df) > 0 else 0
        return f"The query returned a count of {count_value}."
    else:
        return f"The query found {record_count} matching records."


@handle_errors(default_return=None, log_level="warning")
def _generate_generic_summary(query, df, record_count):
    """Generate generic summary using LLM for complex queries with human-friendly output."""
    # Create a sample of the data for the LLM
    sample_data = df.head(3).to_string() if len(df) > 0 else "No data"

    prompt = f"""
You are a friendly data analyst assistant. The user asked: "{query}".
Here are the first few results from the query:
{sample_data}

Total records found: {record_count}

Generate a conversational, human-friendly summary that:
1. Directly answers the user's question
2. Highlights the most important insights from the data
3. Uses natural language (avoid technical jargon)
4. Formats numbers clearly (e.g., "10,953 items" or "$1,258.68")
5. Mentions specific names/categories when relevant
6. Keeps response to 2-3 sentences maximum

Be helpful, clear, and concise.
"""

    summary = call_llm(prompt, max_tokens=LLMTokenLimits.SHORT)
    return summary if summary else f"Found {record_count} records matching your query."


# Helper: Structured Summary Context
@handle_errors(default_return="No data available.", log_level="warning")
def summarize_sql_result(result):
    """Generate quantitative summary from SQL results."""
    if not result or isinstance(result, dict) and "error" in result:
        return "No data available."

    rows = result.get("rows", [])
    if not rows:
        return "Query returned no results."

    # Convert to DataFrame for analysis
    df = pd.DataFrame(rows)

    if df.empty:
        return "Query returned an empty dataset."

    # Generate statistics for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().round(2).to_dict()
        summary_parts = [f"Dataset contains {len(df)} records."]

        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            col_stats = stats[col]
            summary_parts.append(
                f"{col}: avg={col_stats.get('mean', 'N/A')}, "
                f"min={col_stats.get('min', 'N/A')}, "
                f"max={col_stats.get('max', 'N/A')}"
            )

        return " ".join(summary_parts)
    else:
        return f"Query returned {len(df)} records with {len(df.columns)} columns."


# Helper: Collection Selection
def _select_collection(query, default="olist_products"):
    q_lower = query.lower()
    if any(
        word in q_lower for word in ["review", "rating", "score", "quality", "shipping"]
    ):
        return "olist_reviews"
    elif any(word in q_lower for word in ["seller", "vendor", "supplier"]):
        return "olist_sellers"
    elif any(word in q_lower for word in ["order", "purchase", "item"]):
        return "olist_order_items"
    return default


# Helper: Sentiment & Emotion Analysis
def _analyze_reviews(context, top_n=5):
    """Analyze sentiment across multiple reviews and return summary."""
    if not context.strip():
        return None

    reviews = [r.strip() for r in context.split("\n\n") if r.strip()]
    reviews = reviews[:top_n]

    if not reviews:
        return None

    sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    detailed_scores = []
    highlights = {"positive": [], "negative": []}

    for review in reviews:
        analysis = analyze_sentiment_emotion(review[:500])
        if not analysis:
            continue
        sentiment = analysis.get("sentiment", "neutral").lower()
        score = analysis.get("sentiment_score", 0.0)

        if sentiment in sentiment_scores:
            sentiment_scores[sentiment] += score
        detailed_scores.append(analysis)

        if sentiment == "positive":
            highlights["positive"].append(review)
        elif sentiment == "negative":
            highlights["negative"].append(review)

    overall_sentiment = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k])
    overall_score = sentiment_scores[overall_sentiment] / max(1, len(reviews))

    return {
        "overall_sentiment": overall_sentiment,
        "overall_score": overall_score,
        "highlights": highlights,
        "details": detailed_scores,
    }


# Helper: Chunk Long Context
def _chunk_context(ctx, max_words=500):
    words = ctx.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


# Helper: Translate Portuguese category names in SQL results
def _translate_sql_results(sql_result):
    """Translate Portuguese category names in SQL results to English and format nicely."""
    if not sql_result or isinstance(sql_result, dict) and "error" in sql_result:
        return sql_result

    rows = sql_result.get("rows", [])
    if not rows:
        return sql_result

    # Common Portuguese to English category translations
    translations = {
        "moveis_decoracao": "furniture decor",
        "moveis decoracao": "furniture decor",
        "informatica_acessorios": "computer accessories",
        "informatica acessorios": "computer accessories",
        "relogios_presentes": "watches gifts",
        "relogios presentes": "watches gifts",
        "ferramentas_jardim": "garden tools",
        "ferramentas jardim": "garden tools",
        "cama_mesa_banho": "bed bath table",
        "cama mesa banho": "bed bath table",
        "bed_bath_table": "bed bath table",
        "beleza_saude": "health beauty",
        "beleza saude": "health beauty",
        "health_beauty": "health beauty",
        "esporte_lazer": "sports leisure",
        "esporte lazer": "sports leisure",
        "sports_leisure": "sports leisure",
        "telefonia": "telephony",
        "automotivo": "automotive",
        "brinquedos": "toys",
        "cool_stuff": "cool stuff",
        "utilidades_domesticas": "housewares",
        "utilidades domesticas": "housewares",
        "housewares": "housewares",
        "climatizacao": "air conditioning",
        "construcao_ferramentas_seguranca": "construction tools safety",
        "construcao ferramentas seguranca": "construction tools safety",
        "construction_tools_safety": "construction tools safety",
        "pet_shop": "pet shop",
        "pet shop": "pet shop",
        "perfumaria": "perfumery",
        "bebes": "baby",
        "eletronicos": "electronics",
        "papelaria": "stationery",
        "fashion_bolsas_e_acessorios": "fashion bags accessories",
        "fashion bolsas e acessorios": "fashion bags accessories",
        "consoles_games": "consoles games",
        "malas_acessorios": "luggage accessories",
        "malas acessorios": "luggage accessories",
        "alimentos": "food",
        "alimentos_bebidas": "food drinks",
        "alimentos bebidas": "food drinks",
    }

    # Translate rows
    translated_rows = []
    for row in rows:
        new_row = row.copy() if isinstance(row, dict) else row
        if isinstance(new_row, dict):
            # Check for category name fields
            for key in list(new_row.keys()):
                if "category" in key.lower() or "product_name" in key.lower():
                    value = new_row[key]
                    if isinstance(value, str):
                        # Try direct translation
                        lower_value = value.lower().strip()
                        if lower_value in translations:
                            new_row[key] = translations[lower_value]
                        else:
                            # Try replacing underscores with spaces
                            normalized = lower_value.replace("_", " ")
                            if normalized in translations:
                                new_row[key] = translations[normalized]
                            else:
                                # If already English with underscores, just format nicely
                                new_row[key] = lower_value.replace("_", " ").title()
        translated_rows.append(new_row)

    # Update result
    result = sql_result.copy()
    result["rows"] = translated_rows
    return result


# Helper: Format Response
def format_response(response_type, **kwargs):
    """Format unified response structure with proper typing."""
    from typing import Any, Dict

    result: Dict[str, Any] = {"type": response_type}
    result.update(kwargs)
    return result


# Core Query Engine
class QueryEngine:
    def __init__(
        self,
        hybrid_mode=True,
        session_id="default",
        persist_memory=True,
        use_orchestrator=True,
    ):
        # New modular components
        self.executor = ExecutorEngine()
        self.summariser = Summariser()
        self.session_memory = SessionMemory(
            session_id=session_id, max_recent=5, persist=persist_memory
        )
        self.summary_context = ""

        # Legacy compatibility attributes
        self.hybrid_mode = hybrid_mode
        self.session_id = session_id
        self.memory = self.session_memory  # Alias for backward compatibility

        # Orchestrator integration (optional, for conversational mode)
        self.use_orchestrator = use_orchestrator
        self._orchestrator = None
        if use_orchestrator:
            try:
                from backend.orchestrator import Orchestrator

                self._orchestrator = Orchestrator(session_id=session_id)
                logger.info(
                    "[QueryEngine] Orchestrator mode enabled for conversational AI"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize orchestrator: {e}. Running in legacy mode."
                )
                self.use_orchestrator = False

    def execute_query(self, query, params=None):
        if params is None:
            params = {}

        # Input validation
        MAX_QUERY_LENGTH = 2000
        if not query or not isinstance(query, str):
            return {
                "error": "Invalid query: must be a non-empty string",
                "type": "error",
            }

        query = query.strip()
        if not query:
            return {"error": "Query cannot be empty", "type": "error"}

        if len(query) > MAX_QUERY_LENGTH:
            return {
                "error": f"Query too long (max {MAX_QUERY_LENGTH} characters)",
                "type": "error",
            }

        # Preprocess query for intelligence enhancements
        from backend.utils.query_preprocessor import QueryPreprocessor

        preprocessor = QueryPreprocessor()
        enhanced_query, warnings = preprocessor.preprocess(query)

        # Store original query and warnings for context
        original_query = query
        query = enhanced_query
        params["preprocessing_warnings"] = warnings

        try:
            # If orchestrator is enabled, route through it for conversational experience
            if self.use_orchestrator and self._orchestrator:
                return self._execute_with_orchestrator(query, params)

            # Otherwise, use legacy direct execution
            return self._execute_direct(query, params)

        except Exception as e:
            logger.exception("Query execution failed: %s", str(e))
            return {"error": f"Query execution failed: {str(e)}"}

    def _execute_with_orchestrator(self, query, params):
        """Execute query through orchestrator for conversational experience."""
        if self._orchestrator is None:
            logger.warning("Orchestrator is None, falling back to direct execution")
            return self._execute_direct(query, params)
        return self._orchestrator.process_query(query, self)

    def _execute_direct(self, query, params):
        """Direct execution without orchestrator (legacy mode)."""
        try:
            # At start, figure out follow-up detection heuristic
            is_follow_up = False
            last_context = self.memory.get_context(include_recent=1)
            if last_context and any(
                w in query.lower()
                for w in [
                    "that",
                    "those",
                    "this",
                    "last",
                    "previous",
                    "month",
                    "week",
                    "top",
                    "bottom",
                    "again",
                ]
            ):
                is_follow_up = True

            # Add conversational context (use provided context or build from memory)
            provided_context = params.get("context")
            if provided_context:
                context_string = provided_context
            else:
                context_string = self.memory.get_context_string(
                    self.session_id, limit=2
                )

            # Translate to English
            lang = detect_language(query)
            if lang and lang != "en":
                translated_query = translate_to_english(query)
            else:
                translated_query = query

            # Decision planning with enhanced logic and conversation context
            plan = decide_action(translated_query, conversation_history=context_string)

            if not plan or "action" not in plan:
                action = "RAG"
                logger.warning("Planner returned invalid plan, defaulting to RAG mode")
            else:
                action = plan.get("action", "SQL+RAG" if self.hybrid_mode else "RAG")

            # If query is vague and we have context, expand it into a complete query
            # Note: Only expand for data queries (SQL, CODE, RAG, SQL+RAG), not CONVERSATION or METADATA
            if context_string and action in ["SQL", "CODE", "RAG", "SQL+RAG"]:
                from backend.utils.intelligent_planner import IntelligentPlanner

                planner = IntelligentPlanner()
                expanded_query = planner.expand_vague_query(
                    translated_query, context_string
                )
                if expanded_query != translated_query:
                    logger.info(
                        f"Query expanded for better execution: '{translated_query}' → '{expanded_query}'"
                    )
                    translated_query = expanded_query

            q_lower = translated_query.lower()

            # Execute based on action
            if action == "SQL":
                result = self._handle_sql_mode(translated_query, params, context_string)
            elif action in ["RAG", "SQL+RAG"]:
                result = self._handle_rag_mode(
                    translated_query,
                    action,
                    params,
                    translated_query.lower(),
                    context_string,
                )
            elif action == "CODE":
                result = self._handle_code_mode(
                    translated_query, translated_query.lower(), context_string
                )
            elif action == "CONVERSATION":
                result = self._handle_conversation_mode(translated_query)
            elif action == "METADATA":
                result = self._handle_metadata_mode(translated_query)
            else:
                result = {"error": f"Unsupported action type: {action}"}

            # Calculate confidence score
            confidence = self._calculate_result_confidence(result, action)
            result["confidence"] = confidence  # type: ignore
            result["plan"] = plan  # type: ignore
            result["is_follow_up"] = is_follow_up  # type: ignore

            # Add preprocessing warnings if any
            if params.get("preprocessing_warnings"):
                result["warnings"] = params["preprocessing_warnings"]  # type: ignore

            # later, after building `result` (whatever type), store into session memory
            try:
                self.memory.add(query, action, result)
            except Exception:
                # non-fatal
                logger.debug("Failed to persist session memory")

            return result

        except Exception as e:
            logger.exception("Query execution failed: %s", str(e))
            return {"error": f"Query execution failed: {str(e)}"}

    def _calculate_result_confidence(self, result, action):
        """Calculate confidence based on result data."""
        context_length = len(result.get("context", ""))
        sql_result = result.get("sql_result", [])
        sql_count = (
            len(sql_result)
            if isinstance(sql_result, list)
            else (1 if sql_result else 0)
        )

        # For dict results, check if it has data
        if isinstance(sql_result, dict) and "error" not in sql_result:
            sql_count = len(str(sql_result))

        return calculate_confidence(
            action=action,
            context_length=context_length,
            sql_result_count=sql_count,
            retrieval_count=context_length // 100,  # Approximate doc count
        )

    def clear_memory(self) -> None:
        """Clear session memory."""
        self.memory.clear()
        logger.info(f"Memory cleared for session: {self.session_id}")

    def get_session_history(self, limit=None):
        """Get session history."""
        return self.memory.get_history(self.session_id, limit)

    # -------------------------------
    # UPDATE SUMMARY CONTEXT
    # -------------------------------
    def _update_context(self, query, result):
        """Update summary context using the summariser."""
        summary = self.summariser.generate_summary(query, result)
        self.summary_context = summary
        # Store summary in session memory as a special entry
        summary_entry = {
            "response": {"summary": summary, "type": "context_update"},
            "confidence": 1.0,
            "timestamp": datetime.now().isoformat(),
        }
        self.session_memory.add(query, "CONTEXT_UPDATE", summary_entry)

    # -------------------------------
    # SAVE GENERATED CODE
    # -------------------------------
    def _save_generated_code(self, query_text, code):
        """Save generated code to file."""
        os.makedirs("generatedfiles", exist_ok=True)
        filename = f"generatedfiles/{self._clean_filename(query_text)}.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        return filename

    def _clean_filename(self, query_text):
        """Clean query text for use as filename."""
        import re
        from datetime import datetime

        safe = re.sub(r"[^a-zA-Z0-9_]", "_", query_text[:40])
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}"

    def _handle_sql_mode(self, query, params, context_string=""):
        """Handle SQL-only queries with dynamic generation, self-validation, and conversational context."""
        try:
            # Initialize warnings list
            warnings = []

            # Always use dynamic SQL generation
            logger.info("Generating SQL dynamically for query")

            # Prepare context for generation
            prompt_context = (
                f"\n\nPrevious Context:\n{context_string}" if context_string else ""
            )

            # Generate SQL using LLM with context
            sql_result = generate_sql(query, context_info=prompt_context)

            if (
                sql_result
                and sql_result.get("sql")
                and not sql_result["sql"].startswith("-- Error")
            ):
                sql = sql_result["sql"]
                sql_params = {}  # Dynamic SQL typically doesn't need params
                logger.info(f"Dynamic SQL generated: {sql[:100]}...")

                # SELF-VALIDATION: Validate SQL before execution
                from backend.utils.self_validator import SelfValidator

                validator = SelfValidator()
                is_valid, corrected_sql, reason = validator.validate_sql(sql, query)

                if corrected_sql:
                    logger.info(f"SQL auto-corrected: {reason}")
                    sql = corrected_sql
                    warnings.append(f"Auto-corrected: {reason}")
                elif not is_valid:
                    return {"error": f"SQL validation failed: {reason}"}
            else:
                return {"error": "Failed to generate valid SQL query"}

            sql_params = sql_params or {}
            sql_params = {**sql_params, **params}

            # Use safe executor
            sql_result = run_sql_safe(sql, sql_params)
            if isinstance(sql_result, dict) and "error" in sql_result:
                return {"error": f"SQL execution failed: {sql_result['error']}"}

            # Translate Portuguese category names to English
            sql_result = _translate_sql_results(sql_result)

            # Extract thinking process from SQL comments (if any)
            thinking_note = None
            if sql and sql.strip().startswith("--"):
                # Extract first comment line as thinking note
                first_line = sql.strip().split("\n")[0]
                if "selected" in first_line.lower() or "chosen" in first_line.lower():
                    thinking_note = first_line.replace("--", "").strip()

            # Generate structured summary
            summary_context = summarize_sql_result(sql_result)

            # Generate natural language summary for conversational responses
            natural_language_summary = summarize_sql_result_natural_language(
                query, sql_result
            )

            # If query mentions random month, extract and include the selected month
            if "random month" in query.lower():
                import re

                month_match = re.search(
                    r"strftime\('%m',\s*order_purchase_timestamp\)\s*=\s*'(\d+)'", sql
                )
                if month_match:
                    month_num = int(month_match.group(1))
                    month_names = [
                        "January",
                        "February",
                        "March",
                        "April",
                        "May",
                        "June",
                        "July",
                        "August",
                        "September",
                        "October",
                        "November",
                        "December",
                    ]
                    month_name = (
                        month_names[month_num - 1]
                        if 1 <= month_num <= 12
                        else str(month_num)
                    )
                    natural_language_summary = (
                        f"For {month_name} 2017: {natural_language_summary}"
                    )

            # If there's a thinking note from SQL, prepend it to summary
            if thinking_note:
                natural_language_summary = (
                    f"💡 {thinking_note}\n\n{natural_language_summary}"
                )

            # Build narrative summary with context
            # FIXED: Always generate summary from FRESH data, not contaminated context
            rows_text = "\n".join(
                ", ".join(f"{k}: {v}" for k, v in row.items())
                for row in sql_result["rows"]
            )
            fresh_data_for_summary = (
                f"### SQL Query\n{sql}\n\n### Query Results\n{rows_text}"
            )
            summary = self.summariser.generate_summary(query, fresh_data_for_summary)

            return format_response(
                "sql",
                query=sql,
                params=sql_params,
                sql_result=sql_result,
                summary=summary,
                summary_context=summary_context,
                natural_language_summary=natural_language_summary,
                warnings=warnings if warnings else None,
            )
        except Exception as e:
            logger.exception("SQL mode error: %s", str(e))
            return {"error": f"SQL mode error: {str(e)}"}

    def _handle_rag_mode(self, query, action, params, q_lower, context_string=""):
        """Handle RAG and Hybrid SQL+RAG queries with conversational context."""
        try:
            collection = _select_collection(
                query,
                default="olist_reviews" if action == "SQL+RAG" else "olist_products",
            )

            docs = retrieve_by_text(
                query, collection_name=collection, k=15 if action == "SQL+RAG" else 10
            )
            if isinstance(docs, dict) and "error" in docs:
                return {"error": f"Retrieval failed: {docs['error']}"}

            docs_list = docs if isinstance(docs, list) else docs.get("results", [])
            context = "\n\n".join(
                d.get("document", "") for d in docs_list if isinstance(d, dict)
            )

            enhanced_context = context
            if collection == "olist_reviews" and context:
                lang = detect_language(context[:200])
                if lang == "pt":
                    enhanced_context = translate_to_english(context)

            sentiment_summary = None
            if collection == "olist_reviews" and enhanced_context:
                sentiment_summary = _analyze_reviews(enhanced_context, top_n=5)

            sql_result = None
            sql_context = ""
            summary_context = ""
            if self.hybrid_mode and action == "SQL+RAG":
                sql_result, sql_context = fetch_sql_context(q_lower, params)
                if sql_result:
                    # Translate Portuguese category names to English
                    sql_result = _translate_sql_results(sql_result)
                    summary_context = summarize_sql_result(sql_result)

            se_str = self._build_sentiment_string(sentiment_summary)

            # Prepend conversational context if available
            full_context = (
                f"{context_string}\n\n{enhanced_context}"
                if context_string
                else enhanced_context
            )

            # Include summary context prominently in full context if available
            if summary_context:
                full_context = f"DATA SUMMARY (Use this quantitative information to support your answer):\n{summary_context}\n\nRETRIEVED CONTEXT:\n{full_context}"

            answer = self._summarize_or_chunk(full_context, sql_context, query, se_str)

            result = format_response(
                "hybrid" if action == "SQL+RAG" else "rag",
                context=enhanced_context,
                answer=answer,
                summary_context=summary_context,
            )
            if sql_result:
                result["sql_result"] = sql_result  # type: ignore
            if sentiment_summary:
                result["sentiment_summary"] = sentiment_summary  # type: ignore

            return result
        except Exception as e:
            logger.exception("RAG mode error: %s", str(e))
            return {"error": f"RAG mode error: {str(e)}"}

    def _summarize_or_chunk(self, context, sql_context, query, se_str):
        """Decide whether to summarize or chunk based on context length."""
        if sql_context:
            enriched_context = f"{sql_context}\n\n{context}"
            summary = self.summariser.generate_summary(query, enriched_context, se_str)
        else:
            summary = self.summariser.generate_summary(query, context, se_str)

        if len(context) > 1000:
            return summary
        else:
            return self._generate_chunked_answer(context, sql_context, query, se_str)

    def _build_sentiment_string(self, sentiment_summary):
        """Build sentiment string for prompts."""
        if not sentiment_summary or not isinstance(sentiment_summary, dict):
            return ""

        se_str = (
            f"Overall Sentiment: {sentiment_summary.get('overall_sentiment', 'N/A')} "
            f"(score: {sentiment_summary.get('overall_score', 0.0):.2f})\n"
        )
        highlights = sentiment_summary.get("highlights", {})
        if highlights.get("positive"):
            se_str += f"\nSample Positive Review: {highlights['positive'][0][:250]}"
        if highlights.get("negative"):
            se_str += f"\nSample Negative Review: {highlights['negative'][0][:250]}"
        return se_str

    def _generate_chunked_answer(self, context, sql_context, query, se_str):
        """Generate answer from context chunks with conversation history."""
        # Split context into conversation history and retrieved data
        conversation_context = ""
        retrieved_context = context

        # Check if context contains conversation history
        if "Previous conversation:" in context:
            # Split on the first occurrence of a data section
            split_markers = [
                "\n\nReview",
                "\n\nData Summary",
                "\n\nSQL Query",
                "\n\n### ",
            ]
            split_index = len(context)

            for marker in split_markers:
                idx = context.find(marker)
                if idx != -1 and idx < split_index:
                    split_index = idx

            if split_index < len(context):
                conversation_context = context[:split_index].strip()
                retrieved_context = context[split_index:].strip()
            else:
                # Fallback: take first 500 chars as conversation
                conversation_context = context[:500] if len(context) > 500 else ""
                retrieved_context = context[500:] if len(context) > 500 else context

        context_chunks = _chunk_context(retrieved_context)
        final_answer_parts = []

        for i, chunk in enumerate(context_chunks):
            prompt = f"""
You are a helpful assistant analyzing Olist e-commerce data.

{conversation_context}

{sql_context}

Chunk {i+1} of Retrieved Context:
{chunk}

User Query: {query}

{se_str}

Instructions:
1. Consider previous conversation context to provide coherent follow-up answers.
2. Use SQL results if present to support reasoning.
3. Analyze customer trends, satisfaction, or product performance.
4. Highlight specific data points (numbers, categories, reviews).
5. Be clear if information is translated or partial.
Answer:"""
            answer_part = call_llm(prompt, max_tokens=LLMTokenLimits.LONG)
            final_answer_parts.append(answer_part)

        return "\n\n".join(final_answer_parts)

    def _handle_code_mode(self, query, q_lower, context_string=""):
        """Handle code generation queries - generates and executes code."""
        try:
            if any(word in q_lower for word in ["sql", "query", "select", "database"]):
                # Generate SQL code
                result_sql = generate_sql(query, context_info=context_string)
                if not isinstance(result_sql, dict):
                    return {"error": "Failed to generate SQL code"}

                sql_query = result_sql.get("sql", "")
                # Execute the generated SQL
                execution_result = run_sql_safe(sql_query, {})

                return format_response(
                    "code",
                    language="sql",
                    file_path=result_sql.get("path", ""),
                    code=sql_query,
                    execution_result=execution_result,
                    message=f"SQL query generated, saved, and executed",
                )
            else:
                # Generate Python code
                code_text = generate_code(query, context_info=context_string)
                if not isinstance(code_text, str):
                    return {"error": "Failed to generate Python code"}

                # Save the generated code
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Clean query text to remove quotes and invalid filename characters
                clean_query = query.replace('"', "").replace("'", "").replace("`", "")
                slug = (
                    "_".join(clean_query.lower().split()[:5])
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(":", "_")
                    .replace("*", "_")
                    .replace("?", "_")
                    .replace('"', "_")
                    .replace("<", "_")
                    .replace(">", "_")
                    .replace("|", "_")
                )
                file_path = GENERATED_DIR / f"{timestamp}_{slug}.py"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code_text)

                # Execute the generated code
                logger.info(f"Executing generated Python code from {file_path}")
                execution_result = self.executor.execute_python(str(file_path))

                # Log execution
                log_execution("python", code_text, execution_result)

                # Generate AI insights about the results
                conversational_summary = self._generate_code_insights(
                    query, code_text, execution_result
                )

                preview = code_text[:500] + ("..." if len(code_text) > 500 else "")
                return format_response(
                    "code",
                    language="python",
                    file_path=str(file_path),
                    code=preview,
                    execution_result=execution_result,
                    message=f"I've analyzed your request and generated insights from the data.",
                    conversational_summary=conversational_summary,
                )
        except Exception as e:
            logger.exception("Code generation error: %s", str(e))
            return {"error": f"Code generation error: {str(e)}"}

    def _generate_code_insights(self, query, code_text, execution_result):
        """Generate conversational insights about code execution results."""
        try:
            # Prepare context for AI insight generation
            context = f"User Query: {query}\n\nGenerated Code:\n{code_text[:1000]}..."

            if execution_result.get("status") == "success":
                if execution_result.get("output_type") == "image":
                    context += "\n\nResult: Generated visualization plots"
                elif execution_result.get("stdout"):
                    context += (
                        f"\n\nExecution Output:\n{execution_result['stdout'][:1000]}"
                    )
                else:
                    context += "\n\nResult: Code executed successfully"
            else:
                context += (
                    f"\n\nError: {execution_result.get('error', 'Unknown error')}"
                )

            prompt = f"""
You are a data analyst explaining visualization results to a business user. Based on this analysis:

{context}

Provide a detailed, insightful explanation (3-4 sentences) about what the visualizations reveal. Focus on:
- Key patterns and trends visible in the charts
- Comparative insights (which states/categories are leading, what the distribution shows)
- Business implications and actionable recommendations
- Specific numbers or statistics if available in the output

Example tone: "The visualization reveals that São Paulo (SP) leads with over 15,000 orders, followed by Rio de Janeiro and Minas Gerais, showing a clear concentration of business in Brazil's southeast region. This geographic pattern suggests we should focus marketing efforts and inventory distribution in these high-volume states. The significant drop-off after the top 3 states indicates potential for growth in other regions through targeted campaigns."

Keep it friendly, specific, and actionable. Use concrete observations from the data.
"""

            insight = call_llm(prompt, max_tokens=LLMTokenLimits.SHORT)
            return insight if insight else "Analysis completed successfully."

        except Exception as e:
            logger.exception("Error generating code insights: %s", str(e))
            return "Analysis completed. Check the results above for details."

    def _handle_conversation_mode(self, query):
        """Handle conversational queries with friendly responses."""
        try:
            # Generate a friendly conversational response
            prompt = f"""
You are a helpful AI assistant for an e-commerce data analysis system. The user has asked: "{query}"

This appears to be a conversational query (greeting, thanks, etc.), not a data analysis request.
Respond in a friendly, helpful manner. Keep your response brief and conversational.

If it's a greeting, respond warmly and offer to help with data analysis.
If it's thanks, acknowledge and offer further assistance.
If it's a goodbye, respond politely.

Response:"""

            response = call_llm(prompt, max_tokens=LLMTokenLimits.SHORT)
            if not response:
                # Fallback response
                if any(
                    word in query.lower()
                    for word in ["hi", "hello", "hey", "greetings"]
                ):
                    response = "Hello! I'm here to help you analyze your e-commerce data. What would you like to know about your sales, customers, or products?"
                elif any(word in query.lower() for word in ["thank", "thanks"]):
                    response = "You're welcome! I'm here whenever you need help analyzing your e-commerce data."
                elif any(
                    word in query.lower() for word in ["bye", "goodbye", "see you"]
                ):
                    response = "Goodbye! Feel free to come back anytime for data analysis insights."
                else:
                    response = "I'm here to help with your e-commerce data analysis. What would you like to explore?"

            return format_response(
                "conversation",
                response=response,
                message="Conversational response generated",
            )
        except Exception as e:
            logger.exception("Conversation handling error: %s", str(e))
            return {"error": f"Conversation handling error: {str(e)}"}

    def _handle_metadata_mode(self, query):
        """Handle dataset overview and metadata queries."""
        try:
            # Get database schema information
            schema_info = self.get_schema_info()

            # Query database for key statistics
            stats_queries = {
                "total_orders": "SELECT COUNT(*) as count FROM orders",
                "total_customers": "SELECT COUNT(DISTINCT customer_id) as count FROM customers",
                "total_products": "SELECT COUNT(DISTINCT product_id) as count FROM products",
                "total_sellers": "SELECT COUNT(DISTINCT seller_id) as count FROM sellers",
                "date_range": "SELECT MIN(order_purchase_timestamp) as start_date, MAX(order_purchase_timestamp) as end_date FROM orders",
                "total_revenue": "SELECT SUM(payment_value) as total FROM order_payments",
                "avg_order_value": "SELECT AVG(payment_value) as avg_value FROM order_payments",
            }

            stats = {}
            for key, sql_query in stats_queries.items():
                try:
                    result = run_sql_safe(sql_query, {})
                    if result.get("status") == "success" and result.get("rows"):
                        stats[key] = result["rows"][0]
                except Exception as e:
                    logger.warning(f"Failed to get {key}: {e}")
                    stats[key] = {"error": str(e)}

            # Generate comprehensive dataset description using LLM
            prompt = f"""
You are a data analyst assistant. The user asked: "{query}"

Here is comprehensive information about the available dataset:

**Database Schema:**
{schema_info}

**Dataset Statistics:**
- Total Orders: {stats.get('total_orders', {}).get('count', 'N/A')}
- Total Customers: {stats.get('total_customers', {}).get('count', 'N/A')}
- Total Products: {stats.get('total_products', {}).get('count', 'N/A')}
- Total Sellers: {stats.get('total_sellers', {}).get('count', 'N/A')}
- Date Range: {stats.get('date_range', {}).get('start_date', 'N/A')} to {stats.get('date_range', {}).get('end_date', 'N/A')}
- Total Revenue: ${stats.get('total_revenue', {}).get('total', 0):,.2f}
- Average Order Value: ${stats.get('avg_order_value', {}).get('avg_value', 0):,.2f}

**Available Data:**
This is an e-commerce dataset (Olist Brazilian E-Commerce) containing:
1. **Orders** - Customer purchase transactions with timestamps and status
2. **Order Items** - Individual products in each order with pricing
3. **Products** - Product catalog with categories and dimensions
4. **Customers** - Customer information including location (city, state)
5. **Sellers** - Seller information including location
6. **Payments** - Payment transactions and methods
7. **Reviews** - Customer reviews with scores and comments
8. **Category Translation** - Product category names in English

Provide a comprehensive, friendly overview of what this dataset contains and what kinds of analysis can be performed with it. Be specific about the data types and relationships. Make it conversational and helpful.

Response:"""

            response = call_llm(prompt, max_tokens=LLMTokenLimits.CODE)
            if not response:
                # Fallback response
                response = f"""This is the **Olist Brazilian E-Commerce dataset** containing comprehensive information about e-commerce transactions.

**Dataset Overview:**
- 📦 **{stats.get('total_orders', {}).get('count', 'N/A'):,} orders** from {stats.get('total_customers', {}).get('count', 'N/A'):,} customers
- 🛍️ **{stats.get('total_products', {}).get('count', 'N/A'):,} unique products** sold by {stats.get('total_sellers', {}).get('count', 'N/A'):,} sellers
- 💰 **Total revenue**: ${stats.get('total_revenue', {}).get('total', 0):,.2f}
- 📅 **Time period**: {stats.get('date_range', {}).get('start_date', 'N/A')} to {stats.get('date_range', {}).get('end_date', 'N/A')}

**Available Analysis:**
- Sales trends and revenue analysis
- Product category performance
- Customer behavior and satisfaction (via reviews)
- Seller performance metrics
- Geographic distribution of orders
- Payment patterns and preferences

You can ask me to analyze any aspect of this data!"""

            return format_response(
                "metadata",
                response=response,
                schema=schema_info,
                statistics=stats,
                message="Dataset overview generated",
            )
        except Exception as e:
            logger.exception("Metadata handling error: %s", str(e))
            return {"error": f"Metadata handling error: {str(e)}"}

    # -------------------------------
    # SCHEMA INTROSPECTION METHODS
    # -------------------------------
    def get_table_columns(self, table_name):
        """Get column names for a given table."""
        try:
            query = f"PRAGMA table_info({table_name});"
            result = run_sql_safe(query)
            if (
                isinstance(result, dict)
                and "rows" in result
                and isinstance(result["rows"], list)
            ):
                return [
                    row.get("name")
                    for row in result["rows"]
                    if isinstance(row, dict) and row.get("name") is not None
                ]
            return []
        except Exception as e:
            logger.error(f"Failed to get columns for table {table_name}: {str(e)}")
            return []

    def execute_sql_query(self, sql_query, params=None):
        """Execute SQL query with schema-aware error correction."""
        try:
            # First attempt
            result = run_sql_safe(sql_query, params)
            if not isinstance(result, dict) or "error" not in result:
                return {"success": True, "result": result}

            # If error, try to fix it
            error_msg = result.get("error", "")
            if "no such column" in error_msg.lower():
                fixed_query = self.fix_query_with_schema(sql_query, error_msg)
                if fixed_query != sql_query:
                    logger.info(
                        f"Attempting to fix SQL query. Original: {sql_query[:100]}..., Fixed: {fixed_query[:100]}..."
                    )
                    result = run_sql_safe(fixed_query, params)
                    if not isinstance(result, dict) or "error" not in result:
                        return {
                            "success": True,
                            "result": result,
                            "corrected": True,
                            "original_query": sql_query,
                        }

            return {"success": False, "error": error_msg, "result": result}

        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            return {"success": False, "error": str(e)}

    def fix_query_with_schema(self, sql_query, error_msg):
        """Fix SQL query by replacing invalid columns with valid ones based on schema."""
        try:
            # Extract table aliases and invalid columns from error
            import re

            # Common invalid column patterns from AI generation
            invalid_patterns = [
                (
                    r"\boi\.quantity\b",
                    "oi.order_item_id",
                ),  # quantity doesn't exist, use order_item_id
                (
                    r"\boi\.product\b",
                    "oi.product_id",
                ),  # product doesn't exist, use product_id
                (
                    r"\bp\.category\b",
                    "p.product_category_name",
                ),  # category might be wrong
                (r"\bo\.status\b", "o.order_status"),  # status might be misspelled
            ]

            fixed_query = sql_query
            for pattern, replacement in invalid_patterns:
                fixed_query = re.sub(
                    pattern, replacement, fixed_query, flags=re.IGNORECASE
                )

            # If no pattern matches, try to identify table and get valid columns
            if fixed_query == sql_query:
                # Look for table aliases in query
                table_aliases = {}
                alias_pattern = r"(\w+)\s+AS\s+(\w+)|FROM\s+(\w+)\s+(\w+)"
                matches = re.findall(alias_pattern, sql_query, re.IGNORECASE)
                for match in matches:
                    if match[0] and match[1]:  # AS syntax
                        table_aliases[match[1]] = match[0]
                    elif match[2] and match[3]:  # FROM table alias syntax
                        table_aliases[match[3]] = match[2]

                # For common invalid columns, try to find valid alternatives
                if "oi.quantity" in sql_query.lower():
                    # Check if order_items has any numeric columns that could represent quantity
                    columns = self.get_table_columns("olist_order_items_dataset")
                    numeric_cols = [
                        col
                        for col in columns
                        if col is not None
                        and any(
                            term in col.lower() for term in ["price", "freight", "item"]
                        )
                    ]
                    if numeric_cols:
                        fixed_query = sql_query.replace(
                            "oi.quantity", f"oi.{numeric_cols[0]}"
                        )

            return fixed_query

        except Exception as e:
            logger.error(f"Failed to fix query: {str(e)}")
            return sql_query

    # -------------------------------
    # SCHEMA INFO FOR AI GENERATION
    # -------------------------------
    def get_schema_info(self) -> str:
        """Get database schema information for AI generation."""
        try:
            # Use actual table names from database
            tables = [
                "orders",
                "order_items",
                "products",
                "customers",
                "sellers",
                "order_payments",
                "order_reviews",
                "category_translation",
            ]

            schema_parts = []
            for table in tables:
                columns = self.get_table_columns(table)
                if columns:
                    # Filter out None values before joining
                    valid_columns = [col for col in columns if col is not None]
                    if valid_columns:
                        schema_parts.append(f"- {table}: {', '.join(valid_columns)}")

            if schema_parts:
                return "Database Schema:\n" + "\n".join(schema_parts)
            else:
                # Fallback to hardcoded schema if introspection fails
                return """
Database Schema:
- orders: order_id, customer_id, order_status, order_purchase_timestamp, order_delivered_customer_date
- order_items: order_id, order_item_id, product_id, seller_id, price, freight_value
- products: product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm
- customers: customer_id, customer_city, customer_state
- sellers: seller_id, seller_city, seller_state
- order_payments: order_id, payment_type, payment_value
- order_reviews: review_id, order_id, review_score, review_comment_message
- category_translation: product_category_name, product_category_name_english
"""
        except Exception as e:
            logger.error(f"Failed to get schema info: {str(e)}")
            # Return fallback schema
            return """
Database Schema:
- orders: order_id, customer_id, order_status, order_purchase_timestamp, order_delivered_customer_date
- order_items: order_id, order_item_id, product_id, seller_id, price, freight_value
- products: product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm
- customers: customer_id, customer_city, customer_state
- sellers: seller_id, seller_city, seller_state
- order_payments: order_id, payment_type, payment_value
- order_reviews: review_id, order_id, review_score, review_comment_message
- category_translation: product_category_name, product_category_name_english
"""
