from backend.sql_executor import run_sql_safe
from backend.features.code_generator import generate_sql
import logging

logger = logging.getLogger(__name__)


def fetch_sql_context(query_lower, params):
    """
    Fetch SQL context dynamically by generating SQL for the query.

    Args:
        query_lower: Lowercased query string
        params: Query parameters dict

    Returns:
        tuple: (sql_result, sql_context_string)
    """
    try:
        # Generate SQL dynamically using the LLM
        logger.info(f"Generating SQL for hybrid query: {query_lower}")

        sql_generation_result = generate_sql(query_lower)

        if not sql_generation_result or "error" in sql_generation_result:
            logger.warning(f"SQL generation failed: {sql_generation_result}")
            return None, ""

        sql = sql_generation_result.get("sql")  # Changed from "code" to "sql"
        if not sql or sql.startswith("-- Error"):
            logger.warning(f"No valid SQL code generated: {sql}")
            return None, ""

        # Execute the generated SQL
        sql_result = run_sql_safe(sql)

        if isinstance(sql_result, dict) and sql_result.get("status") == "success":
            rows = sql_result.get("rows", [])
            if rows:
                # Format results for context
                sql_context = f"\n\nSQL Query Results:\n{rows[:10]}\n"
                logger.info(f"SQL executed successfully, got {len(rows)} rows")
                return sql_result, sql_context
            else:
                logger.warning("SQL returned no rows")
                return None, ""
        else:
            error = (
                sql_result.get("error", "Unknown error")
                if isinstance(sql_result, dict)
                else "Invalid result"
            )
            logger.error(f"SQL execution failed: {error}")
            return None, ""

    except Exception as e:
        logger.exception(f"Error fetching SQL context: {e}")
        return None, ""
