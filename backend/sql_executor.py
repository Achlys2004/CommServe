import sqlite3
from pathlib import Path
import pandas as pd
import logging
import json
from datetime import datetime
from config import DB_PATH

logger = logging.getLogger(__name__)

ALLOWED_STATEMENTS = {"select", "with", "pragma"}


def log_execution(query_type, query, result):
    """Log execution to JSON file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"execution_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": query_type, "query": query, "result": result}, f, indent=2)
    return str(path)


def validate_sql(sql):
    """Validate SQL query for safety and relevance."""
    if not sql or not isinstance(sql, str):
        return False

    sql_lower = sql.lower().strip()

    # Allow PRAGMA table_info queries for schema introspection
    if sql_lower.startswith("pragma table_info(") and sql_lower.endswith(");"):
        return True

    allowed_tables = [
        "orders",
        "order_items",
        "products",
        "customers",
        "sellers",
        "order_payments",
        "payments",  # Keep for backward compatibility
        "order_reviews",
        "category_translation",
        "sqlite_master",  # Allow system table queries for introspection
    ]

    # Check for allowed tables
    has_valid_table = any(table in sql_lower for table in allowed_tables)

    # Basic safety checks
    dangerous_keywords = [
        "drop",
        "delete",
        "update",
        "insert",
        "alter",
        "create",
        "truncate",
    ]
    has_dangerous = any(keyword in sql_lower for keyword in dangerous_keywords)

    return has_valid_table and not has_dangerous


def run_sql_safe(sql, params=None):
    """Execute SQL with enhanced safety and logging."""
    try:
        # Validate SQL first
        if not validate_sql(sql):
            logger.warning(f"SQL validation failed for query: {sql[:100]}...")
            return {"status": "failed", "error": "SQL query failed validation"}

        result = run_sql(sql, params)

        # Log results for verification
        if isinstance(result, dict) and "error" not in result:
            rows_count = len(result.get("rows", []))
            logger.info(f"SQL executed successfully, {rows_count} rows returned")
            if rows_count > 0:
                logger.info(f"Executed Query Result (Preview): {result['rows'][:3]}")
            result["status"] = "success"

            # Log execution to JSON file
            log_execution("sql", sql, result)
        elif isinstance(result, dict) and "error" in result:
            logger.warning(f"SQL execution error: {result['error']}")
            result["status"] = "failed"

        return result

    except Exception as e:
        logger.exception(f"SQL execution failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_sql(sql, params=None, limit=None):
    """
    Execute SQL query. If limit is None, respect the query's own LIMIT clause.
    If limit is provided and query has no LIMIT, apply the given limit.
    """
    # Remove leading comments before validation
    sql_lines = sql.strip().split("\n")
    sql_without_comments = "\n".join(
        line for line in sql_lines if not line.strip().startswith("--")
    )
    sql_stripped = sql_without_comments.strip().lower()

    if not sql_stripped:
        return {"error": "Empty SQL query after removing comments."}

    first_word = sql_stripped.split()[0]
    if first_word not in ALLOWED_STATEMENTS:
        return {"error": "Only SELECT/WITH queries are allowed."}
    if not DB_PATH.parent.exists():
        return {"error": f"Data directory not found at {DB_PATH.parent}"}
    if not DB_PATH.exists():
        return {"error": f"Database not found at {DB_PATH}"}

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(sql, conn, params=params)

        # Only apply limit if explicitly provided AND query has no LIMIT clause
        if limit is not None and "limit" not in sql_stripped:
            if len(df) > limit:
                df = df.head(limit)

        result = {"columns": df.columns.tolist(), "rows": df.to_dict(orient="records")}
    except Exception as e:
        result = {"error": str(e)}
    finally:
        conn.close()
    return result


TEMPLATES = {
    "top_categories": """
        SELECT ct.product_category_name_english AS category, SUM(oi.price) AS revenue
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_purchase_timestamp BETWEEN :start AND :end
        GROUP BY category
        ORDER BY revenue DESC
        LIMIT :limit;
    """,
    "most_sold_categories": """
        SELECT ct.product_category_name_english AS category, COUNT(oi.order_item_id) AS total_sold
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_purchase_timestamp BETWEEN :start AND :end
        GROUP BY category
        ORDER BY total_sold DESC
        LIMIT :limit;
    """,
    "avg_order_value_by_category": """
        WITH order_totals AS (
            SELECT o.order_id, SUM(oi.price) AS order_total
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.order_purchase_timestamp BETWEEN :start AND :end
            GROUP BY o.order_id
        )
        SELECT ct.product_category_name_english AS category, AVG(t.order_total) AS avg_order_value
        FROM order_totals t
        JOIN order_items oi ON t.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        GROUP BY category
        ORDER BY avg_order_value DESC
        LIMIT :limit;
    """,
    "top_customers_by_revenue": """
        SELECT c.customer_id, c.customer_city, c.customer_state, SUM(oi.price) AS total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_purchase_timestamp BETWEEN :start AND :end
        GROUP BY c.customer_id, c.customer_city, c.customer_state
        ORDER BY total_spent DESC
        LIMIT :limit;
    """,
    "top_sellers_by_revenue": """
        SELECT s.seller_id, s.seller_city, s.seller_state, SUM(oi.price) AS revenue
        FROM sellers s
        JOIN order_items oi ON s.seller_id = oi.seller_id
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_purchase_timestamp BETWEEN :start AND :end
        GROUP BY s.seller_id, s.seller_city, s.seller_state
        ORDER BY revenue DESC
        LIMIT :limit;
    """,
    "revenue_by_payment_method": """
        SELECT p.payment_type, SUM(p.payment_value) AS revenue
        FROM payments p
        JOIN orders o ON p.order_id = o.order_id
        WHERE o.order_purchase_timestamp BETWEEN :start AND :end
        GROUP BY p.payment_type
        ORDER BY revenue DESC;
    """,
    "orders_by_customer_city": """
        SELECT c.customer_city, COUNT(o.order_id) AS order_count
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.order_purchase_timestamp BETWEEN :start AND :end
        GROUP BY c.customer_city
        ORDER BY order_count DESC
        LIMIT :limit;
    """,
    "avg_review_score_by_category": """
        SELECT ct.product_category_name_english AS category, AVG(r.review_score) AS avg_score
        FROM order_reviews r
        JOIN order_items oi ON r.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        GROUP BY category
        ORDER BY avg_score DESC
        LIMIT :limit;
    """,
    "top_products_by_revenue": """
        SELECT 
            p.product_id,
            ct.product_category_name_english AS category,
            p.product_category_name AS category_pt,
            SUM(oi.price) AS revenue,
            COUNT(DISTINCT oi.order_id) AS order_count,
            ROUND(AVG(oi.price), 2) AS avg_price
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_purchase_timestamp BETWEEN :start AND :end
        GROUP BY p.product_id, category, category_pt
        ORDER BY revenue DESC
        LIMIT :limit;
    """,
}
