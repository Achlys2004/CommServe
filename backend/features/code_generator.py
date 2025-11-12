import os
import json
import logging
from datetime import datetime
from pathlib import Path
from backend.llm_client import call_llm
from backend.llm_config import LLMTokenLimits
from backend.utils.code_validator import validate_and_fix_code, extract_python_code
from config import GENERATED_DIR

logger = logging.getLogger(__name__)


def load_schema():
    """Load database schema from JSON file."""
    schema_path = (
        Path(__file__).parent.parent.parent / "data" / "schema.json"
    )  # Go up two levels to project root, then to data/
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # Fallback to basic schema if file not found
        return {
            "tables": {
                "orders": {
                    "columns": [
                        "order_id",
                        "customer_id",
                        "order_status",
                        "order_purchase_timestamp",
                    ]
                },
                "order_items": {"columns": ["order_id", "product_id", "price"]},
                "products": {"columns": ["product_id", "product_category_name"]},
                "customers": {
                    "columns": ["customer_id", "customer_city", "customer_state"]
                },
                "sellers": {"columns": ["seller_id", "seller_city"]},
                "order_payments": {
                    "columns": ["order_id", "payment_type", "payment_value"]
                },
                "order_reviews": {"columns": ["order_id", "review_score"]},
                "category_translation": {
                    "columns": [
                        "product_category_name",
                        "product_category_name_english",
                    ]
                },
            }
        }


def get_basic_schema_context() -> str:
    """Fallback basic schema context."""
    schema_text = """## Database Tables:
- orders: order_id, customer_id, order_status, order_purchase_timestamp, order_delivered_customer_date
- order_items: order_id, order_item_id, product_id, seller_id, price, freight_value
- products: product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm
- customers: customer_id, customer_unique_id, customer_city, customer_state
- sellers: seller_id, seller_city, seller_state
- order_payments: order_id, payment_sequential, payment_type, payment_installments, payment_value
- order_reviews: review_id, order_id, review_score, review_comment_message
- category_translation: product_category_name, product_category_name_english
"""
    return schema_text


def _generate_fallback_code(query: str) -> str:
    """
    Generate a simple, guaranteed-to-work fallback visualization code.
    Used when LLM-generated code fails validation.
    """
    # Extract date if mentioned in query
    date_filter = ""
    query_lower = query.lower()

    # Look for specific months
    months = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }

    year = "2017"  # default
    month = "01"  # default

    for month_name, month_num in months.items():
        if month_name in query_lower:
            month = month_num
            break

    if "2016" in query_lower:
        year = "2016"
    elif "2018" in query_lower:
        year = "2018"

    date_filter = f"{year}-{month}"

    return f"""import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to database
conn = sqlite3.connect('data/olist.db')

try:
    # Query for {date_filter} sales data
    query = \"\"\"
    SELECT 
        strftime('%Y-%m-%d', o.order_purchase_timestamp) as date,
        COUNT(DISTINCT o.order_id) as order_count,
        SUM(CAST(oi.price AS REAL)) as total_revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE strftime('%Y-%m', o.order_purchase_timestamp) = '{date_filter}'
    AND o.order_status = 'delivered'
    GROUP BY date
    ORDER BY date
    LIMIT 100
    \"\"\"
    
    df = pd.read_sql(query, conn)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Daily orders
    ax1.plot(df['date'], df['order_count'], marker='o', linewidth=2)
    ax1.set_title(f'Daily Orders - {date_filter}', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Orders')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily revenue
    ax2.bar(df['date'], df['total_revenue'], color='green', alpha=0.7)
    ax2.set_title(f'Daily Revenue - {date_filter}', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Revenue (R$)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print insights
    print("=" * 50)
    print(f"Sales Analysis for {{date_filter}}")
    print("=" * 50)
    print(f"Total Orders: {{df['order_count'].sum():,}}")
    print(f"Total Revenue: R$ {{df['total_revenue'].sum():,.2f}}")
    print(f"Average Daily Orders: {{df['order_count'].mean():.1f}}")
    print(f"Average Daily Revenue: R$ {{df['total_revenue'].mean():,.2f}}")
    print(f"Peak Orders: {{df['order_count'].max()}} on {{df.loc[df['order_count'].idxmax(), 'date']}}")
    print("=" * 50)
    
finally:
    conn.close()
"""


def generate_sql(query, context_info="", schema_info=None):
    try:
        # Load schema
        schema = load_schema()
        dataset_info = schema.get("dataset_info", {})
        date_range = dataset_info.get("date_range", "September 2016 - August 2018")

        # Build context string for the prompt
        context_section = ""
        if context_info:
            context_section = f"""
{context_info}

Important Instructions:
- If this is a follow-up query, use the previous context to understand what the user is referring to
- Generate ONLY ONE SQL query, even if the request mentions multiple things
- For follow-ups like "bottom 5" after "top 5", change ORDER BY ... DESC to ASC
"""

        # Build schema section with detailed context
        if schema_info:
            schema_section = schema_info
        else:
            schema_section = get_basic_schema_context()

        prompt = f"""
You are an expert SQL developer working with the Olist e-commerce dataset in SQLite.

DATASET CONTEXT:
- Date Range: {date_range}
- Database: SQLite (use SQLite-specific syntax)
- Currency: Brazilian Real (R$)

{schema_section}

CRITICAL SCHEMA RULES (MUST FOLLOW):
1. ✅ Table is 'order_payments' NOT 'payments'
2. ✅ City/state columns are in 'customers' table, NOT 'orders'
3. ✅ For geographic queries: FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE c.customer_city = ...
4. ✅ Always use LEFT JOIN for category_translation with COALESCE
5. ✅ Filter by o.order_status = 'delivered' for completed orders

IMPORTANT - SQLite Syntax:
- Dates: Use strftime() for date operations
- Quarters: Calculate using CASE with month ranges (SQLite has no %q format)
- No DATEDIFF - use julianday() for date arithmetic
- Always use proper JOINs and COALESCE for NULL handling

User Request: {query}{context_section}

QUALITY REQUIREMENTS:
1. Generate syntactically correct SQLite query
2. Return meaningful, labeled columns (use AS aliases)
3. Join category_translation for product names (use COALESCE for NULL)
4. Filter for 'delivered' status when analyzing completed orders
5. Use appropriate aggregations (COUNT, SUM, AVG)
6. When user requests "random" month/year/period, include the selected value as the first column in results
7. Return ONLY the SQL query, no explanations

Generate the SQL query now:"""

        sql_text = call_llm(prompt, max_tokens=LLMTokenLimits.SQL)

        if not sql_text or not sql_text.strip():
            sql_text = "-- Error: Failed to generate SQL"

        # Clean up the response (remove markdown if present)
        sql_text = sql_text.strip()
        if sql_text.startswith("```sql"):
            sql_text = sql_text[6:]
        if sql_text.endswith("```"):
            sql_text = sql_text[:-3]
        sql_text = sql_text.strip()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "_".join(query.lower().split()[:5]).replace("/", "_").replace("\\", "_")
        file_path = GENERATED_DIR / f"{timestamp}_{slug}.sql"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sql_text)

        return {"path": str(file_path), "sql": sql_text}

    except Exception as e:
        error_sql = f"-- Error generating SQL: {str(e)}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = GENERATED_DIR / f"{timestamp}_error.sql"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(error_sql)
        return {"path": str(file_path), "sql": error_sql}


def generate_code(query, context_info=""):
    try:
        # Get comprehensive schema context for the LLM
        schema_context = get_basic_schema_context()

        prompt = f"""You are an expert Python developer. Generate clean, executable Python code for data analysis.

DATABASE SCHEMA:
{schema_context}

{"CONTEXT: " + context_info if context_info else ""}

USER REQUEST: {query}

REQUIREMENTS:
1. Database: Connect to 'data/olist.db' using sqlite3
2. Visualizations: Use matplotlib/seaborn (do NOT call plt.show())
3. Data handling: Use pandas DataFrames
4. Insights: Print 3-5 specific findings with numbers
5. Error handling: Use try-except blocks
6. Syntax: MUST be valid Python - balanced parentheses, closed strings

CODE STRUCTURE (follow exactly):
```
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to database
conn = sqlite3.connect('data/olist.db')

try:
    # Query 1: [describe what it does]
    query1 = \"\"\"
    SELECT ... FROM orders
    WHERE strftime('%Y-%m', order_purchase_timestamp) = '2017-01'
    AND order_status = 'delivered'
    LIMIT 1000
    \"\"\"
    df1 = pd.read_sql(query1, conn)
    
    # Create visualization 1
    plt.figure(figsize=(10, 6))
    # ... plotting code ...
    plt.title('Title')
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.tight_layout()
    
    # Print insights
    print("=== Analysis Results ===")
    print(f"Total records: {{len(df1)}}")
    
finally:
    conn.close()
```

CRITICAL RULES:
- Keep queries SIMPLE with LIMIT clause
- Use triple-quoted strings for SQL (easier to read)
- Test each line mentally - does it compile?
- Close ALL parentheses before moving to next line
- Use simple variable names (df, query, fig)
- NO complex nested function calls
- Print clear insights with {{}} formatting

Generate ONLY the Python code (no explanations, no markdown):"""

        # Try to generate code with retry logic and validation
        max_attempts = 3
        code_text = None

        for attempt in range(max_attempts):
            code_text = call_llm(prompt, max_tokens=LLMTokenLimits.CODE)

            if not code_text or not code_text.strip():
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Empty response from LLM, attempt {attempt + 1}/{max_attempts}"
                    )
                    continue
                code_text = "# Error: Failed to generate code"
                break

            # Extract code from markdown
            code_text = extract_python_code(code_text)
            code_text = code_text.strip()
            if code_text.startswith("```"):
                code_text = code_text[3:].strip()
            if code_text.endswith("```"):
                code_text = code_text[:-3].strip()

            # Validate the generated code
            is_valid, fixed_code, error_msg = validate_and_fix_code(code_text)

            if is_valid:
                code_text = fixed_code
                logger.info(
                    f"Generated code validated successfully on attempt {attempt + 1}"
                )
                break
            else:
                logger.warning(
                    f"Attempt {attempt + 1}: Code validation failed - {error_msg}"
                )

                if attempt < max_attempts - 1:
                    # Try again with a simpler prompt emphasizing syntax correctness
                    prompt = f"""The previous code had syntax error: {error_msg}

Generate SIMPLE, VALID Python code for: {query}

Use this exact structure:
```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('data/olist.db')
try:
    query = \"\"\"SELECT * FROM orders WHERE strftime('%Y-%m', order_purchase_timestamp) = '2017-01' LIMIT 100\"\"\"
    df = pd.read_sql(query, conn)
    
    plt.figure(figsize=(10, 6))
    df['column'].value_counts().head(10).plot(kind='bar')
    plt.title('Title')
    plt.tight_layout()
    
    print(f"Total: {{len(df)}}")
finally:
    conn.close()
```

CRITICAL: Balance ALL parentheses. Close ALL strings. Keep it SIMPLE.
Generate code now:"""
                    continue
                else:
                    # Last attempt failed, use the error message
                    logger.error(
                        f"All {max_attempts} attempts failed to generate valid code"
                    )
                    break

        # Ensure code_text is never None
        if not code_text:
            code_text = "# Error: Failed to generate code"

        # Final validation check
        is_valid, fixed_code, error_msg = validate_and_fix_code(code_text)

        if is_valid:
            code_text = fixed_code
            logger.info("Final code validation successful")
        else:
            # Validation still failed after all attempts - use fallback
            logger.warning(f"Final validation failed: {error_msg}")
            logger.info("Using guaranteed fallback visualization template")
            code_text = _generate_fallback_code(query)

            # Validate the fallback (should always work)
            is_valid, fixed_code, error_msg = validate_and_fix_code(code_text)
            if is_valid:
                code_text = fixed_code
                logger.info("Fallback code validated successfully")
            else:
                # Even fallback failed (very unlikely) - last resort error message
                logger.error(f"Even fallback code failed validation: {error_msg}")
                error_details = f"\nSyntax Error Details:\n- Message: {error_msg}\n"
                error_msg_repr = repr(error_details)
                code_text = (
                    "# Syntax error in generated code\nimport sqlite3\nimport pandas as pd\n\nprint('Code generation encountered a syntax error.')\nprint("
                    + error_msg_repr
                    + ")\nprint()\nprint('This usually happens when:')\nprint('1. The AI response contains unterminated strings')\nprint('2. Unbalanced brackets or parentheses')\nprint('3. Invalid character encoding')\nprint()\nprint('Please try rephrasing your request with more specific details.')\nprint('Example: \"Create a bar chart showing top 10 cities by order count\"')\n"
                )

        return code_text

    except Exception as gen_error:
        error_msg = repr(str(gen_error))
        return (
            "# Error generating code\nimport sqlite3\nimport pandas as pd\n\nprint('Code generation failed.')\nprint('Error:', "
            + error_msg
            + ")\n"
        )


def get_detailed_schema_context() -> str:
    # Get comprehensive database schema information with column types and relationships.
    # This provides the LLM with complete awareness of the database structure.
    try:
        schema = load_schema()
        context_parts = []

        # Dataset overview
        dataset_info = schema.get("dataset_info", {})
        context_parts.append("## Olist Brazilian E-Commerce Dataset")
        context_parts.append(
            f"- Date Range: {dataset_info.get('date_range', '2016-2018')}"
        )
        context_parts.append(
            f"- Description: {dataset_info.get('description', 'E-commerce dataset')}"
        )
        context_parts.append("")

        # Key facts
        key_facts = dataset_info.get("key_facts", [])
        if key_facts:
            context_parts.append("## Key Facts:")
            for fact in key_facts:
                context_parts.append(f"- {fact}")
            context_parts.append("")

        # Tables with detailed column information
        context_parts.append("## Database Tables:")
        for table_name, table_info in schema["tables"].items():
            context_parts.append(f"### {table_name}")
            context_parts.append(
                f"**Description:** {table_info.get('description', 'No description')}"
            )

            # Primary key info
            pk = table_info.get("primary_key", [])
            if isinstance(pk, list):
                pk_str = ", ".join(pk)
            else:
                pk_str = str(pk)
            context_parts.append(f"**Primary Key:** {pk_str}")

            # Columns
            context_parts.append("**Columns:**")
            for col in table_info["columns"]:
                # Try to infer data types from column names
                col_type = infer_column_type(col)
                context_parts.append(f"  - `{col}`: {col_type}")

            context_parts.append("")

        # Relationships
        relationships = schema.get("relationships", [])
        if relationships:
            context_parts.append("## Table Relationships:")
            for rel in relationships:
                if (
                    isinstance(rel, dict)
                    and "from" in rel
                    and "to" in rel
                    and "type" in rel
                ):
                    from_table, from_col = rel["from"].split(".")
                    to_table, to_col = rel["to"].split(".")
                    rel_type = rel["type"]
                    context_parts.append(
                        f"- `{from_table}.{from_col}` → `{to_table}.{to_col}` ({rel_type})"
                    )
            context_parts.append("")

        # Common JOIN patterns
        context_parts.append("## Common JOIN Patterns:")
        context_parts.append("SQL Examples:")
        context_parts.append("-- Geographic analysis (cities/states)")
        context_parts.append("FROM orders o")
        context_parts.append("JOIN customers c ON o.customer_id = c.customer_id")
        context_parts.append("")
        context_parts.append("-- Product analysis with categories")
        context_parts.append("FROM order_items oi")
        context_parts.append("JOIN products p ON oi.product_id = p.product_id")
        context_parts.append(
            "LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name"
        )
        context_parts.append("")
        context_parts.append("-- Complete order analysis")
        context_parts.append("FROM orders o")
        context_parts.append("JOIN customers c ON o.customer_id = c.customer_id")
        context_parts.append("JOIN order_items oi ON o.order_id = oi.order_id")
        context_parts.append("JOIN products p ON oi.product_id = p.product_id")
        context_parts.append("LEFT JOIN order_payments op ON o.order_id = op.order_id")
        context_parts.append("LEFT JOIN order_reviews r ON o.order_id = r.order_id")
        context_parts.append("")

        return "\n".join(context_parts)

    except Exception as e:
        logger.warning(f"Could not load detailed schema context: {e}")
        return get_basic_schema_context()


def infer_column_type(column_name: str) -> str:
    # Infer likely data type from column name.
    col_lower = column_name.lower()

    # ID columns
    if col_lower.endswith("_id") or col_lower in [
        "order_id",
        "customer_id",
        "product_id",
        "seller_id",
    ]:
        return "TEXT (Primary/Foreign Key)"

    # Date/Timestamp columns
    if "date" in col_lower or "timestamp" in col_lower or "at" in col_lower:
        return "TIMESTAMP/DATE"

    # Numeric columns
    if any(
        word in col_lower
        for word in [
            "price",
            "value",
            "weight",
            "length",
            "height",
            "width",
            "score",
            "sequential",
            "installments",
        ]
    ):
        return "REAL/INTEGER (Numeric)"

    # Text columns
    if any(
        word in col_lower
        for word in [
            "name",
            "city",
            "state",
            "status",
            "type",
            "comment",
            "message",
            "title",
        ]
    ):
        return "TEXT (String)"

    # Default
    return "TEXT/REAL/INTEGER"
