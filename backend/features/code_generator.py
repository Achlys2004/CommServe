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
6. Return ONLY the SQL query, no explanations

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

        prompt = (
            "You are an expert Python developer for data analysis and visualization.\n\n"
            + schema_context
            + "\n\n"
            + (f"Context: {context_info}\n\n" if context_info else "")
            + "User Request: "
            + query
            + "\n\nGenerate a complete, executable Python script that:\n1. Connects to SQLite database at 'data/olist.db'\n2. Executes appropriate SQL queries using correct table/column names\n3. Creates meaningful visualizations (matplotlib/seaborn)\n4. Calculates and prints key insights with specific numbers\n5. Focuses on actionable business insights\n6. Handles errors gracefully with try-except blocks\n7. Closes database connections properly\n\nCRITICAL REQUIREMENTS:\n- Use EXACT table and column names from the schema above\n- Start with imports (no explanatory text before code)\n- Use CAST(column AS REAL) for numeric aggregations in SQL\n- Convert DataFrame columns to numeric: pd.to_numeric(df['col'], errors='coerce')\n- Handle NaN values properly with .notna() and .dropna()\n- Use errorbar=None instead of ci=None in seaborn\n- For seaborn barplot, use hue parameter instead of palette for color mapping\n- Do NOT call plt.show() (plots are captured automatically)\n- Join category_translation for English product category names\n- Use COALESCE for NULL handling in SQL\n- Filter 'delivered' orders for completed transactions\n- Print 3-5 specific, actionable insights with actual numbers\n- Include detailed print statements with statistics\n- ALL strings must be properly terminated with matching quotes\n- ALL brackets/parentheses must be balanced\n- NO markdown formatting in code (remove ```)\n- Code must compile without syntax errors\n\nFor customer behavior analysis, focus on:\n- Geographic patterns (cities/states from customers table)\n- Spending patterns (order values, payment methods from order_payments)\n- Order frequency and timing (timestamps from orders)\n- Review scores and satisfaction (from order_reviews)\n- Business recommendations based on data\n\nReturn ONLY executable Python code:"
        )

        # Try to generate code with retry logic
        max_attempts = 2
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
            else:
                # Got valid response, break out of retry loop
                break

        # Ensure code_text is never None
        if not code_text:
            code_text = "# Error: Failed to generate code"

        # Robust code extraction from markdown using the validator
        code_text = extract_python_code(code_text)

        # Remove any remaining markdown artifacts
        code_text = code_text.strip()
        if code_text.startswith("```"):
            code_text = code_text[3:].strip()
        if code_text.endswith("```"):
            code_text = code_text[:-3].strip()

        # Validate and auto-fix common issues
        is_valid, fixed_code, error_msg = validate_and_fix_code(code_text)

        if is_valid:
            code_text = fixed_code
            logger.info("Generated code validated successfully")
        else:
            # Validation failed, try LLM fix
            logger.warning(f"Code validation failed: {error_msg}")

            fix_prompt = (
                "The following Python code has a syntax error:\n\n```python\n"
                + code_text
                + "\n```\n\nError: "
                + error_msg
                + "\n\nFix the syntax error and return ONLY the corrected Python code without any explanations or markdown.\nEnsure all strings are properly terminated and all brackets/parentheses are balanced.\nReturn executable Python code starting with imports."
            )

            try:
                fixed_code = call_llm(fix_prompt, max_tokens=LLMTokenLimits.CODE)
                if fixed_code:
                    fixed_code = fixed_code.strip()

                    # Robust extraction for fixed code
                    if "```python" in fixed_code:
                        parts = fixed_code.split("```python")
                        if len(parts) > 1:
                            fixed_code = parts[1].split("```")[0].strip()
                    elif "```py" in fixed_code:
                        parts = fixed_code.split("```py")
                        if len(parts) > 1:
                            fixed_code = parts[1].split("```")[0].strip()
                    elif fixed_code.startswith("```"):
                        lines = fixed_code.split("\n")
                        start_idx = 1  # Skip first ``` line
                        end_idx = len(lines)
                        for i in range(len(lines) - 1, -1, -1):
                            if lines[i].strip() == "```":
                                end_idx = i
                                break
                        fixed_code = "\n".join(lines[start_idx:end_idx]).strip()

                    # Remove any remaining artifacts
                    if fixed_code.startswith("```"):
                        fixed_code = fixed_code[3:].strip()
                    if fixed_code.endswith("```"):
                        fixed_code = fixed_code[:-3].strip()

                    # Validate the fix
                    compile(fixed_code, "<string>", "exec")
                    code_text = fixed_code
                    logger.info("Code auto-corrected successfully")
            except Exception as e:
                logger.error(f"Could not fix code: {e}")
                # Return a more helpful error script with the actual error details
                error_details = (
                    "\nSyntax Error Details:\n- Message: " + error_msg + "\n"
                )
                error_msg_repr = repr(error_details)
                error_script = (
                    "# Syntax error in generated code\nimport sqlite3\nimport pandas as pd\n\nprint('Code generation encountered a syntax error.')\nprint("
                    + error_msg_repr
                    + ")\nprint()\nprint('This usually happens when:')\nprint('1. The AI response contains unterminated strings')\nprint('2. Unbalanced brackets or parentheses')\nprint('3. Invalid character encoding')\nprint()\nprint('Please try rephrasing your request with more specific details.')\nprint('Example: \"Create a bar chart showing top 10 cities by order count\"')\n"
                )
                code_text = error_script

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
