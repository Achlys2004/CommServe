"""
Direct Backend Testing - Test QueryEngine with diverse queries
"""

import sys

sys.path.insert(0, ".")

from backend.query_engine import QueryEngine
import json

# Initialize engine
engine = QueryEngine(use_orchestrator=True, session_id="test_session")

# Test cases
test_cases = [
    {
        "name": "Simple Count (No Insights Expected)",
        "query": "How many orders are there?",
        "check": ["Should return a simple number", "Should NOT show AI insights"],
    },
    {
        "name": "Category Translation Test",
        "query": "Show me top 5 product categories by sales",
        "check": [
            "Categories should be in ENGLISH",
            "NOT Portuguese (informatica_acessorios, etc)",
        ],
    },
    {
        "name": "Date Filtering",
        "query": "What were the sales in December 2017?",
        "check": ["Should filter by month and year", "Should return sales amount"],
    },
    {
        "name": "Quarterly Aggregation (SQLite)",
        "query": "Show me quarterly revenue for 2017",
        "check": [
            "Should use SQLite quarter calculation (NOT %q)",
            "Should group by quarters",
        ],
    },
    {
        "name": "NULL Handling",
        "query": "Show me average order value by product category including uncategorized",
        "check": [
            "Should use COALESCE for NULLs",
            "Should show 'Uncategorized' or similar for NULLs",
        ],
    },
    {
        "name": "Complex Analysis (Insights Expected)",
        "query": "Which states have the most repeat customers?",
        "check": [
            "Should show AI insights explaining patterns",
            "Should join customers and orders tables",
        ],
    },
    {
        "name": "Visualization Request",
        "query": "Create a chart showing sales by month in 2017",
        "check": [
            "Should generate CODE type",
            "Should create visualization",
            "Should show AI insights",
        ],
    },
    {
        "name": "Ambiguous Query (AI Understanding)",
        "query": "What's trending?",
        "check": [
            "AI should interpret as product/sales trends",
            "Should provide meaningful analysis",
        ],
    },
]

print("\n" + "=" * 80)
print("DIRECT BACKEND TESTING - QueryEngine Validation")
print("=" * 80)

results = {"passed": 0, "failed": 0, "warnings": []}

for i, test in enumerate(test_cases, 1):
    print(f"\n\n{'#'*80}")
    print(f"TEST {i}/{len(test_cases)}: {test['name']}")
    print(f"{'#'*80}")
    print(f"Query: {test['query']}")
    print(f"\nExpectations:")
    for check in test["check"]:
        print(f"  - {check}")

    print(f"\n{'-'*80}")

    try:
        # Execute query
        result = engine.execute_query(test["query"])

        print(f"\nResult Type: {result.get('type', 'unknown')}")

        # Check for errors
        if "error" in result:
            print(f"\n[ERROR] {result['error']}")
            results["failed"] += 1
            results["warnings"].append(f"Test {i}: Query failed with error")
            continue

        # Analyze SQL queries
        if result.get("type") == "sql":
            sql = result.get("sql", "")
            print(f"\nGenerated SQL (first 300 chars):")
            print(sql[:300] + "..." if len(sql) > 300 else sql)

            # Check for category translation JOIN
            if "category" in test["query"].lower():
                if "category_translation" in sql.lower():
                    print("\n[OK] Uses category_translation JOIN")
                else:
                    print(
                        "\n[WARNING] Missing category_translation JOIN - Portuguese names may appear!"
                    )
                    results["warnings"].append(
                        f"Test {i}: No category_translation JOIN"
                    )

            # Check for COALESCE
            if (
                "uncategorized" in test["query"].lower()
                or "including" in test["query"].lower()
            ):
                if "COALESCE" in sql.upper() or "IFNULL" in sql.upper():
                    print("[OK] Uses COALESCE/IFNULL for NULL handling")
                else:
                    print("[WARNING] No NULL handling detected")
                    results["warnings"].append(f"Test {i}: No NULL handling in SQL")

            # Check for SQLite quarter calculation
            if "quarter" in test["query"].lower():
                if "%q" in sql:
                    print(
                        "[ERROR] Uses MySQL %q instead of SQLite quarter calculation!"
                    )
                    results["warnings"].append(f"Test {i}: Wrong quarter syntax (%q)")
                elif "strftime" in sql.lower() or "CASE" in sql:
                    print("[OK] Uses SQLite-compatible quarter calculation")
                else:
                    print("[WARNING] Quarter calculation method unclear")

            # Check results
            result_data = result.get("result", {})
            if isinstance(result_data, dict):
                rows = result_data.get("rows", [])
                columns = result_data.get("columns", [])
            else:
                rows = []
                columns = []
                print(f"[WARNING] Result data is not a dict: {type(result_data)}")

            print(f"\nColumns: {columns}")
            print(f"Rows returned: {len(rows)}")

            if rows and len(rows) > 0:
                print(f"Sample row: {rows[0]}")

                # Check for Portuguese text (not translated)
                for val in rows[0]:
                    if isinstance(val, str):
                        portuguese_chars = ["ç", "ã", "õ", "á", "é", "í", "ó", "ú"]
                        if any(char in val.lower() for char in portuguese_chars):
                            if "informatica" in val.lower() or "moveis" in val.lower():
                                print(
                                    f"\n[ERROR] Portuguese category name detected: '{val}'"
                                )
                                print("         Translations are NOT working!")
                                results["warnings"].append(
                                    f"Test {i}: Portuguese text in results"
                                )
                                break

        # Check for CODE execution
        elif result.get("type") == "code":
            result_data = result.get("result", {})
            if isinstance(result_data, dict):
                output = result_data.get("output", "")
                has_plot = result_data.get("plot", False)
            else:
                output = str(result_data) if result_data else ""
                has_plot = False
                print(f"[WARNING] Code result data is not a dict: {type(result_data)}")

            print(f"\nCode Output (first 300 chars):")
            print(output[:300] if output else "[No output]")

            if has_plot:
                print("\n[OK] Visualization generated")
            else:
                print("\n[WARNING] No visualization in CODE result")
                results["warnings"].append(f"Test {i}: No plot generated")

        # Check for RAG response
        elif result.get("type") == "rag":
            response = result.get("response", "")
            print(f"\nRAG Response (first 300 chars):")
            print(response[:300] if response else "[No response]")

        # Check for AI insights
        ai_insights = result.get("ai_insights", "")
        if ai_insights:
            print(f"\n[AI INSIGHTS] {len(ai_insights)} chars")
            print(f"Preview: {ai_insights[:150]}...")

            # For simple queries, insights might be unnecessary
            if "how many" in test["query"].lower() and "simple" in test["name"].lower():
                print("[INFO] Simple query has insights - may be unnecessary")
        else:
            print("\n[NO INSIGHTS]")
            # For complex queries, insights are expected
            if (
                "complex" in test["name"].lower()
                or "visualization" in test["name"].lower()
            ):
                print("[WARNING] Complex query without insights")
                results["warnings"].append(f"Test {i}: No insights for complex query")

        # Check humanized response
        humanized = result.get("humanized_response", "")
        if humanized:
            print(f"\n[HUMANIZED] {len(humanized)} chars")
            print(f"Preview: {humanized[:150]}...")

        results["passed"] += 1

    except Exception as e:
        print(f"\n[EXCEPTION] {str(e)}")
        import traceback

        traceback.print_exc()
        results["failed"] += 1

print("\n\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Completed: {results['passed']}/{len(test_cases)}")
print(f"Failed: {results['failed']}/{len(test_cases)}")
print(f"Warnings: {len(results['warnings'])}")

if results["warnings"]:
    print(f"\n{'='*80}")
    print("WARNINGS & ISSUES FOUND:")
    print("=" * 80)
    for warning in results["warnings"]:
        print(f"  - {warning}")

print(f"\n{'='*80}")
print("KEY ISSUES TO FIX:")
print("=" * 80)
print("1. Portuguese category names NOT translated to English")
print("2. Missing category_translation JOINs in SQL")
print("3. NULL handling (COALESCE) not always applied")
print("4. AI insights shown/hidden incorrectly")
print("5. Visualizations not generated when requested")
print("=" * 80)
