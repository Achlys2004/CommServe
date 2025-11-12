"""Quick test to check why category query returns empty"""

import sqlite3

conn = sqlite3.connect("data/olist.db")

# Test the generated SQL
sql = """
SELECT
    COALESCE(ct.product_category_name_english, p.product_category_name) AS category_name,
    COUNT(oi.order_id) AS total_orders,
    SUM(oi.price) AS total_sales,
    AVG(oi.price) AS average_price
FROM
    orders o
JOIN
    order_items oi ON o.order_id = oi.order_id
JOIN
    products p ON oi.product_id = p.product_id
LEFT JOIN
    category_translation ct ON p.product_category_name = ct.product_category_name
WHERE
    o.order_status = 'delivered'
GROUP BY
    category_name
ORDER BY
    total_sales DESC
LIMIT 5
"""

print("Testing generated SQL...")
try:
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    print(f"\nColumns: {columns}")
    print(f"Rows: {len(results)}")

    if results:
        print("\nTop 5 categories:")
        for row in results:
            print(f"  {row}")
    else:
        print("\nNo results! Debugging...")

        # Check if products table has data
        cursor.execute("SELECT COUNT(*) FROM products LIMIT 1")
        print(f"Products count: {cursor.fetchone()[0]}")

        # Check if order_items links to products
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM order_items oi 
            JOIN products p ON oi.product_id = p.product_id
        """
        )
        print(f"Order items with products: {cursor.fetchone()[0]}")

        # Check category translation
        cursor.execute(
            "SELECT product_category_name, product_category_name_english FROM category_translation LIMIT 5"
        )
        print("\nSample translations:")
        for row in cursor.fetchall():
            print(f"  {row[0]} → {row[1]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
finally:
    conn.close()
