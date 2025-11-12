#!/usr/bin/env python3
"""
Add database indexes to improve query performance.
Run this once to optimize the SQLite database for faster queries.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "olist.db"


def add_indexes():
    """Add indexes to common join columns for better query performance."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    indexes = [
        # Orders table
        "CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(order_status)",
        "CREATE INDEX IF NOT EXISTS idx_orders_purchase_date ON orders(order_purchase_timestamp)",
        # Order items table
        "CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id)",
        "CREATE INDEX IF NOT EXISTS idx_order_items_seller_id ON order_items(seller_id)",
        # Products table
        "CREATE INDEX IF NOT EXISTS idx_products_product_id ON products(product_id)",
        "CREATE INDEX IF NOT EXISTS idx_products_category ON products(product_category_name)",
        # Customers table
        "CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_customers_state ON customers(customer_state)",
        # Sellers table
        "CREATE INDEX IF NOT EXISTS idx_sellers_seller_id ON sellers(seller_id)",
        # Payments table
        "CREATE INDEX IF NOT EXISTS idx_payments_order_id ON order_payments(order_id)",
        # Reviews table
        "CREATE INDEX IF NOT EXISTS idx_reviews_order_id ON order_reviews(order_id)",
        # Category translation table
        "CREATE INDEX IF NOT EXISTS idx_category_translation ON category_translation(product_category_name)",
    ]

    print(f"Adding indexes to {DB_PATH}...")
    for idx, sql in enumerate(indexes, 1):
        try:
            cursor.execute(sql)
            print(
                f"  [{idx}/{len(indexes)}] Created {sql.split('idx_')[1].split(' ON')[0]}"
            )
        except Exception as e:
            print(f"  [{idx}/{len(indexes)}] Error: {e}")

    conn.commit()
    conn.close()
    print("\nDatabase indexing complete! Queries should be faster now.")


if __name__ == "__main__":
    add_indexes()
