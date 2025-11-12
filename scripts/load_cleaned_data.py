"""
Load cleaned data from data/cleaned directory into SQLite database
"""

import os
import pandas as pd
import sqlite3
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
DB_PATH = BASE_DIR / "data" / "olist.db"

# Table mappings
tables = {
    "order_items": "olist_order_items_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",  # Use order_payments as table name
    "customers": "olist_customers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_data():
    """Load cleaned CSV files into SQLite database"""

    # Remove existing database to start fresh
    if DB_PATH.exists():
        print(f"Removing existing database: {DB_PATH}")
        os.remove(DB_PATH)

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    print("\n" + "=" * 80)
    print("LOADING CLEANED DATA INTO DATABASE")
    print("=" * 80 + "\n")

    for table_name, csv_name in tables.items():
        csv_path = CLEANED_DIR / csv_name

        if not csv_path.exists():
            print(f"❌ Missing: {csv_path}")
            continue

        print(f"📊 Loading {csv_name} -> table '{table_name}'")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Strip whitespace from column names
        df.columns = [col.strip() for col in df.columns]

        # Convert date columns
        for col in df.columns:
            if "date" in col.lower() or "timestamp" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

        # Load into database
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        print(f"   ✓ Loaded {len(df):,} rows")

    # Create indexes for performance
    print("\n📑 Creating indexes...")
    cur = conn.cursor()

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_items_order_id ON order_items(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_items_product_id ON order_items(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_payments_order_id ON order_payments(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_customers_id ON customers(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_products_id ON products(product_id);",
    ]

    for idx_sql in indexes:
        cur.execute(idx_sql)

    conn.commit()
    print("   ✓ Indexes created")

    # Verify the schema
    print("\n" + "=" * 80)
    print("DATABASE SCHEMA VERIFICATION")
    print("=" * 80 + "\n")

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables_in_db = [row[0] for row in cur.fetchall()]

    print(f"📋 Tables in database: {', '.join(tables_in_db)}\n")

    # Show key table schemas
    for table in [
        "orders",
        "customers",
        "order_payments",
        "products",
        "category_translation",
    ]:
        if table in tables_in_db:
            cur.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cur.fetchall()]
            print(f"   {table}: {', '.join(cols)}")

    conn.close()

    print("\n" + "=" * 80)
    print(f"✅ DATABASE READY: {DB_PATH}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    load_data()
