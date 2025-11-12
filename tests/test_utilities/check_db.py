import sqlite3

conn = sqlite3.connect("data/olist.db")
cursor = conn.cursor()

# Get all tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print("Tables:", tables)

# Check category_translation table
cursor.execute("PRAGMA table_info(category_translation)")
columns = cursor.fetchall()
print("category_translation columns:", columns)

# Check products table to see how they relate
cursor.execute("PRAGMA table_info(products)")
products_columns = cursor.fetchall()
print("products columns:", products_columns)

conn.close()
