import sqlite3

conn = sqlite3.connect("data/olist.db")
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print("Tables:", tables)

# Check orders table
cursor.execute("PRAGMA table_info(orders)")
print("\nOrders columns:", [row[1] for row in cursor.fetchall()])

# Check customers table
cursor.execute("PRAGMA table_info(customers)")
print("Customers columns:", [row[1] for row in cursor.fetchall()])

# Check payments table (order_payments)
cursor.execute("PRAGMA table_info(order_payments)")
print("Payments columns:", [row[1] for row in cursor.fetchall()])

conn.close()
