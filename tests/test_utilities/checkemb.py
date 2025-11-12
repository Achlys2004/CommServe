import chromadb
from pathlib import Path
import sys

# Add current directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMBEDDINGS_DIR as CHROMA_DIR

# Initialize the persistent client
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Get collections
try:
    prod_collection = client.get_collection("olist_products")
    print(f"Products: {prod_collection.count()}")
except Exception as e:
    print(f"Error getting products collection: {e}")

try:
    review_collection = client.get_collection("olist_reviews")
    print(f"Reviews: {review_collection.count()}")
except Exception as e:
    print(f"Error getting reviews collection: {e}")

try:
    item_collection = client.get_collection("olist_order_items")
    print(f"Order Items: {item_collection.count()}")
except Exception as e:
    print(f"Error getting order items collection: {e}")

try:
    seller_collection = client.get_collection("olist_sellers")
    print(f"Sellers: {seller_collection.count()}")
except Exception as e:
    print(f"Error getting sellers collection: {e}")
