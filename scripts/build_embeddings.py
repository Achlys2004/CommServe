import os
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from config import DB_PATH, EMBEDDINGS_DIR as CHROMA_DIR


def fetch_products():
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql(
        """
        SELECT 
            product_id,
            product_category_name,
            product_name_lenght,
            product_description_lenght,
            product_photos_qty,
            product_weight_g,
            product_length_cm,
            product_height_cm,
            product_width_cm
        FROM products
        """,
        conn,
    )
    conn.close()

    string_cols = data.select_dtypes(include=["object"]).columns
    data[string_cols] = data[string_cols].fillna("")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)

    data["text"] = (
        "Category: "
        + data["product_category_name"].fillna("unknown")
        + " | Name length: "
        + data["product_name_lenght"].astype(str)
        + " | Description length: "
        + data["product_description_lenght"].astype(str)
        + " | Photos: "
        + data["product_photos_qty"].astype(str)
        + " | Dimensions (cm): "
        + data["product_length_cm"].astype(str)
        + "x"
        + data["product_height_cm"].astype(str)
        + "x"
        + data["product_width_cm"].astype(str)
    )
    return data[["product_id", "text", "product_category_name"]]


def fetch_reviews():
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql(
        """
        SELECT 
            review_id,
            order_id,
            review_score,
            review_comment_title,
            review_comment_message,
            review_creation_date
        FROM order_reviews
        """,
        conn,
    )
    conn.close()

    string_cols = data.select_dtypes(include=["object"]).columns
    data[string_cols] = data[string_cols].fillna("")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)

    data["text"] = (
        "Review (Score: "
        + data["review_score"].astype(str)
        + ") - "
        + data["review_comment_title"].fillna("")
        + " "
        + data["review_comment_message"].fillna("")
    )
    return data[["review_id", "order_id", "text", "review_score"]]


def fetch_order_items():
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql(
        """
        SELECT 
            order_id,
            order_item_id,
            product_id,
            seller_id,
            shipping_limit_date,
            price,
            freight_value
        FROM order_items
        """,
        conn,
    )
    conn.close()

    string_cols = data.select_dtypes(include=["object"]).columns
    data[string_cols] = data[string_cols].fillna("")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)

    data["text"] = (
        "Order Item ID: "
        + data["order_item_id"].astype(str)
        + " | Product ID: "
        + data["product_id"].astype(str)
        + " | Seller: "
        + data["seller_id"].astype(str)
        + " | Price: "
        + data["price"].astype(str)
        + " | Freight: "
        + data["freight_value"].astype(str)
        + " | Ship by: "
        + data["shipping_limit_date"].astype(str)
    )
    return data[["order_id", "order_item_id", "text", "price", "freight_value"]]


def fetch_sellers():
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql(
        """
        SELECT 
            seller_id,
            seller_zip_code_prefix,
            seller_city,
            seller_state
        FROM sellers
        """,
        conn,
    )
    conn.close()

    string_cols = data.select_dtypes(include=["object"]).columns
    data[string_cols] = data[string_cols].fillna("")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)

    data["text"] = (
        "Seller ID: "
        + data["seller_id"]
        + " | City: "
        + data["seller_city"]
        + " | State: "
        + data["seller_state"]
        + " | Zip: "
        + data["seller_zip_code_prefix"].astype(str)
    )
    return data[["seller_id", "text", "seller_city", "seller_state"]]


def build():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    batch_size = 5000  # ChromaDB max batch size is ~5461, so use 5000

    # products
    print("Embedding: products ...")
    products = fetch_products()
    prod_collection = client.get_or_create_collection(
        name="olist_products", embedding_function=embed_func  # type: ignore
    )
    prod_ids = products["product_id"].astype(str).tolist()
    prod_docs = products["text"].tolist()
    prod_metas = products[["product_category_name"]].to_dict(orient="records")  # type: ignore
    for i in range(0, len(prod_ids), batch_size):
        batch_ids = prod_ids[i : i + batch_size]
        batch_docs = prod_docs[i : i + batch_size]
        batch_metas = prod_metas[i : i + batch_size]
        prod_collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)  # type: ignore

    # reviews
    print("Embedding: order_reviews ...")
    reviews = fetch_reviews()
    review_collection = client.get_or_create_collection(
        name="olist_reviews", embedding_function=embed_func  # type: ignore
    )
    rev_ids = (
        reviews["review_id"].astype(str) + "_" + reviews["order_id"].astype(str)
    ).tolist()

    rev_docs = reviews["text"].tolist()
    rev_metas = reviews[["order_id", "review_score"]].astype(str).to_dict(orient="records")  # type: ignore
    for i in range(0, len(rev_ids), batch_size):
        batch_ids = rev_ids[i : i + batch_size]
        batch_docs = rev_docs[i : i + batch_size]
        batch_metas = rev_metas[i : i + batch_size]
        review_collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)  # type: ignore

    # order_items
    print("Embedding: order_items ...")
    items = fetch_order_items()
    item_collection = client.get_or_create_collection(
        name="olist_order_items", embedding_function=embed_func  # type: ignore
    )
    item_ids = (
        items["order_id"].astype(str) + "_" + items["order_item_id"].astype(str)
    ).tolist()
    item_docs = items["text"].tolist()
    item_metas = items[["price", "freight_value"]].to_dict(orient="records")  # type: ignore
    for i in range(0, len(item_ids), batch_size):
        batch_ids = item_ids[i : i + batch_size]
        batch_docs = item_docs[i : i + batch_size]
        batch_metas = item_metas[i : i + batch_size]
        item_collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)  # type: ignore

    # sellers
    print("Embedding: sellers ...")
    sellers = fetch_sellers()
    seller_collection = client.get_or_create_collection(
        name="olist_sellers", embedding_function=embed_func  # type: ignore
    )
    seller_ids = sellers["seller_id"].astype(str).tolist()
    seller_docs = sellers["text"].tolist()
    seller_metas = sellers[["seller_city", "seller_state"]].to_dict(orient="records")  # type: ignore
    for i in range(0, len(seller_ids), batch_size):
        batch_ids = seller_ids[i : i + batch_size]
        batch_docs = seller_docs[i : i + batch_size]
        batch_metas = seller_metas[i : i + batch_size]
        seller_collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)  # type: ignore

    print(f"\nEmbeddings built successfully!")
    print(f"Stored in: {CHROMA_DIR.resolve()}")
    print(f"Collections: {client.list_collections()}")


if __name__ == "__main__":
    build()
