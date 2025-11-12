import chromadb
from pathlib import Path
from config import EMBEDDINGS_DIR as CHROMA_DIR


def get_chroma_client():
    if not CHROMA_DIR.exists():
        return {"error": f"Chroma embeddings directory not found at {CHROMA_DIR}"}
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client
    except Exception as e:
        return {"error": f"Failed to initialize Chroma client: {e}"}


def retrieve_by_text(
    query, collection_name="olist_products", k=5, metadata_filters=None
):
    try:
        client = get_chroma_client()
        if isinstance(client, dict) and "error" in client:
            return client
        col = client.get_collection(collection_name)  # type: ignore

        fetch_k = min(k * 3, 50)
        results = col.query(
            query_texts=[query], n_results=fetch_k, where=metadata_filters
        )
        docs = []

        ids = (results.get("ids") or [[]])[0]
        docs_text = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]  # Lower is better

        DISTANCE_THRESHOLD = 1.5
        for i, doc_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else float("inf")
            if distance < DISTANCE_THRESHOLD:
                docs.append(
                    {
                        "id": doc_id,
                        "document": docs_text[i] if i < len(docs_text) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                        "distance": distance,
                    }
                )

        docs = docs[:k]
        return {"results": docs}

    except Exception as e:
        return {"error": f"Chroma retrieval error: {str(e)}"}
