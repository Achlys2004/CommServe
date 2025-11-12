from backend.query_engine import QueryEngine
from backend.llm_client import call_llm
from backend.retriever import retrieve_by_text


def route_query(req):
    query = req.get("query")
    params = req.get("params", {})
    session_id = req.get("session_id", "default")

    if not query:
        return {"error": "Missing query parameter"}

    # Initialize query engine with session
    query_engine = QueryEngine(hybrid_mode=True, session_id=session_id)

    # Handle memory reset
    if req.get("mode") == "reset":
        query_engine.clear_memory()
        return {"mode": "reset", "message": "Session memory cleared successfully"}

    # Handle history request
    if req.get("mode") == "history":
        history = query_engine.get_session_history(limit=req.get("limit", 10))
        return {"mode": "history", "data": history}

    # Retrieval-only mode
    if req.get("mode") == "retrieve":
        docs = retrieve_by_text(query)
        return {"mode": "retrieve", "data": docs}

    # Direct LLM mode
    if req.get("mode") == "llm":
        answer = call_llm(query)
        return {"mode": "llm", "data": {"answer": answer}}

    # Default intelligent query execution with memory
    result = query_engine.execute_query(query, params)
    return {"mode": "auto", "data": result}
