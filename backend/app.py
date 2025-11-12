import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import time
from backend.query_router import route_query
from backend.llm_client import _key_status, TIER_KEYS

app = FastAPI(title="CommServe API", description="AI-powered data analysis API")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected exceptions globally."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Unexpected issue occurred",
            "detail": str(exc) if app.debug else "Internal server error",
        },
    )


class QueryReq(BaseModel):
    query: str = ""
    params: dict = {}
    mode: Optional[str] = None
    session_id: str = "default"


class FeedbackReq(BaseModel):
    query: str
    action_taken: str
    was_correct: bool
    correct_action: Optional[str] = None
    implicit_signals: Optional[dict] = None


@app.post("/query")
async def query_endpoint(req):
    result = route_query(req.dict())
    if "error" in result:
        return {"status": "error", "detail": result["error"]}
    else:
        return {"status": "success", "data": result}


@app.post("/feedback")
async def feedback_endpoint(req: FeedbackReq):
    """Record feedback about query classification for learning."""
    try:
        from backend.utils.intelligent_planner import IntelligentPlanner

        planner = IntelligentPlanner()
        planner.record_feedback(
            req.query,
            req.action_taken,
            req.was_correct,
            req.correct_action or None,
            req.implicit_signals or {},
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/learning-stats")
async def learning_stats():
    """Get learning system statistics."""
    try:
        from backend.utils.intelligent_planner import IntelligentPlanner

        planner = IntelligentPlanner()
        stats = planner.get_statistics()
        return {"status": "success", "data": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/reset")
async def reset_session(session_id="default"):
    """Reset session memory."""
    result = route_query({"mode": "reset", "session_id": session_id, "query": ""})
    return {"status": "success", "message": result.get("message")}


@app.get("/history")
async def get_history(session_id="default", limit=10):
    """Get session history."""
    result = route_query(
        {"mode": "history", "session_id": session_id, "query": "", "limit": limit}
    )
    return {"status": "success", "data": result.get("data")}


@app.get("/health")
async def health_check():
    """Health check endpoint with system status."""
    current_time = time.time()

    # Check LLM tier status
    tier_status = {}
    available_tiers = 0
    for tier, config in TIER_KEYS.items():
        status = _key_status[tier]
        is_available = (
            status["available"]
            and config["key"]
            and current_time >= status["cooldown_until"]
        )
        tier_status[tier] = {
            "provider": config["provider"],
            "available": is_available,
            "failure_count": status["failure_count"],
            "cooldown_remaining": max(0, int(status["cooldown_until"] - current_time)),
        }
        if is_available:
            available_tiers += 1

    # Check database
    db_healthy = os.path.exists("data/olist.db")

    # Check embeddings
    embeddings_healthy = os.path.exists("embeddings/chroma")

    # Overall health status
    is_healthy = available_tiers > 0 and db_healthy

    return {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": current_time,
        "components": {
            "llm_tiers": {
                "available": available_tiers,
                "total": len(TIER_KEYS),
                "details": tier_status,
            },
            "database": {"healthy": db_healthy, "path": "data/olist.db"},
            "embeddings": {"healthy": embeddings_healthy, "path": "embeddings/chroma"},
        },
    }


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
