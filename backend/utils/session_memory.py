import json
import logging
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from backend.features.summariser import Summariser

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("logs")  # persistent store location (already exists in your repo)
MEMORY_DIR.mkdir(exist_ok=True)


class SessionMemory:
    """
    Advanced session memory manager with persistence and compression.
    - Keeps recent raw history entries (max_recent)
    - Compresses older entries into a single summarized blob.
    - Persists per-session JSON to disk.
    - Maintains backward compatibility with existing interface.
    """

    def __init__(
        self, session_id: str = "default", max_recent: int = 3, persist: bool = True
    ):
        self.session_id = session_id
        self.max_recent = max_recent  # Keep last 3 queries + responses
        self.persist = persist
        self.summariser = Summariser()
        self.recent = (
            []
        )  # list of {"query":..., "action":..., "result":..., "timestamp":...}
        self.compressed_summary = None
        self._load_from_disk()

    def add_exchange(
        self,
        session_id: str,
        query: str,
        response: dict,
        action: str,
        confidence: float = 0.0,
    ) -> None:
        """Add a query-response exchange to session history."""
        # Convert response to result format expected by SessionMemory
        result = {
            "response": response,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }
        self.add(query, action, result)

    def get_history(self, session_id, limit=None):
        """Retrieve session history."""
        # Convert internal format to expected format
        recent = self.recent
        if limit:
            recent = recent[-limit:]

        history = []
        for entry in recent:
            history.append(
                {
                    "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                    "query": entry["query"],
                    "response": entry["result"].get("response", {}),
                    "action": entry["action"],
                    "confidence": entry["result"].get("confidence", 0.0),
                }
            )
        return history

    def get_context_string(self, session_id: str, limit: int = 2) -> str:
        """Build a formatted context string from recent history."""
        return self.get_context(include_recent=limit)

    def get_last_query(self, session_id):
        """Get the most recent query from session."""
        if self.recent:
            return self.recent[-1]["query"]
        return None

    def clear_session(self, session_id: str) -> None:
        """Clear all history for a session."""
        self.clear()
        logger.info(f"Cleared session: {session_id}")

    def get_all_sessions(self):
        """Get list of all active session IDs."""
        return [self.session_id]

    def add(self, query, action, result):
        entry = {
            "query": query,
            "action": action,
            "result": result,
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
        }
        self.recent.append(entry)

        # Auto-summarize after every 3 interactions
        if len(self.recent) >= 3 and len(self.recent) % 3 == 0:
            self._auto_summarize_recent()

        if self.persist:
            self._save_to_disk()

    def get_context(self, include_recent: int = 2) -> str:
        """
        Return a context string suitable for LLM prompts:
        [compressed_summary (if exists)] + last `include_recent` successful items
        """
        parts = []
        if self.compressed_summary:
            parts.append(f"Session summary: {self.compressed_summary}")

        # Only include successful interactions (those without error)
        successful_items = []
        for e in reversed(self.recent):
            if len(successful_items) >= include_recent:
                break
            result = e.get("result", {})
            response = result.get("response", {})
            # Skip if response is an error string or contains error
            if isinstance(response, str) and "failed" in response.lower():
                continue
            if isinstance(response, dict) and "error" in response:
                continue
            successful_items.append(e)

        # Reverse back to chronological order
        successful_items.reverse()

        for e in successful_items:
            parts.append(
                f"Query: {e['query']}\nAction: {e['action']}\nResultPreview: {self._shorten_result(e['result'])}"
            )
        return "\n\n".join(parts).strip()

    def _auto_summarize_recent(self):
        """Auto-summarize the most recent 3 interactions and add to compressed summary."""
        if len(self.recent) < 3:
            return

        # Get the last 3 interactions
        recent_three = self.recent[-3:]
        text = "\n\n".join(
            [
                f"Q: {e['query']}\nA: {self._shorten_result(e['result'])}"
                for e in recent_three
            ]
        )

        # Create summary prompt
        summary_prompt = (
            f"Summarize these recent conversation interactions concisely:\n\n{text}"
        )

        try:
            # Use summariser to create a summary
            summary = self.summariser.generate_summary(
                "Summarize recent interactions", summary_prompt, ""
            )

            # Append to existing compressed summary
            if self.compressed_summary:
                self.compressed_summary += f"\n\nRecent summary: {summary}"
            else:
                self.compressed_summary = f"Recent summary: {summary}"

            logger.info(f"Auto-summarized {len(recent_three)} recent interactions")

        except Exception as e:
            logger.warning(f"Failed to auto-summarize recent interactions: {e}")
            # Fallback: just append the raw text
            if self.compressed_summary:
                self.compressed_summary += f"\n\n{text}"
            else:
                self.compressed_summary = text

    def clear(self):
        self.recent = []
        self.compressed_summary = None
        if self.persist:
            self._save_to_disk()

    def _shorten_result(self, result):
        # result may be SQL result with rows; provide small preview
        if not result:
            return ""
        try:
            response = result.get("response", {})
            if isinstance(response, dict) and "rows" in response:
                rows = response["rows"]
                preview = json.dumps(rows[:2], ensure_ascii=False)
                return preview
            return str(response)[:400]
        except Exception:
            return str(result)[:400]

    def _path(self) -> Path:
        return MEMORY_DIR / f"session_{self.session_id}.json"

    def _save_to_disk(self):
        payload = {"recent": self.recent, "compressed_summary": self.compressed_summary}
        try:
            with open(self._path(), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_from_disk(self):
        try:
            p = self._path()
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.recent = payload.get("recent", [])
                self.compressed_summary = payload.get("compressed_summary")
        except Exception:
            # ignore load failures and start fresh
            self.recent = []
            self.compressed_summary = None


# Global session memory instance
_memory_instance = None


def get_session_memory():
    """Get or create the global session memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = SessionMemory(max_recent=3)
    return _memory_instance


@lru_cache(maxsize=500)
def detect_follow_up(current_query, previous_query):
    """Detect if current query is a follow-up and enrich it with context."""
    if not previous_query:
        return False, current_query

    current_lower = current_query.lower().strip()

    # Follow-up indicators
    follow_up_patterns = [
        "what about",
        "how about",
        "and",
        "also",
        "last",
        "previous",
        "same",
        "that",
        "it",
        "this",
        "them",
        "more",
        "other",
    ]

    # Check if query is short and contains follow-up patterns
    is_follow_up = len(current_query.split()) < 8 and any(
        pattern in current_lower for pattern in follow_up_patterns
    )

    if is_follow_up:
        # Enrich the query with previous context
        enriched_query = (
            f"{current_query} (Context: Previously asked about '{previous_query}')"
        )
        logger.info(f"Follow-up detected. Enriched: {enriched_query}")
        return True, enriched_query

    return False, current_query


def calculate_confidence(
    action: str,
    context_length: int = 0,
    sql_result_count: int = 0,
    retrieval_count: int = 0,
) -> float:
    """Calculate confidence score based on available data."""
    base_confidence = 0.5

    # SQL confidence based on result count
    if action in ["SQL", "SQL+RAG"]:
        if sql_result_count > 0:
            base_confidence += 0.3
        if sql_result_count > 5:
            base_confidence += 0.1

    # RAG confidence based on context length and retrieval count
    if action in ["RAG", "SQL+RAG"]:
        if retrieval_count > 0:
            base_confidence += 0.2
        if context_length > 500:
            base_confidence += 0.1
        if retrieval_count >= 10:
            base_confidence += 0.1

    # CODE confidence is moderate by default
    if action == "CODE":
        base_confidence = 0.7

    return min(base_confidence, 1.0)
