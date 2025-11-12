"""
Orchestrator - Multi-Tier AI Query Router with Context Management
Manages conversation flow, intent routing, and tier fallback logic.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from backend.planner import decide_action
from backend.utils.session_memory import SessionMemory
from backend.llm_client import call_llm
from backend.llm_config import LLMTokenLimits

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Intelligent query orchestrator that manages multi-tier AI stack:
    - Tier 1: SQL Agent (structured queries)
    - Tier 2: Python Analysis Agent (statistical/visualization)
    - Tier 3: Conversational LLM Agent (explanations, insights)

    Handles context persistence, intent routing, and fallback logic.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.memory = SessionMemory(session_id=session_id, max_recent=5)
        self.tier_history = []  # Track which tier was used

    def process_query(self, query: str, query_engine: Any) -> Dict[str, Any]:
        """
        Main orchestration entry point. Routes query through appropriate tier
        with context awareness and fallback handling.

        Args:
            query: User's natural language query
            query_engine: QueryEngine instance for execution

        Returns:
            Enriched response with conversational context
        """
        logger.info(f"[Orchestrator] Processing query: {query[:100]}")

        # Step 1: Build conversational context from memory
        context = self._build_context()

        # Step 2: Classify intent and route to appropriate tier
        intent = decide_action(query, context)
        action = intent.get("action", "CONVERSATION")
        confidence = intent.get("confidence", 0.5)

        logger.info(f"[Orchestrator] Intent: {action} (confidence: {confidence:.2f})")

        # Step 3: Execute through query engine (use internal direct execution to avoid recursion)
        result = query_engine._execute_direct(query, {"context": context})

        # Step 4: Add conversational humanization
        humanized_response = self._humanize_response(
            query=query, result=result, action=action, context=context
        )

        # Step 5: Store interaction in memory
        self.memory.add_exchange(
            session_id=self.session_id,
            query=query,
            response=humanized_response,
            action=action,
            confidence=confidence,
        )

        # Track tier usage
        self.tier_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "tier": self._map_action_to_tier(action),
                "action": action,
            }
        )

        return humanized_response

    def _build_context(self) -> str:
        """Build concise context summary from recent interactions."""
        context = self.memory.get_context(include_recent=3)

        if not context:
            return ""

        # Add conversational markers
        return f"\n[Previous Context]:\n{context}\n"

    def _enhance_query_with_context(self, query: str, context: str) -> str:
        """
        Append context to query for better agent understanding.
        Handles follow-up questions like "Now show only electronics".
        """
        if not context:
            return query

        # Check if query is a follow-up (uses pronouns, references "now", "also", etc.)
        follow_up_markers = ["now", "also", "what about", "how about", "and", "show me"]
        is_follow_up = any(marker in query.lower() for marker in follow_up_markers)

        if is_follow_up:
            return f"{context}\nCurrent query: {query}"

        return query

    def _humanize_response(
        self, query: str, result: Dict[str, Any], action: str, context: str
    ) -> Dict[str, Any]:
        """
        Transform technical response into human-friendly, conversational format.
        Intelligently decides when insights add value.
        """
        # Use intelligent insight decision system
        from backend.utils.self_validator import SelfValidator

        validator = SelfValidator()

        should_show, reason = validator.should_show_insights(query, result)

        logger.info(f"Insight decision for query '{query}': {should_show} - {reason}")

        # Only generate insights if they would add value
        if should_show:
            # Extract the actual result
            sql_result = result.get("sql_result", {})
            execution_result = result.get("execution_result", {})
            response_text = result.get("response", "")

            commentary = None

            # SQL queries - analyze the data results
            if sql_result and sql_result.get("rows"):
                commentary = self._generate_insight_commentary(
                    query=query, sql_result=sql_result, action=action, context=context
                )
            # Code execution - analyze the output/visualizations
            elif execution_result and execution_result.get("status") == "success":
                commentary = self._generate_code_insights(
                    query, execution_result, context
                )
            # RAG/text queries - use the response text
            elif response_text and action in ["RAG", "SQL+RAG"]:
                commentary = response_text  # RAG already has conversational response

            # Add commentary if generated
            if commentary:
                result["conversational_summary"] = commentary
                result["personality_active"] = True

        # Add metadata about orchestration
        result["orchestrator_meta"] = {
            "tier_used": self._map_action_to_tier(action),
            "action": action,
            "context_used": bool(context),
            "session_id": self.session_id,
            "insights_shown": should_show,
            "insight_reason": reason,
        }

        return result

    def _generate_insight_commentary(
        self, query: str, sql_result: Dict[str, Any], action: str, context: str
    ) -> Optional[str]:
        """
        Generate human-like insights and commentary about the data results.
        This is where we add the "Sci personality" - conversational and insightful.
        """
        rows = sql_result.get("rows", [])
        if not rows:
            return "Hmm, no data came back for that one. Might want to check if the filters are too restrictive."

        # Build a concise data summary
        data_summary = self._summarize_results_for_llm(rows, query)

        # Create prompt for conversational commentary
        prompt = f"""You are Sci, a friendly data analyst having a natural conversation with a colleague.

User asked: "{query}"

Data summary: {data_summary}

{f"Context: {context}" if context else ""}

Write a natural, conversational insight (1-2 sentences) that:
- If a random period was selected, explicitly mention which month/year/period was chosen
- Highlights 1-2 key findings with specific numbers
- Adds practical business meaning
- Sounds like a colleague sharing insights
- Uses contractions and natural language

Examples:
✅ "Your top category is bed_bath_table with 11,115 orders - that's dominating! Health_beauty comes in second with 9,670 orders."
✅ "Average order value is $159.83, so there's room to boost revenue through bundling strategies."

Keep under 80 words:"""

        try:
            commentary = call_llm(prompt, max_tokens=LLMTokenLimits.MEDIUM)

            # Check if commentary is actually an error message
            error_indicators = [
                "I apologize, but I'm experiencing",
                "Hey there! I'm getting a lot of requests",
                "Hmm, I'm having trouble connecting",
                "Oops! I ran into an unexpected issue",
            ]

            if commentary:
                is_error = any(
                    commentary.startswith(indicator) for indicator in error_indicators
                )
            else:
                is_error = False

            if is_error or not commentary:
                logger.warning(
                    "LLM returned error message instead of commentary, skipping insights"
                )
                return None  # Return None to skip showing insights

            return commentary
        except Exception as e:
            logger.warning(f"Failed to generate commentary: {e}")
            return None  # Return None instead of fallback message

    def _summarize_results_for_llm(self, rows: list, query: str) -> str:
        """Create concise summary of results for LLM commentary generation."""
        if not rows:
            return "No data returned from query"

        # Get basic stats
        total_rows = len(rows)

        # Check for random selections in query
        has_random = "random" in query.lower()
        selected_period = None

        if has_random and rows:
            # Look for period/date columns in first row
            first_row = rows[0]
            for key, value in first_row.items():
                key_lower = key.lower()
                if any(
                    period in key_lower
                    for period in ["month", "year", "period", "date"]
                ):
                    selected_period = f"{key}: {value}"
                    break
                # Also check if value looks like a date/month
                elif isinstance(value, str) and (
                    len(value) == 7 and value.count("-") == 1
                ):  # YYYY-MM format
                    selected_period = f"Selected {key}: {value}"
                    break

        # Extract numeric columns and calculate basic stats
        numeric_cols = []
        for row in rows[:3]:  # Sample first few rows
            for key, value in row.items():
                try:
                    float(value)  # Check if numeric
                    if key not in numeric_cols:
                        numeric_cols.append(key)
                except (ValueError, TypeError):
                    pass

        # Build summary
        summary_parts = [f"Total rows: {total_rows}"]

        # Add selected period if found
        if selected_period:
            summary_parts.insert(0, selected_period)

        # Add sample data
        if rows:
            sample = rows[0]
            sample_text = ", ".join([f"{k}: {v}" for k, v in sample.items()])
            summary_parts.append(f"Sample: {sample_text}")

        # Add numeric summaries if available
        for col in numeric_cols[:3]:  # Limit to 3 columns
            try:
                values = [
                    float(row.get(col, 0)) for row in rows if row.get(col) is not None
                ]
                if values:
                    avg_val = sum(values) / len(values)
                    summary_parts.append(f"Avg {col}: {avg_val:.2f}")
            except:
                pass

        return " | ".join(summary_parts)

    def _generate_code_insights(
        self, query: str, execution_result: Dict[str, Any], context: str
    ) -> Optional[str]:
        """
        Generate insights for code execution results (visualizations, analysis).
        """
        # Extract execution details
        stdout = execution_result.get("stdout", "")
        has_viz = execution_result.get("output_type") == "image"
        images_count = len(execution_result.get("images", []))

        # Build a summary of what was done
        summary_parts = []
        if stdout:
            # Extract key statistics from stdout
            summary_parts.append(f"Analysis output:\n{stdout[:500]}")

        if has_viz and images_count > 0:
            summary_parts.append(f"Generated {images_count} visualization(s)")

        result_summary = (
            "\n\n".join(summary_parts)
            if summary_parts
            else "Code executed successfully"
        )

        # Create prompt for conversational insights
        prompt = f"""You are Sci, a conversational data analyst AI. 
You just ran Python code to analyze the Olist e-commerce dataset.

User asked: "{query}"

Code execution results:
{result_summary}

Context: {context if context else "This is the first analysis."}

Generate a brief, insightful response (2-4 sentences) that:
1. Explains the key patterns or trends found in the data
2. Provides business context or actionable insights
3. Uses specific numbers or statistics from the output
4. Sounds natural and conversational

Keep it under 150 words. Be analytical but friendly."""

        try:
            commentary = call_llm(prompt, max_tokens=LLMTokenLimits.MEDIUM)

            # Check if commentary is actually an error message
            error_indicators = [
                "I apologize, but I'm experiencing",
                "Hey there! I'm getting a lot of requests",
                "Hmm, I'm having trouble connecting",
                "Oops! I ran into an unexpected issue",
            ]

            if commentary:
                is_error = any(
                    commentary.startswith(indicator) for indicator in error_indicators
                )
            else:
                is_error = False

            if is_error or not commentary:
                logger.warning(
                    "LLM returned error message instead of code insights, skipping"
                )
                return None  # Return None to skip showing insights

            return commentary
        except Exception as e:
            logger.warning(f"Failed to generate code insights: {e}")
            return None  # Return None instead of fallback message

    def _map_action_to_tier(self, action: str) -> str:
        """Map action type to tier for tracking."""
        tier_mapping = {
            "SQL": "Tier-1-SQL",
            "CODE": "Tier-2-Python",
            "RAG": "Tier-3-Conversational",
            "SQL+RAG": "Tier-1-Hybrid",
            "CONVERSATION": "Tier-3-Conversational",
            "METADATA": "Tier-3-Info",
        }
        return tier_mapping.get(action, "Unknown")

    def get_conversation_history(self, limit: int = 10) -> list:
        """Retrieve formatted conversation history."""
        return self.memory.get_history(self.session_id, limit=limit)

    def clear_conversation(self):
        """Clear conversation memory."""
        self.memory.clear_session(self.session_id)
        self.tier_history = []
        logger.info(
            f"[Orchestrator] Cleared conversation for session: {self.session_id}"
        )

    def get_context_summary(self) -> str:
        """Get human-readable context summary."""
        history = self.get_conversation_history(limit=5)

        if not history:
            return "No conversation history yet."

        summary_lines = []
        for entry in history[-3:]:  # Last 3 exchanges
            query = entry.get("query", "")
            action = entry.get("action", "")
            summary_lines.append(f"- {query[:50]}... ({action})")

        return "\n".join(summary_lines)
