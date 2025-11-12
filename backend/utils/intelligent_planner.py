"""
Intelligent Planner with Few-Shot Learning
Enhances classification using similar past queries and continuous learning.
"""

from backend.llm_client import call_llm
from backend.llm_config import LLMTokenLimits
from backend.utils.query_learning import QueryLearningSystem
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)


class IntelligentPlanner:
    """Enhanced query planner with few-shot learning capabilities."""

    def __init__(self):
        self.learning_system = QueryLearningSystem()

    def decide_action(self, user_query: str, conversation_history: str = "") -> dict:
        """
        Classify query using intelligent few-shot learning.
        Falls back to original planner logic if learning system fails.
        """
        # Check for clear METADATA indicators first
        query_lower = user_query.lower()
        metadata_keywords = [
            "dataset",
            "data",
            "available",
            "tables",
            "columns",
            "structure",
            "overview",
            "information",
            "what can",
            "what do you have",
            "tell me about",
            "describe",
            "schema",
            "fields",
            "explain",
            "simple words",
            "simpler",
            "easier to understand",
        ]

        # Check conversation history for dataset context
        history_lower = conversation_history.lower() if conversation_history else ""
        has_dataset_context = any(
            word in history_lower for word in ["dataset", "data", "olist", "e-commerce"]
        )

        # Detect vague/incomplete queries that likely need context from history
        vague_indicators = [
            "what about",
            "how about",
            "what if",
            "and what about",
            "also show",
            "now show",
        ]

        # Check if query is vague/incomplete (too short or starts with vague pattern)
        is_vague = any(pattern in query_lower for pattern in vague_indicators) or (
            len(user_query.split()) <= 4
            and any(word in query_lower for word in ["about", "for", "in", "from"])
        )

        # If query seems incomplete/vague AND we have conversation history, inherit action
        if is_vague and conversation_history:
            # Try to extract the last action from conversation history using multiple patterns
            last_action = None

            # Look for explicit "Action: X" pattern
            for action_type in ["SQL", "CODE", "RAG", "SQL+RAG", "METADATA"]:
                if f"Action: {action_type}" in conversation_history:
                    last_action = action_type
                    break

            # If no explicit action found, look at the previous query content to infer
            if not last_action and "Query:" in conversation_history:
                # Extract the last query from history
                lines = conversation_history.split("\n")
                for line in reversed(lines):
                    if line.startswith("Query:"):
                        prev_query = line.replace("Query:", "").strip()
                        prev_lower = prev_query.lower()

                        # Infer action from previous query patterns
                        if any(
                            word in prev_lower
                            for word in [
                                "show",
                                "top",
                                "most",
                                "count",
                                "how many",
                                "average",
                                "total",
                                "list",
                            ]
                        ):
                            last_action = "SQL"
                        elif any(
                            word in prev_lower
                            for word in ["chart", "plot", "visuali", "graph"]
                        ):
                            last_action = "CODE"
                        elif any(
                            word in prev_lower
                            for word in [
                                "why",
                                "explain",
                                "sentiment",
                                "feel",
                                "opinion",
                            ]
                        ):
                            last_action = "RAG"
                        break

            if (
                last_action
                and last_action != "METADATA"
                and last_action != "CONVERSATION"
            ):
                logger.info(
                    f"Follow-up detected: '{user_query}' inheriting action '{last_action}' from context"
                )
                return self._build_result(
                    last_action,
                    f"Vague/incomplete query detected - inheriting '{last_action}' action from previous context",
                    0.88,
                    use_cache=False,
                )

        # Check for visualization/chart requests FIRST (these take priority)
        visualization_keywords = [
            "chart",
            "plot",
            "graph",
            "visuali",
            "create a",
            "generate a",
            "draw",
            "diagram",
        ]

        if any(keyword in query_lower for keyword in visualization_keywords):
            # This is clearly a CODE request for visualization
            return self._build_result(
                "CODE",
                "Query explicitly requests charts/visualizations",
                0.95,
                use_cache=False,
            )

        if any(keyword in query_lower for keyword in metadata_keywords) or (
            has_dataset_context
            and any(
                word in query_lower
                for word in ["explain", "simple", "simpler", "easier"]
            )
        ):
            # Additional check: if it's asking about the dataset itself (not specific analysis)
            analysis_keywords = [
                "analyze",
                "show",
                "find",
                "calculate",
                "average",
                "count",
                "top",
                "best",
                "worst",
            ]
            if not any(keyword in query_lower for keyword in analysis_keywords):
                return self._build_result(
                    "METADATA",
                    "Query asks about dataset structure, available data, or overview",
                    0.9,
                    use_cache=False,
                )

        try:
            # Find similar past queries
            similar_queries = self.learning_system.find_similar_queries(
                user_query, top_k=3
            )

            # Check for high-confidence match
            if similar_queries and similar_queries[0]["similarity"] > 0.95:
                logger.info(
                    f"High-confidence match found: {similar_queries[0]['query']}"
                )
                return self._build_result(
                    similar_queries[0]["action"],
                    similar_queries[0]["reasoning"],
                    similar_queries[0]["confidence"],
                    use_cache=True,
                )

            # Use few-shot learning with LLM
            result = self._classify_with_examples(
                user_query, similar_queries, conversation_history
            )

            # Store this query for future learning
            self.learning_system.add_example(
                user_query, result["action"], result["confidence"], result["reason"]
            )

            return result

        except Exception as e:
            logger.error(f"Error in intelligent planner: {e}")
            # Fallback to original planner WITHOUT recursion
            from backend.planner import decide_action as fallback_planner

            try:
                return fallback_planner(user_query, conversation_history)
            except Exception as fallback_error:
                logger.error(f"Fallback planner also failed: {fallback_error}")
                # Return safe default action
                return self._build_result(
                    "CONVERSATIONAL",
                    "Falling back to conversational mode due to errors",
                    0.5,
                    use_cache=False,
                )

    def _classify_with_examples(
        self, query: str, similar_queries: list, conversation_history: str = ""
    ) -> dict:
        """Classify query using few-shot learning from similar examples."""

        # Build examples text
        examples_text = ""
        if similar_queries:
            examples_text = "\n\n**Similar Past Queries (for reference):**\n"
            for i, ex in enumerate(similar_queries, 1):
                examples_text += f"{i}. Query: \"{ex['query']}\"\n"
                examples_text += f"   → Action: {ex['action']}\n"
                examples_text += f"   → Reasoning: {ex['reasoning']}\n"
                examples_text += f"   → Similarity: {ex['similarity']:.2f} | Success Rate: {ex['success_rate']:.0%}\n\n"

        classification_prompt = f"""You are an intelligent query router for an e-commerce data analysis system.

**Available Actions:**
1. **SQL** - For queries requiring structured data analysis (counts, aggregations, filtering, ratings)
   - Examples: "Show top 5 products", "How many orders in 2018?", "Products with worst ratings"
   - IMPORTANT: Queries about "best/worst", "most loved/hated", "highest/lowest rated" use SQL to analyze review_score

2. **RAG** - For qualitative insights from review TEXT (comments, feedback)
   - Examples: "Why are customers unhappy?", "What do review COMMENTS say?"
   - NOTE: If asking about sentiment SCORES (numeric), use SQL instead

3. **CODE** - For complex analysis, visualizations, or custom transformations
   - Examples: "Generate Python code to analyze...", "Create a correlation matrix"

4. **SQL+RAG** - For queries needing both structured data AND contextual insights
   - Examples: "Show top products and explain why they're popular"

5. **CONVERSATION** - For greetings, thanks, clarifications, off-topic questions
   - Examples: "Hi", "Thank you", "Can you help me?"
   - NOTE: ONLY use this for non-data queries. If it relates to data analysis, use another action.

6. **METADATA** - For queries about the dataset itself (structure, available data, schema, overview)
   - Examples: "What data do you have?", "Tell me about this dataset", "What can I analyze?", "Available tables?", "Dataset overview", "What information is available?"
   - KEY: Questions asking WHAT the dataset contains, its structure, or capabilities

**CRITICAL: Handling Follow-up Questions**
- If the query is vague (e.g., "what about january?", "show me more", "what about 2017?"), look at the conversation history
- Inherit the action type from the previous query if it's a continuation
- Examples:
  - Previous: "what are the most loved products" (SQL) → Current: "what about in january?" → Action: SQL (filtering by time)
  - Previous: "show sales trends" (SQL) → Current: "what about 2017?" → Action: SQL (filtering by year)
  - Previous: "create a chart" (CODE) → Current: "make it for june" → Action: CODE (updating time filter)

**Dataset Context:**
E-commerce (Olist Brazilian) dataset with: orders, products, customers, sellers, payments, reviews
Can analyze: sales, revenue, trends, geography, sentiment, customer behavior
{examples_text}
**Conversation History:**
{conversation_history if conversation_history else "None - Fresh query"}

**Current Query:** "{query}"

**Your Task:**
Analyze the query intent and classify it. If this is a follow-up question (like "what about X?"), inherit the action from the previous query. Learn from the similar queries above. Respond with valid JSON only (no markdown):

{{
  "action": "SQL|RAG|CODE|SQL+RAG|CONVERSATION|METADATA",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this action was chosen"
}}

JSON Response:"""

        try:
            llm_response = call_llm(
                classification_prompt, max_tokens=LLMTokenLimits.SHORT
            )

            if not llm_response:
                return self._fallback_classification(query)

            # Check if response is an error message (starts with conversational text)
            error_indicators = [
                "I apologize, but I'm experiencing",
                "Hey there! I'm getting a lot of requests",
                "Hmm, I'm having trouble connecting",
                "Oops! I ran into an unexpected issue",
                "All LLM providers failed",
                "[SYSTEM NOTICE]",
                "[ERROR:",
            ]

            is_error_response = any(
                llm_response.startswith(indicator) for indicator in error_indicators
            )

            if is_error_response:
                logger.warning(
                    f"LLM returned error message instead of JSON: {llm_response[:100]}..."
                )
                return self._fallback_classification(query)

            # Clean response
            llm_response = llm_response.strip()
            if llm_response.startswith("```"):
                llm_response = llm_response.split("```")[1]
                if llm_response.startswith("json"):
                    llm_response = llm_response[4:]
            llm_response = llm_response.strip()

            # Parse response
            classification = json.loads(llm_response)

            action = classification.get("action", "RAG")
            confidence = classification.get("confidence", 0.5)
            reasoning = classification.get("reasoning", "LLM-based classification")

            # Validate action
            valid_actions = [
                "SQL",
                "RAG",
                "CODE",
                "SQL+RAG",
                "CONVERSATION",
                "METADATA",
            ]
            if action not in valid_actions:
                logger.warning(f"Invalid action '{action}' from LLM, using fallback")
                return self._fallback_classification(query)

            return self._build_result(action, reasoning, confidence)

        except Exception as e:
            logger.exception(f"Error in LLM classification: {e}")
            return self._fallback_classification(query)

    def _build_result(
        self, action: str, reasoning: str, confidence: float, use_cache: bool = False
    ) -> dict:
        """Build standardized result dictionary."""
        valid_actions = ["SQL", "RAG", "CODE", "SQL+RAG", "CONVERSATION", "METADATA"]
        scores = {a: 0.0 for a in valid_actions}
        scores[action] = confidence

        return {
            "action": action,
            "reason": reasoning,
            "scores": scores,
            "confidence": confidence,
            "sql_template": None,
            "retrieval_query": None,
            "visualization": None,
            "requires_context": False,
            "from_cache": use_cache,
        }

    def _fallback_classification(self, user_query: str) -> dict:
        """Fallback classification using keyword matching."""
        q = user_query.lower().strip()

        # Greetings
        if any(
            word in q for word in ["hi", "hello", "hey", "thanks", "thank you", "bye"]
        ):
            return self._build_result("CONVERSATION", "Greeting detected", 0.9)

        # Metadata - Enhanced patterns
        if any(
            phrase in q
            for phrase in [
                "tell me about",
                "what data",
                "what's the data",
                "whats the data",
                "dataset",
                "schema",
                "about the data",
                "describe the data",
                "data about",
            ]
        ):
            return self._build_result("METADATA", "Dataset overview request", 0.85)

        # Code
        if any(
            phrase in q
            for phrase in ["generate code", "create code", "python code", "script"]
        ):
            return self._build_result("CODE", "Code generation requested", 0.9)

        # SQL (including sentiment-based queries)
        if any(
            word in q
            for word in [
                "how many",
                "count",
                "total",
                "sum",
                "average",
                "top",
                "bottom",
                "show",
                "list",
                "most",
                "least",
                "best",
                "worst",
                "hated",
                "loved",
                "rated",
                "rating",
            ]
        ):
            return self._build_result("SQL", "Structured data query detected", 0.8)

        # Default to RAG for text analysis
        return self._build_result("RAG", "Default text analysis", 0.6)

    def record_feedback(
        self,
        query: str,
        action_taken: str,
        was_correct: bool,
        correct_action: Optional[str] = None,
        implicit_signals: Optional[dict] = None,
    ):
        """Record feedback to improve future classifications."""
        self.learning_system.record_feedback(
            query, action_taken, was_correct, correct_action, implicit_signals
        )

    def get_statistics(self) -> dict:
        """Get learning system statistics."""
        return self.learning_system.get_statistics()

    def expand_vague_query(self, query: str, conversation_history: str) -> str:
        """
        Expand a vague follow-up query into a complete query using conversation history.

        Example:
            Previous: "Show me the top 5 most sold product categories from january 2017"
            Vague: "what about february 2017?"
            Expanded: "Show me the top 5 most sold product categories from february 2017"
        """
        if not conversation_history or "Query:" not in conversation_history:
            return query

        # Extract the last query from history
        lines = conversation_history.split("\n")
        prev_query = None
        for line in reversed(lines):
            if line.startswith("Query:"):
                prev_query = line.replace("Query:", "").strip()
                break

        if not prev_query:
            return query

        # Check if current query is actually vague
        query_lower = query.lower()
        vague_patterns = ["what about", "how about", "what if", "also show", "now show"]
        is_vague = any(pattern in query_lower for pattern in vague_patterns)

        if not is_vague:
            return query

        # Use LLM to intelligently expand the query
        expansion_prompt = f"""You are helping expand a vague follow-up query into a complete query using context.

Previous Query: "{prev_query}"

Follow-up Query: "{query}"

Task: Rewrite the follow-up query as a complete, standalone query that includes the context from the previous query.

Rules:
1. Keep the SAME intent/action from the previous query (e.g., if it was "show top 5", keep "show top 5")
2. Replace the time period/filter with the new one from the follow-up
3. Make it a complete, clear query
4. Return ONLY the expanded query, nothing else

Expanded Query:"""

        try:
            expanded = call_llm(expansion_prompt, max_tokens=LLMTokenLimits.SHORT)
            if expanded and len(expanded.strip()) > len(query):
                # Validate the expansion makes sense
                if any(
                    word in expanded.lower()
                    for word in ["show", "get", "list", "find", "top", "most"]
                ):
                    logger.info(f"Expanded vague query: '{query}' → '{expanded}'")
                    return expanded.strip()
        except Exception as e:
            logger.warning(f"Failed to expand vague query: {e}")

        # Fallback: return original query
        return query
