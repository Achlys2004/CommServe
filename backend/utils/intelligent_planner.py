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
            logger.exception(f"Error in intelligent planner: {e}")
            # Fallback to original planner
            from backend.planner import decide_action as fallback_planner

            return fallback_planner(user_query, conversation_history)

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

5. **CONVERSATION** - For greetings, thanks, clarifications
   - Examples: "Hi", "Thank you", "Can you help me?"

6. **METADATA** - For queries about the dataset itself (structure, available data, schema, overview)
   - Examples: "What data do you have?", "Tell me about this dataset", "What can I analyze?", "Available tables?", "Dataset overview", "What information is available?"
   - KEY: Questions asking WHAT the dataset contains, its structure, or capabilities

**Dataset Context:**
E-commerce (Olist Brazilian) dataset with: orders, products, customers, sellers, payments, reviews
Can analyze: sales, revenue, trends, geography, sentiment, customer behavior
{examples_text}
**Conversation History:**
{conversation_history if conversation_history else "None - Fresh query"}

**Current Query:** "{query}"

**Your Task:**
Analyze the query intent and classify it. Learn from the similar queries above. Respond with valid JSON only (no markdown):

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
