from transformers import pipeline
from backend.llm_client import call_llm
from backend.llm_config import LLMTokenLimits
import logging

logger = logging.getLogger(__name__)


class Summariser:
    def __init__(self):
        self.summariser = pipeline("summarization", model="facebook/bart-large-cnn")
        self.use_llm_for_hybrid = True  # Use LLM for better hybrid summaries

    def generate_summary(self, query, result, sentiment_emotion_str=""):
        """Generate summary - ALWAYS uses LLM for better ChatGPT-quality responses."""

        # FIXED: Always use LLM for ChatGPT-quality summaries
        # BART produces low-quality concatenated text like "Query: X Results: Y"
        return self._generate_llm_summary(query, result, sentiment_emotion_str)

    def _generate_llm_summary(self, query, result, sentiment_emotion_str=""):
        """Generate summary using LLM for better quality."""
        try:
            # Extract key information
            result_str = str(result)

            # Limit context size
            max_context = 3000
            if len(result_str) > max_context:
                # Try to extract SQL results if present
                if "SQL Query Results:" in result_str:
                    sql_start = result_str.find("SQL Query Results:")
                    sql_section = result_str[sql_start : sql_start + 1500]
                    result_str = sql_section
                else:
                    result_str = result_str[:max_context]

            prompt = f"""Analyze this data and provide a clear, concise answer to the user's question.

User Question: {query}

Data Available:
{result_str}

{sentiment_emotion_str}

Instructions:
1. If SQL data is present, highlight the TOP results (best/most sold/highest rated)
2. Be specific with numbers, names, and rankings
3. If sentiment data is available, mention overall customer satisfaction
4. Keep response under 150 words
5. Answer the question directly - don't just say "here's data"
6. Provide a single, coherent response without repetition

Your Answer:"""

            logger.info(f"🔍 SUMMARISER PROMPT (first 500 chars):\n{prompt[:500]}")
            answer = call_llm(prompt, max_tokens=LLMTokenLimits.MEDIUM)
            logger.info(f"🤖 LLM RESPONSE: {answer if answer else 'None returned'}")
            return (
                answer if answer else "Analysis complete - see data above for details."
            )

        except Exception as e:
            logger.exception(f"LLM summary generation failed: {e}")
            return f"Query: {query}. Result: See data above for detailed analysis."

    def summarise_sql_result(self, rows, columns):
        if not rows:
            return "No results found for this query."
        sample = rows[:3]
        return (
            f"Sample data: {sample}. {len(rows)} rows returned with columns {columns}."
        )
