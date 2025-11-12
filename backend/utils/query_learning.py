"""
Query Learning System - Intelligent Few-Shot Classification with Feedback Loop
Learns from past queries to improve classification accuracy over time.
"""

import json
import sqlite3
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from backend.llm_client import call_llm
from backend.llm_config import LLMTokenLimits

logger = logging.getLogger(__name__)


class QueryLearningSystem:
    """Manages query examples, embeddings, and continuous learning."""

    def __init__(self, db_path: str = "data/query_learning.db"):
        self.db_path = db_path
        self._init_database()
        self._embeddings_cache = {}

    def _init_database(self):
        """Initialize SQLite database for storing query examples."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query examples table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL,
                reasoning TEXT,
                embedding TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Query feedback table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                was_correct BOOLEAN,
                correct_action TEXT,
                user_feedback TEXT,
                implicit_signals TEXT,
                reward_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indices for faster lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_examples_action 
            ON query_examples(action)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_examples_success 
            ON query_examples(success_count DESC)
        """
        )

        conn.commit()
        conn.close()

        # Initialize with golden examples if database is empty
        self._seed_golden_examples()

    def _seed_golden_examples(self):
        """Seed database with high-quality example queries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if we already have examples
        cursor.execute("SELECT COUNT(*) FROM query_examples")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return

        golden_examples = [
            # SQL - Numeric queries
            (
                "show top 5 products",
                "SQL",
                1.0,
                "Aggregation query requiring order counts",
            ),
            ("how many orders in 2018", "SQL", 1.0, "Count query with date filter"),
            ("average order value by state", "SQL", 1.0, "Aggregation with grouping"),
            (
                "total revenue last month",
                "SQL",
                1.0,
                "Sum aggregation with date filter",
            ),
            ("list all customers from SP", "SQL", 1.0, "Simple select with filter"),
            ("most sold product", "SQL", 1.0, "Aggregation to find maximum"),
            ("products with lowest ratings", "SQL", 1.0, "Aggregation on review_score"),
            ("most hated product", "SQL", 1.0, "Lowest average review_score analysis"),
            (
                "best rated products",
                "SQL",
                1.0,
                "Highest average review_score analysis",
            ),
            ("count orders by status", "SQL", 1.0, "Count with grouping"),
            # RAG - Text analysis
            (
                "why are customers unhappy",
                "RAG",
                0.95,
                "Qualitative analysis of review comments",
            ),
            (
                "what do reviews say about shipping",
                "RAG",
                0.95,
                "Text analysis of review messages",
            ),
            ("customer feedback on quality", "RAG", 0.95, "Sentiment from review text"),
            (
                "common complaints in reviews",
                "RAG",
                0.95,
                "Pattern extraction from text",
            ),
            ("what makes customers happy", "RAG", 0.95, "Positive sentiment analysis"),
            # CODE - Analysis/visualization
            (
                "generate python code to analyze",
                "CODE",
                1.0,
                "Explicit code generation request",
            ),
            (
                "create a correlation matrix",
                "CODE",
                0.9,
                "Complex analysis requiring code",
            ),
            ("build prediction model", "CODE", 0.9, "Machine learning task"),
            ("write code to visualize", "CODE", 1.0, "Explicit visualization request"),
            # SQL+RAG - Hybrid
            (
                "show top products and explain why they're popular",
                "SQL+RAG",
                0.85,
                "Needs both stats and text insights",
            ),
            (
                "sales trends and customer sentiment",
                "SQL+RAG",
                0.85,
                "Combines numeric and text analysis",
            ),
            # CONVERSATION
            ("hi", "CONVERSATION", 1.0, "Simple greeting"),
            ("thank you", "CONVERSATION", 1.0, "Acknowledgment"),
            ("hello", "CONVERSATION", 1.0, "Greeting"),
            # METADATA
            ("what data do you have", "METADATA", 1.0, "Dataset information request"),
            ("tell me about this dataset", "METADATA", 1.0, "Schema overview request"),
            ("available tables", "METADATA", 1.0, "Database structure query"),
        ]

        for query, action, confidence, reasoning in golden_examples:
            cursor.execute(
                """
                INSERT INTO query_examples (query, action, confidence, reasoning, success_count)
                VALUES (?, ?, ?, ?, ?)
            """,
                (query, action, confidence, reasoning, 5),
            )  # Pre-seed with success count

        conn.commit()
        conn.close()
        logger.info(f"Seeded {len(golden_examples)} golden examples")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using multi-tier fallback with caching."""
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]

        try:
            # Use the multi-tier fallback system from llm_client
            from backend.llm_client import embed_texts

            embeddings = embed_texts([text])
            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                self._embeddings_cache[text] = embedding
                return embedding
            return None

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_array = np.array(a)
        b_array = np.array(b)
        return np.dot(a_array, b_array) / (
            np.linalg.norm(a_array) * np.linalg.norm(b_array)
        )

    def find_similar_queries(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find most similar past queries using semantic similarity."""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all examples with embeddings
        cursor.execute(
            """
            SELECT id, query, action, confidence, reasoning, embedding, 
                   success_count, failure_count
            FROM query_examples
            WHERE embedding IS NOT NULL
            ORDER BY success_count DESC
            LIMIT 100
        """
        )

        results = []
        for row in cursor.fetchall():
            (
                example_id,
                example_query,
                action,
                confidence,
                reasoning,
                embedding_str,
                success,
                failure,
            ) = row

            if embedding_str:
                try:
                    example_embedding = json.loads(embedding_str)
                    similarity = self.cosine_similarity(
                        query_embedding, example_embedding
                    )

                    # Calculate quality score (combine similarity with success rate)
                    total = success + failure
                    success_rate = success / total if total > 0 else 0.5
                    quality_score = similarity * 0.7 + success_rate * 0.3

                    results.append(
                        {
                            "id": example_id,
                            "query": example_query,
                            "action": action,
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "similarity": similarity,
                            "quality_score": quality_score,
                            "success_rate": success_rate,
                        }
                    )
                except json.JSONDecodeError:
                    continue

        conn.close()

        # Sort by quality score and return top-k
        results.sort(key=lambda x: x["quality_score"], reverse=True)
        return results[:top_k]

    def add_example(
        self, query: str, action: str, confidence: float, reasoning: str = ""
    ):
        """Add a new query example to the learning system."""
        embedding = self.get_embedding(query)
        embedding_str = json.dumps(embedding) if embedding else None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if similar query already exists
        cursor.execute(
            """
            SELECT id FROM query_examples 
            WHERE query = ? AND action = ?
        """,
            (query, action),
        )

        if cursor.fetchone():
            # Update existing
            cursor.execute(
                """
                UPDATE query_examples 
                SET confidence = ?, reasoning = ?, last_used = CURRENT_TIMESTAMP
                WHERE query = ? AND action = ?
            """,
                (confidence, reasoning, query, action),
            )
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO query_examples 
                (query, action, confidence, reasoning, embedding, last_used)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (query, action, confidence, reasoning, embedding_str),
            )

        conn.commit()
        conn.close()

    def record_feedback(
        self,
        query: str,
        action_taken: str,
        was_correct: bool,
        correct_action: Optional[str] = None,
        implicit_signals: Optional[Dict] = None,
    ):
        """Record feedback about a query classification."""
        # Calculate reward score
        reward = self._calculate_reward(was_correct, implicit_signals)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Store feedback
        cursor.execute(
            """
            INSERT INTO query_feedback 
            (query, action_taken, was_correct, correct_action, implicit_signals, reward_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                query,
                action_taken,
                was_correct,
                correct_action,
                json.dumps(implicit_signals) if implicit_signals else None,
                reward,
            ),
        )

        # Update example success/failure counts
        if was_correct:
            cursor.execute(
                """
                UPDATE query_examples 
                SET success_count = success_count + 1
                WHERE query = ? AND action = ?
            """,
                (query, action_taken),
            )
        else:
            cursor.execute(
                """
                UPDATE query_examples 
                SET failure_count = failure_count + 1
                WHERE query = ? AND action = ?
            """,
                (query, action_taken),
            )

            # Add corrected example if provided
            if correct_action:
                embedding = self.get_embedding(query)
                embedding_str = json.dumps(embedding) if embedding else None

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO query_examples 
                    (query, action, confidence, reasoning, embedding, success_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        query,
                        correct_action,
                        0.8,
                        "Corrected from user feedback",
                        embedding_str,
                        1,
                    ),
                )

        conn.commit()
        conn.close()

    def _calculate_reward(
        self, was_correct: bool, implicit_signals: Optional[Dict] = None
    ) -> float:
        """Calculate reward score from explicit and implicit signals."""
        if was_correct:
            return 1.0

        if not implicit_signals:
            return 0.0

        # Use implicit signals to estimate correctness
        score = 0.0

        if implicit_signals.get("has_results"):
            score += 0.3
        if implicit_signals.get("user_clicked_results"):
            score += 0.3
        if implicit_signals.get("execution_success"):
            score += 0.2
        if not implicit_signals.get("user_refined_query"):
            score += 0.2

        return score

    def get_statistics(self) -> Dict:
        """Get learning system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM query_examples")
        total_examples = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM query_feedback")
        total_feedback = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT action, COUNT(*) 
            FROM query_examples 
            GROUP BY action
        """
        )
        examples_by_action = dict(cursor.fetchall())

        cursor.execute(
            """
            SELECT AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END)
            FROM query_feedback
            WHERE created_at > datetime('now', '-7 days')
        """
        )
        recent_accuracy = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_examples": total_examples,
            "total_feedback": total_feedback,
            "examples_by_action": examples_by_action,
            "recent_accuracy": round(recent_accuracy * 100, 2),
            "last_updated": datetime.now().isoformat(),
        }

    def export_training_data(self, output_path: str = "training_data.jsonl"):
        """Export high-quality examples for potential fine-tuning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get high-quality examples
        cursor.execute(
            """
            SELECT query, action, reasoning
            FROM query_examples
            WHERE success_count > failure_count
            AND success_count >= 2
            ORDER BY success_count DESC
        """
        )

        with open(output_path, "w", encoding="utf-8") as f:
            for query, action, reasoning in cursor.fetchall():
                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a query classifier for an e-commerce analysis system.",
                        },
                        {"role": "user", "content": query},
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {"action": action, "reasoning": reasoning}
                            ),
                        },
                    ]
                }
                f.write(json.dumps(training_example) + "\n")

        conn.close()
        logger.info(f"Exported training data to {output_path}")
