"""
Query Preprocessor - Intelligent query enhancement and correction
"""

import re
from datetime import datetime


class QueryPreprocessor:
    """Enhances and corrects user queries before processing."""

    def __init__(self):
        # Known data range
        self.data_start_year = 2016
        self.data_end_year = 2018
        self.current_year = datetime.now().year

    def preprocess(self, query: str) -> tuple[str, list[str]]:
        """
        Preprocess query to fix common issues.

        Returns:
            tuple: (enhanced_query, list_of_warnings)
        """
        warnings = []
        enhanced = query

        # Fix date references
        enhanced, date_warnings = self._fix_date_references(enhanced)
        warnings.extend(date_warnings)

        # Add context hints
        enhanced, context_warnings = self._add_context_hints(enhanced)
        warnings.extend(context_warnings)

        # Fix common typos
        enhanced = self._fix_common_typos(enhanced)

        return enhanced, warnings

    def _fix_date_references(self, query: str) -> tuple[str, list[str]]:
        """Fix references to years outside the data range."""
        warnings = []
        enhanced = query

        # Check for year mentions
        year_pattern = r"\b(20\d{2})\b"
        years_mentioned = re.findall(year_pattern, query)

        for year_str in years_mentioned:
            year = int(year_str)
            if year < self.data_start_year or year > self.data_end_year:
                warnings.append(
                    f"⚠️ Note: Data is only available for {self.data_start_year}-{self.data_end_year}. "
                    f"Adjusting query from {year} to use available data."
                )
                # Replace with nearest valid year
                if year > self.data_end_year:
                    enhanced = enhanced.replace(year_str, str(self.data_end_year))
                else:
                    enhanced = enhanced.replace(year_str, str(self.data_start_year))

        # Check for Q1, Q2, Q3, Q4 without year
        quarter_pattern = r"\b[Qq]([1-4])\b"
        if re.search(quarter_pattern, query):
            # If no year specified with quarter, add context
            if not any(str(y) in query for y in range(2010, 2025)):
                warnings.append(
                    f"📅 Note: Using {self.data_end_year} for quarter analysis (data range: {self.data_start_year}-{self.data_end_year})"
                )

        return enhanced, warnings

    def _add_context_hints(self, query: str) -> tuple[str, list[str]]:
        """Add helpful context hints based on query type."""
        warnings = []
        query_lower = query.lower()

        # Customer reactions / reviews
        if any(
            word in query_lower
            for word in ["reaction", "sentiment", "feeling", "opinion"]
        ):
            if "review" not in query_lower:
                warnings.append(
                    "💡 Tip: Analyzing customer reactions from review scores and comments"
                )

        # Geographic/location queries
        if any(
            word in query_lower
            for word in ["location", "where", "geographic", "region"]
        ):
            if "state" not in query_lower and "city" not in query_lower:
                warnings.append("💡 Tip: Analyzing by Brazilian states and cities")

        # Sales/revenue without timeframe
        if any(
            word in query_lower for word in ["sales", "revenue", "sold"]
        ) and not any(
            word in query_lower
            for word in ["month", "year", "quarter", "week", "2016", "2017", "2018"]
        ):
            warnings.append(
                f"💡 Tip: Analyzing data from {self.data_start_year}-{self.data_end_year} period"
            )

        return query, warnings

    def _fix_common_typos(self, query: str) -> str:
        """Fix common typos and misspellings."""
        corrections = {
            "visualise": "visualize",
            "analyse": "analyze",
            "vidual": "visual",
            "custmer": "customer",
            "prodcut": "product",
        }

        enhanced = query
        for typo, correct in corrections.items():
            # Case-insensitive replacement
            enhanced = re.sub(
                r"\b" + typo + r"\b", correct, enhanced, flags=re.IGNORECASE
            )

        return enhanced
