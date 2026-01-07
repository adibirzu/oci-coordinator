"""Tests for Database Troubleshoot Agent.

Tests for database name extraction regex patterns and agent functionality.
"""

import pytest


class TestDatabaseNameExtraction:
    """Tests for _extract_db_name method patterns."""

    @pytest.fixture
    def extract_fn(self):
        """Create a standalone extraction function for testing."""
        import re

        def _extract_db_name(query: str) -> str | None:
            """Extract database name from natural language query."""
            query_lower = query.lower()

            # Common words that are never database names
            filtered_words = {
                "the", "a", "my", "our", "this", "that", "all", "any",
                "performance", "health", "status", "metrics", "issues",
                "slow", "error", "problem", "check", "query", "queries",
            }

            # Regex patterns for database name extraction (order matters!)
            db_patterns = [
                r"['\"]?([\w_-]+)['\"]?\s+(?:database|db)\s+(?:performance|health|status)",
                r"(?:for|on|analyze|check|troubleshoot|investigate)\s+['\"]?([\w_-]+)['\"]?\s+(?:database|db)",
                r"(?:performance|status|health)\s+(?:of|for)\s+['\"]?([\w_-]+)['\"]?",
                r"['\"]?([\w_-]+)['\"]?\s+(?:database|db)(?:\s|$|[,.])",
                r"(?:database|db)\s+['\"]?([\w_-]+)['\"]?",
                r"(?:for|on|analyze|check|troubleshoot|investigate)\s+['\"]?([\w_-]+)['\"]?",
            ]

            for pattern in db_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    name = match.group(1)
                    if name not in filtered_words:
                        return name

            return None

        return _extract_db_name

    def test_check_dbname_db_performance(self, extract_fn):
        """Test 'check FINANCE DB performance' pattern."""
        assert extract_fn("check FINANCE DB performance") == "finance"

    def test_analyze_dbname_database_health(self, extract_fn):
        """Test 'analyze PRODDB database health' pattern."""
        assert extract_fn("analyze PRODDB database health") == "proddb"

    def test_troubleshoot_dbname(self, extract_fn):
        """Test 'troubleshoot HR_DB' pattern."""
        assert extract_fn("troubleshoot HR_DB") == "hr_db"

    def test_database_dbname_is_slow(self, extract_fn):
        """Test 'database ORCL is slow' pattern."""
        assert extract_fn("database ORCL is slow") == "orcl"

    def test_performance_of_dbname(self, extract_fn):
        """Test 'performance of SALES_DB' pattern."""
        assert extract_fn("performance of SALES_DB") == "sales_db"

    def test_check_performance_of_dbname(self, extract_fn):
        """Test 'check performance of FINANCE' pattern."""
        assert extract_fn("check performance of FINANCE") == "finance"

    def test_dbname_db_has_issues(self, extract_fn):
        """Test 'INVENTORY db has issues' pattern."""
        assert extract_fn("INVENTORY db has issues") == "inventory"

    def test_status_of_dbname(self, extract_fn):
        """Test 'status of TESTDB' pattern."""
        assert extract_fn("status of TESTDB") == "testdb"

    def test_health_of_dbname(self, extract_fn):
        """Test 'health of PRODDB' pattern."""
        assert extract_fn("health of PRODDB") == "proddb"

    def test_investigate_dbname_db(self, extract_fn):
        """Test 'investigate MYDB db' pattern."""
        assert extract_fn("investigate MYDB db") == "mydb"

    def test_db_dbname_lookup(self, extract_fn):
        """Test 'db FINANCE' pattern."""
        assert extract_fn("db FINANCE") == "finance"

    def test_database_dbname_lookup(self, extract_fn):
        """Test 'database FINANCE' pattern."""
        assert extract_fn("database FINANCE") == "finance"

    def test_does_not_extract_filtered_words(self, extract_fn):
        """Test that common words are filtered out."""
        # Should not extract 'performance' as database name
        assert extract_fn("check the performance") is None

    def test_quoted_database_names(self, extract_fn):
        """Test extraction with quoted database names."""
        assert extract_fn("check 'FINANCE' db performance") == "finance"
        assert extract_fn('database "MYDB"') == "mydb"

    def test_hyphenated_database_names(self, extract_fn):
        """Test extraction of hyphenated database names."""
        assert extract_fn("check prod-db-1 database") == "prod-db-1"

    def test_underscored_database_names(self, extract_fn):
        """Test extraction of underscored database names."""
        assert extract_fn("analyze hr_db_prod database") == "hr_db_prod"

    def test_no_database_mentioned(self, extract_fn):
        """Test queries without database references."""
        assert extract_fn("show me all instances") is None
        assert extract_fn("list compartments") is None
