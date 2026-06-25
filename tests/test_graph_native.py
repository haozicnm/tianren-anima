"""P0.2: Graph-native storage tests — schema migration, fact extraction, graph search."""

import pytest
import sqlite3
import tempfile
import os
import sys

# Add the source path
src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src_path)


class TestSchemaMigration:
    """Verify migration 003 correctly transforms the temporal_graph schema."""

    def setup_method(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def teardown_method(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _run_initial_schema(self):
        """Run the 001 and 002 migrations."""
        mig_dir = os.path.join(src_path, "anima", "migrations")

        for fname in ["001_initial.sql", "002_add_nature.sql"]:
            path = os.path.join(mig_dir, fname)
            if os.path.exists(path):
                sql = open(path).read()
                self.conn.executescript(sql)

    def _run_migration_003(self):
        """Run the 003 migration."""
        path = os.path.join(src_path, "anima", "migrations", "003_fix_temporal_graph.sql")
        if os.path.exists(path):
            sql = open(path).read()
            self.conn.executescript(sql)

    def test_temporal_facts_column_renames(self):
        """003 should rename 'obj' to 'object' and add new columns."""
        self._run_initial_schema()
        self._run_migration_003()

        # Verify new column names
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(temporal_facts)").fetchall()]
        assert "object" in cols, "Should have 'object' column after migration"
        assert "obj" not in cols, "Old 'obj' column should be gone"
        assert "user_id" in cols, "Should have 'user_id' column"
        assert "last_updated" in cols, "Should have 'last_updated' column"
        assert "memory_id" in cols, "Should have 'memory_id' column"

    def test_temporal_edges_column_renames(self):
        """003 should add 'id' PK and rename 'relation' to 'relation_type'."""
        self._run_initial_schema()
        self._run_migration_003()

        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(temporal_edges)").fetchall()]
        assert "id" in cols, "Should have 'id' column"
        assert "relation_type" in cols, "Should have 'relation_type' column"
        assert "relation" not in cols, "Old 'relation' column should be gone"

    def test_data_preservation_on_migration(self):
        """Data should survive the migration."""
        self._run_initial_schema()

        # Insert test data using OLD schema columns
        conn = self.conn
        conn.execute(
            "INSERT INTO temporal_facts(id, subject, predicate, obj, valid_from, valid_to, confidence) "
            "VALUES ('f1', 'Alice', 'knows', 'Bob', 1000, NULL, 0.9)"
        )
        conn.execute(
            "INSERT INTO temporal_edges(source_id, target_id, relation, valid_from, valid_to, weight) "
            "VALUES ('f1', 'f1', 'self_ref', 1000, NULL, 1.0)"
        )

        self._run_migration_003()

        # Verify data migrated correctly
        fact = conn.execute("SELECT * FROM temporal_facts WHERE id='f1'").fetchone()
        assert fact["object"] == "Bob", "Data should be in 'object' column"
        assert fact["subject"] == "Alice"
        assert fact["confidence"] == 0.9

        edge = conn.execute("SELECT * FROM temporal_edges LIMIT 1").fetchone()
        assert edge["relation_type"] == "self_ref", "Data should be in 'relation_type' column"
        assert edge["id"] is not None, "Should have auto-generated id"


class TestFactExtraction:
    """Test SPO fact extraction from content."""

    def test_extract_type_of_fact(self):
        from anima.ops.extract_facts import extract_candidate_facts

        facts = extract_candidate_facts("Alice is a software engineer at Google.")
        assert len(facts) >= 1
        # At least one should be the type_of pattern
        type_facts = [f for f in facts if f["predicate"] == "is_a"]
        assert len(type_facts) >= 1

    def test_extract_has_fact(self):
        from anima.ops.extract_facts import extract_candidate_facts

        facts = extract_candidate_facts("Bob has a red bicycle and a blue car.")
        assert len(facts) >= 1
        has_facts = [f for f in facts if f["predicate"] == "has"]
        assert len(has_facts) >= 1

    def test_extract_action_fact(self):
        from anima.ops.extract_facts import extract_candidate_facts

        facts = extract_candidate_facts("Charlie built the prototype in two weeks.")
        assert len(facts) >= 1
        action_facts = [f for f in facts if f["relation_type"] == "action"]
        assert len(action_facts) >= 1

    def test_extract_empty_content(self):
        from anima.ops.extract_facts import extract_candidate_facts

        facts = extract_candidate_facts("hmm ok thanks bye")
        assert facts == [], "Should return empty for content without proper nouns"

    def test_extract_deduplication(self):
        from anima.ops.extract_facts import extract_candidate_facts

        # Same fact appearing twice
        facts = extract_candidate_facts(
            "Alice is a developer. Alice is a developer."
        )
        # Should deduplicate
        type_facts = [f for f in facts if f["predicate"] == "is_a"]
        assert len(type_facts) == 1, "Should deduplicate identical facts"

    def test_extract_multiple_patterns(self):
        from anima.ops.extract_facts import extract_candidate_facts

        facts = extract_candidate_facts(
            "Alice is a data scientist. She works at OpenAI and has a PhD in ML. "
            "Alice built the recommendation system and said the results were promising."
        )
        # Should extract from multiple patterns
        assert len(facts) >= 2, "Should extract multiple facts from rich content"


class TestMemoryFactBridge:
    """Test that facts are linked to memories via memory_id."""

    @pytest.mark.asyncio
    async def test_link_and_retrieve_facts(self):
        """Insert a fact with memory_id, then retrieve it."""
        import uuid
        from anima.core.db import db, q
        from anima.temporal_graph import insert_fact, get_facts_for_memory

        # Use a temp in-memory database
        # This test requires the DB to be set up
        # Skip for unit test; focus on integration
        pytest.skip("Requires configured database — tested in integration")


class TestGraphSearchEnrichment:
    """Test that search results include graph context."""

    @pytest.mark.asyncio
    async def test_enrich_adds_graph_fields(self):
        from anima.memory.hsg import enrich_with_graph_context

        results = [
            {"id": "test-mem-1", "content": "Alice works at OpenAI", "score": 0.9},
            {"id": "test-mem-2", "content": "Bob is a researcher", "score": 0.7},
        ]

        enriched = await enrich_with_graph_context(results)

        assert len(enriched) == 2
        for r in enriched:
            assert "graph_facts" in r, "Should have graph_facts field"
            assert "graph_edges" in r, "Should have graph_edges field"
            assert isinstance(r["graph_facts"], list)
            assert isinstance(r["graph_edges"], list)

    @pytest.mark.asyncio
    async def test_enrich_empty_results(self):
        from anima.memory.hsg import enrich_with_graph_context

        enriched = await enrich_with_graph_context([])
        assert enriched == []
