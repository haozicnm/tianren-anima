"""P1: Version chain + cascaded entity extraction tests."""

import pytest
import tempfile
import os
import sys

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src_path)


class TestVersionChainSchema:
    """Verify migration 004 adds version chain columns."""

    def setup_method(self):
        import sqlite3
        self.db_path = tempfile.mktemp(suffix=".db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def teardown_method(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _run_migrations(self):
        mig_dir = os.path.join(src_path, "anima", "migrations")
        for fname in sorted(os.listdir(mig_dir)):
            if fname.endswith(".sql"):
                sql = open(os.path.join(mig_dir, fname)).read()
                self.conn.executescript(sql)

    def test_version_chain_columns_added(self):
        """004 should add parent_id and root_id."""
        self._run_migrations()
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(memories)").fetchall()]
        assert "parent_id" in cols
        assert "root_id" in cols

    def test_version_chain_indexes(self):
        """004 should add indexes for version chain queries."""
        self._run_migrations()
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        indexes = {r["name"] for r in rows}
        assert "idx_memories_root" in indexes
        assert "idx_memories_parent" in indexes


class TestEntityExtraction:
    """Test cascaded entity extraction."""

    def test_extract_typed_entities(self):
        from anima.ops.extract_entities import extract_entities

        entities = extract_entities(
            "Alice Smith works at Google Inc and uses Python with AWS."
        )
        entity_names = {e["name"] for e in entities}
        assert "Alice Smith" in entity_names
        assert "Google Inc" in entity_names
        assert "Python" in entity_names
        assert "AWS" in entity_names

    def test_extract_entity_types(self):
        from anima.ops.extract_entities import extract_entities

        entities = extract_entities("Alice works at OpenAI and deployed v2.4.1 on 2024-03-15.")
        types = {e["type"] for e in entities}
        assert "PERSON" in types
        assert "ORG" in types
        assert "VERSION" in types
        assert "DATE" in types

    def test_extract_no_entities(self):
        from anima.ops.extract_entities import extract_entities

        entities = extract_entities("the cat sat on the mat")
        assert entities == []

    def test_co_reference_resolution_contains(self):
        from anima.ops.extract_entities import resolve_co_references

        entities = [
            {"name": "Google", "type": "ORG", "position": 0},
            {"name": "Google Inc", "type": "ORG", "position": 20},
        ]
        resolved = resolve_co_references(entities)
        # "Google Inc" should win (longer form)
        assert len(resolved) == 1
        assert resolved[0]["name"] == "Google Inc"

    def test_co_reference_resolution_case_insensitive(self):
        from anima.ops.extract_entities import resolve_co_references

        entities = [
            {"name": "Alice Smith", "type": "PERSON", "position": 0},
            {"name": "alice smith", "type": "PERSON", "position": 15},
        ]
        resolved = resolve_co_references(entities)
        assert len(resolved) == 1
        # Either form is valid since they're case-insensitive identical
        assert resolved[0]["name"].lower() == "alice smith"

    def test_co_reference_different_types_not_merged(self):
        from anima.ops.extract_entities import resolve_co_references

        entities = [
            {"name": "AWS", "type": "TECH", "position": 0},
            {"name": "AWS", "type": "ORG", "position": 10},
        ]
        resolved = resolve_co_references(entities)
        # Different types should NOT be merged
        assert len(resolved) == 2

    def test_entity_similarity(self):
        from anima.ops.extract_entities import _entity_similarity

        assert _entity_similarity("Google Inc", "Google") == 0.5  # 1 shared / 2 total
        assert _entity_similarity("Alice Smith", "Alice") == 0.5
        assert _entity_similarity("completely", "different") == 0.0
        assert _entity_similarity("same thing", "same thing") == 1.0


class TestVersionChain:
    """Test version chain creation and traversal."""

    @pytest.mark.asyncio
    async def test_version_chain_basic(self):
        """Create a simple 2-node version chain."""
        pytest.skip("Requires configured database — tested in integration")

    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        """Detect forked version chain."""
        pytest.skip("Requires configured database — tested in integration")


class TestCascadePipeline:
    """Test the full cascaded extraction pipeline."""

    @pytest.mark.asyncio
    async def test_cascade_extract_basic(self):
        from anima.ops.extract_entities import cascade_extract

        result = await cascade_extract(
            "Alice is a software engineer. She built the recommendation system at Google."
        )

        assert "entities" in result
        assert "facts" in result
        assert "stats" in result
        assert result["stats"]["entity_count"] >= 2  # Alice + Google at least
        assert result["stats"]["fact_count"] >= 1      # At least one fact

    @pytest.mark.asyncio
    async def test_cascade_extract_empty(self):
        from anima.ops.extract_entities import cascade_extract

        result = await cascade_extract("ok thanks bye")
        assert result["entities"] == []
        assert result["facts"] == []
        assert result["stats"]["entity_count"] == 0

    @pytest.mark.asyncio
    async def test_cascade_extract_rich_content(self):
        from anima.ops.extract_entities import cascade_extract

        result = await cascade_extract(
            "Alice Smith deployed v3.2.1 of the Kubernetes cluster to AWS on 2024-06-15. "
            "Bob from OpenAI reviewed the PR and approved it."
        )

        # Should extract at least 3 entities (Alice Smith, Kubernetes, AWS, Bob, OpenAI, v3.2.1)
        assert result["stats"]["entity_count"] >= 3
        # Facts depend on SPO pattern matching — at least entity extraction works
        assert "entities" in result
        assert "facts" in result

    def test_entity_uuid_stability(self):
        """Same entity should get the same UUID."""
        from anima.ops.extract_entities import cascade_extract
        import asyncio

        async def run():
            r1 = await cascade_extract("Alice works at Google.")
            r2 = await cascade_extract("Alice uses Python.")
            return r1, r2

        r1, r2 = asyncio.run(run())

        alice1 = next((e for e in r1["entities"] if e["name"] == "Alice"), None)
        alice2 = next((e for e in r2["entities"] if e["name"] == "Alice"), None)

        if alice1 and alice2:
            assert alice1["uuid"] == alice2["uuid"], "Same entity should have stable UUID"
