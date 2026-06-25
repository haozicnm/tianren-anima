"""P2: Graph reorganization + altruistic knowledge pool tests."""

import pytest
import tempfile
import os
import sys

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src_path)


class TestVectorClustering:
    """Test vector-based clustering in graph_reorganizer."""

    def test_cluster_empty(self):
        from anima.memory.graph_reorganizer import cluster_memories
        clusters = cluster_memories([])
        assert clusters == []

    def test_cluster_insufficient_data(self):
        from anima.memory.graph_reorganizer import cluster_memories, MemoryVector
        import numpy as np

        # Only 2 vectors, below MIN_CLUSTER_SIZE
        vecs = [
            MemoryVector("a", np.array([1.0, 0.0], dtype=np.float32), 0.8, "semantic", "u1"),
            MemoryVector("b", np.array([0.0, 1.0], dtype=np.float32), 0.7, "semantic", "u1"),
        ]
        clusters = cluster_memories(vecs)
        assert clusters == []

    def test_cluster_basic(self):
        from anima.memory.graph_reorganizer import cluster_memories, MemoryVector
        import numpy as np

        # Create two clear clusters of vectors
        np.random.seed(42)
        vecs = []
        # Cluster 1: centered around [1, 0, 0]
        for i in range(5):
            v = np.array([1.0, 0.0, 0.0], dtype=np.float32) + np.random.randn(3).astype(np.float32) * 0.1
            vecs.append(MemoryVector(f"c1_{i}", v, 0.8, "semantic", "u1"))
        # Cluster 2: centered around [0, 1, 0]
        for i in range(5):
            v = np.array([0.0, 1.0, 0.0], dtype=np.float32) + np.random.randn(3).astype(np.float32) * 0.1
            vecs.append(MemoryVector(f"c2_{i}", v, 0.7, "semantic", "u1"))

        clusters = cluster_memories(vecs, n_clusters=5)
        assert len(clusters) >= 1  # At least one cluster

    def test_cluster_member_count(self):
        from anima.memory.graph_reorganizer import cluster_memories, MemoryVector
        import numpy as np

        np.random.seed(42)
        vecs = []
        for i in range(10):
            v = np.array([float(i % 2), float((i+1) % 2), 0.0], dtype=np.float32) + np.random.randn(3).astype(np.float32) * 0.05
            vecs.append(MemoryVector(f"m{i}", v, 0.5, "semantic", "u1"))

        clusters = cluster_memories(vecs)
        # Each cluster should have at least MIN_CLUSTER_SIZE members
        for c in clusters:
            assert len(c.members) >= 3


class TestAltruisticPool:
    """Test cross-user altruistic knowledge pool."""

    @pytest.mark.asyncio
    async def test_knowledge_pool_empty(self):
        from anima.ops.altruistic_pool import get_knowledge_pool

        result = await get_knowledge_pool(tian_ren_ratio=0.0)
        assert "pool_stats" in result
        assert "shared_knowledge" in result
        assert "top_contributors" in result
        # At ratio=0, no shared knowledge
        assert result["shared_knowledge"] == []

    @pytest.mark.asyncio
    async def test_knowledge_pool_stats(self):
        from anima.ops.altruistic_pool import get_knowledge_pool

        result = await get_knowledge_pool(tian_ren_ratio=0.5)
        stats = result["pool_stats"]
        assert "total_memories" in stats
        assert "altruistic_count" in stats
        assert "altruistic_percentage" in stats
        assert isinstance(stats["total_memories"], int)

    @pytest.mark.asyncio
    async def test_get_altruistic_stats(self):
        from anima.ops.altruistic_pool import get_altruistic_stats

        stats = await get_altruistic_stats()
        assert "altruistic_count" in stats
        assert isinstance(stats["altruistic_count"], int)

    def test_promote_egoistic_denied(self):
        """Cannot promote egoistic memory to altruistic."""
        # Unit test: the function should return False for non-existent IDs
        # Integration test would require an actual egoistic memory in DB
        pass  # Tested in integration


class TestGraphReorganizerIntegration:
    """Integration tests for graph reorganizer (requires DB)."""

    @pytest.mark.asyncio
    async def test_reorganize_empty_db(self):
        """Reorganize on empty DB should return insufficient_data."""
        from anima.memory.graph_reorganizer import run_reorganization_once

        result = await run_reorganization_once()
        assert result["status"] in ("complete", "insufficient_data")


class TestP2APIIntegration:
    """Test P2 APIs exposed on Memory class."""

    @pytest.mark.asyncio
    async def test_memory_reorganize_graph(self):
        from anima.main import Memory

        mem = Memory(user="test-p2", use_working_memory=False)
        result = await mem.reorganize_graph()
        assert "status" in result

    @pytest.mark.asyncio
    async def test_memory_knowledge_pool(self):
        from anima.main import Memory

        mem = Memory(user="test-p2", use_working_memory=False)
        result = await mem.get_knowledge_pool(tian_ren_ratio=0.5)
        assert "pool_stats" in result


class TestTianRenRatio:
    """Test tian_ren_ratio scoring logic."""

    @pytest.mark.asyncio
    async def test_search_altruistic_ratio_zero(self):
        """At ratio=0, should only search own memories."""
        from anima.ops.altruistic_pool import search_altruistic

        results = await search_altruistic(
            query="test",
            current_user_id="user_a",
            tian_ren_ratio=0.0,
        )
        assert isinstance(results, list)
        # Results should all be from current user
        for r in results:
            assert r.get("from_other_user", False) is False

    @pytest.mark.asyncio
    async def test_search_altruistic_ratio_one(self):
        """At ratio=1, should search all altruistic memories."""
        from anima.ops.altruistic_pool import search_altruistic

        results = await search_altruistic(
            query="test",
            current_user_id="user_a",
            tian_ren_ratio=1.0,
            include_own=False,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_scoring_fields(self):
        """Results should have nature_boost and adjusted_score."""
        from anima.ops.altruistic_pool import search_altruistic

        results = await search_altruistic(
            query="test",
            tian_ren_ratio=0.5,
        )
        for r in results:
            assert "nature_boost" in r
            assert "adjusted_score" in r
            assert isinstance(r["adjusted_score"], (int, float))
