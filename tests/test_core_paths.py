"""
Core path tests for OpenMemory: add, search, delete_all, classify, decay.
Covers regressions #180 (NULL user_id in MCP query) and #186 (delete_all + search).

Run:
    OM_DB_URL=sqlite:///:memory: OM_EMBED_KIND=synthetic python -m pytest tests/test_core_paths.py -v
"""

import os
import sys
import math
import asyncio
import pytest
from pathlib import Path

# ── env setup (must happen before openmemory imports) ──────────────────
os.environ.setdefault("OM_EMBED_KIND", "synthetic")
os.environ.setdefault("OM_VEC_DIM", "1024")

# ── pure-function imports (no DB side-effects) ────────────────────────
from anima.memory.hsg import classify_content, calc_decay, compute_simhash
from anima.core.constants import SECTOR_CONFIGS


# ═══════════════════════════════════════════════════════════════════════
# 1. classify_content() — sector detection
# ═══════════════════════════════════════════════════════════════════════

CLASSIFY_CASES = [
    # (content, metadata, expected_primary, reason)
    (
        "I feel really happy today and excited about the weekend!",
        None,
        "emotional",
        "emotion words trigger emotional sector",
    ),
    (
        "The capital of France is Paris. The Eiffel Tower was built in 1889.",
        None,
        "semantic",
        "factual/definitional content triggers semantic",
    ),
    (
        "To deploy the app, first run npm install, then npm run build, and finally push to production.",
        None,
        "procedural",
        "step-by-step instructions trigger procedural",
    ),
    (
        "Yesterday I went to the park and saw a beautiful sunset. It reminded me of my childhood.",
        None,
        "episodic",
        "temporal markers + personal recollection trigger episodic",
    ),
    (
        "I've been thinking about my career and I realize I need to improve my skills and grow as a person.",
        None,
        "reflective",
        "introspection/insight triggers reflective",
    ),
    (
        "Just regular text with no strong sector signals at all.",
        None,
        "semantic",
        "default fallback is semantic when no patterns match",
    ),
    (
        "I was so angry when the build failed AGAIN!! Then I realized the fix was simple.",
        None,
        "emotional",
        "emotional words + !! should outweigh single reflective word",
    ),
    # metadata sector override
    (
        "Just some random words here.",
        {"sector": "procedural"},
        "procedural",
        "metadata sector override takes precedence",
    ),
    (
        "Nothing special.",
        {"sector": "episodic"},
        "episodic",
        "explicit metadata sector works",
    ),
]


@pytest.mark.parametrize("content,metadata,expected,reason", CLASSIFY_CASES)
def test_classify_content(content, metadata, expected, reason):
    result = classify_content(content, metadata)
    assert result["primary"] == expected, (
        f"{reason}\n  content: {content!r}\n"
        f"  got primary={result['primary']!r}, expected={expected!r}\n"
        f"  confidence={result['confidence']:.3f}"
    )
    assert 0.0 <= result["confidence"] <= 1.0, "confidence must be in [0, 1]"
    assert isinstance(result["additional"], list), "additional must be a list"
    for sec in result["additional"]:
        assert sec in SECTOR_CONFIGS, f"additional sector {sec!r} not in SECTOR_CONFIGS"


def test_classify_content_returns_all_keys():
    """classify_content must always return primary, additional, confidence."""
    result = classify_content("hello world")
    for key in ("primary", "additional", "confidence"):
        assert key in result, f"missing key {key!r}"


def test_classify_content_metadata_none():
    """Passing None for metadata must not crash."""
    result = classify_content("test content", None)
    assert result["primary"] in SECTOR_CONFIGS


# ═══════════════════════════════════════════════════════════════════════
# 2. calc_decay() — formula correctness
# ═══════════════════════════════════════════════════════════════════════

def test_calc_decay_no_decay_at_day_zero():
    """At 0 days, calc_decay should return the initial salience (plus
    minimal reinforcement)."""
    for sector in SECTOR_CONFIGS:
        result = calc_decay(sector, 0.5, 0.0)
        # At days=0: exp(-lambda * 0) = 1, so decayed = init_sal * 1 = init_sal
        # Plus alpha_reinforce * (1 - exp(-lambda*0)) = alpha_reinforce * 0 = 0
        assert result == pytest.approx(0.5, abs=0.01), (
            f"sector={sector}: expected ~0.5, got {result}"
        )


def test_calc_decay_decays_over_time():
    """Salience should decrease as days pass."""
    for sector in SECTOR_CONFIGS:
        fresh = calc_decay(sector, 1.0, 0.0)
        aged = calc_decay(sector, 1.0, 30.0)
        assert aged < fresh, (
            f"sector={sector}: aged({aged}) should be < fresh({fresh})"
        )


def test_calc_decay_bounded_0_to_1():
    """Output must always be in [0, 1]."""
    for sector in SECTOR_CONFIGS:
        for days in (0, 1, 7, 30, 100, 365):
            for init in (0.0, 0.3, 0.5, 0.8, 1.0):
                result = calc_decay(sector, init, days)
                assert 0.0 <= result <= 1.0, (
                    f"sector={sector} init={init} days={days} → {result}"
                )


def test_calc_decay_sector_differences():
    """Semantic (lambda=0.005) decays slower than emotional (lambda=0.02)."""
    sem = calc_decay("semantic", 1.0, 30.0)
    emo = calc_decay("emotional", 1.0, 30.0)
    assert sem > emo, (
        f"semantic({sem}) should decay slower than emotional({emo})"
    )


def test_calc_decay_reflective_slowest():
    """Reflective (lambda=0.001) should be the slowest decay."""
    ref = calc_decay("reflective", 1.0, 50.0)
    for sector in ("emotional", "procedural", "episodic", "semantic"):
        other = calc_decay(sector, 1.0, 50.0)
        assert ref >= other - 0.001, (
            f"reflective({ref}) should be >= {sector}({other}) at 50 days"
        )


def test_calc_decay_unknown_sector_returns_init():
    """Unknown sector should return initial salience unchanged."""
    result = calc_decay("nonexistent", 0.7, 100.0)
    assert result == 0.7


def test_calc_decay_segment_scaling():
    """When seg_idx and max_seg are provided, decay lambda scales by segment ratio."""
    no_seg = calc_decay("semantic", 1.0, 10.0)
    early_seg = calc_decay("semantic", 1.0, 10.0, seg_idx=0, max_seg=10)
    late_seg = calc_decay("semantic", 1.0, 10.0, seg_idx=9, max_seg=10)

    # Early segment (seg_idx=0): lambda *= (1 - sqrt(0/10)) = 1 → same as no_seg
    assert early_seg == pytest.approx(no_seg, abs=0.01)

    # Late segment (seg_idx=9): lambda *= (1 - sqrt(9/10)) ≈ (1-0.949) = 0.051*lambda
    # So it decays slower, meaning late_seg > no_seg
    assert late_seg > no_seg, (
        f"late segment({late_seg}) should decay slower than no segment({no_seg})"
    )


def test_calc_decay_monotonic():
    """calc_decay should be monotonically decreasing over time."""
    sector = "semantic"
    prev = calc_decay(sector, 1.0, 0.0)
    for days in (1, 3, 7, 14, 30):
        curr = calc_decay(sector, 1.0, days)
        assert curr <= prev, f"not monotonic at days={days}: {curr} > {prev}"
        prev = curr


# ═══════════════════════════════════════════════════════════════════════
# 3.  Integration tests — memory.add / search / delete_all
# ═══════════════════════════════════════════════════════════════════════

# In-memory SQLite DB is set via env var before import.
# Each test class uses its own Memory instance scoped to avoid cross-test
# pollution.  We reset the DB module between test classes to get a fresh
# in-memory database.

def _new_memory():
    """Create a fresh Memory instance with in-memory SQLite.

    Patches the global DB singleton so that each call gets a new connection
    to a new in-memory database.
    """
    os.environ["OM_DB_URL"] = "sqlite:///:memory:"
    os.environ["OM_EMBED_KIND"] = os.environ.get("OM_EMBED_KIND", "synthetic")

    import anima.core.config as configmod
    configmod.env.database_url = "sqlite:///:memory:"

    import anima.core.db as dbmod
    # Close any existing connection and reset
    if dbmod.db.conn is not None:
        try:
            dbmod.db.conn.close()
        except Exception:
            pass
    dbmod.db.conn = None

    # Clear HSG cache
    import anima.memory.hsg as hsgmod
    hsgmod.cache.clear()

    from anima.client import Memory
    return Memory()


class TestMemoryAddSearch:
    """Core path: add → search roundtrip."""

    @pytest.mark.asyncio
    async def test_add_returns_id(self):
        """memory.add() must return a result dict with an 'id' key."""
        mem = _new_memory()
        result = await mem.add(
            "The sky is blue and beautiful today.",
            user_id="test_user_1",
        )
        assert "id" in result or "root_memory_id" in result, (
            f"add() result missing id: {list(result.keys())}"
        )
        mem_id = result.get("id") or result.get("root_memory_id")
        assert mem_id is not None
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    @pytest.mark.asyncio
    async def test_add_with_tags_and_meta(self):
        """memory.add() should accept tags and meta."""
        mem = _new_memory()
        result = await mem.add(
            "Complete the quarterly report by Friday.",
            user_id="test_user_1",
            tags=["work", "urgent"],
            meta={"priority": 10, "project": "Q2"},
        )
        mem_id = result.get("id") or result.get("root_memory_id")
        assert mem_id is not None

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """memory.search() returns a list of result dicts."""
        mem = _new_memory()
        # Add a unique memory to search for
        await mem.add(
            "Purple elephants dance on the moon every Tuesday.",
            user_id="test_user_1",
        )
        results = await mem.search(
            "purple elephants moon",
            user_id="test_user_1",
            limit=5,
        )
        assert isinstance(results, list), f"expected list, got {type(results)}"
        assert len(results) > 0, "search returned empty results"
        # Each result should have expected keys
        for r in results:
            assert "id" in r, f"result missing id: {list(r.keys())}"
            assert "content" in r, f"result missing content"
            assert "score" in r, f"result missing score"

    @pytest.mark.asyncio
    async def test_search_user_isolation(self):
        """Search for user A should not return user B's memories."""
        mem = _new_memory()
        await mem.add("User A's secret recipe.", user_id="user_a")
        await mem.add("User B's public notes.", user_id="user_b")

        results_a = await mem.search("recipe", user_id="user_a", limit=10)
        results_b = await mem.search("recipe", user_id="user_b", limit=10)

        # user_a should find their memory, user_b should not
        a_contents = [r["content"] for r in results_a]
        assert any("secret recipe" in c for c in a_contents), (
            f"user_a should find their recipe: {a_contents}"
        )

        b_contents = [r["content"] for r in results_b]
        assert not any("secret recipe" in c for c in b_contents), (
            f"user_b should NOT find user_a's recipe: {b_contents}"
        )

    @pytest.mark.asyncio
    async def test_search_with_limit(self):
        """memory.search() should respect the limit parameter."""
        mem = _new_memory()
        for i in range(5):
            await mem.add(f"Limit test memory number {i}.", user_id="test_user_limit")

        results = await mem.search("limit test", user_id="test_user_limit", limit=3)
        assert len(results) <= 3, f"limit=3 but got {len(results)} results"


class TestDeleteAll:
    """Regression #186: delete_all must actually remove memories so
    subsequent searches return nothing."""

    @pytest.mark.asyncio
    async def test_delete_all_clears_user_memories(self):
        """Add memories, delete_all, verify search returns empty."""
        mem = _new_memory()
        uid = "delete_test_user"

        # Seed several memories
        for i in range(3):
            await mem.add(f"Delete test memory {i}.", user_id=uid)

        # Verify they exist
        before = await mem.search("Delete test", user_id=uid, limit=10)
        assert len(before) >= 1, f"expected >=1 results before delete, got {len(before)}"

        # Delete all for this user
        await mem.delete_all(user_id=uid)

        # Verify they're gone
        after = await mem.search("Delete test", user_id=uid, limit=10)
        assert len(after) == 0, (
            f"#186 regression: expected 0 results after delete_all, got {len(after)}"
        )

    @pytest.mark.asyncio
    async def test_delete_all_only_affects_target_user(self):
        """delete_all(user_id=X) must not delete user Y's memories."""
        mem = _new_memory()
        uid_a = "delete_user_a"
        uid_b = "delete_user_b"

        await mem.add("Alpaca farms in Peru are thriving.", user_id=uid_a)
        await mem.add("Banana smoothies are delicious and healthy.", user_id=uid_b)

        # Sanity: both can find their own memories before delete
        a_before = await mem.search("Alpaca farms Peru", user_id=uid_a, limit=10)
        b_before = await mem.search("Banana smoothies delicious", user_id=uid_b, limit=10)
        assert len(a_before) >= 1, "user A should find their memory before delete"
        assert len(b_before) >= 1, "user B should find their memory before delete"

        # Delete user A only
        await mem.delete_all(user_id=uid_a)

        # User B should still have their memory
        b_results = await mem.search("Banana smoothies", user_id=uid_b, limit=10)
        assert len(b_results) >= 1, (
            f"#186: user B's memories should survive delete_all(user A)"
        )

        # User A should be empty
        a_results = await mem.search("Alpaca Peru", user_id=uid_a, limit=10)
        assert len(a_results) == 0, (
            f"#186: user A's memories should be gone after delete_all"
        )

    @pytest.mark.asyncio
    async def test_delete_all_history_empty(self):
        """After delete_all, history() should also return empty."""
        mem = _new_memory()
        uid = "history_del_user"
        await mem.add("History test memory.", user_id=uid)
        await mem.delete_all(user_id=uid)

        hist = mem.history(user_id=uid, limit=20)
        assert len(hist) == 0, f"history should be empty after delete_all, got {len(hist)}"


class TestMCPNullUserId:
    """Regression #180: MCP query with NULL user_id must not crash."""

    @pytest.mark.asyncio
    async def test_search_with_none_user_id(self):
        """Calling search with user_id=None must not raise an exception."""
        mem = _new_memory()
        # Add a memory for some user
        await mem.add("Public knowledge.", user_id="public_user")

        # Search with explicit None user_id — this simulates what the
        # MCP tool openmemory_query does when user_id is not provided.
        try:
            results = await mem.search("knowledge", user_id=None, limit=5)
        except Exception as e:
            pytest.fail(f"#180 regression: search(user_id=None) raised {type(e).__name__}: {e}")

        # Should return results (no user filter applied)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_mcp_store_with_null_user_id(self):
        """MCP openmemory_store with user_id=None must work (defaults to 'anonymous')."""
        mem = _new_memory()
        result = await mem.add("MCP null user test.", user_id=None)
        mem_id = result.get("id") or result.get("root_memory_id")
        assert mem_id is not None

    @pytest.mark.asyncio
    async def test_delete_all_with_null_user_id_is_noop(self):
        """delete_all with user_id=None (no default) should be a safe no-op."""
        mem = _new_memory()
        try:
            await mem.delete_all(user_id=None)
        except Exception as e:
            pytest.fail(f"delete_all(user_id=None) raised {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 4.  compute_simhash — deterministic and content-sensitive
# ═══════════════════════════════════════════════════════════════════════

class TestSimhash:
    def test_simhash_deterministic(self):
        """Same input always produces same simhash."""
        h1 = compute_simhash("hello world")
        h2 = compute_simhash("hello world")
        assert h1 == h2

    def test_simhash_different_for_different_inputs(self):
        """Different content should produce different simhashes."""
        h1 = compute_simhash("The quick brown fox.")
        h2 = compute_simhash("A completely different sentence.")
        assert h1 != h2

    def test_simhash_returns_hex_string(self):
        """Simhash must be a hex string."""
        h = compute_simhash("test")
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_add_empty_string(self):
        """Adding an empty string should not crash."""
        mem = _new_memory()
        result = await mem.add("", user_id="edge_user")
        assert result is not None

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        """Searching with an empty query should not crash."""
        mem = _new_memory()
        await mem.add("Some content.", user_id="edge_user")
        results = await mem.search("", user_id="edge_user", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_nonexistent_user(self):
        """Searching for a user who has no memories returns empty list."""
        mem = _new_memory()
        results = await mem.search("anything", user_id="nonexistent_user_42", limit=5)
        assert results == []

    def test_classify_content_empty_string(self):
        """Classify empty string should return semantic default."""
        result = classify_content("")
        assert result["primary"] in SECTOR_CONFIGS

    def test_calc_decay_zero_salience(self):
        """Zero salience gets reinforced slightly but stays small."""
        result = calc_decay("semantic", 0.0, 30.0)
        # calc_decay adds alpha_reinforce * (1 - exp(-lambda*days)) as a floor
        # so zero salience rebounds very slightly; expectation: < 0.05
        assert 0.0 <= result < 0.05, f"expected near-zero, got {result}"
