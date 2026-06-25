"""
Test Anima WorkingMemory bridge — P0: FIFO trace → consolidation pipeline.
"""
import asyncio
import time
import pytest
from openmemory import Memory
from anima.memory.working import WorkingMemoryItem


@pytest.mark.asyncio
async def test_working_memory_basic():
    """Working memory buffers items before promotion."""
    mem = Memory(user="wm_test", use_working_memory=True)
    await mem.delete_all(user_id="wm_test")

    wm = mem.working_memory
    assert wm is not None
    assert wm.size == 0

    for i in range(10):
        res = await mem.add(f"Working memory item {i}", user_id="wm_test")
        assert res["status"] == "buffered"
        assert res["working_memory"] is True

    assert wm.size == 10
    await mem.delete_all(user_id="wm_test")


@pytest.mark.asyncio
async def test_working_memory_auto_promotion():
    """Auto-promotion triggers when buffer reaches capacity."""
    mem = Memory(user="wm_test2", use_working_memory=True)
    await mem.delete_all(user_id="wm_test2")
    wm = mem.working_memory

    for i in range(19):
        await mem.add(f"Auto-promo item {i}", user_id="wm_test2")
    assert wm.size == 19

    await mem.add("Auto-promo item 19 (trigger)", user_id="wm_test2")
    assert wm.size <= wm.capacity
    assert wm.stats["promoted_total"] >= 10

    results = await mem.search("Auto-promo item")
    assert len(results) > 0, "Promoted items should be searchable in LTM"

    await mem.delete_all(user_id="wm_test2")


@pytest.mark.asyncio
async def test_working_memory_flush():
    """Explicit flush promotes all remaining items."""
    mem = Memory(user="wm_test3", use_working_memory=True)
    await mem.delete_all(user_id="wm_test3")
    wm = mem.working_memory

    items = ["Quantum computing basics", "Rust async patterns", "Python decorators",
             "Kubernetes pod lifecycle", "SQL query optimization"]
    for item in items:
        await mem.add(item, user_id="wm_test3")
    assert wm.size == 5

    promoted = await mem.flush()
    assert len(promoted) == 5
    assert wm.size == 0

    results = await mem.search("Quantum")
    assert len(results) >= 1
    results2 = await mem.search("Kubernetes")
    assert len(results2) >= 1

    await mem.delete_all(user_id="wm_test3")


@pytest.mark.asyncio
async def test_working_memory_search():
    """Search returns both working and permanent memory results."""
    mem = Memory(user="wm_test4", use_working_memory=True)
    await mem.delete_all(user_id="wm_test4")

    await mem.add("Python async patterns", user_id="wm_test4", nature="altruistic")
    await mem.add("Rust ownership rules", user_id="wm_test4", nature="altruistic")
    await mem.flush()
    await mem.add("JavaScript event loop", user_id="wm_test4")

    results = await mem.search("async patterns")
    assert len(results) > 0
    results2 = await mem.search("event loop")
    assert len(results2) > 0
    results3 = await mem.search("Python", nature="altruistic")
    assert len(results3) > 0

    await mem.delete_all(user_id="wm_test4")


@pytest.mark.asyncio
async def test_working_memory_salience_filter():
    """Low-salience items are discarded, high-salience promoted."""
    mem = Memory(user="wm_test5", use_working_memory=True)
    await mem.delete_all(user_id="wm_test5")
    wm = mem.working_memory

    for i in range(5):
        item = WorkingMemoryItem(
            id="", content=f"High value insight {i}",
            user_id="wm_test5", nature="altruistic", salience=0.8,
        )
        await wm.add(item)

    for i in range(5):
        item = WorkingMemoryItem(
            id="", content=f"Low value noise {i}",
            user_id="wm_test5", nature="egoistic", salience=0.2,
        )
        await wm.add(item)

    for i in range(10):
        await mem.add(f"Filler item {i}", user_id="wm_test5")

    stats = wm.stats
    assert stats["promoted_total"] > 0

    results = await mem.search("High value insight")
    assert len(results) > 0, "High salience items should be promoted"

    await mem.delete_all(user_id="wm_test5")


if __name__ == "__main__":
    asyncio.run(test_working_memory_basic())
    asyncio.run(test_working_memory_auto_promotion())
    asyncio.run(test_working_memory_flush())
    asyncio.run(test_working_memory_search())
    asyncio.run(test_working_memory_salience_filter())
    print("\n🎉 All WorkingMemory tests passed!")
