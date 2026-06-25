"""
Anima WorkingMemory Bridge (P0)

MemOS-inspired FIFO pipeline: trace → consolidation → permanent.
WorkingMemory acts as a transient buffer (cap 20) that auto-promotes
memories to LongTermMemory when full or on explicit flush.

FIFO semantics:
  - New memories always land in WorkingMemory first
  - When buffer is full, the oldest N items are evaluated:
    * Items with high salience (>0.5) are promoted to LTM with decay boost
    * Items with low salience are either discarded or saved with decay penalty
  - On explicit flush(), all items are promoted
  - Search queries BOTH WorkingMemory (fresh) AND LongTermMemory (permanent)
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────

WORKING_MEMORY_CAP = 20          # Max items in working memory before promotion
LONG_TERM_MEMORY_CAP = 1500      # Max items in long-term memory
PROMOTION_BATCH_SIZE = 10        # How many items to promote when buffer full
SALIENCE_PROMOTION_THRESHOLD = 0.5  # Min salience to promote to LTM
PROMOTION_DECAY_BOOST = 0.05     # Decay boost for promoted items
DISCARD_DECAY_PENALTY = 0.03     # Decay penalty for items below threshold


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class WorkingMemoryItem:
    """Transient memory item before permanent storage."""
    id: str
    content: str
    user_id: str
    nature: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    salience: float = 0.5          # Initial salience (will be recalculated on promotion)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0           # How many times this was returned in search results
    last_accessed_at: float = 0.0
    primary_sector: str = "semantic"
    source_session: Optional[str] = None


# ── WorkingMemory ────────────────────────────────────────────────────────

class WorkingMemory:
    """
    MemOS-inspired trace buffer — a FIFO pipeline between agent conversation
    and permanent memory storage.

    Usage:
        wm = WorkingMemory(permanent_store=memory.add_to_ltm, user_id="abc")

        # Add memories (auto-promotes when full)
        wm.add(WorkingMemoryItem(...))

        # Promote all remaining items
        wm.flush()

        # Search both working + permanent memory
        working_results = wm.search(query)
    """

    def __init__(
        self,
        permanent_store=None,   # async callable(content, user_id, nature, **kwargs) -> id
        user_id: str = "default",
        capacity: int = WORKING_MEMORY_CAP,
        promotion_batch: int = PROMOTION_BATCH_SIZE,
    ):
        self._buffer: List[WorkingMemoryItem] = []
        self._promote_fn = permanent_store
        self.user_id = user_id
        self.capacity = capacity
        self.promotion_batch = promotion_batch
        self._promotion_count: int = 0
        self._discard_count: int = 0
        self._item_counter: int = 0

    # ── Public API ─────────────────────────────────────────────────

    async def add(self, item: WorkingMemoryItem) -> str:
        """Add item to working memory. Auto-promotes if buffer full."""
        if not item.id:
            self._item_counter += 1
            item.id = f"wm_{self.user_id}_{int(time.time()*1000)}_{self._item_counter}"

        self._buffer.append(item)
        logger.debug("WorkingMemory: added %s (%d/%d)", item.id[:16], len(self._buffer), self.capacity)

        promoted_ids = []
        if len(self._buffer) >= self.capacity:
            promoted_ids = await self._promote_oldest(self.promotion_batch)

        return item.id

    async def flush(self, force: bool = False) -> List[str]:
        """Promote all working memory items to permanent storage."""
        if not self._buffer:
            return []
        return await self._promote_oldest(len(self._buffer), force=force)

    def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        nature: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search working memory for matches.
        Uses simple keyword overlap + recency scoring.
        """
        import re
        from collections import Counter

        results = []
        query_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        if not query_tokens:
            return results

        uid = user_id or self.user_id
        now = time.time()

        for item in self._buffer:
            if uid and item.user_id != uid:
                continue
            if nature and item.nature != nature:
                continue

            content_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", item.content.lower()))
            overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)

            # Recency boost: newer items score higher
            age_hours = (now - item.created_at) / 3600.0
            recency = max(0.1, 1.0 - age_hours / 24.0)  # Decay over 24 hours

            score = overlap * 0.7 + recency * 0.3

            if overlap > 0:
                item.access_count += 1
                item.last_accessed_at = now
                results.append({
                    "id": f"wm:{item.id}",
                    "content": item.content,
                    "score": score,
                    "nature": item.nature,
                    "source": "working_memory",
                    "access_count": item.access_count,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self._buffer),
            "capacity": self.capacity,
            "promoted_total": self._promotion_count,
            "discarded_total": self._discard_count,
            "items": [
                {
                    "id": item.id[:16],
                    "nature": item.nature,
                    "salience": item.salience,
                    "access_count": item.access_count,
                    "age_seconds": time.time() - item.created_at,
                }
                for item in self._buffer[-10:]  # Show last 10
            ],
        }

    # ── Internal ───────────────────────────────────────────────────

    async def _promote_oldest(self, count: int, force: bool = False) -> List[str]:
        """Promote the oldest N items from the buffer to permanent storage."""
        if not self._promote_fn:
            logger.warning("WorkingMemory: no promote function set, dropping %d items", count)
            self._discard_count += count
            self._buffer = self._buffer[count:]
            return []

        batch = self._buffer[:count]
        self._buffer = self._buffer[count:]

        promoted = []
        for item in batch:
            try:
                if force or item.salience >= SALIENCE_PROMOTION_THRESHOLD:
                    # Promote to LTM with decay boost
                    boost = PROMOTION_DECAY_BOOST if item.access_count > 0 else 0
                    await self._promote_fn(
                        content=item.content,
                        user_id=item.user_id,
                        nature=item.nature,
                        tags=item.tags,
                        meta={
                            **(item.meta or {}),
                            "wm_source": "working_memory_bridge",
                            "wm_promoted_at": time.time(),
                            "wm_access_count": item.access_count,
                            "wm_salience": item.salience,
                            "wm_boost": boost,
                            "wm_source_session": item.source_session,
                        },
                    )
                    self._promotion_count += 1
                    promoted.append(item.id)
                    logger.debug("WorkingMemory: promoted %s (salience=%.2f, access=%d)", item.id[:16], item.salience, item.access_count)
                else:
                    # Low salience, discard
                    self._discard_count += 1
                    logger.debug("WorkingMemory: discarded %s (salience=%.2f < %.2f)", item.id[:16], item.salience, SALIENCE_PROMOTION_THRESHOLD)
            except Exception as e:
                logger.error("WorkingMemory: promotion failed for %s: %s", item.id[:16], e)
                # Keep failed items in buffer for retry
                self._buffer.insert(0, item)

        return promoted


# ── Factory ──────────────────────────────────────────────────────────────

def create_working_memory_bridge(
    memory_add_fn,   # async (content, user_id, nature, **kwargs) -> id
    user_id: str = "default",
    capacity: int = WORKING_MEMORY_CAP,
) -> WorkingMemory:
    """Create a WorkingMemory instance wired to the permanent store."""
    return WorkingMemory(
        permanent_store=memory_add_fn,
        user_id=user_id,
        capacity=capacity,
    )
