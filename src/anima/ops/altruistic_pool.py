"""
P2.2: Cross-User Altruistic Knowledge Pool — shared knowledge across users.

Builds on the nature spectrum (egoistic/altruistic/hybrid) and decay modifiers:
- egoistic: 1.5x decay (fast forgetting, user-private)
- altruistic: 0.3x decay (slow forgetting, shared across users)
- hybrid: 1.0x decay (normal)

Cross-user search with tian_ren_ratio [0,1]:
  0 = pure egoistic (only own memories)
  0.5 = balanced
  1 = pure altruistic (all altruistic memories from all users)

Implements:
- search_altruistic: cross-user search of altruistic memory pool
- get_knowledge_pool: aggregated altruistic knowledge with tian_ren_ratio ranking
- get_altruistic_stats: statistics about the shared knowledge pool
"""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional

from ..core.db import db, q
from ..memory.hsg import hsg_query

logger = logging.getLogger(__name__)


# ── Cross-User Search ────────────────────────────────────────────────

async def search_altruistic(
    query: str,
    current_user_id: Optional[str] = None,
    limit: int = 10,
    tian_ren_ratio: float = 0.5,
    min_salience: float = 0.1,
    include_own: bool = True,
) -> List[Dict[str, Any]]:
    """Cross-user search of the altruistic knowledge pool.

    Args:
        query: Search query
        current_user_id: Current user (for excluding/weighting own memories)
        limit: Max results
        tian_ren_ratio: 0=纯利己, 0.5=平衡, 1=纯利他
        min_salience: Minimum salience threshold
        include_own: Whether to include current user's own altruistic memories

    Returns ranked results with cross-user attribution.
    """
    # Build filter based on tian_ren_ratio
    filters = {}

    if tian_ren_ratio == 0:
        # Pure egoistic — search only current user's memories, all natures
        if current_user_id:
            filters["user_id"] = current_user_id
    elif tian_ren_ratio < 0.5:
        # Bias toward egoistic — include current user's own + altruistic
        if current_user_id:
            filters["user_id"] = current_user_id
        filters["nature"] = None  # Don't filter by nature
    elif tian_ren_ratio < 1.0:
        # Bias toward altruistic — search altruistic pool across all users
        filters["nature"] = "altruistic"
        if not include_own and current_user_id:
            filters["exclude_user_id"] = current_user_id
    else:
        # Pure altruistic — all altruistic memories from all users
        filters["nature"] = "altruistic"

    # Use existing hsg_query with appropriate filters
    # Note: hsg_query doesn't natively support cross-user or nature filter
    # We intercept with a post-filter approach
    
    # First, get results from hsg_query
    raw_results = await hsg_query(query, limit * 2, filters)
    
    # Post-process: filter by user_id for ratio=0 (own only)
    if tian_ren_ratio == 0 and current_user_id:
        raw_results = [
            r for r in raw_results
            if r.get("user_id") == current_user_id
        ]
    elif not include_own and current_user_id:
        raw_results = [
            r for r in raw_results
            if r.get("user_id") != current_user_id
        ]
    
    # Apply tian_ren_ratio scoring
    scored = []
    for r in raw_results:
        nature = r.get("nature", "hybrid")
        user_id = r.get("user_id", "anonymous")
        
        # Base score from hsg
        base_score = r.get("score", 0.5)
        
        # Tian-ren ratio modifier: altruistic memories get boosted at high ratio
        if nature == "altruistic":
            nature_boost = 1.0 + tian_ren_ratio  # 1.0 → 2.0 boost at ratio=1
        elif nature == "egoistic":
            nature_boost = 1.0 - tian_ren_ratio  # 0.0 → 1.0 at ratio=0
        else:
            nature_boost = 1.0
        
        # Cross-user bonus: memories from other users are more interesting at high ratio
        cross_user_bonus = 1.0
        if current_user_id and user_id != current_user_id:
            cross_user_bonus = 1.0 + tian_ren_ratio * 1.5  # Up to 2.5x at ratio=1
        
        adjusted_score = base_score * nature_boost * cross_user_bonus
        
        scored.append({
            **r,
            "nature_boost": round(nature_boost, 2),
            "cross_user_bonus": round(cross_user_bonus, 2),
            "adjusted_score": round(adjusted_score, 4),
            "from_other_user": (current_user_id is not None and user_id != current_user_id),
        })
    
    # Sort by adjusted score
    scored.sort(key=lambda x: x["adjusted_score"], reverse=True)
    
    return scored[:limit]


# ── Knowledge Pool ───────────────────────────────────────────────────

async def get_knowledge_pool(
    user_id: Optional[str] = None,
    tian_ren_ratio: float = 0.5,
    limit: int = 50,
) -> Dict[str, Any]:
    """Get the aggregated altruistic knowledge pool.

    Returns:
    - shared_knowledge: top altruistic memories from all users
    - pool_stats: statistics about the shared pool
    - top_contributors: users contributing most altruistic memories
    """
    # Count altruistic memories
    altruistic_count = db.fetchone(
        "SELECT COUNT(*) as cnt FROM memories WHERE nature='altruistic'"
    )
    
    # Get top altruistic memories across all users
    if tian_ren_ratio > 0:
        rows = db.fetchall(
            """SELECT id, content, user_id, nature, salience, primary_sector, updated_at
               FROM memories
               WHERE nature='altruistic' AND salience >= 0.1
               ORDER BY salience DESC
               LIMIT ?""",
            (limit,)
        )
        
        shared_knowledge = [
            {
                "id": r["id"],
                "content": r["content"][:200],
                "user_id": r["user_id"],
                "nature": r["nature"],
                "salience": r["salience"],
                "primary_sector": r["primary_sector"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
    else:
        shared_knowledge = []
    
    # Get top contributors
    contributors = db.fetchall(
        """SELECT user_id, COUNT(*) as contribution_count, AVG(salience) as avg_salience
           FROM memories
           WHERE nature='altruistic'
           GROUP BY user_id
           ORDER BY contribution_count DESC
           LIMIT 10"""
    )
    
    # Pool statistics
    total_memories = db.fetchone("SELECT COUNT(*) as cnt FROM memories")["cnt"] or 0
    hybrid_count = db.fetchone("SELECT COUNT(*) as cnt FROM memories WHERE nature='hybrid'")["cnt"] or 0
    egoistic_count = db.fetchone("SELECT COUNT(*) as cnt FROM memories WHERE nature='egoistic'")["cnt"] or 0
    alt_count = altruistic_count["cnt"] or 0 if altruistic_count else 0
    
    return {
        "shared_knowledge": shared_knowledge,
        "pool_stats": {
            "total_memories": total_memories,
            "altruistic_count": alt_count,
            "hybrid_count": hybrid_count,
            "egoistic_count": egoistic_count,
            "altruistic_percentage": round(alt_count / max(total_memories, 1) * 100, 1),
            "tian_ren_ratio": tian_ren_ratio,
        },
        "top_contributors": [
            {
                "user_id": c["user_id"],
                "contributions": c["contribution_count"],
                "avg_salience": round(c["avg_salience"], 3) if c["avg_salience"] else 0,
            }
            for c in contributors
        ],
    }


async def get_altruistic_stats() -> Dict[str, Any]:
    """Get statistics about the altruistic knowledge pool."""
    pool = await get_knowledge_pool(tian_ren_ratio=1.0)
    return pool["pool_stats"]


async def promote_to_altruistic(memory_id: str) -> bool:
    """Promote a memory to altruistic nature (make it shareable).
    
    Only hybrid memories can be promoted (egoistic stays private).
    """
    mem = q.get_mem(memory_id)
    if not mem:
        return False
    
    nature = mem["nature"] if "nature" in mem.keys() else "hybrid"
    if nature == "egoistic":
        logger.warning("Cannot promote egoistic memory %s to altruistic", memory_id[:16])
        return False
    
    now = int(time.time() * 1000)
    db.execute(
        "UPDATE memories SET nature='altruistic', updated_at=? WHERE id=?",
        (now, memory_id)
    )
    db.commit()
    
    logger.info("Promoted memory %s to altruistic pool", memory_id[:16])
    return True


async def demote_to_private(memory_id: str) -> bool:
    """Demote a memory from altruistic to hybrid (remove from shared pool)."""
    mem = q.get_mem(memory_id)
    if not mem:
        return False
    
    if "nature" not in mem.keys() or mem["nature"] != "altruistic":
        return False
    
    now = int(time.time() * 1000)
    db.execute(
        "UPDATE memories SET nature='hybrid', updated_at=? WHERE id=?",
        (now, memory_id)
    )
    db.commit()
    return True
