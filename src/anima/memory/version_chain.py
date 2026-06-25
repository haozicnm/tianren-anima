"""
P1.1: Version Chain — supermemory-inspired memory versioning with conflict detection.

Each memory gets a parent_id and root_id, forming a version chain:
  root_id → v1 (parent_id=NULL) → v2 (parent_id=v1.id) → v3 (parent_id=v2.id)

Conflict detection: when two memories share the same root_id but have different
parent_id chains (forked), they are flagged as conflicting versions.
"""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..core.db import db, q

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────

@dataclass
class VersionNode:
    """A single node in the version chain."""
    id: str
    content: str
    parent_id: Optional[str]
    root_id: Optional[str]
    version: int
    salience: float
    updated_at: int
    nature: Optional[str] = None
    primary_sector: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> "VersionNode":
        return cls(
            id=row["id"],
            content=(row["content"] if "content" in row.keys() else "")[:100],
            parent_id=row["parent_id"] if "parent_id" in row.keys() else None,
            root_id=row["root_id"] if "root_id" in row.keys() else None,
            version=row["version"] if "version" in row.keys() else 1,
            salience=row["salience"] if "salience" in row.keys() else 0.5,
            updated_at=row["updated_at"] if "updated_at" in row.keys() else 0,
            nature=row["nature"] if "nature" in row.keys() else None,
            primary_sector=row["primary_sector"] if "primary_sector" in row.keys() else None,
        )


@dataclass
class VersionChain:
    """The full version chain for a root memory."""
    root_id: str
    nodes: List[VersionNode] = field(default_factory=list)
    has_conflicts: bool = False
    conflict_nodes: List[str] = field(default_factory=list)  # IDs of conflicting nodes

    @property
    def latest(self) -> Optional[VersionNode]:
        return self.nodes[-1] if self.nodes else None

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def is_forked(self) -> bool:
        """True if the chain has forked (multiple children at any node)."""
        children_count: Dict[str, int] = {}
        for n in self.nodes:
            if n.parent_id:
                children_count[n.parent_id] = children_count.get(n.parent_id, 0) + 1
        return any(c > 1 for c in children_count.values())


# ── Core API ─────────────────────────────────────────────────────────

async def create_version(
    content: str,
    parent_id: str,
    user_id: Optional[str] = None,
    nature: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new version of an existing memory.

    Generates a new memory with parent_id pointing to the previous version.
    root_id is inherited from the parent's root_id (or parent's id if parent has no root).
    """
    parent_mem = q.get_mem(parent_id)
    if not parent_mem:
        raise ValueError(f"Parent memory {parent_id} not found")

    import uuid
    from .hsg import add_hsg_memory, compute_simhash

    new_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    root_id = parent_mem["root_id"] or parent_id
    new_version = (parent_mem["version"] if "version" in parent_mem.keys() else 1) + 1

    # Create the new version as a regular memory with version chain metadata
    simhash = compute_simhash(content)
    from ..core.config import env
    from ..utils.chunking import chunk_text
    from .hsg import classify_content, extract_essence, calc_mean_vec, create_single_waypoint, embed_multi_sector
    from .embed import embed_multi_sector as embed_ms
    from ..core.constants import SECTOR_CONFIGS
    from ..core.vector_store import vector_store as store
    from ..utils.vectors import vec_to_buf

    cls = classify_content(content)
    stored = extract_essence(content, cls["primary"], env.summary_max_length)
    sec_cfg = SECTOR_CONFIGS[cls["primary"]]
    init_sal = max(0.0, min(1.0, (parent_mem["salience"] or 0.5) + 0.05))

    # Keep parent's nature if not overridden
    effective_nature = nature or (parent_mem["nature"] if "nature" in parent_mem.keys() else None)

    q.ins_mem(
        id=new_id,
        user_id=user_id or (parent_mem["user_id"] if "user_id" in parent_mem.keys() else "anonymous"),
        nature=effective_nature,
        segment=parent_mem["segment"] if "segment" in parent_mem.keys() else 0,
        content=stored,
        simhash=simhash,
        primary_sector=cls["primary"],
        tags=parent_mem["tags"] if "tags" in parent_mem.keys() else None,
        meta=parent_mem["meta"] if "meta" in parent_mem.keys() else None,
        created_at=now,
        updated_at=now,
        last_seen_at=now,
        salience=init_sal,
        decay_lambda=sec_cfg["decay_lambda"],
        version=new_version,
        mean_dim=None,
        mean_vec=None,
        compressed_vec=None,
        feedback_score=0,
    )

    # Set version chain fields after insert (avoid schema issues with ON CONFLICT)
    db.execute(
        "UPDATE memories SET parent_id=?, root_id=? WHERE id=?",
        (parent_id, root_id, new_id)
    )
    db.commit()

    # Create waypoint from parent to new version
    from .hsg import create_contextual_waypoints
    await create_contextual_waypoints(new_id, [parent_id], base_wt=0.8, user_id=user_id)

    return {
        "id": new_id,
        "parent_id": parent_id,
        "root_id": root_id,
        "version": new_version,
        "content": stored,
    }


async def get_version_chain(memory_id: str) -> VersionChain:
    """Retrieve the full version chain for a memory.

    Follows root_id → all descendants. Detects forks and conflicts.
    """
    # Find the root
    mem = q.get_mem(memory_id)
    if not mem:
        return VersionChain(root_id=memory_id)

    root_id = (mem["root_id"] if "root_id" in mem.keys() else None) or memory_id

    # Get all memories in the chain
    all_rows = db.fetchall(
        """SELECT id, content, parent_id, root_id, version, salience, updated_at, nature, primary_sector
           FROM memories WHERE root_id=? OR id=?
           ORDER BY version ASC, updated_at ASC""",
        (root_id, root_id)
    )

    nodes = [VersionNode.from_row(r) for r in all_rows]
    chain = VersionChain(root_id=root_id, nodes=nodes)

    # Detect conflicts: nodes that share the same parent_id (fork)
    parent_children: Dict[str, List[str]] = {}
    for n in nodes:
        pid = n.parent_id or "ROOT"
        if pid not in parent_children:
            parent_children[pid] = []
        parent_children[pid].append(n.id)

    for pid, children in parent_children.items():
        if len(children) > 1:
            # Keep the latest version (highest salience) as canonical, mark others as conflicts
            sorted_children = sorted(
                children,
                key=lambda cid: next((n.salience for n in nodes if n.id == cid), 0),
                reverse=True,
            )
            chain.conflict_nodes.extend(sorted_children[1:])
            chain.has_conflicts = True

    return chain


async def detect_conflicts() -> List[Dict[str, Any]]:
    """Scan all memories for version chain conflicts (forks).

    Returns list of conflict groups with their conflicting nodes.
    """
    # Find root_ids that have multiple children from the same parent
    rows = db.fetchall("""
        SELECT root_id, parent_id, COUNT(*) as child_count
        FROM memories
        WHERE parent_id IS NOT NULL AND root_id IS NOT NULL
        GROUP BY root_id, parent_id
        HAVING child_count > 1
    """)

    conflicts = []
    for r in rows:
        children = db.fetchall(
            "SELECT id, content, parent_id, version, salience FROM memories "
            "WHERE root_id=? AND parent_id=? ORDER BY salience DESC",
            (r["root_id"], r["parent_id"])
        )
        conflict_group = {
            "root_id": r["root_id"],
            "parent_id": r["parent_id"],
            "fork_count": r["child_count"],
            "nodes": [dict(c) for c in children],
        }
        conflicts.append(conflict_group)

    return conflicts


async def merge_versions(
    root_id: str,
    primary_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge conflicting versions by marking the primary as canonical.

    Non-primary versions are kept but marked with reduced salience.
    Returns the canonical version and merged stats.
    """
    chain = await get_version_chain(root_id)
    if not chain.has_conflicts:
        return {"status": "no_conflicts", "root_id": root_id}

    # If no primary specified, use the highest salience version
    if not primary_version_id:
        primary_version_id = max(
            [n.id for n in chain.nodes if n.id not in chain.conflict_nodes],
            key=lambda cid: next((n.salience for n in chain.nodes if n.id == cid), 0),
        )

    now = int(time.time() * 1000)
    merged_count = 0

    # Downgrade conflicting versions
    for conflict_id in chain.conflict_nodes:
        db.execute(
            "UPDATE memories SET salience=salience*0.3, updated_at=? WHERE id=?",
            (now, conflict_id)
        )
        merged_count += 1

    # Boost the canonical version
    db.execute(
        "UPDATE memories SET salience=MIN(1.0, salience+0.1), updated_at=? WHERE id=?",
        (now, primary_version_id)
    )
    db.commit()

    return {
        "status": "merged",
        "root_id": root_id,
        "canonical_id": primary_version_id,
        "merged_count": merged_count,
    }


async def realias_parent(memory_id: str, new_parent_id: str) -> bool:
    """Change a memory's parent in the version chain (re-parenting).
    
    Useful for correcting incorrectly linked versions.
    """
    mem = q.get_mem(memory_id)
    parent = q.get_mem(new_parent_id)
    if not mem or not parent:
        return False

    # Ensure they share the same root
    mem_root = (mem["root_id"] if "root_id" in mem.keys() else None) or memory_id
    parent_root = (parent["root_id"] if "root_id" in parent.keys() else None) or new_parent_id
    if mem_root != parent_root:
        logger.warning("realias_parent: root mismatch (%s vs %s)", mem_root, parent_root)
        return False

    db.execute(
        "UPDATE memories SET parent_id=?, updated_at=? WHERE id=?",
        (new_parent_id, int(time.time() * 1000), memory_id)
    )
    db.commit()
    return True
