"""
P2.1: Graph Reorganizer — MemOS-inspired vector clustering for memory graph reorganization.

Periodically reorganizes the memory graph by:
1. Loading memory vectors and clustering them (centroid-based, no scipy needed)
2. Creating/updating waypoints based on cluster co-membership
3. Pruning weak edges below threshold
4. Boosting high-connectivity cluster centers

Uses mean_vec from existing embeddings — no extra embedding calls needed.
"""

from __future__ import annotations

import asyncio
import time
import math
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

import numpy as np

from ..core.db import db, q
from ..utils.vectors import buf_to_vec, cos_sim

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────

MAX_CLUSTERS = 20          # Max number of clusters
MIN_CLUSTER_SIZE = 3       # Min memories per cluster
MAX_ITERATIONS = 10        # K-means iterations
WAYPOINT_BOOST = 0.1       # Boost for within-cluster edges
PRUNE_THRESHOLD = 0.05     # Prune edges below this weight
SALIENCE_BOOST = 0.05      # Salience boost for cluster centroids


# ── Data Types ───────────────────────────────────────────────────────

@dataclass
class MemoryVector:
    """A memory with its vector embedding."""
    id: str
    vector: np.ndarray
    salience: float
    primary_sector: str
    user_id: str
    nature: Optional[str] = None

@dataclass
class Cluster:
    """A cluster of related memories."""
    id: int
    centroid: np.ndarray
    members: List[str] = field(default_factory=list)
    sectors: Dict[str, int] = field(default_factory=dict)  # sector → count


# ── Vector Loading ───────────────────────────────────────────────────

async def load_memory_vectors(
    user_id: Optional[str] = None,
    min_salience: float = 0.1,
    limit: int = 1000,
) -> List[MemoryVector]:
    """Load memories with valid mean_vec from the database."""
    if user_id:
        rows = db.fetchall(
            """SELECT id, mean_vec, salience, primary_sector, user_id, nature
               FROM memories
               WHERE mean_vec IS NOT NULL AND salience >= ? AND user_id = ?
               ORDER BY salience DESC
               LIMIT ?""",
            (min_salience, user_id, limit)
        )
    else:
        rows = db.fetchall(
            """SELECT id, mean_vec, salience, primary_sector, user_id, nature
               FROM memories
               WHERE mean_vec IS NOT NULL AND salience >= ?
               ORDER BY salience DESC
               LIMIT ?""",
            (min_salience, limit)
        )
    
    logger.debug("load_memory_vectors: user_id=%s, got %d rows", user_id, len(rows))
    vectors = []
    for r in rows:
        try:
            mv = r["mean_vec"]
            vlist = buf_to_vec(mv)
            vec = np.array(vlist, dtype=np.float32)
            if len(vec) > 0:
                vectors.append(MemoryVector(
                    id=r["id"],
                    vector=vec,
                    salience=r["salience"] or 0.5,
                    primary_sector=r["primary_sector"] or "semantic",
                    user_id=r["user_id"] or "anonymous",
                    nature=r["nature"] if "nature" in r.keys() else None,
                ))
        except Exception as e:
            logger.debug("load_memory_vectors: failed for %s: %s", r["id"][:12], e)
            continue
    
    return vectors


# ── Clustering ───────────────────────────────────────────────────────

def cluster_memories(
    vectors: List[MemoryVector],
    n_clusters: Optional[int] = None,
) -> List[Cluster]:
    """Centroid-based clustering of memory vectors.
    
    Uses a simple iterative algorithm:
    1. Pick centroids from top-salience memories
    2. Assign each vector to nearest centroid
    3. Recompute centroids
    4. Repeat until convergence
    """
    if len(vectors) < MIN_CLUSTER_SIZE:
        return []
    
    k = min(n_clusters or MAX_CLUSTERS, len(vectors) // MIN_CLUSTER_SIZE)
    k = max(1, k)
    
    # Step 1: Initialize centroids from highest-salience, diverse memories
    centroids = []
    used = set()
    
    # Pick first centroid: highest salience
    sorted_vecs = sorted(vectors, key=lambda v: v.salience, reverse=True)
    centroids.append(sorted_vecs[0].vector.copy())
    used.add(sorted_vecs[0].id)
    
    # Pick remaining centroids: farthest from existing (k-means++ inspired)
    for _ in range(1, k):
        best_dist = -1
        best_vec = None
        for v in sorted_vecs:
            if v.id in used:
                continue
            min_dist = min(
                cos_sim(v.vector, c) for c in centroids
            )
            # We want LOW similarity (far from existing centroids)
            if (1.0 - min_dist) > best_dist:
                best_dist = 1.0 - min_dist
                best_vec = v
        
        if best_vec is None:
            break
        centroids.append(best_vec.vector.copy())
        used.add(best_vec.id)
    
    k = len(centroids)
    
    # Step 2-4: Iterative assignment + update
    assignments = {}  # vector_id → cluster_idx
    
    for _ in range(MAX_ITERATIONS):
        changed = False
        
        # Assign each vector to nearest centroid
        for v in vectors:
            best_idx = 0
            best_sim = -1
            for i, c in enumerate(centroids):
                sim = cos_sim(v.vector, c)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            
            if assignments.get(v.id) != best_idx:
                assignments[v.id] = best_idx
                changed = True
        
        if not changed:
            break
        
        # Recompute centroids
        for i in range(k):
            members = [v for v in vectors if assignments.get(v.id) == i]
            if members:
                centroids[i] = np.mean([v.vector for v in members], axis=0)
    
    # Build clusters
    clusters = []
    for i in range(k):
        members = [v for v in vectors if assignments.get(v.id) == i]
        if len(members) >= MIN_CLUSTER_SIZE:
            sector_counts = {}
            for v in members:
                s = v.primary_sector
                sector_counts[s] = sector_counts.get(s, 0) + 1
            
            clusters.append(Cluster(
                id=i,
                centroid=centroids[i],
                members=[v.id for v in members],
                sectors=sector_counts,
            ))
    
    return clusters


# ── Graph Reorganization ─────────────────────────────────────────────

async def reorganize_graph(
    user_id: Optional[str] = None,
    boost_salience: bool = True,
    prune_edges: bool = True,
) -> Dict[str, Any]:
    """Full graph reorganization pipeline.
    
    1. Load vectors → cluster → create/update waypoints
    2. Boost salience of cluster centroids
    3. Prune weak waypoint edges
    """
    start_time = time.time()
    
    # Load vectors
    vectors = await load_memory_vectors(user_id=user_id)
    logger.info("Reorganizer: loaded %d vectors", len(vectors))
    
    if len(vectors) < MIN_CLUSTER_SIZE * 2:
        return {"status": "insufficient_data", "vector_count": len(vectors)}
    
    # Cluster
    clusters = cluster_memories(vectors)
    logger.info("Reorganizer: found %d clusters", len(clusters))
    
    # Apply cluster-based graph updates
    stats = {
        "vector_count": len(vectors),
        "cluster_count": len(clusters),
        "waypoints_created": 0,
        "waypoints_updated": 0,
        "edges_pruned": 0,
        "centroids_boosted": 0,
    }
    
    now = int(time.time() * 1000)
    
    for cluster in clusters:
        member_ids = cluster.members
        
        # Create/update waypoints among cluster members (bidirectional)
        for i, src_id in enumerate(member_ids):
            for dst_id in member_ids[i+1:]:
                # Check existing waypoint
                existing = db.fetchone(
                    "SELECT weight FROM waypoints WHERE src_id=? AND dst_id=?",
                    (src_id, dst_id)
                )
                
                if existing:
                    new_weight = min(1.0, float(existing["weight"]) + WAYPOINT_BOOST)
                    db.execute(
                        "UPDATE waypoints SET weight=?, updated_at=? WHERE src_id=? AND dst_id=?",
                        (new_weight, now, src_id, dst_id)
                    )
                    stats["waypoints_updated"] += 1
                else:
                    db.execute(
                        "INSERT INTO waypoints(src_id, dst_id, user_id, weight, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (src_id, dst_id, user_id or "anonymous", 0.5, now, now)
                    )
                    stats["waypoints_created"] += 1
        
        # Boost centroid memory (closest to centroid = highest mean similarity)
        if boost_salience and len(member_ids) >= 3:
            centroid_member = max(
                member_ids,
                key=lambda mid: cos_sim(
                    next(v.vector for v in vectors if v.id == mid),
                    cluster.centroid
                )
            )
            mem = q.get_mem(centroid_member)
            if mem:
                new_sal = min(1.0, (mem["salience"] or 0.5) + SALIENCE_BOOST)
                db.execute(
                    "UPDATE memories SET salience=?, updated_at=? WHERE id=?",
                    (new_sal, now, centroid_member)
                )
                stats["centroids_boosted"] += 1
    
    # Prune weak edges
    if prune_edges:
        db.execute(
            "DELETE FROM waypoints WHERE weight < ?",
            (PRUNE_THRESHOLD,)
        )
        stats["edges_pruned"] = db.conn.total_changes
        db.commit()
    
    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 2)
    logger.info("Reorganizer: complete in %.1fs — %s", elapsed, stats)
    
    return {"status": "complete", **stats}


# ── Scheduler ────────────────────────────────────────────────────────

_reorg_task: Optional[asyncio.Task] = None
_interval_minutes: int = 60  # Default: every hour


async def _reorg_loop(user_id: Optional[str] = None):
    """Background loop for periodic graph reorganization."""
    while True:
        try:
            await reorganize_graph(user_id=user_id)
        except Exception as e:
            logger.error("Reorganizer: loop error: %s", e)
        await asyncio.sleep(_interval_minutes * 60)


def start_reorganizer(
    user_id: Optional[str] = None,
    interval_minutes: int = 60,
):
    """Start periodic graph reorganization."""
    global _reorg_task, _interval_minutes
    
    if _reorg_task and not _reorg_task.done():
        logger.info("Reorganizer: already running")
        return
    
    _interval_minutes = interval_minutes
    _reorg_task = asyncio.create_task(_reorg_loop(user_id=user_id))
    logger.info("Reorganizer: started (every %d min)", interval_minutes)


def stop_reorganizer():
    """Stop periodic graph reorganization."""
    global _reorg_task
    if _reorg_task:
        _reorg_task.cancel()
        _reorg_task = None
        logger.info("Reorganizer: stopped")


async def run_reorganization_once(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Run one reorganization cycle synchronously."""
    return await reorganize_graph(user_id=user_id)
