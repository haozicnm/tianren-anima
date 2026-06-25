# Health Check & Backfill Scripts

## Full Health Check (run from execute_code)

```python
import os, sqlite3

db_path = os.path.expanduser("~/.hermes/openmemory.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 1. Table overview
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    cnt = cur.fetchone()[0]
    print(f"  {t}: {cnt}")

# 2. Memories by nature
cur.execute("SELECT nature, COUNT(*) FROM memories GROUP BY nature ORDER BY COUNT(*) DESC")
for n, c in cur.fetchall():
    print(f"  {n or 'NULL'}: {c}")

# 3. Vector coverage (vectors table, not memories.vector column)
cur.execute("SELECT sector, COUNT(*) FROM vectors GROUP BY sector ORDER BY COUNT(*) DESC")
for s, c in cur.fetchall():
    print(f"  {s}: {c}")

# 4. Missing vectors
cur.execute("SELECT COUNT(DISTINCT id) FROM vectors")
with_vec = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM memories")
total = cur.fetchone()[0]
missing = total - with_vec
print(f"  Vector coverage: {with_vec}/{total} ({missing} missing)")

# 5. Graph stats
for table in ["waypoints", "temporal_facts", "temporal_edges"]:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"  {table}: {cur.fetchone()[0]}")

# 6. Decay stats
cur.execute("SELECT AVG(decay_lambda), MIN(decay_lambda), MAX(decay_lambda) FROM memories WHERE decay_lambda > 0")
row = cur.fetchone()
print(f"  Decay: avg={row[0]:.4f}, min={row[1]:.4f}, max={row[2]:.4f}")

# 7. DB size
print(f"  DB size: {os.path.getsize(db_path)/1024/1024:.1f} MB")

# 8. Find empty content memories
cur.execute("SELECT id FROM memories WHERE LENGTH(content) = 0")
empty = cur.fetchall()
if empty:
    print(f"  ⚠️ Empty content memories: {len(empty)} — DELETE FROM memories WHERE LENGTH(content) = 0")

conn.close()
```

## Temporal Edges Backfill (numpy matrix approach)

For populating `temporal_edges` when the table is empty but memories exist with `mean_vec`.

```python
import os, sys, sqlite3, time, uuid
import numpy as np

sys.path.insert(0, os.path.expanduser("~/dev/OpenMemory/packages/openmemory-py/src"))
from openmemory.utils.vectors import buf_to_vec, cos_sim

db_path = os.path.expanduser("~/.hermes/openmemory.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Load all memories with mean_vec
cur.execute("SELECT id, mean_vec FROM memories WHERE mean_vec IS NOT NULL AND mean_vec != ''")
rows = cur.fetchall()

ids, vecs = [], []
for r in rows:
    try:
        ids.append(r["id"])
        vecs.append(np.array(buf_to_vec(r["mean_vec"]), dtype=np.float32))
    except:
        pass

# Pairwise similarity matrix (O(n²) but fast for n<5000)
vec_matrix = np.stack(vecs)
normed = vec_matrix / np.maximum(np.linalg.norm(vec_matrix, axis=1, keepdims=True), 1e-8)
sim_matrix = normed @ normed.T

TOP_K, MIN_SIM = 5, 0.5
now = int(time.time() * 1000)
batch = []

for i in range(len(ids)):
    sims = sim_matrix[i].copy()
    sims[i] = -1  # exclude self
    for j in np.argsort(sims)[-TOP_K:][::-1]:
        if sims[j] >= MIN_SIM:
            batch.append((str(uuid.uuid4()), ids[i], ids[j], "semantic_neighbor", now, None, float(sims[j]), '{"source":"backfill"}'))

cur.executemany(
    "INSERT OR IGNORE INTO temporal_edges(id, source_id, target_id, relation_type, valid_from, valid_to, weight, metadata) VALUES (?,?,?,?,?,?,?,?)",
    batch
)
conn.commit()
print(f"Created {len(batch)} temporal edges for {len(ids)} memories")

# Verify
cur.execute("SELECT COUNT(*) FROM temporal_edges")
cur.execute("SELECT COUNT(DISTINCT source_id) FROM temporal_edges")
print(f"Total edges: {cur.fetchone()[0]}, source memories: {cur.fetchone()[0]}")
conn.close()
```

## Orphan Cleanup

```python
# Clean orphaned waypoints (src/dst memory deleted)
db.execute("DELETE FROM waypoints WHERE src_id NOT IN (SELECT id FROM memories) OR dst_id NOT IN (SELECT id FROM memories)")

# Clean orphaned vectors
db.execute("DELETE FROM vectors WHERE id NOT IN (SELECT id FROM memories)")

# Clean orphaned temporal_edges
db.execute("DELETE FROM temporal_edges WHERE source_id NOT IN (SELECT id FROM memories) OR target_id NOT IN (SELECT id FROM memories)")

# Clean orphaned temporal_facts
db.execute("DELETE FROM temporal_facts WHERE memory_id IS NOT NULL AND memory_id NOT IN (SELECT id FROM memories)")
```

## Search Verification

```python
from memory_search import memory_search  # via Hermes tool

# Test general search
results = memory_search(query="矛盾论 主要矛盾", limit=5)
# Expect: top results are 毛选 content with similarity > 0.5

# Test nature-filtered search
results = memory_search(query="矛盾论", nature="altruistic", limit=3)
# Expect: all results have nature="altruistic"

# Check graph_facts and graph_edges in results
for r in results:
    print(f"  facts={len(r.get('graph_facts', []))}, edges={len(r.get('graph_edges', []))}")
```
