# Zvec Integration into OpenMemory/Anima

## Overview

Zvec (alibaba/zvec, ★12.2K) is an embedded (in-process) vector database with C++ core. Integrated as an alternative `VectorStore` backend for OpenMemory, replacing SQLite brute-force cosine similarity search with HNSW indexing.

## Files Modified

- `openmemory/core/vector/zvec_store.py` — new `ZvecVectorStore` class implementing `VectorStore` interface
- `openmemory/core/vector_store.py` — factory function: `elif backend == "zvec":` branch
- Env var: `OPENMEMORY_VECTOR_STORE=zvec` to enable

## HNSW Configuration Pitfalls

### Metric Type (CRITICAL)
Zvec `HnswIndexParam` defaults to `MetricType.IP` (inner product). OpenMemory uses **cosine similarity**. Must explicitly set:

```python
from zvec import HnswIndexParam, MetricType

HnswIndexParam(
    metric_type=MetricType.COSINE,  # NOT default IP!
    m=16,
    ef_construction=200,
)
```

Without this, search results have wrong similarity scores (negative or >1.0).

### Doc API (NOT dict-like)
Zvec `Doc` objects are NOT dictionaries. Cannot use `.get()`:

```python
# WRONG — raises AttributeError
doc.get("id")
doc.get("fields", {}).get("memory_id", "")

# CORRECT
doc.id                    # document ID
doc.fields                # dict of field values
doc.vectors               # dict of vector values
```

### Collection Schema
```python
schema = CollectionSchema(
    name=f"memory_{sector}",
    fields=[
        FieldSchema("memory_id", DataType.STRING, nullable=False, index_param=InvertIndexParam()),
        FieldSchema("user_id", DataType.STRING, nullable=True, index_param=InvertIndexParam()),
    ],
    vectors=[
        VectorSchema("embedding", DataType.VECTOR_FP32, dimension=dim,
                     index_param=HnswIndexParam(metric_type=MetricType.COSINE, m=16, ef_construction=200)),
    ],
)
```

### Collection Lifecycle
```python
# Create new
collection = zvec.create_and_open(path=path, schema=schema, option=CollectionOption(read_only=False, enable_mmap=True))

# Open existing
collection = zvec.open(path=path)

# Close
collection.close()
```

### Delete-then-Insert for Updates
Zvec has no native update. Pattern for overwriting:
```python
try:
    collection.insert([doc])
except Exception as e:
    if "already exists" in str(e).lower():
        collection.delete([id])
        collection.insert([doc])
```

## db.conn Guard
`storeVector` queries `db.conn` for user_id fallback. In test environments, `db.conn` may be `None`:

```python
if user_id is None and db.conn is not None:
    try:
        mem = db.conn.execute("SELECT user_id FROM memories WHERE id=?", (id,)).fetchone()
        if mem:
            user_id = mem["user_id"]
    except Exception:
        pass
```

## Performance Benchmarks (dim=1024)

| Data Size | Zvec HNSW | SQLite Brute-Force | Speedup |
|-----------|-----------|-------------------|---------|
| 1,000 | 0.57ms | 2.92ms | 5.1x |
| 5,000 | 1.47ms | 14.94ms | 10.2x |
| 10,000 | 2.71ms | 30.87ms | 11.4x |

Anima current: ~8,683 vectors → expected 10x+ retrieval speedup.

Store throughput: ~6,000-12,000 vec/s (1024-dim).

## Sector Architecture
Each HSG sector (semantic/episodic/reflective/emotional/procedural) gets its own zvec Collection:
- `~/.openmemory/zvec/sector_semantic/`
- `~/.openmemory/zvec/sector_episodic/`
- etc.

Collections are lazily initialized on first access per sector.

## Fallback Strategy
`ZvecVectorStore.search()` falls back to SQLite brute-force (`_fallback_search`) on any zvec error. This ensures zero-downtime migration — if zvec has issues, search still works.

## Import Requirements
```
pip install zvec  # v0.5.0, 74.9 MB wheel (C++ core)
```

Requires Python 3.10-3.14, Linux/macOS/Windows x86_64/ARM64.
