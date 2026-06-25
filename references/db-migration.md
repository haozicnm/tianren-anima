# DB Migration: Dev → Production

When the dev database (with all labeled memories) is at a different path than the Hermes plugin's active database.

## Finding the databases

The Hermes plugin uses `openmemory-py`'s `Memory(user=...)` which creates a SQLite DB based on the `user` parameter. Check multiple paths:

```bash
find ~ -name "openmemory.db" 2>/dev/null
# Typical paths:
# ~/.hermes/openmemory.db          — Hermes plugin active
# ~/dev/OpenMemory/packages/openmemory-py/openmemory.db — dev/test
```

## Checking what's where

```python
import sqlite3

for db_path in ["~/.hermes/openmemory.db", "~/dev/OpenMemory/packages/openmemory-py/openmemory.db"]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Verify schema compatibility
    cur.execute("PRAGMA table_info(memories)")
    hermes_cols = {r[1] for r in cur.fetchall()}

    cur.execute("SELECT nature, COUNT(*) FROM memories GROUP BY nature")
    print(f"\n{db_path}:")
    for row in cur.fetchall():
        print(f"  nature={row[0]}: {row[1]}")

    conn.close()
```

## Migration script

When schemas match, copy memories that don't exist yet:

```python
import sqlite3

hermes_db = "~/.hermes/openmemory.db"
dev_db = "~/dev/OpenMemory/packages/openmemory-py/openmemory.db"

dev_conn = sqlite3.connect(dev_db)
hermes_conn = sqlite3.connect(hermes_db)

# Get IDs from both
dev_ids = set(r[0] for r in dev_conn.execute("SELECT id FROM memories"))
hermes_ids = set(r[0] for r in hermes_conn.execute("SELECT id FROM memories"))
new_ids = dev_ids - hermes_ids

# Copy
if new_ids:
    dev_conn.row_factory = None  # ensure plain tuples
    rows = dev_conn.execute(
        f"SELECT * FROM memories WHERE id IN ({','.join('?'*len(new_ids))})",
        list(new_ids)
    ).fetchall()

    placeholders = ",".join("?" * len(rows[0]))
    hermes_conn.executemany(f"INSERT INTO memories VALUES ({placeholders})", rows)
    hermes_conn.commit()

dev_conn.close()
hermes_conn.close()
```

## Pitfall: vectors table column mismatch

If the vectors table has different column names across databases (e.g. `id` vs `memory_id`), skip vector migration. Vectors will be regenerated on next `embed` call.

## Always backup first

```bash
cp ~/.hermes/openmemory.db ~/.hermes/openmemory.db.bak
```
