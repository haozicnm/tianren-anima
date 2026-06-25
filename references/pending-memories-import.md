# Pending Memories Import

When LM Studio goes offline, the reflect engine queues memories to `~/.hermes/pending_memories.json` instead of writing to DB. Import them when LM Studio comes back online.

## Diagnosis

```bash
# 1. Check LM Studio is online
curl -s http://localhost:11434/v1/models | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"

# 2. Check pending count
python3 -c "
import json, os
with open(os.path.expanduser('~/.hermes/pending_memories.json')) as f:
    data = json.load(f)
actual = [m for m in data if m.get('content') and m.get('content') != '__META__']
print(f'Pending: {len(actual)}')
"

# 3. Check DB current count
python3 -c "
import sqlite3
conn = sqlite3.connect(os.path.expanduser('~/.hermes/openmemory.db'))
print(f'Memories: {conn.execute(\"SELECT COUNT(*) FROM memories\").fetchone()[0]}')
print(f'Vectors: {conn.execute(\"SELECT COUNT(*) FROM vectors\").fetchone()[0]}')
"
```

## Import Script

```bash
cd ~/dev/OpenMemory/packages/openmemory-py

# CRITICAL: 4 slashes for absolute path (pitfall #32)
OM_DB_URL="sqlite:////home/horizon/.hermes/openmemory.db" \
OM_EMBED_KIND=openai OM_VEC_DIM=1024 \
OPENAI_API_KEY=lm-studio \
OM_OPENAI_BASE_URL=http://localhost:11434/v1 \
OM_OPENAI_MODEL=text-embedding-bge-m3 \
python3 -c "
import json, os, sys, asyncio

with open(os.path.expanduser('~/.hermes/pending_memories.json')) as f:
    data = json.load(f)
actual = [m for m in data if m.get('content') and m.get('content') != '__META__']
print(f'Importing: {len(actual)} memories')

sys.path.insert(0, 'src')
from openmemory import Memory
mem = Memory(user='horizon')

async def import_all():
    ok, fail = 0, 0
    for i, m in enumerate(actual):
        try:
            await mem.add(content=m['content'], user_id='horizon', nature=m.get('nature', 'hybrid'))
            ok += 1
            if (i+1) % 10 == 0: print(f'  {i+1}/{len(actual)}')
        except Exception as e:
            fail += 1
            print(f'  FAIL [{i}]: {str(e)[:100]}')
    print(f'Done: {ok} ok, {fail} fail')

asyncio.run(import_all())
"
```

## Post-Import

```bash
# 1. Verify count increased
python3 -c "
import sqlite3
conn = sqlite3.connect(os.path.expanduser('~/.hermes/openmemory.db'))
print(f'Memories: {conn.execute(\"SELECT COUNT(*) FROM memories\").fetchone()[0]}')
print(f'Vectors: {conn.execute(\"SELECT COUNT(*) FROM vectors\").fetchone()[0]}')
"

# 2. Backup and clear pending
cp ~/.hermes/pending_memories.json ~/.hermes/pending_memories.json.bak.$(date +%Y%m%d)
echo '[]' > ~/.hermes/pending_memories.json

# 3. Verify searchability
# Use memory_search tool for a term from the imported content
```

## Common Pitfalls

- **3 slashes = relative path**: `sqlite:///path` → `path` (relative). Use `sqlite:////path` (4 slashes) for absolute.
- **Wrong DB**: If count doesn't increase after import, check `PRAGMA database_list` — you may have written to a different DB.
- **__META__ entry**: First entry in pending_memories.json may be metadata (`content: "__META__"`), skip it.
