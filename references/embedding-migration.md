# Embedding Provider Migration — Batch Re-embedding

2026-06-20 实测：1082 条 768→1024-dim (bge-m3)，25 秒完成。

## 迁移脚本核心模式

```python
import sqlite3, struct, asyncio, httpx

DB_PATH = "/home/horizon/.hermes/openmemory.db"
LM_URL  = "http://localhost:11434/v1"
MODEL   = "text-embedding-bge-m3"
BATCH   = 10  # LM Studio batch embed

async def migrate():
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        "SELECT id, content FROM memories WHERE mean_dim=768 AND content IS NOT NULL AND content != ''"
    ).fetchall()
    
    client = httpx.AsyncClient(base_url=LM_URL, timeout=60.0)
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start+BATCH]
        resp = await client.post("/embeddings", json={"model": MODEL, "input": [r[1] for r in batch]})
        embeddings = sorted(resp.json()["data"], key=lambda x: x["index"])
        for i, r in enumerate(batch):
            vec = embeddings[i]["embedding"]
            blob = struct.pack(f"{len(vec)}f", *vec)
            db.execute("UPDATE memories SET mean_vec=?, mean_dim=? WHERE id=?", (blob, len(vec), r[0]))
        if start % 50 == 0: db.commit()
    db.commit()
    db.execute("DELETE FROM vectors WHERE dim=768")  # 清旧 per-sector 向量
    db.commit()
    db.close()

asyncio.run(migrate())
```

## 关键注意

1. **Hermes DB 无 `prev_mean_vec` 列**：先 `PRAGMA table_info(memories)` 确认 schema，不要假设列名。
2. **只更新 mean_vec，不重建 per-sector 向量**：旧记忆的 per-sector 向量被删除。搜索通过 vector_store 的 mean_vec fallback 覆盖。
3. **批量 10 条/次**：LM Studio batch endpoint 支持 10 条/次，单条 ~100ms，批量 ~50ms/条。
4. **备份**：迁移前 `cp openmemory.db openmemory.db.bak.bge-m3`。
5. **验证**：迁移后 `SELECT mean_dim, COUNT(*) FROM memories WHERE mean_vec IS NOT NULL GROUP BY mean_dim` 确认分布。

## 回退

```bash
cp ~/.hermes/openmemory.db.bak.bge-m3 ~/.hermes/openmemory.db
```
