from typing import List, Dict, Optional, Any
from .core.db import db, q
from .memory.hsg import hsg_query
from .memory.working import WorkingMemory, WorkingMemoryItem, create_working_memory_bridge
from .ops.ingest import ingest_document
from .openai_handler import OpenAIRegistrar

class Memory:
    def __init__(self, user: Optional[str] = None, use_working_memory: bool = False):
        self.default_user = user
        self.use_working_memory = use_working_memory
        db.connect()
        self._openai = OpenAIRegistrar(self)

        # P0: WorkingMemory bridge (MemOS-inspired FIFO pipeline)
        if use_working_memory:
            self._wm = create_working_memory_bridge(
                memory_add_fn=self._add_to_ltm,
                user_id=user or "default",
            )
        else:
            self._wm = None

    @property
    def openai(self):
        return self._openai

    @property
    def working_memory(self) -> Optional[WorkingMemory]:
        """Access the WorkingMemory bridge for inspection/flushing."""
        return self._wm

    async def _add_to_ltm(
        self,
        content: str,
        user_id: str = "",
        nature: Optional[str] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Direct write to LongTermMemory (called by WorkingMemory on promotion)."""
        uid = user_id or self.default_user or "default"
        res = await ingest_document(
            "text", content,
            meta=meta or {},
            user_id=uid,
            tags=tags or [],
            nature=nature,
        )
        if "root_memory_id" in res:
            res["id"] = res["root_memory_id"]
        return res

    async def add(
        self,
        content: str,
        user_id: Optional[str] = None,
        nature: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        uid = user_id or self.default_user

        if self._wm:
            # P0: Route through WorkingMemory first
            from .memory.hsg import classify_content

            qc = classify_content(content)
            primary_sector = qc.get("primary", "semantic") if qc else "semantic"

            item = WorkingMemoryItem(
                id="",
                content=content,
                user_id=uid or "default",
                nature=nature,
                tags=kwargs.get("tags", []),
                meta=kwargs.get("meta", {}),
                primary_sector=primary_sector,
                salience=0.5,  # Initial salience, recalculated on promotion
            )
            wm_id = await self._wm.add(item)

            # Check if auto-promotion happened
            if self._wm.size < self._wm.capacity:
                return {"id": wm_id, "status": "buffered", "working_memory": True}
            else:
                return {"id": wm_id, "status": "promoted", "working_memory": True}
        else:
            # Direct to permanent storage (legacy mode)
            res = await ingest_document(
                "text", content,
                meta=kwargs.get("meta"),
                user_id=uid,
                tags=kwargs.get("tags"),
                nature=nature,
            )
            if "root_memory_id" in res:
                res["id"] = res["root_memory_id"]
            return res

    async def flush(self) -> List[str]:
        """Promote all WorkingMemory items to LongTermMemory. Call on session end."""
        if self._wm:
            return await self._wm.flush(force=True)
        return []

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        nature: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        uid = user_id or self.default_user
        filters = kwargs.copy()
        filters["user_id"] = uid
        if nature:
            filters["nature"] = nature

        # P0: Search both WorkingMemory AND LongTermMemory
        ltm_results = await hsg_query(query, limit, filters)

        if self._wm:
            wm_results = self._wm.search(query, limit=limit, user_id=uid, nature=nature)
            # Merge: WorkingMemory results first (fresher), then LTM
            # Deduplicate by content similarity
            seen_contents = set()
            merged = []
            for r in wm_results:
                sig = r["content"][:80]
                if sig not in seen_contents:
                    seen_contents.add(sig)
                    merged.append(r)
            for r in ltm_results:
                sig = r["content"][:80]
                if sig not in seen_contents:
                    seen_contents.add(sig)
                    merged.append(r)
            return merged[:limit]
        else:
            return ltm_results

    async def get(self, memory_id: str):
        # Handle working memory IDs
        if memory_id.startswith("wm:") and self._wm:
            real_id = memory_id[3:]
            for item in self._wm._buffer:
                if item.id == real_id:
                    return {
                        "id": memory_id,
                        "content": item.content,
                        "nature": item.nature,
                        "source": "working_memory",
                    }
        return q.get_mem(memory_id)

    async def delete(self, memory_id: str):
        if memory_id.startswith("wm:"):
            # Cannot delete from working memory explicitly — will be filtered on flush
            return
        q.del_mem(memory_id)

    async def delete_all(self, user_id: str = None):
        uid = user_id or self.default_user
        if self._wm:
            self._wm._buffer.clear()
        if uid:
            q.del_mem_by_user(uid)
            from .memory.hsg import cache as hsg_cache
            hsg_cache.clear()

    # ── P2: Graph Reorganization ─────────────────────────────────────

    async def reorganize_graph(self) -> Dict[str, Any]:
        """P2.1: Run graph reorganization (vector clustering + edge pruning)."""
        from .memory.graph_reorganizer import run_reorganization_once
        return await run_reorganization_once(user_id=self.default_user)

    def start_reorganizer(self, interval_minutes: int = 60):
        """P2.1: Start periodic graph reorganization."""
        from .memory.graph_reorganizer import start_reorganizer
        start_reorganizer(user_id=self.default_user, interval_minutes=interval_minutes)

    def stop_reorganizer(self):
        """P2.1: Stop periodic graph reorganization."""
        from .memory.graph_reorganizer import stop_reorganizer
        stop_reorganizer()

    # ── P2: Cross-User Altruistic Pool ───────────────────────────────

    async def search_altruistic(
        self,
        query: str,
        limit: int = 10,
        tian_ren_ratio: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """P2.2: Cross-user search of altruistic knowledge pool."""
        from .ops.altruistic_pool import search_altruistic as sa
        return await sa(
            query=query,
            current_user_id=self.default_user,
            limit=limit,
            tian_ren_ratio=tian_ren_ratio,
        )

    async def get_knowledge_pool(
        self,
        tian_ren_ratio: float = 0.5,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """P2.2: Get aggregated altruistic knowledge pool."""
        from .ops.altruistic_pool import get_knowledge_pool as gkp
        return await gkp(
            user_id=self.default_user,
            tian_ren_ratio=tian_ren_ratio,
            limit=limit,
        )

    async def promote_to_altruistic(self, memory_id: str) -> bool:
        """P2.2: Promote a memory to the altruistic shared pool."""
        from .ops.altruistic_pool import promote_to_altruistic
        return await promote_to_altruistic(memory_id)

    async def demote_to_private(self, memory_id: str) -> bool:
        """P2.2: Remove a memory from the altruistic shared pool."""
        from .ops.altruistic_pool import demote_to_private
        return await demote_to_private(memory_id)

    def history(self, user_id: str = None, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        uid = user_id or self.default_user
        rows = q.all_mem_by_user(uid, limit, offset)
        return [dict(r) for r in rows]

    def source(self, name: str):
        from . import connectors

        sources = {
            "github": connectors.github_connector,
            "notion": connectors.notion_connector,
            "google_drive": connectors.google_drive_connector,
            "google_sheets": connectors.google_sheets_connector,
            "google_slides": connectors.google_slides_connector,
            "onedrive": connectors.onedrive_connector,
            "web_crawler": connectors.web_crawler_connector,
        }

        if name not in sources:
            raise ValueError(f"unknown source: {name}. available: {list(sources.keys())}")

        return sources[name](user_id=self.default_user)

def run_mcp():
    import asyncio
    from .ai.mcp import run_mcp_server
    try:
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        print("Server mode removed. Use 'mcp' for agentic usage.")
    elif len(sys.argv) > 1 and sys.argv[1] == "mcp":
        run_mcp()
    else:
        print("天人·Anima Memory Engine")
        print("Usage: python -m anima.main mcp")
