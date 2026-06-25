-- 004_add_version_chain.sql
-- P1.1: Version chain support (supermemory-inspired)
-- parent_id: previous version of this memory
-- root_id: original memory in the chain (same across all versions)

ALTER TABLE memories ADD COLUMN parent_id TEXT REFERENCES memories(id);
ALTER TABLE memories ADD COLUMN root_id TEXT REFERENCES memories(id);

-- Index for version chain traversal and conflict detection
CREATE INDEX IF NOT EXISTS idx_memories_root ON memories(root_id);
CREATE INDEX IF NOT EXISTS idx_memories_parent ON memories(parent_id);
