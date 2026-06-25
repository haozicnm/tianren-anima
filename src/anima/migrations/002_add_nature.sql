-- 002_add_nature.sql
-- Heavenly-Human Harmony: nature spectrum (egoistic/altruistic/hybrid)
ALTER TABLE memories ADD COLUMN nature TEXT;
CREATE INDEX IF NOT EXISTS idx_memories_nature ON memories(nature);
