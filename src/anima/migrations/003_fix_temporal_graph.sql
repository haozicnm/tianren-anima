-- 003_fix_temporal_graph.sql
-- P0.2: Fix temporal_graph schema mismatches + add memory-fact bridge

-- temporal_facts: fix column name (obj→object), add user_id, last_updated, memory_id
CREATE TABLE IF NOT EXISTS temporal_facts_new (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from INTEGER NOT NULL,
    valid_to INTEGER,
    confidence REAL,
    last_updated INTEGER,
    memory_id TEXT,
    metadata TEXT,
    FOREIGN KEY(memory_id) REFERENCES memories(id)
);
INSERT OR IGNORE INTO temporal_facts_new (id, subject, predicate, object, valid_from, valid_to, confidence, metadata)
    SELECT id, subject, predicate, obj, valid_from, valid_to, confidence, metadata FROM temporal_facts;
DROP TABLE temporal_facts;
ALTER TABLE temporal_facts_new RENAME TO temporal_facts;

-- temporal_edges: add id PK, rename relation→relation_type
CREATE TABLE IF NOT EXISTS temporal_edges_new (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    valid_from INTEGER NOT NULL,
    valid_to INTEGER,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata TEXT,
    FOREIGN KEY(source_id) REFERENCES temporal_facts(id),
    FOREIGN KEY(target_id) REFERENCES temporal_facts(id)
);
-- Generate UUIDs for existing edges (SQLite hex_random_blob → uuid format)
INSERT OR IGNORE INTO temporal_edges_new (id, source_id, target_id, relation_type, valid_from, valid_to, weight, metadata)
    SELECT 
        lower(hex(randomblob(4)) || '-' || hex(randomblob(2)) || '-' || hex(randomblob(2)) || '-' || hex(randomblob(2)) || '-' || hex(randomblob(6))),
        source_id, target_id, relation, valid_from, valid_to, weight, metadata 
    FROM temporal_edges;
DROP TABLE temporal_edges;
ALTER TABLE temporal_edges_new RENAME TO temporal_edges;

-- Indexes for graph-native search
CREATE INDEX IF NOT EXISTS idx_temporal_facts_user ON temporal_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_temporal_facts_memory ON temporal_facts(memory_id);
CREATE INDEX IF NOT EXISTS idx_temporal_facts_object ON temporal_facts(object);
CREATE INDEX IF NOT EXISTS idx_temporal_edges_source ON temporal_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_temporal_edges_target ON temporal_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_temporal_edges_relation ON temporal_edges(relation_type);
