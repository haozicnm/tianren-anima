# Fact Extraction Bridge — Regex SPO Pipeline → Cascaded Entity Extraction

P0.2 → P1.2 evolution: started with regex-only SPO extraction, upgraded to cascaded entity extraction with co-reference resolution.

## Pattern Types (regex SPO)

### English Patterns (P0.2 original)

| Type | Pattern | Example | Confidence |
|------|---------|---------|------------|
| `type_of` | X is/was Y | "Alice is a data scientist" → `Alice is_a data scientist` | 0.7 |
| `has` | X has/have Y | "Bob has a PhD" → `Bob has PhD` | 0.6 |
| `relation` | X works_at/for/in Y | "Alice works at OpenAI" → `Alice works_at OpenAI` | 0.65 |
| `action` | X built/created/developed Y | "Alice built the prototype" → `Alice built prototype` | 0.7 |
| `statement` | X said/stated/claimed Y | "Alice said results were promising" → `Alice said results were promising` | 0.5 |

### Chinese Patterns (added 2026-06-21)

| Type | Pattern | Example | Confidence |
|------|---------|---------|------------|
| `type_of` | X 是 Y | "Hermes Agent 是一个开源 AI agent 框架" → `Hermes Agent is_a 开源 AI agent 框架` | 0.7 |
| `action` | X 支持/使用/包含/需要/基于 Y | "workflow-engine 支持 34 种节点" → `workflow-engine 支持 34 种节点` | 0.7 |
| `relation` | X 由 Y 创建/开发 | "框架由 Nous Research 创建" → `框架 created_by Nous Research` | 0.65 |
| `has` | X 的 Y 是 Z | "DeepSeek 的缓存是前缀匹配" → `DeepSeek has 前缀匹配` | 0.6 |
| `action` | X 被重命名为 Y | "OpenMemory 被重命名为 tianren-anima" → `OpenMemory 重命名为 tianren-anima` | 0.7 |
| `action` | X → Y | "P0→P1→P2" → `P0 action P1` | 0.7 |
| `action` | English 是/支持 Chinese | "LM Studio 使用 bge-m3" → `LM Studio 使用 bge-m3` | 0.7 |

**Key design decisions for Chinese patterns:**
- Subject allows spaces (`[\w\-\. ]{1,30}`) for multi-word names like "Hermes Agent", "LM Studio"
- Terminated by CJK/ASCII punctuation (`，。；`), not just English punctuation
- Chinese stop words added to noise filter: `的|了|在|是|有|和|与|或|但|如果|因为|所以|可以|需要|应该|已经|还是|就是|不是|没有`
- `(?<![\u4e00-\u9fff])` lookbehind was tested but rejected — it prevented matching "Hermes Agent 是" because "Agent" ends with ASCII
- Coverage: ~23% for conversational content (245/1086 memories). Regex works best on prose, not "User: ... \nAssistant: ..." format

## Regex Design Notes

**Subject** always requires capitalized first word: `([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)`
This ensures we only extract named entities, not generic nouns.

**Object** for `type_of` uses lowercase-tolerant: `([a-zA-Z][\w\s]{1,40}?)` — terminated by stop words (`at|in|for|with|from|by|and|but|which`) to clip trailing context.

**Deduplication**: `{subject}|{predicate}|{object}` signature stored in a set to prevent duplicates.

**Limits**: max 10 facts per memory. Skip subjects matching noise words (`the|a|an|this|that|it|they|he|she|we|you|i`). Skip if subject > 80 chars or object > 200 chars.

## P1.2 Upgrade: Cascaded Entity Extraction

File: `ops/extract_entities.py`

The cascade pipeline wraps regex SPO extraction with entity recognition:

```
Phase 1: extract_entities(text)           — regex entity types (PERSON/ORG/TECH/DATE/VERSION)
Phase 2: resolve_co_references(entities)  — merge same entity, different forms
Phase 3: hash → UUID                       — MD5({name}:{type})[:12] for stable UUIDs
Phase 4: extract_candidate_facts(content) — reuse P0.2 regex SPO + insert_fact(..., memory_id)
```

### Entity Types

| Type | Examples | Patterns |
|------|----------|----------|
| PERSON | Alice, Bob Chen, 伟哥, 若溪姐 | Full names: `[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+`; single capped: `[A-Z][a-z]+`; CJK names with titles: `[\u4e00-\u9fff]{2,4}(哥\|姐\|弟\|妹\|老师\|博士\|教授)` |
| ORG | Google Inc, OpenAI, 盖亚团队 | Known list + suffix; CJK: `[\u4e00-\u9fff]{2,10}(公司\|大学\|研究院\|实验室\|团队\|协会)` |
| TECH | Python, Kubernetes, AWS, 记忆引擎 | Known tech keywords + project names (MiMo/Hermes/LM Studio); CJK: `[\u4e00-\u9fff]{2,6}(引擎\|框架\|模型\|系统\|平台\|数据库)` |
| DATE | 2024-06-15, 2026年6月20日 | ISO dates + named months + Chinese date format |
| VERSION | v3.2.1, 2.4.0-beta | Semantic version pattern |
| CONCEPT | 实践论, 天人合一框架 | CJK abstract concepts: `[\u4e00-\u9fff]{2,8}(论\|主义\|思想\|哲学\|框架\|体系\|模型\|方法\|策略\|原则)` |

### PERSON Stop-Word Filtering

The single-word PERSON pattern `[A-Z][a-z]+` is too broad and would match sentence-initial words like "She", "He", "The", "And", "But". Filter with:

```python
PERSON_STOP_WORDS = {
    "She", "He", "It", "They", "We", "You", "I", "Me", "This", "That",
    "The", "A", "An", "And", "But", "Or", "If", "In", "On", "At", "To",
    "For", "With", "From", "By", "As", "Is", "Was", "Are", "Be", "Has",
    "Have", "Had", "Do", "Does", "Did", "Will", "Would", "Can", "Could",
    "Not", "No", "Yes", "So", "Just", "Now", "Then", "Also",
    # Chinese stop words
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "什么", "怎么", "为什么", "是", "的", "了", "在",
    "有", "和", "与", "或", "但", "如果", "因为", "所以", "可以",
    "需要", "应该", "已经", "还是", "就是", "不是", "没有",
}
```

### Co-reference Resolution

| Rule | Behavior |
|------|----------|
| Exact match (case-insensitive) | Merge, keep first occurrence |
| Contains (n1 in n2) | Skip shorter, add longer at its turn |
| Contains (n2 in n1) | Skip shorter, keep longer as canonical |
| Jaccard similarity > 0.8 | Merge to longer form |
| Different types | Never merge (AWS=TECH ≠ AWS=ORG) |

**Critical pitfall**: when n1 in n2, do NOT set `canonical = e2` and append. Instead mark `i` as merged with `canonical = None` and let `e2` be added when its own loop turn comes. Otherwise `e2` appears twice in results.

### LLM-Assisted Mode (opt-in)

When API keys are configured:
1. Delegates entity recognition to LLM (`adapter.chat()` with entity extraction prompt)
2. Merges LLM entities with regex entities (LLM takes priority, regex augments)
3. Falls back to pure regex when LLM is unavailable

## Integration Flow

```
add_hsg_memory(content) 
    → asyncio.create_task(_extract_facts_for_memory(mid, content, user_id))
        → try: extract_and_link_entities(mid, content, user_id)  # P1.2 cascade
        → except: extract_and_link_facts(mid, content, user_id)  # P0.2 regex fallback
            → for each candidate: insert_fact(..., memory_id=mid)
                → temporal_facts table row with memory_id backlink
```

Fire-and-forget: fact extraction never blocks memory ingestion. Failures are logged at DEBUG level.

## Known Limitations

1. **Conversational content has low SPO density**: regex patterns match ~23% of memories. "User: ... \nAssistant: ..." format lacks explicit subject-verb-object structures. LLM-assisted extraction would improve this significantly.
2. No LLM-assisted entity extraction by default (opt-in only with API keys)
3. No OWL ontology validation (P2 candidate)
4. No cross-fact contradiction detection
5. Entity resolution is purely token-based (no semantic disambiguation)
6. Chinese patterns may produce noisy facts from long compound sentences — the non-greedy `?` on object matching helps but doesn't eliminate all noise

## Backfill Script

For memories migrated directly (bypassing `add_hsg_memory()`), run `~/backfill_extractions.py`:

```python
os.environ["OM_DB_URL"] = "sqlite:///" + os.path.expanduser("~/.hermes/openmemory.db")
# ... then call _extract_facts_for_memory() for each memory
```

The script skips memories that already have facts (by checking `temporal_facts.memory_id`), so it's safe to re-run.
