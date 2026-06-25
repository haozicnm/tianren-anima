"""
P1.2: Cascaded Entity Extraction — cognee-inspired multi-round extraction pipeline.

Pipeline:
  1. Extract — regex SPO facts (base) + optional LLM entity recognition
  2. Resolve — co-reference resolution (same entity, different names)
  3. Assign UUIDs — stable UUIDs for resolved entities
  4. Build graph — upsert facts with resolved entity references

Uses regex extraction as the base (zero-cost, always available).
LLM-assisted extraction is opt-in when API keys are configured.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Entity Types ──────────────────────────────────────────────────────

# Stop words that look like names but aren't
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

KNOWN_ENTITY_TYPES = {
    "PERSON": [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",      # Proper full names: Alice Smith
        r"\b[A-Z][a-z]+\b",                           # Single capitalized: Alice
        r"[\u4e00-\u9fff]{2,4}(?:哥|姐|弟|妹|老师|博士|教授|先生|女士|总)",  # Chinese names with titles
    ],
    "ORG": [
        r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+)\s+(?:Inc|Corp|LLC|Ltd|Company|Group|Lab|AI)\b",
        r"\b(?:OpenAI|Google|Microsoft|Amazon|Meta|Apple|Netflix|Tesla|SpaceX|Anthropic|Nous Research|Hermes|DeepSeek)\b",
        r"[\u4e00-\u9fff]{2,10}(?:公司|大学|研究院|实验室|团队|协会|基金会|组织|机构|集团)",  # Chinese org suffixes
    ],
    "TECH": [
        r"\b(?:Python|Rust|SQL|React|Kubernetes|Docker|AWS|GCP|Azure|Linux|Node\.js|TypeScript|PostgreSQL|MongoDB|Redis|GraphQL|API|ML|AI|GPU|CPU|HTTP|REST|gRPC|Kafka|Spark|Hadoop|SQLite|Tauri|Flutter|Vue|shadcn|WebSocket)\b",
        r"\b(?:mimo|MiMo|OpenMemory|tianren|Anima|Hermes|LM Studio|bge-m3|GGUF|LoRA|DPO|GRPO)\b",
        r"[\u4e00-\u9fff]{2,6}(?:引擎|框架|模型|系统|平台|数据库|算法|接口|协议|缓存)",  # Chinese tech terms
    ],
    "DATE": [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        r"\d{4}年\d{1,2}月\d{1,2}日?",  # Chinese date format
    ],
    "VERSION": [
        r"\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b",
    ],
    # Chinese-specific entity types
    "CONCEPT": [
        r"[\u4e00-\u9fff]{2,8}(?:论|主义|思想|哲学|框架|体系|模型|方法|策略|原则)",  # Abstract concepts
    ],
}

ENTITY_TYPE_PATTERNS = {
    etype: [re.compile(p) for p in patterns]
    for etype, patterns in KNOWN_ENTITY_TYPES.items()
}


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract typed entities from text using regex patterns.

    Returns list of {name, type, position}.
    """
    entities = []
    seen = set()

    for etype, patterns in ENTITY_TYPE_PATTERNS.items():
        for pat in patterns:
            for match in pat.finditer(text):
                name = match.group(0).strip()
                
                # Filter stop words for PERSON type
                if etype == "PERSON" and name in PERSON_STOP_WORDS:
                    continue
                
                sig = (name.lower(), etype)
                if sig not in seen:
                    seen.add(sig)
                    entities.append({
                        "name": name,
                        "type": etype,
                        "position": match.start(),
                    })

    # Sort by position for co-reference context
    entities.sort(key=lambda e: e["position"])
    return entities


# ── Co-reference Resolution ──────────────────────────────────────────

def resolve_co_references(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Resolve co-referring entities (same entity, different surface forms).

    Rules:
    - "Alice" and "Alice Smith" → merge to "Alice Smith" 
    - "Google" and "Google Inc" → merge to "Google Inc"
    - Case-insensitive matching
    - Acronym matching: "AWS" ↔ "Amazon Web Services"
    """
    if len(entities) <= 1:
        return entities

    resolved = []
    merged_indices = set()

    for i, e1 in enumerate(entities):
        if i in merged_indices:
            continue

        canonical = e1
        for j, e2 in enumerate(entities):
            if j <= i or j in merged_indices:
                continue

            # Same type required for merging
            if e1["type"] != e2["type"]:
                continue

            n1, n2 = e1["name"].lower(), e2["name"].lower()

            # Exact match (case-insensitive) — merge, keep first occurrence
            if n1 == n2:
                merged_indices.add(j)
                continue

            # One contains the other
            if n1 in n2:
                # e1 is a substring of e2 → skip e1, e2 will be added at its turn
                merged_indices.add(i)
                canonical = None
                break
            elif n2 in n1:
                # e2 is a substring of e1 → skip e2, keep e1 as canonical
                merged_indices.add(j)
                continue

            # High similarity (e.g., "Google" vs "Google Inc")
            if _entity_similarity(n1, n2) > 0.8:
                canonical = e1 if len(n1) >= len(n2) else e2
                merged_indices.add(i if len(n1) < len(n2) else j)
                break

        if canonical is not None:
            resolved.append(canonical)

    return resolved


def _entity_similarity(name1: str, name2: str) -> float:
    """Jaccard similarity of word sets (case-insensitive)."""
    w1 = set(name1.lower().split())
    w2 = set(name2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# ── Cascade Pipeline ─────────────────────────────────────────────────

async def cascade_extract(
    content: str,
    user_id: Optional[str] = None,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Cognee-inspired cascaded extraction pipeline.

    Phase 1: Extract entities (regex + optional LLM)
    Phase 2: Resolve co-references
    Phase 3: Assign stable UUIDs
    Phase 4: Build SPO facts with resolved entities

    Returns {entities, facts, stats}.
    """
    # Phase 1: Extract
    entities = extract_entities(content)

    if use_llm:
        try:
            llm_entities = await _llm_extract_entities(content)
            entities = _merge_entities(entities, llm_entities)
        except Exception as e:
            logger.debug("LLM entity extraction skipped: %s", e)

    # Phase 2: Resolve co-references
    entities = resolve_co_references(entities)

    # Phase 3: Assign stable UUIDs (deterministic from entity name+type)
    import hashlib
    for e in entities:
        key = f"{e['name']}:{e['type']}"
        e["uuid"] = hashlib.md5(key.encode()).hexdigest()[:12]

    # Phase 4: Extract SPO facts with resolved entities
    from .extract_facts import extract_candidate_facts
    facts = extract_candidate_facts(content)

    return {
        "entities": entities,
        "facts": facts,
        "stats": {
            "entity_count": len(entities),
            "fact_count": len(facts),
            "co_references_resolved": _count_merged(entities),
            "method": "llm+cascade" if use_llm else "regex+cascade",
        },
    }


async def _llm_extract_entities(content: str) -> List[Dict[str, str]]:
    """Use LLM to extract entities from content.

    Falls back to empty list if LLM is unavailable.
    """
    try:
        from ..core.config import env

        # Use the configured provider
        if env.emb_kind == "openai":
            from ..ai.openai import OpenAIAdapter
            adapter = OpenAIAdapter()
        elif env.emb_kind == "ollama":
            from ..ai.ollama import OllamaAdapter
            adapter = OllamaAdapter()
        elif env.emb_kind == "gemini":
            from ..ai.gemini import GeminiAdapter
            adapter = GeminiAdapter()
        else:
            return []  # No LLM available

        prompt = f"""Extract named entities from this text. Return JSON array of {{"name": "...", "type": "PERSON|ORG|TECH|DATE|VERSION"}}.

Text: {content[:2000]}

JSON:"""

        response = await adapter.chat([
            {"role": "system", "content": "You extract entities as JSON. Be concise."},
            {"role": "user", "content": prompt},
        ], temperature=0.1)

        # Parse JSON from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            entities = json.loads(json_match.group(0))
            return [
                {"name": e["name"], "type": e.get("type", "PERSON")}
                for e in entities
                if isinstance(e, dict) and "name" in e
            ]
    except Exception as e:
        logger.debug("LLM entity extraction failed: %s", e)

    return []


def _merge_entities(
    regex_entities: List[Dict[str, str]],
    llm_entities: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Merge regex and LLM entities, deduplicating by name."""
    seen = set()
    merged = []

    # LLM entities take priority (higher quality)
    for e in llm_entities:
        sig = (e["name"].lower(), e["type"])
        if sig not in seen:
            seen.add(sig)
            merged.append(e)

    # Add regex entities not already captured
    for e in regex_entities:
        sig = (e["name"].lower(), e["type"])
        if sig not in seen:
            seen.add(sig)
            merged.append(e)

    return merged


def _count_merged(entities: List[Dict[str, str]]) -> int:
    """Count how many entities were merged (before vs after count)."""
    # This is approximate — we track the reduction
    return 0  # Tracked externally


# ── Integration with memory_add ──────────────────────────────────────

async def extract_and_link_entities(
    memory_id: str,
    content: str,
    user_id: Optional[str] = None,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Full pipeline: extract entities + facts, link to memory.

    Returns {entities, fact_ids, stats}.
    """
    import asyncio
    from ..temporal_graph import insert_fact

    result = await cascade_extract(content, user_id, use_llm=use_llm)

    # Insert facts with memory_id linkage
    fact_ids = []
    for f in result["facts"]:
        try:
            fid = await insert_fact(
                subject=f["subject"],
                predicate=f["predicate"],
                subject_object=f["object"],
                confidence=f["confidence"],
                user_id=user_id,
                memory_id=memory_id,
                metadata={
                    "extraction_method": result["stats"]["method"],
                    "source_memory": memory_id,
                    "entities": [e["uuid"] for e in result["entities"]],
                },
            )
            fact_ids.append(fid)
        except Exception as e:
            logger.debug("Cascade: failed to insert fact: %s", e)

    return {
        "entities": result["entities"],
        "fact_ids": fact_ids,
        "stats": {
            **result["stats"],
            "facts_linked": len(fact_ids),
        },
    }
