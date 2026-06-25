"""
P0.2: Fact Extraction Bridge — cognee-inspired SPO extraction pipeline.

Extracts simple subject-predicate-object triples from memory content
and links them to the source memory via temporal_facts.memory_id.

The extraction is regex-based (no LLM calls) for speed and cost.
For more complex entity extraction, the cognee-style LLM pipeline
can be added in P1 (cascaded entity extraction).
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Any, Optional

from ..core.db import db, q
from ..temporal_graph import insert_fact, get_facts_for_memory

logger = logging.getLogger(__name__)

# ── SPO Extraction Patterns ──────────────────────────────────────────

# Pattern: <subject> <verb> <object> for common fact structures
FACT_PATTERNS = [
    # "X is Y" / "X was Y" — match both capitalized and lowercase proper nouns
    (r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(is|was|are|were)\s+(a\s+|an\s+|the\s+)?([a-zA-Z][\w\s]{1,40}?)(?:[.,;]|\s+(?:at|in|for|with|from|by|and|but|which|$))',
     'type_of'),
    # "X has Y"
    (r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(has|have|had)\s+(a\s+|an\s+|the\s+)?([A-Za-z][\w\s]{2,30}?)(?:[.,;]|\s+and|\s+but|\s+which)',
     'has'),
    # "X works at Y" / "X lives in Y"
    (r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(works?\s+(?:at|for|in|as)|lives?\s+(?:in|at)|studies?\s+(?:at|in))\s+([A-Za-z][\w\s]{2,40}?)(?:[.,;]|\s+and|$)',
     'relation'),
    # "X uses Y" / "X built Y" / "X created Y"
    (r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(uses?|built|created|developed|designed|implemented|deployed|chose|selected|prefers?)\s+(a\s+|an\s+|the\s+)?([A-Za-z][\w\s]{2,40}?)(?:[.,;]|\s+and|\s+but|$)',
     'action'),
    # "X said Y" / "X thinks Y" / "X believes Y"
    (r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(said|says|stated|thinks?|believes?|argues?|claims?)\s+(?:that\s+)?(.{5,60}?)(?:[.,;]|\s+and\s+[A-Z]|$)',
     'statement'),

    # ── Chinese SPO Patterns ──────────────────────────────────────
    # "X 是 Y" — is/type_of (allow spaces in subject for multi-word names)
    (r'([A-Za-z][\w\-\. ]{1,30}|[\u4e00-\u9fff]{2,15})\s*是\s*(?:一个|一种|一款)?\s*([\u4e00-\u9fff\w][\u4e00-\u9fff\w ]{1,40}?)(?:[，。；,\.;\n]|$)',
     'type_of'),
    # "X 支持/包含/使用/需要/基于 Y" — action verbs
    (r'([\u4e00-\u9fff\w][\u4e00-\u9fff\w \-\.]{1,20})\s*(支持|包含|使用|需要|基于|采用|提供|实现|运行|部署|开发|创建|构建|设计|维护|管理)\s*(?:了|着|过)?\s*([\u4e00-\u9fff\w][\u4e00-\u9fff\w ]{1,30}?)(?:[，。；,\.;\n]|$)',
     'action'),
    # "X 由 Y 创建/开发/设计" — passive: created by
    (r'([\u4e00-\u9fff\w][\u4e00-\u9fff\w \-\.]{1,20})\s*由\s*([\u4e00-\u9fff\w][\u4e00-\u9fff\w ]{1,20})\s*(创建|开发|设计|构建|维护|管理)',
     'relation'),
    # "X 的 Y 是 Z" — possessive fact
    (r'([A-Za-z][\w\-\.]{1,20}|[\u4e00-\u9fff]{2,8})\s*的\s*([\u4e00-\u9fff]{2,8})\s*是\s*([\u4e00-\u9fff\w][\u4e00-\u9fff\w ]{1,30}?)(?:[，。；,\.;\n]|$)',
     'has'),
    # "X 被重命名为/转换为/升级为 Y" — passive transformation
    (r'([\u4e00-\u9fff\w][\u4e00-\u9fff\w \-\.]{1,20})\s*被\s*(?:重命名为|转换为|升级为|迁移到|改为)\s*([\u4e00-\u9fff\w][\u4e00-\u9fff\w \-\.]{1,20})',
     'action'),
    # "X → Y" — arrow transformation
    (r'([\u4e00-\u9fff\w][\u4e00-\u9fff\w \-\.]{1,20})\s*(?:→|->)\s*([\u4e00-\u9fff\w][\u4e00-\u9fff\w \-\.]{1,20})',
     'action'),
    # English mixed with Chinese: "X (tool) 是/支持 Y" — require subject to be ASCII-dominated
    (r'([A-Za-z][\w\-\.]{2,30})\s*(?:是|支持|使用|包含|提供)\s*([\u4e00-\u9fff\w][\u4e00-\u9fff\w ]{1,30}?)(?:[，。；,\.;\n]|$)',
     'action'),
]


def extract_candidate_facts(text: str) -> List[Dict[str, Any]]:
    """Extract candidate SPO triples from text using regex patterns.
    
    Returns list of {subject, predicate, object, relation_type, confidence}.
    """
    facts = []
    seen = set()
    
    for pattern, rel_type in FACT_PATTERNS:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            # Normalize groups by rel_type
            subj = pred = obj = None
            if rel_type == 'type_of':
                subj = groups[0].strip()
                # English: 4 groups (subj, verb, article, obj); Chinese: 2 groups (subj, obj)
                if len(groups) >= 5 and groups[3]:
                    obj = groups[3].strip()
                elif len(groups) >= 5 and groups[4]:
                    obj = groups[4].strip()
                else:
                    obj = groups[-1].strip()
                pred = "is_a"
                conf = 0.7
            elif rel_type == 'has':
                subj = groups[0].strip()
                obj = groups[-1].strip()
                pred = "has"
                conf = 0.6
            elif rel_type == 'relation':
                subj = groups[0].strip()
                pred = groups[-1].strip().replace(' ', '_')
                obj = groups[1].strip() if len(groups) == 3 else groups[2].strip()
                conf = 0.65
            elif rel_type == 'action':
                subj = groups[0].strip()
                pred = groups[1].strip()
                obj = groups[-1].strip()
                conf = 0.7
            elif rel_type == 'statement':
                subj = groups[0].strip()
                pred = groups[1].strip()
                obj = groups[2].strip().rstrip('.').strip()
                conf = 0.5
            else:
                continue
            
            # Deduplicate
            sig = f"{subj}|{pred}|{obj}"
            if sig in seen:
                continue
            seen.add(sig)
            
            # Filter out low-quality extractions
            if len(subj) < 2 or len(obj) < 2:
                continue
            if len(subj) > 80 or len(obj) > 200:
                continue
            # Skip if subject/object look like noise
            if re.match(r'^(the|a|an|this|that|it|they|he|she|we|you|i)$', subj, re.I):
                continue
            if re.match(r'^(的|了|在|是|有|和|与|或|但|如果|因为|所以|可以|需要|应该|已经|还是|就是|不是|没有|这个|那个|什么|怎么|为什么)$', subj):
                continue
            
            facts.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "relation_type": rel_type,
                "confidence": conf,
            })
    
    return facts


async def extract_and_link_facts(
    memory_id: str,
    content: str,
    user_id: Optional[str] = None,
    max_facts: int = 10,
) -> List[str]:
    """Extract SPO facts from memory content and link them to the memory.
    
    Called after each memory_add to build the knowledge graph.
    
    Returns list of fact IDs created.
    """
    candidates = extract_candidate_facts(content)
    
    if not candidates:
        return []
    
    # Limit number of facts per memory to avoid explosion
    candidates = candidates[:max_facts]
    
    fact_ids = []
    for c in candidates:
        try:
            fid = await insert_fact(
                subject=c["subject"],
                predicate=c["predicate"],
                subject_object=c["object"],
                confidence=c["confidence"],
                user_id=user_id,
                memory_id=memory_id,
                metadata={
                    "extraction_method": "regex_spo",
                    "source_memory": memory_id,
                },
            )
            fact_ids.append(fid)
        except Exception as e:
            logger.debug("Fact extraction: failed to insert fact for memory %s: %s", memory_id[:16], e)
    
    if fact_ids:
        logger.debug("Fact extraction: linked %d facts to memory %s", len(fact_ids), memory_id[:16])
    
    return fact_ids


async def get_memory_knowledge_graph(memory_id: str) -> Dict[str, Any]:
    """Get the knowledge graph (facts + waypoints) for a memory."""
    # Get linked temporal facts
    facts = await get_facts_for_memory(memory_id)
    
    # Get waypoint connections (incoming + outgoing)
    incoming = db.fetchall(
        "SELECT src_id, weight FROM waypoints WHERE dst_id=? ORDER BY weight DESC LIMIT 20",
        (memory_id,)
    )
    outgoing = db.fetchall(
        "SELECT dst_id, weight FROM waypoints WHERE src_id=? ORDER BY weight DESC LIMIT 20",
        (memory_id,)
    )
    
    return {
        "memory_id": memory_id,
        "facts": facts,
        "incoming_edges": [{"from": r["src_id"], "weight": r["weight"]} for r in incoming],
        "outgoing_edges": [{"to": r["dst_id"], "weight": r["weight"]} for r in outgoing],
    }
