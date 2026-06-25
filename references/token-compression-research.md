# Token Compression Research

Two complementary approaches: engineering compression (Headroom) and semantic compression (BabelTele).

---

## 1. Headroom (chopratejas/headroom) — Engineering Compression

> ★39,739 | Apache 2.0 | Rust+Python+TS | Netflix 工程师 Tejas Chopra
> 起因：$287 API 账单，发现 76% token 仅用于读取冗余数据

### Architecture

```
请求 → CacheAligner(检测易失内容) → ContentRouter(类型识别)
    ↓
SmartCrusher(JSON) / CodeAware(AST) / Kompress(BERT) / LogCompressor / DiffCompressor
    ↓
CCR Store (BLAKE3哈希, SQLite/Redis, 30min TTL)
    ↓
压缩后 → LLM → 可 headroom_retrieve 按需还原
```

### Six Compression Algorithms

| Algorithm | Target | Tech | Ratio |
|-----------|--------|------|-------|
| SmartCrusher | JSON arrays | Statistical analysis + lossless CSV/lossy row-drop + CCR sentinel | 70-90% |
| CodeAware | Source code | tree-sitter AST: keep signatures, compress bodies (30+ langs) | 60-80% |
| Kompress | Plain text | ModernBERT ONNX token-level keep/drop classifier (261MB, f1=0.913) | 40-60% |
| LogCompressor | Build logs | Format detection + level classification + conservative dedup | 80-95% |
| DiffCompressor | Git diff | Keep change lines, compress context | 60-80% |
| SearchCompressor | Search results | High-score keep + similarity dedup | 70-85% |

### CCR (Compress-Cache-Retrieve) — Core Innovation

"Lossy on wire, lossless end-to-end" — compress aggressively, store originals with BLAKE3 hash, LLM can call `headroom_retrieve(hash, query)` to get originals on demand. Max 3 retrieval rounds.

4 components: Tool Injection, Response Handler (intercepts LLM calls), Context Tracker (cross-turn memory), Batch Processor.

### CacheAligner — KV Cache Optimization

Detect-only (PR-A2: never mutates). Uses stdlib parsers (uuid, datetime, base64) to identify volatile content in system prompts. Warnings only — dynamic context should go to user messages (live zone), not system prompt (cache zone).

### Integration Modes

| Mode | Command | Friction |
|------|---------|----------|
| MCP server | `headroom mcp serve` | Zero — 3 tools: compress/retrieve/stats |
| HTTP proxy | `headroom proxy --port 8787` | Change base_url only |
| Wrap CLI | `headroom wrap claude\|codex\|cursor\|aider\|copilot` | One command |
| Python/TS lib | `compress(messages)` | Direct API |

### Hermes MCP Setup (2026-06-20 verified)

Install: `pip install headroom-ai` (v0.26.0)

Config in `~/.hermes/config.yaml`:
```yaml
mcp_servers:
  headroom:
    command: /home/horizon/.hermes/hermes-agent/venv/bin/headroom
    args:
    - mcp
    - serve
    timeout: 60
    connect_timeout: 30
```

⚠️ `hermes config set` stores lists as YAML strings. Fix with Python yaml.dump (see pitfall #37 in SKILL.md).

Tools registered: `mcp_headroom_headroom_compress`, `mcp_headroom_headroom_retrieve`, `mcp_headroom_headroom_stats`

### Limitations

- ML model dependency (261MB download on first use)
- Rust PyO3 extensions needed for SmartCrusher/Diff/Log/Search compressors
- CCR overhead: storage + multi-round retrieval latency
- Short text (<100 chars) not compressed
- Already-cached prefixes not compressed (prefix-frozen)

---

## 2. BabelTele — Semantic "AI Language" Compression

> arXiv 2606.19857 (2026-06-18) | 上海交大 + 悉尼大学 + 合肥工大 + 西安交大 + 南京大学
> 论文：《大语言模型并不总是需要可读语言》

### Core Idea

当前 LLM 交互使用**为人类设计的自然语言**，包含大量冗余。BabelTele 反其道行之——**为模型设计语言**，不考虑人类可读性。

融合多语言词汇 + 数学符号 + 逻辑运算符 + 表情符号，生成高度密集的"模型语言"。

### Results

| Metric | Value |
|--------|-------|
| Compression ratio | **27.9%** (text reduced to ~1/4) |
| Semantic retention | **99.5%** |
| Multi-agent communication | ~40% token reduction, >96% task completion |
| Zero-shot transfer | One model's compressed text understood by another without training |

### Comparison

| Method | Compression | Semantic |
|--------|-------------|----------|
| NL summary | ~50% | ~90% |
| LLMLingua-2 | ~35% | ~95% |
| **BabelTele** | **27.9%** | **99.5%** |

### Applicability to TianRen-Anima

| Use Case | Fit | Notes |
|----------|-----|-------|
| Memory injection compression | ✅ High | Compress memories to 1/3 before injecting into system prompt |
| Cross-session memory transfer | ✅ High | Store memories in AI language, models understand on retrieval |
| Complement to Headroom | ✅ Synergy | Headroom handles structured data (JSON/code/logs), BabelTele handles natural language |

**Combined potential**: Headroom (tool output compression) + BabelTele (memory injection compression) = 60-80% total token savings.

### Status

**Paper only (arXiv preprint)** — no code released yet. Worth monitoring for implementation.

---

## Applicability to Hermes + TianRen-Anima

| Aspect | Headroom | BabelTele |
|--------|----------|-----------|
| Structured data (JSON/code/logs) | ✅ Direct | ❌ Not designed for |
| Natural language memory | ⚠️ Partial (Kompress 40-60%) | ✅ Optimized (72% compression) |
| MCP integration | ✅ Ready | ❌ Not available |
| Code available | ✅ pip install headroom-ai | ❌ Paper only |
| Current Hermes cache hit | 98.3% (56:1) — prefix stable | N/A |

### Recommended Priority

1. **MCP mode** (Headroom) — zero friction, already configured
2. **Monitor BabelTele** — watch for code release, integrate into memory injection pipeline
3. **Proxy mode** (Headroom) — change Hermes base_url for full compression
4. **Combined approach** — Headroom for tool outputs + BabelTele for memory = maximum savings
