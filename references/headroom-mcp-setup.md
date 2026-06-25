# Headroom MCP Integration for Hermes

## Setup Steps

### 1. Install headroom-ai

```bash
/home/horizon/.hermes/hermes-agent/venv/bin/pip install headroom-ai
```

Version 0.26.0 as of 2026-06-20. Pulls in: litellm, tiktoken, regex, fastuuid, ast-grep-cli.

### 2. Configure MCP server in Hermes

```bash
hermes config set mcp_servers.headroom.command "/home/horizon/.hermes/hermes-agent/venv/bin/headroom"
hermes config set mcp_servers.headroom.args '["mcp", "serve"]'
hermes config set mcp_servers.headroom.timeout 60
hermes config set mcp_servers.headroom.connect_timeout 30
```

**PITFALL**: `hermes config set` stores list args as YAML strings (`'[\"mcp\", \"serve\"]'`), not lists. The MCP client may misparse this. Fix with Python:

```python
import yaml
with open('/home/horizon/.hermes/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['mcp_servers']['headroom']['args'] = ['mcp', 'serve']
with open('/home/horizon/.hermes/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
```

### 3. Restart Gateway

Cannot restart from inside Gateway. Use cron job:

```python
cronjob(action='create', name='headroom-mcp-restart',
    schedule='<ISO timestamp 1min from now>',
    no_agent=True, script='hermes gateway restart')
```

### 4. Verify

```bash
hermes tools | grep headroom
# Expected: mcp_headroom_headroom_compress, mcp_headroom_headroom_retrieve, mcp_headroom_headroom_stats
```

## MCP Tools Provided

| Tool | Function |
|------|----------|
| `headroom_compress` | Compress content on-demand (JSON/code/text/logs) |
| `headroom_retrieve` | Retrieve original uncompressed content by hash |
| `headroom_stats` | Session compression statistics |

## Proxy Mode (Full Auto-Compression)

For automatic compression of ALL API traffic (not just manual tool calls):

```bash
headroom proxy --port 8787 --mode cache
```

Then point Hermes base_url to proxy. But this conflicts with DeepSeek's native caching — the proxy rewrites request structure, which may break prefix-cache alignment.

**Recommendation**: Use MCP mode for manual compression of specific content (tool outputs, long text). Skip proxy mode for now — DeepSeek cache hit rate is already 98.3%.

## Modes

| Mode | Priority | Behavior |
|------|----------|----------|
| `token` (default) | Max compression | Prior turns may be rewritten |
| `cache` | Prefix stability | Freeze prior turns for cache hit |

## Compression Rates by Content Type

| Type | Rate | Algorithm |
|------|------|-----------|
| JSON arrays | 70-90% | SmartCrusher (Rust) |
| Source code | 60-80% | tree-sitter AST |
| Build logs | 80-95% | LogCompressor |
| Git diff | 60-80% | DiffCompressor |
| Search results | 70-85% | SearchCompressor |
| Plain text | 40-60% | ModernBERT ML model |
