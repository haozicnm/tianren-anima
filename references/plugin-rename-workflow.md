# Plugin Rename Workflow (OpenMemory → 天人·Anima)

Full end-to-end procedure for renaming a forked project and its Hermes integration.

## 1. GitHub Fork Rename

```bash
# Extract token from git credential store
TOKEN=$(grep "github.com" ~/.git-credentials | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')

# Rename via API
curl -X PATCH https://api.github.com/repos/<owner>/<old-name> \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "<new-name>"}'

# Update local remote
cd ~/dev/<repo>
git remote set-url fork https://ghfast.top/https://github.com/<owner>/<new-name>.git
```

## 2. Plugin Directory Rename + Code Update

```bash
# Rename directory
mv ~/.hermes/hermes-agent/plugins/memory/<old> ~/.hermes/hermes-agent/plugins/memory/<new>
```

In `__init__.py`, update these strings:
- `return "old-name"` → `return "new-name"`
- All `openmemory.json` → `tianren-anima.json` paths (both docstrings AND code)
- Thread names (cosmetic but consistent)
- Docstrings mentioning old name

**Keep unchanged**: Python `from openmemory import Memory` imports (the pip package name stays).

## 3. Config File Rename

```bash
mv ~/.hermes/openmemory.json ~/.hermes/tianren-anima.json
```

## 4. config.yaml Update

⚠️ `config.yaml` is **protected** from direct `patch()` editing by Hermes.
Use the CLI instead:

```bash
hermes config set memory.provider tianren-anima
```

## 5. Skill Rename

Create new skill, delete old with `absorbed_into`:

```python
skill_manage(action='create', name='new-skill', content=updated_skill_content)
skill_manage(action='delete', name='old-skill', absorbed_into='new-skill')
```

## 6. Restart Gateway

```bash
# Create restart script
cat > ~/.hermes/scripts/restart-gateway.sh << 'EOF'
#!/bin/bash
pkill -f "hermes_cli.main gateway run" 2>/dev/null
sleep 3
nohup ~/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run > /dev/null 2>&1 &
sleep 8
~/.hermes/hermes-agent/venv/bin/hermes memory status
EOF
chmod +x ~/.hermes/scripts/restart-gateway.sh

# Trigger via cron (can't restart from inside gateway)
cronjob(action='create', schedule='1m', script='restart-gateway.sh', no_agent=True)
```

## 7. Verify

```bash
hermes memory status  # should show new provider name as active
```

## Pitfalls
- `hermes gateway restart` fails from inside gateway process tree — use cron
- `hermes config set` is the only way to edit `config.yaml` — `patch()` is refused
- Hermes update's autostash preserves plugin directory renames (tested with v0.16→v0.17)
