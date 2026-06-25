# WSL2 开机自启配置

Hermes Gateway 依赖 WSL2 内的 systemd 用户服务自启动。完整启动链：

```
Windows 开机 → WSL2 启动 → systemd (user) → hermes-gateway.service
```

## 当前配置

### wsl.conf (`/etc/wsl.conf`)
```ini
[boot]
systemd=true
```

### .wslconfig (`C:\Users\haozi\.wslconfig`)
```ini
[wsl2]
networkingMode=mirrored
memory=8GB
```

### systemd 用户服务
```
~/.config/systemd/user/hermes-gateway.service
```
- Symlink in `default.target.wants/` → enabled ✅
- `Restart=always`, `RestartSec=5`
- Linger 必须启用：`loginctl enable-linger horizon`

### 验证当前状态
```bash
# Gateway 是否由 systemd 管理
cat /proc/$(pgrep -f "gateway run")/status | grep PPid
# 应为 113 (systemd --user)

# 服务状态
export XDG_RUNTIME_DIR=/run/user/$(id -u)
systemctl --user status hermes-gateway.service
```

## ⚠️ 关键缺口：WSL 不会 Windows 开机自启

WSL2 默认不在 Windows 启动时自动启动。需要 Windows 侧机制：

### 方案 A：Windows 计划任务（推荐）
```powershell
# PowerShell (管理员)
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Deepin -- echo started"
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName "WSL-Deepin-AutoStart" -Action $action -Trigger $trigger -Settings $settings
```

### 方案 B：Startup 脚本
在 `C:\Users\haozi\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\` 放 `.bat`：
```bat
wsl -d Deepin -- echo "WSL started"
```

### 方案 C：Windows 服务（NSSM）
用 NSSM 注册 WSL 为 Windows 服务，支持开机自启 + 崩溃重启。

## .profile 保活机制（systemd 的 fallback）

当 systemd user session 不可用时（D-Bus 未启动），`.profile` 中的保活脚本是唯一可靠的自启方式：

```bash
# ~/.profile 中的 gateway 保活逻辑
GATEWAY_PIDFILE="$HOME/.hermes/gateway.pid"
if [ ! -f "$GATEWAY_PIDFILE" ] || ! kill -0 "$(cat "$GATEWAY_PIDFILE")" 2>/dev/null; then
    nohup /home/horizon/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run --replace > /dev/null 2>&1 &
    echo $! > "$GATEWAY_PIDFILE"
fi
```

**触发条件**：用户登录（打开 WSL 终端窗口）时执行。
**PID 锁**：防止重复启动。
**`--replace`**：替换已有进程。

⚠️ 如果不打开终端，gateway 不会启动。这是 systemd 失效时的降级方案。

## D-Bus Session 不可达诊断

`systemctl --user` 报 `Failed to connect to bus: No such file or directory` 表示 user D-Bus session 未启动。

**诊断**：
```bash
# 检查 XDG_RUNTIME_DIR
ls -la /run/user/$(id -u)/
# 如果目录不存在，systemd user session 没启动

# 检查 D-Bus socket
ls /run/user/$(id -u)/bus 2>/dev/null || echo "D-Bus socket 不存在"

# 检查 gateway 父进程
cat /proc/$(pgrep -f "gateway run")/status | grep PPid
# PPid=1 → 由 init/systemd 管理（正常）
# PPid=<其他> → 由 shell 启动（.profile fallback）
```

**修复**：
```bash
# 需要 sudo
sudo systemctl start user@1000.service
# 之后 systemctl --user 就可用了
systemctl --user status hermes-gateway.service
```

## Gateway 重启限制

**从 gateway 内部无法重启自己**：所有终端命令都是 gateway 的子进程，kill gateway 会连带杀掉当前会话。

**重启方式**：
1. **独立终端**：`wsl -d Deepin -- bash -c "kill -9 \$(pgrep -f 'hermes.*gateway run')"`
2. **Windows PowerShell**：`wsl -d Deepin -- kill -9 $(wsl -d Deepin -- pgrep -f 'hermes.*gateway run')`
3. **重启 WSL**：`wsl --shutdown` 然后重新打开终端（触发 .profile）
4. **systemctl**（需要 D-Bus 可用）：`systemctl --user restart hermes-gateway.service`

## 环境信息
- WSL 内核: 6.6.87.2-microsoft-standard-WSL2
- 发行版: Deepin 25
- CPU: AMD Ryzen 7 5800H (16 核)
- RAM: 8GB (WSL 限制)
- GPU: 无直通 (LM Studio 跑 CPU)
- Docker: v26.1.5 运行中
