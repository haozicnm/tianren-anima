# MiniMind 训练管线 — 天人·Anima 记忆系统本地小模型

## 概述

用 MiniMind (64M LLM) 的 Transformer 架构训练本地小模型，替代大模型 API 做记忆操作：
1. **Nature Classifier** — 分类记忆为 egoistic/hybrid/altruistic (已完成, 86.6%)
2. **Memory Encoder** — 替代 bge-m3 embedding 服务 (待做)
3. **Memory Distiller** — 替代反思引擎的大模型调用 (待做)

## 项目路径

```
~/minimind-memory/
├── data/
│   ├── nature_classify.jsonl      # Nature 分类训练数据 (含增强)
│   ├── search_pairs.jsonl         # 搜索对 (对比学习 encoder)
│   ├── knowledge_facts.jsonl      # SPO 三元组
│   └── reflection_pairs.jsonl     # 反思训练数据
├── models/
│   ├── nature_classifier.pt       # 训练好的分类器权重
│   └── tokenizer.json             # 字符级 tokenizer
├── prepare_data.py                # 数据导出脚本
├── augment_data.py                # 数据增强脚本
└── train_classifier.py            # 分类器训练脚本
```

## 训练数据来源

从 `~/.hermes/openmemory.db` 导出:

| 数据集 | 来源 | 规模 |
|--------|------|------|
| nature_classify | memories 表 (nature 字段) | 1762→1992 (含增强) |
| search_pairs | temporal_edges (weight≥0.6) | 5000 对 |
| knowledge_facts | temporal_facts | 418 条 |
| reflection_pairs | temporal_edges 连通分量 | 2 对 (簇太大连通) |

## Nature 分类器训练结果

| 指标 | 数值 |
|------|------|
| 模型参数 | 12.4M (4层, hidden=512, heads=8) |
| 验证准确率 | 86.6% |
| egoistic F1 | 0.899 (最佳) |
| hybrid F1 | 0.667 (最难分, 边界模糊) |
| altruistic F1 | 0.918 |
| 训练时间 | 2.5 分钟 (RTX 3060, 30 epochs) |

### 数据分布问题

原始分布严重不均: altruistic 72.9% vs egoistic 1.4%

增强策略:
1. **模板填充** (200条): 从现有 egoistic 提取关键词, 填充到 EGOISTIC_PATTERNS 模板
2. **Pseudo-extraction** (23条): 从 hybrid 中提取含第一人称的句子
3. **同义替换** (7条): SYNONYMS 字典替换

最终分布: altruistic 65.3%, hybrid 20.9%, egoistic 13.8%

### 过拟合问题

train_acc 98% vs val_acc 86%, 明显过拟合。解决方案:
- 更多训练数据 (当前 1992 条偏少)
- 加 dropout (当前 0.1)
- 用 BPE tokenizer 替代字符级 (MiniMind 自带)
- Label smoothing

## MiniMind 架构适配要点

### RoPE 维度修复

MiniMind 的 `precompute_freqs_cis` 输出需要 cat 两倍:
```python
# 正确: 输出 [seq, head_dim]
cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
# 错误: 输出 [seq, head_dim//2], 会导致 q*k 维度不匹配
```

### Attention Mask 与 fp16

`scaled_dot_product_attention` 的 attn_mask 必须与 query dtype 一致:
```python
# 正确: float mask with -inf
mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
mask = (1.0 - mask) * torch.finfo(torch.float16).min
# 错误: long int mask, 会报 dtype mismatch
```

### PyTorch 2.x API 变更

```python
# 正确 (PyTorch 2.x)
from torch.amp import autocast, GradScaler
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    ...

# 旧写法 (deprecated)
from torch.cuda.amp import autocast, GradScaler
```

## 类别权重计算

处理不平衡分布的关键: loss 函数加 class weights
```python
weights = [total / (count[label] + 1) for label in labels]
weights = weights / sum(weights) * len(labels)  # 归一化
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

## 数据增强模式 (egoistic 不足时)

```python
# 1. 模板填充
EGOISTIC_PATTERNS = [
    "我偏好{tech}",
    "我不喜欢{approach}，更倾向于{alternative}",
    "我的{tool}配置是{config}",
    ...
]
FILL = {"tech": ["Python", "Rust", ...], "task": [...], ...}

# 2. 从 hybrid 提取 pseudo-egoistic
# 关键词: "我", "我们", "用户说", "用户要求", "I think", "I prefer"

# 3. 同义替换
SYNONYMS = {"喜欢": ["偏好", "倾向", "偏爱"], "修复": ["解决", "处理", "修正"], ...}
```
