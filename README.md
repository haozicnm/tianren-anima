# 天人·Anima

> 记忆即人格。

认知记忆引擎，让 AI Agent 从"安全利己"走向"主动创造"。

## 核心特性

- **天人合一三层架构** — L1 nature 标注 → L2 衰减修正 → L3 天人比排序
- **HSG 五扇区模型** — episodic / semantic / procedural / emotional / reflective
- **MiniMind 小模型集成** — 本地 12M 参数分类器 + 7M 参数编码器，零 API 依赖
- **知识图谱** — SPO 三元组 + 语义近邻边 + k-means 聚类
- **版本链** — 记忆版本控制 + 分叉检测 + 合并
- **跨用户利他池** — 利他记忆跨用户共享，天人比动态调节
- **自愈能力** — 自动检测 embedding 服务、自动降级、自动恢复

## 架构

```
用户对话 → WorkingMemory (FIFO) → Nature Classifier → Fact Extraction
         → add_hsg_memory (5-sector embedding + graph edges)
         → hsg_query (vector + BM25 + graph traversal + decay + tian_ren_ratio)
```

## 安装

```bash
pip install -e .
```

## 使用

```python
from anima import HSGMemory

mem = HSGMemory(user="horizon", use_working_memory=True)

# 写入
await mem.add("天人合一是张载《正蒙》中的命题", nature="altruistic")
await mem.add("我偏好用 Rust 做系统编程", nature="egoistic")

# 刷新工作记忆到长期记忆
await mem.flush()

# 搜索 (天人比: 0=纯利己, 0.5=平衡, 1=纯利他)
results = await mem.search("天人合一", nature="altruistic", tian_ren_ratio=0.7)

# 知识池
pool = await mem.get_knowledge_pool(tian_ren_ratio=0.5)
```

## MiniMind 模型

| 模型 | 参数 | 用途 | 指标 |
|------|------|------|------|
| Nature Classifier | 12.4M | egoistic/hybrid/altruistic 分类 | 86.6% 准确率 |
| Memory Encoder | 7.4M | 256 维向量编码 | R@10=86% |

训练脚本在 `scripts/`。

## 目录结构

```
tianren-anima/
├── src/anima/
│   ├── core/       # db, config, constants, types, vector_store
│   ├── memory/     # hsg, decay, embed, working, version_chain, graph_reorganizer
│   ├── graph/      # temporal_graph (store, query, types)
│   ├── ops/        # extract_facts, extract_entities, altruistic_pool, ingest
│   ├── models/     # MiniMind classifier, encoder (inference)
│   ├── migrations/ # SQL 迁移
│   └── utils/      # vectors, text, keyword, chunking
├── tests/
├── data/           # 训练数据
├── models/         # 训练好的权重
├── scripts/        # 训练脚本
└── references/     # 设计文档
```

## License

Apache 2.0
