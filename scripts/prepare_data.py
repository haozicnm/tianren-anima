#!/usr/bin/env python3
"""
天人·Anima 记忆系统 → MiniMind 训练数据导出

导出三类训练数据:
1. nature_classify.jsonl  — Nature 分类数据 (content → egoistic/hybrid/altruistic)
2. search_pairs.jsonl    — 搜索对数据 (query, positive, negative) 用于对比学习
3. knowledge_facts.jsonl — 知识图谱数据 (subject, predicate, object)
"""

import sqlite3
import json
import os
import random
import hashlib
from collections import Counter
from datetime import datetime

DB_PATH = os.path.expanduser("~/.hermes/openmemory.db")
OUTPUT_DIR = os.path.expanduser("~/minimind-memory/data")

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ─────────────────────────────────────────────
# 1. Nature Classification Dataset
# ─────────────────────────────────────────────

def export_nature_classifier(conn):
    """
    导出 nature 分类训练数据
    
    格式: {"text": "...", "label": "egoistic|hybrid|altruistic"}
    
    问题: nature 分布严重不均
      altruistic: 1324 (72.9%)
      hybrid: 450 (24.8%)
      egoistic: 25 (1.4%)
      NULL: 17 (0.9%)
    
    解决: 
      1. 导出原始数据
      2. 对 egoistic 做数据增强
      3. 对 NULL 用规则预标注
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT id, content, nature 
        FROM memories 
        WHERE content IS NOT NULL AND LENGTH(content) > 10
        ORDER BY id
    """)
    
    rows = cur.fetchall()
    dataset = []
    skipped = 0
    
    for row in rows:
        content = row["content"].strip()
        nature = row["nature"]
        
        # 处理 NULL nature → 用规则预标注
        if nature is None:
            nature = rule_classify(content)
            if nature is None:
                skipped += 1
                continue
        
        # 跳过太短的
        if len(content) < 20:
            skipped += 1
            continue
        
        # 清理内容 (去掉 system prompt 前缀等)
        cleaned = clean_content(content)
        if len(cleaned) < 20:
            skipped += 1
            continue
        
        dataset.append({
            "text": cleaned[:2000],  # 截断到 2000 字符
            "label": nature,
            "id": row["id"],
        })
    
    # 统计
    label_counts = Counter(d["label"] for d in dataset)
    print(f"  原始数据: {len(dataset)} 条 (跳过 {skipped})")
    print(f"  分布: {dict(label_counts)}")
    
    return dataset

def rule_classify(content):
    """规则预标注 NULL nature 的记忆"""
    c = content.lower()
    
    egoistic_patterns = [
        "偏好", "我不", "我喜欢", "我不喜欢", "我的习惯",
        "preference", "i prefer", "i don't", "my habit",
        "密码", "token", "secret", "api key",
        "调试过程", "debug", "排查",
    ]
    altruistic_patterns = [
        "研究表明", "论文", "技术发现", "架构", "最佳实践",
        "研究表明", "research", "paper", "architecture", "best practice",
        "解决方案", "模式", "框架", "原理",
        "毛选", "金刚经", "哲学",
    ]
    
    for p in egoistic_patterns:
        if p in c:
            return "egoistic"
    for p in altruistic_patterns:
        if p in c:
            return "altruistic"
    return "hybrid"  # 默认

def clean_content(content):
    """清理记忆内容"""
    # 去掉 [IMPORTANT: ...] 前缀
    if content.startswith("[IMPORTANT:"):
        end = content.find("]")
        if end > 0:
            content = content[end+1:].strip()
    
    # 去掉 system prompt 标记
    for prefix in ["[SYSTEM]", "[USER]", "[ASSISTANT]"]:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
    
    # 去掉过长的 JSON 块
    if content.startswith("{") and len(content) > 500:
        try:
            json.loads(content)
            return ""  # 纯 JSON 不适合做分类训练
        except:
            pass
    
    return content.strip()

def augment_egoistic(dataset):
    """
    数据增强: egoistic 太少 (25条), 需要扩增
    
    策略:
    1. 同义改写 (手动模板)
    2. 从 altruistic 中提取"伪 egoistic" (带个人色彩的表述)
    """
    egoistic_templates = [
        "我偏好使用 {tech} 来做 {task}",
        "我不喜欢 {approach} 的方式，更倾向于 {alternative}",
        "我的工作习惯是 {habit}",
        "个人经验：{insight}",
        "我发现 {finding} 对我很有用",
        "我的配置：{config}",
        "我习惯用 {tool} 处理 {task}",
        "我不太推荐 {method}，因为 {reason}",
    ]
    
    # 从现有 altruistic 中提取技术名词填充模板
    techs = ["Python", "Rust", "SQLite", "Docker", "Git", "VS Code", "Neovim"]
    tasks = ["数据处理", "API 开发", "部署", "测试", "调试", "文档编写"]
    approaches = ["MVC", "微服务", "单体架构", "TDD", "BDD"]
    alternatives = ["更简洁的方案", "直接用标准库", "自己实现"]
    habits = ["先写测试再写代码", "每天早上看日志", "用 TODO 管理任务"]
    tools = ["terminal", "curl", "jq", "ripgrep", "fzf"]
    methods = ["过度封装", "过早优化", "盲目跟风"]
    reasons = ["增加了复杂度", "收益不大", "维护成本高"]
    
    augmented = []
    for template in egoistic_templates:
        for _ in range(3):  # 每个模板生成 3 个变体
            text = template.format(
                tech=random.choice(techs),
                task=random.choice(tasks),
                approach=random.choice(approaches),
                alternative=random.choice(alternatives),
                habit=random.choice(habits),
                insight=f"{random.choice(techs)} 在 {random.choice(tasks)} 场景下效果很好",
                config=f"{random.choice(techs)} + {random.choice(tools)}",
                tool=random.choice(tools),
                finding=f"{random.choice(techs)} 的 {random.choice(['性能', '易用性', '稳定性'])} 比预期好",
                method=random.choice(methods),
                reason=random.choice(reasons),
            )
            augmented.append({
                "text": text,
                "label": "egoistic",
                "id": f"aug_{hashlib.md5(text.encode()).hexdigest()[:8]}",
                "augmented": True,
            })
    
    return augmented

# ─────────────────────────────────────────────
# 2. Search Pairs Dataset
# ─────────────────────────────────────────────

def export_search_pairs(conn):
    """
    导出搜索对训练数据 (用于对比学习 encoder)
    
    正例: temporal_edges 中的 semantic_neighbor 对
    负例: 随机采样的非近邻记忆
    """
    cur = conn.cursor()
    
    # 获取所有有向量的记忆
    cur.execute("""
        SELECT id, content, nature 
        FROM memories 
        WHERE mean_vec IS NOT NULL AND LENGTH(content) > 20
    """)
    memories = {row["id"]: dict(row) for row in cur.fetchall()}
    memory_ids = list(memories.keys())
    
    # 获取 temporal_edges 作为正例对
    cur.execute("""
        SELECT source_id, target_id, weight
        FROM temporal_edges
        WHERE weight >= 0.6
        ORDER BY weight DESC
    """)
    edges = cur.fetchall()
    
    pairs = []
    seen = set()
    
    for edge in edges:
        src, dst, weight = edge["source_id"], edge["target_id"], edge["weight"]
        
        if src not in memories or dst not in memories:
            continue
        
        pair_key = tuple(sorted([src, dst]))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        
        # 正例对
        positive_pair = {
            "query": memories[src]["content"][:500],
            "positive": memories[dst]["content"][:500],
            "weight": weight,
            "src_id": src,
            "dst_id": dst,
        }
        
        # 随机负例 (与 query 不在同一个簇的)
        neg_id = random.choice(memory_ids)
        attempts = 0
        while neg_id == src or neg_id == dst and attempts < 10:
            neg_id = random.choice(memory_ids)
            attempts += 1
        
        positive_pair["negative"] = memories[neg_id]["content"][:500]
        positive_pair["neg_id"] = neg_id
        
        pairs.append(positive_pair)
        
        if len(pairs) >= 5000:  # 限制数量
            break
    
    print(f"  搜索对: {len(pairs)} 对")
    print(f"  权重分布: min={min(p['weight'] for p in pairs):.3f}, max={max(p['weight'] for p in pairs):.3f}")
    
    return pairs

# ─────────────────────────────────────────────
# 3. Knowledge Facts Dataset
# ─────────────────────────────────────────────

def export_knowledge_facts(conn):
    """导出 SPO 三元组"""
    cur = conn.cursor()
    cur.execute("""
        SELECT tf.subject, tf.predicate, tf.object, tf.confidence,
               m.content as memory_content
        FROM temporal_facts tf
        LEFT JOIN memories m ON tf.memory_id = m.id
        WHERE tf.confidence >= 0.5
        ORDER BY tf.confidence DESC
    """)
    
    facts = []
    for row in cur.fetchall():
        facts.append({
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "confidence": row["confidence"],
            "context": (row["memory_content"] or "")[:200],
        })
    
    print(f"  知识三元组: {len(facts)} 条")
    
    # 谓词分布
    pred_counts = Counter(f["predicate"] for f in facts)
    print(f"  谓词分布: {dict(pred_counts.most_common(10))}")
    
    return facts

# ─────────────────────────────────────────────
# 4. Reflection Pairs (for Distiller training)
# ─────────────────────────────────────────────

def export_reflection_pairs(conn):
    """
    导出反思对: 从记忆簇中构造 (多条记忆 → 一条摘要)
    
    用 temporal_edges 的连通分量作为"簇",
    每个簇的中心记忆作为"摘要"候选
    """
    cur = conn.cursor()
    
    # 构建邻接表
    cur.execute("SELECT source_id, target_id, weight FROM temporal_edges WHERE weight >= 0.5")
    edges = cur.fetchall()
    
    adj = {}
    for e in edges:
        src, dst = e["source_id"], e["target_id"]
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, []).append(src)
    
    # 找连通分量 (简单 BFS)
    visited = set()
    clusters = []
    
    for node in adj:
        if node in visited:
            continue
        cluster = []
        queue = [node]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            cluster.append(n)
            for neighbor in adj.get(n, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(cluster) >= 3:  # 至少 3 个节点才算簇
            clusters.append(cluster)
    
    # 每个簇构造一条训练数据
    cur.execute("SELECT id, content FROM memories WHERE content IS NOT NULL")
    contents = {row["id"]: row["content"] for row in cur.fetchall()}
    
    pairs = []
    for cluster in clusters:
        # 簇内记忆
        cluster_contents = [contents[c][:300] for c in cluster if c in contents]
        if len(cluster_contents) < 3:
            continue
        
        # 中心记忆 (度数最高的节点) 作为"摘要"
        center = max(cluster, key=lambda n: len(adj.get(n, [])))
        summary = contents.get(center, "")[:300]
        
        pairs.append({
            "inputs": cluster_contents[:10],  # 最多 10 条
            "summary": summary,
            "cluster_size": len(cluster),
            "center_id": center,
        })
    
    print(f"  反思簇: {len(clusters)} 个连通分量")
    print(f"  训练对: {len(pairs)} 对 (簇大小 >= 3)")
    if pairs:
        sizes = [p["cluster_size"] for p in pairs]
        print(f"  簇大小: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")
    
    return pairs

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("天人·Anima → MiniMind 训练数据导出")
    print("=" * 60)
    
    conn = connect()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Nature Classification
    print("\n[1/4] Nature 分类数据...")
    nature_data = export_nature_classifier(conn)
    augmented = augment_egoistic(nature_data)
    nature_full = nature_data + augmented
    random.shuffle(nature_full)
    
    # 保存
    path = os.path.join(OUTPUT_DIR, "nature_classify.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in nature_full:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  → 保存: {path} ({len(nature_full)} 条, 含 {len(augmented)} 条增强)")
    
    # 统计最终分布
    final_counts = Counter(d["label"] for d in nature_full)
    print(f"  → 最终分布: {dict(final_counts)}")
    
    # 2. Search Pairs
    print("\n[2/4] 搜索对数据...")
    try:
        search_data = export_search_pairs(conn)
        path = os.path.join(OUTPUT_DIR, "search_pairs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in search_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  → 保存: {path}")
    except Exception as e:
        print(f"  ⚠️ 搜索对导出失败: {e}")
        print(f"  → 跳过，后续用 waypoint 聚类替代")
    
    # 3. Knowledge Facts
    print("\n[3/4] 知识图谱数据...")
    facts_data = export_knowledge_facts(conn)
    path = os.path.join(OUTPUT_DIR, "knowledge_facts.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in facts_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  → 保存: {path}")
    
    # 4. Reflection Pairs
    print("\n[4/4] 反思训练数据...")
    reflection_data = export_reflection_pairs(conn)
    path = os.path.join(OUTPUT_DIR, "reflection_pairs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in reflection_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  → 保存: {path}")
    
    conn.close()
    
    # 汇总
    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        with open(fpath) as fh:
            lines = sum(1 for _ in fh)
        print(f"  {f}: {lines} 条, {size/1024:.1f} KB")

if __name__ == "__main__":
    main()
