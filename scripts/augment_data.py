#!/usr/bin/env python3
"""
Nature 分类数据增强

问题: egoistic 只有 45 条 (2.6%), 需要扩增到至少 200 条
策略:
1. 模板填充 (从现有 egoistic 记忆中提取关键词填充模板)
2. 回译风格改写 (中文同义替换)
3. 上下文注入 (从 hybrid 中提取带个人色彩的句子)
"""

import json
import os
import random
import re
from collections import Counter

DATA_DIR = os.path.expanduser("~/minimind-memory/data")

def load_nature_data():
    path = os.path.join(DATA_DIR, "nature_classify.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

# ─────────────────────────────────────────────
# 从现有 egoistic 记忆中提取模式
# ─────────────────────────────────────────────

EGOISTIC_PATTERNS = [
    # 个人偏好
    "我偏好{tech}",
    "我喜欢用{tech}做{task}",
    "我不喜欢{approach}，更倾向于{alternative}",
    "我的习惯是{habit}",
    "个人经验：{tech}在{scenario}下表现更好",
    "我选择{tech}是因为{reason}",
    "我不推荐{method}，因为{reason}",
    
    # 个人状态
    "我今天在调试{tech}的{issue}",
    "我刚发现{finding}",
    "我的{tool}配置是{config}",
    "我需要修复{issue}",
    "我在用{tech}处理{task}时遇到了{error}",
    
    # 个人决策
    "我决定用{tech}替代{old_tech}",
    "我选择{approach}而不是{alternative}，因为{reason}",
    "经过测试，我更喜欢{tech}",
    "我放弃了{method}，改用{alternative}",
    
    # 记忆系统操作
    "我存了一条关于{topic}的记忆",
    "我更新了{config}的配置",
    "我修复了{component}的{issue}",
    "我删除了过时的{item}",
]

# 填充词库
FILL = {
    "tech": [
        "Python", "Rust", "TypeScript", "Go", "SQLite", "PostgreSQL", "Redis",
        "Docker", "Kubernetes", "Nginx", "VS Code", "Neovim", "tmux", "Git",
        "Hermes Agent", "MiMo", "LM Studio", "OpenMemory", "Axum", "Tauri",
        "Flutter", "Vue.js", "React", "FastAPI", "Express", "tokio",
    ],
    "task": [
        "数据处理", "API开发", "前端开发", "部署", "测试", "调试", "文档编写",
        "性能优化", "数据库迁移", "CI/CD配置", "日志分析", "监控告警",
        "记忆管理", "向量搜索", "知识图谱", "模型训练",
    ],
    "approach": [
        "MVC", "微服务", "单体架构", "TDD", "BDD", "DDD",
        "OOP", "函数式", "响应式", "事件驱动",
    ],
    "alternative": [
        "更简洁的方案", "直接用标准库", "自己实现", "用现有的轮子",
        "Rust重写", "脚本自动化", "配置驱动",
    ],
    "habit": [
        "先写测试再写代码", "每天早上看日志", "用TODO管理任务",
        "commit前跑lint", "写完代码先review再提交",
        "调试时先看错误日志", "设计前先画架构图",
    ],
    "scenario": [
        "高并发", "大数据量", "低延迟", "内存受限", "网络不稳定",
        "生产环境", "开发环境", "CI环境",
    ],
    "reason": [
        "性能更好", "生态更成熟", "学习曲线低", "社区活跃",
        "类型安全", "零成本抽象", "编译时检查", "内存安全",
    ],
    "method": [
        "过度封装", "过早优化", "盲目跟风", "不写测试",
        "硬编码配置", "忽略错误处理", "不做代码审查",
    ],
    "issue": [
        "内存泄漏", "性能瓶颈", "并发问题", "类型错误",
        "配置错误", "依赖冲突", "编译失败", "测试超时",
    ],
    "tool": [
        "terminal", "curl", "jq", "ripgrep", "fzf", "bat", "exa",
        "htop", "tmux", "zsh", "starship",
    ],
    "config": [
        "Rust编译参数", "Nginx反向代理", "Docker网络", "Git hooks",
        "ESLint规则", "TypeScript strict mode", "SQLite WAL模式",
    ],
    "topic": [
        "天人合一", "衰减模型", "向量搜索", "知识图谱",
        "版本控制", "CI/CD", "性能优化",
    ],
    "component": [
        "反思引擎", "embedding服务", "搜索模块", "分类器",
        "数据库连接", "缓存层", "API网关",
    ],
    "finding": [
        "SQLite WAL模式比journal快2x", "Rust编译比Go慢但运行更快",
        "向量搜索用HNSW比暴力搜索快10x", "bge-m3在短文本上效果更好",
        "记忆衰减模型比LRU更符合人类遗忘规律",
    ],
    "error": [
        "超时错误", "内存溢出", "死锁", "连接断开",
        "编码问题", "权限错误", "版本不兼容",
    ],
    "old_tech": [
        "JavaScript", "MySQL", "MongoD", "Webpack", "REST", "GraphQL",
    ],
    "item": [
        "配置", "缓存", "日志", "临时文件", "旧版本",
    ],
}

def fill_template(template):
    """随机填充模板"""
    result = template
    for key, values in FILL.items():
        if "{" + key + "}" in result:
            result = result.replace("{" + key + "}", random.choice(values))
    return result

def generate_augmented_egoistic(n=200):
    """生成 n 条增强的 egoistic 数据"""
    samples = []
    seen = set()
    
    while len(samples) < n:
        template = random.choice(EGOISTIC_PATTERNS)
        text = fill_template(template)
        
        # 去重
        key = text[:50]
        if key in seen:
            continue
        seen.add(key)
        
        samples.append({
            "text": text,
            "label": "egoistic",
            "id": f"aug_ego_{len(samples):04d}",
            "augmented": True,
            "method": "template_fill",
        })
    
    return samples

# ─────────────────────────────────────────────
# 从 hybrid 中提取 pseudo-egoistic
# ─────────────────────────────────────────────

def extract_pseudo_egoistic(data, n=50):
    """
    从 hybrid 中找带个人色彩的句子
    特征: 包含"我"、"我们"、"用户"等第一人称
    """
    egoistic_keywords = [
        "我", "我们", "用户说", "用户要求", "用户纠正",
        "I think", "I prefer", "we should", "let me",
    ]
    
    candidates = []
    for d in data:
        if d["label"] != "hybrid":
            continue
        text = d["text"]
        # 检查是否包含个人色彩关键词
        for kw in egoistic_keywords:
            if kw in text:
                # 提取包含关键词的句子
                sentences = re.split(r'[。！？\n]', text)
                for sent in sentences:
                    if kw in sent and len(sent) > 15 and len(sent) < 200:
                        candidates.append({
                            "text": sent.strip(),
                            "label": "egoistic",
                            "id": f"pseudo_ego_{len(candidates):04d}",
                            "augmented": True,
                            "method": "extract_from_hybrid",
                            "source_id": d.get("id"),
                        })
                        break
                break
    
    # 随机采样 n 条
    if len(candidates) > n:
        candidates = random.sample(candidates, n)
    
    return candidates

# ─────────────────────────────────────────────
# 回译风格改写 (中文同义替换)
# ─────────────────────────────────────────────

SYNONYMS = {
    "喜欢": ["偏好", "倾向", "偏爱", "青睐"],
    "不喜欢": ["不推荐", "不建议", "不太认可"],
    "使用": ["采用", "运用", "利用", "借助"],
    "修复": ["解决", "处理", "修正", "调整"],
    "发现": ["注意到", "观察到", "意识到"],
    "选择": ["决定", "倾向", "选用"],
    "问题": ["毛病", "缺陷", "不足", "瓶颈"],
    "方案": ["策略", "办法", "路径", "思路"],
    "性能": ["效率", "速度", "表现"],
    "配置": ["设置", "参数", "选项"],
}

def synonym_augment(text, n=3):
    """同义替换增强"""
    results = []
    for _ in range(n):
        new_text = text
        for word, synonyms in SYNONYMS.items():
            if word in new_text and random.random() < 0.5:
                new_text = new_text.replace(word, random.choice(synonyms), 1)
        if new_text != text:
            results.append(new_text)
    return results

def augment_existing_egoistic(data, n_per_sample=3):
    """对现有 egoistic 记忆做同义替换"""
    augmented = []
    for d in data:
        if d["label"] != "egoistic" or d.get("augmented"):
            continue
        variants = synonym_augment(d["text"], n=n_per_sample)
        for v in variants:
            augmented.append({
                "text": v,
                "label": "egoistic",
                "id": f"syn_aug_{len(augmented):04d}",
                "augmented": True,
                "method": "synonym_replace",
                "source_id": d.get("id"),
            })
    return augmented

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Nature 分类数据增强")
    print("=" * 60)
    
    # 加载原始数据
    data = load_nature_data()
    original_counts = Counter(d["label"] for d in data)
    print(f"\n原始数据: {len(data)} 条")
    print(f"  分布: {dict(original_counts)}")
    
    # 1. 模板填充增强 egoistic
    print("\n[1/3] 模板填充增强 egoistic...")
    template_aug = generate_augmented_egoistic(200)
    print(f"  生成: {len(template_aug)} 条")
    
    # 2. 从 hybrid 提取 pseudo-egoistic
    print("\n[2/3] 从 hybrid 提取 pseudo-egoistic...")
    pseudo_aug = extract_pseudo_egoistic(data, 50)
    print(f"  提取: {len(pseudo_aug)} 条")
    
    # 3. 对现有 egoistic 做同义替换
    print("\n[3/3] 同义替换增强现有 egoistic...")
    syn_aug = augment_existing_egoistic(data, 3)
    print(f"  生成: {len(syn_aug)} 条")
    
    # 合并
    all_aug = template_aug + pseudo_aug + syn_aug
    # 去重 (按 text 前50字符)
    seen = set()
    unique_aug = []
    for d in all_aug:
        key = d["text"][:50]
        if key not in seen:
            seen.add(key)
            unique_aug.append(d)
    
    print(f"\n增强总计: {len(all_aug)} → 去重后 {len(unique_aug)} 条")
    
    # 合并原始 + 增强
    final_data = data + unique_aug
    random.shuffle(final_data)
    
    # 最终分布
    final_counts = Counter(d["label"] for d in final_data)
    print(f"\n最终数据: {len(final_data)} 条")
    for label, count in sorted(final_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 40 // len(final_data))
        pct = count / len(final_data) * 100
        print(f"  {label:12}: {count:5} ({pct:5.1f}%) {bar}")
    
    # 保存
    output_path = os.path.join(DATA_DIR, "nature_classify_augmented.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n保存: {output_path}")
    
    # 也更新原文件
    original_path = os.path.join(DATA_DIR, "nature_classify.jsonl")
    with open(original_path, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"更新: {original_path}")
    
    # 样本展示
    print("\n=== 增强样本展示 ===")
    for method in ["template_fill", "extract_from_hybrid", "synonym_replace"]:
        samples = [d for d in unique_aug if d.get("method") == method]
        if samples:
            print(f"\n  [{method}] ({len(samples)} 条)")
            for s in samples[:2]:
                print(f"    {s['text'][:80]}...")

if __name__ == "__main__":
    main()
