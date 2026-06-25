#!/usr/bin/env python3
"""
MiniMind Nature Classifier — 天性分类器训练

用 MiniMind 的 Transformer 架构做 3 分类:
  egoistic / hybrid / altruistic

模型参数: ~64M (4层, hidden=512, heads=8)
训练数据: nature_classify.jsonl (~2000 条)
硬件: RTX 3060 Laptop 6GB
"""

import json
import os
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

class Config:
    # 数据
    data_path = os.path.expanduser("~/minimind-memory/data/nature_classify.jsonl")
    output_dir = os.path.expanduser("~/minimind-memory/models")
    
    # 模型 (64M 参数)
    vocab_size = 10000        # 词表大小 (后面用 tokenizer 动态设)
    hidden_size = 512         # 隐藏层维度
    num_layers = 4            # Transformer 层数
    num_heads = 8             # 注意力头数
    head_dim = 64             # 每头维度
    intermediate_size = 1024  # FFN 中间层
    max_seq_len = 256         # 最大序列长度
    num_classes = 3           # 分类数
    dropout = 0.1
    
    # 训练
    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 0.01
    epochs = 30
    warmup_steps = 100
    max_grad_norm = 1.0
    
    # 标签
    label2id = {"egoistic": 0, "hybrid": 1, "altruistic": 2}
    id2label = {0: "egoistic", 1: "hybrid", 2: "altruistic"}

cfg = Config()

# ─────────────────────────────────────────────
# Simple Tokenizer (字符级, 不依赖外部库)
# ─────────────────────────────────────────────

class CharTokenizer:
    """简单字符级 tokenizer, 不需要 sentencepiece"""
    
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.char2id = {"<pad>": 0, "<unk>": 1, "<cls>": 2, "<sep>": 3}
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = max_vocab
    
    def fit(self, texts):
        """从文本中构建词表"""
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)
        
        # 按频率排序, 取 top max_vocab
        for char, _ in char_freq.most_common(self.max_vocab - 4):
            if char not in self.char2id:
                idx = len(self.char2id)
                self.char2id[char] = idx
                self.id2char[idx] = char
        
        self.vocab_size = len(self.char2id)
        print(f"  词表大小: {self.vocab_size}")
    
    def encode(self, text, max_len=256):
        """编码文本为 token id 序列"""
        ids = [self.char2id["<cls>"]]
        for char in text[:max_len - 2]:
            ids.append(self.char2id.get(char, self.char2id["<unk>"]))
        ids.append(self.char2id["<sep>"])
        
        # Padding
        while len(ids) < max_len:
            ids.append(0)
        
        return ids[:max_len]
    
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2id": self.char2id, "max_vocab": self.max_vocab}, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["max_vocab"])
        tok.char2id = data["char2id"]
        tok.id2char = {int(v): k for k, v in tok.char2id.items()}
        tok.vocab_size = len(tok.char2id)
        return tok

# ─────────────────────────────────────────────
# MiniMind Transformer (Classification)
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.weight * x.float() * norm).type_as(x)

def precompute_freqs_cis(dim, end, theta=1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [end, dim]
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # [end, dim]
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, cos, sin, mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_emb(q, k, cos[:L], sin[:L])
        
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0)
        attn = attn.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(attn)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
    
    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class MiniMindClassifier(nn.Module):
    """MiniMind Transformer + Classification Head"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        
        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes),
        )
        
        # Precompute RoPE
        cos, sin = precompute_freqs_cis(config.head_dim, config.max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        
        # Init weights
        self.apply(self._init_weights)
        
        # Count params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  模型参数: {total / 1e6:.1f}M (可训练: {trainable / 1e6:.1f}M)")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L]
        returns: logits [B, num_classes]
        """
        x = self.embed(input_ids)  # [B, L, H]
        
        # Causal mask (虽然分类不需要, 保持架构一致性)
        mask = None
        if attention_mask is not None:
            # 0 的位置被 mask 掉 (用 -inf)
            mask = attention_mask.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, L]
            mask = (1.0 - mask) * torch.finfo(torch.float16).min
        
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, mask)
        
        x = self.norm(x)
        
        # 用 [CLS] token (第一个位置) 做分类
        cls_vec = x[:, 0, :]  # [B, H]
        logits = self.cls_head(cls_vec)  # [B, num_classes]
        
        return logits

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class NatureDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.tokenizer.encode(item["text"], self.max_len)
        attention_mask = [1 if t != 0 else 0 for t in input_ids]
        label = cfg.label2id[item["label"]]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def compute_class_weights(data):
    """计算类别权重 (处理不平衡分布)"""
    counts = Counter(d["label"] for d in data)
    total = len(data)
    weights = []
    for label in ["egoistic", "hybrid", "altruistic"]:
        w = total / (counts[label] + 1)  # +1 避免除零
        weights.append(w)
    # 归一化
    w_tensor = torch.tensor(weights)
    w_tensor = w_tensor / w_tensor.sum() * len(weights)
    return w_tensor

def train_epoch(model, loader, optimizer, scaler, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    # Per-class metrics
    per_class = {}
    for label_name, label_id in cfg.label2id.items():
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == label_id and l == label_id)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == label_id and l != label_id)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != label_id and l == label_id)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[label_name] = {"precision": precision, "recall": recall, "f1": f1}
    
    return total_loss / total, correct / total, per_class

def main():
    print("=" * 60)
    print("MiniMind Nature Classifier 训练")
    print("=" * 60)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 加载数据
    print(f"\n加载数据: {cfg.data_path}")
    with open(cfg.data_path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    print(f"  总数据: {len(data)} 条")
    
    label_counts = Counter(d["label"] for d in data)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")
    
    # 构建 tokenizer
    print("\n构建 tokenizer...")
    tokenizer = CharTokenizer(max_vocab=10000)
    tokenizer.fit([d["text"] for d in data])
    
    # 保存 tokenizer
    os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(cfg.output_dir, "tokenizer.json"))
    
    # 更新 vocab_size
    cfg.vocab_size = tokenizer.vocab_size
    
    # 划分 train/val
    random.seed(42)
    random.shuffle(data)
    split = int(len(data) * 0.85)
    train_data = data[:split]
    val_data = data[split:]
    print(f"\n训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # 数据集
    train_set = NatureDataset(train_data, tokenizer, cfg.max_seq_len)
    val_set = NatureDataset(val_data, tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    
    # 模型
    print("\n构建模型...")
    model = MiniMindClassifier(cfg).to(device)
    
    # 类别权重
    class_weights = compute_class_weights(train_data).to(device)
    print(f"  类别权重: {class_weights.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # 学习率调度
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / (cfg.epochs * len(train_loader) - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    
    # 训练
    print(f"\n开始训练 ({cfg.epochs} epochs)...")
    print("-" * 60)
    
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch)
        val_loss, val_acc, per_class = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        
        # 打印
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
              f"LR: {lr:.2e} | {elapsed:.1f}s")
        
        # Per-class metrics
        if epoch % 5 == 0 or epoch == 1:
            for name, m in per_class.items():
                print(f"  {name:12}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "head_dim": cfg.head_dim,
                    "intermediate_size": cfg.intermediate_size,
                    "max_seq_len": cfg.max_seq_len,
                    "num_classes": cfg.num_classes,
                    "vocab_size": cfg.vocab_size,
                    "dropout": cfg.dropout,
                },
                "label2id": cfg.label2id,
                "id2label": cfg.id2label,
                "val_acc": best_val_acc,
                "epoch": best_epoch,
            }, os.path.join(cfg.output_dir, "nature_classifier.pt"))
            print(f"  → 保存最佳模型 (val_acc={best_val_acc:.3f})")
        
        scheduler.step()
    
    # 最终结果
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.3f} (epoch {best_epoch})")
    print(f"模型保存: {cfg.output_dir}/nature_classifier.pt")
    print("=" * 60)
    
    # 最终评估
    checkpoint = torch.load(os.path.join(cfg.output_dir, "nature_classifier.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_acc, final_per_class = evaluate(model, val_loader, criterion, device)
    print(f"\n最终验证结果:")
    print(f"  整体准确率: {final_acc:.3f}")
    for name, m in final_per_class.items():
        print(f"  {name:12}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

if __name__ == "__main__":
    main()
