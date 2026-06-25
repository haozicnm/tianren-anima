#!/usr/bin/env python3
"""
MiniMind Memory Encoder — 记忆编码器训练

用对比学习 (InfoNCE) 训练一个 256 维向量编码器
替代 bge-m3 embedding 服务

输入: 记忆文本
输出: 256 维归一化向量
训练数据: search_pairs.jsonl (query, positive, negative)
"""

import json, os, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

class Config:
    data_path = os.path.expanduser("~/minimind-memory/data/search_pairs.jsonl")
    tokenizer_path = os.path.expanduser("~/minimind-memory/models/tokenizer.json")
    output_dir = os.path.expanduser("~/minimind-memory/models")
    
    # 模型
    vocab_size = 10000
    hidden_size = 384
    num_layers = 4
    num_heads = 6
    head_dim = 64
    intermediate_size = 768
    max_seq_len = 128       # encoder 可以短一些
    embed_dim = 256          # 输出向量维度
    dropout = 0.1
    
    # 训练
    batch_size = 64
    learning_rate = 5e-4
    weight_decay = 0.01
    epochs = 50
    warmup_steps = 200
    max_grad_norm = 1.0
    temperature = 0.07      # InfoNCE 温度参数

cfg = Config()

# ─────────────────────────────────────────────
# Tokenizer (复用分类器的)
# ─────────────────────────────────────────────

class CharTokenizer:
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.char2id = {}
        self.id2char = {}
        self.vocab_size = max_vocab
    
    def encode(self, text, max_len=128):
        ids = [self.char2id.get("<cls>", 2)]
        for char in text[:max_len - 2]:
            ids.append(self.char2id.get(char, self.char2id.get("<unk>", 1)))
        ids.append(self.char2id.get("<sep>", 3))
        while len(ids) < max_len:
            ids.append(0)
        return ids[:max_len]
    
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
# Transformer Encoder
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
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

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
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               dropout_p=self.dropout.p if self.training else 0)
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

class MemoryEncoder(nn.Module):
    """
    Transformer Encoder + mean pooling + projection
    
    输出: L2-normalized 256-dim 向量
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        
        # Projection head (for training only, can be removed at inference)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.embed_dim),
        )
        
        cos, sin = precompute_freqs_cis(config.head_dim, config.max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        
        self.apply(self._init_weights)
        total = sum(p.numel() for p in self.parameters())
        print(f"  模型参数: {total / 1e6:.1f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def encode(self, input_ids, attention_mask=None):
        """返回 L2-normalized 向量"""
        x = self.embed(input_ids)
        
        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
            mask = (1.0 - mask) * torch.finfo(torch.float16).min
        
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, mask)
        
        x = self.norm(x)
        
        # Mean pooling (只 pool 非 padding 位置)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Projection
        vec = self.projection(x)  # [B, embed_dim]
        
        # L2 normalize
        vec = F.normalize(vec, p=2, dim=-1)
        
        return vec
    
    def forward(self, input_ids, attention_mask=None):
        return self.encode(input_ids, attention_mask)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TripletDataset(Dataset):
    """(query, positive, negative) 三元组"""
    
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        q_ids = self.tokenizer.encode(item["query"], self.max_len)
        p_ids = self.tokenizer.encode(item["positive"], self.max_len)
        n_ids = self.tokenizer.encode(item["negative"], self.max_len)
        
        q_mask = [1 if t != 0 else 0 for t in q_ids]
        p_mask = [1 if t != 0 else 0 for t in p_ids]
        n_mask = [1 if t != 0 else 0 for t in n_ids]
        
        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "q_mask": torch.tensor(q_mask, dtype=torch.long),
            "p_ids": torch.tensor(p_ids, dtype=torch.long),
            "p_mask": torch.tensor(p_mask, dtype=torch.long),
            "n_ids": torch.tensor(n_ids, dtype=torch.long),
            "n_mask": torch.tensor(n_mask, dtype=torch.long),
        }

# ─────────────────────────────────────────────
# InfoNCE Loss
# ─────────────────────────────────────────────

def info_nce_loss(q_vec, p_vec, n_vecs, temperature=0.07):
    """
    InfoNCE: 对于每个 query, positive 应该比所有 negative 更近
    
    q_vec: [B, D]
    p_vec: [B, D]
    n_vecs: [B, K, D] (K 个 negative)
    """
    # 正例相似度
    pos_sim = (q_vec * p_vec).sum(dim=-1) / temperature  # [B]
    
    # 负例相似度
    neg_sim = torch.bmm(n_vecs, q_vec.unsqueeze(-1)).squeeze(-1) / temperature  # [B, K]
    
    # 合并: [positive, negative_1, ..., negative_K]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+K]
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)  # 正例在第 0 位
    
    return F.cross_entropy(logits, labels)

def in_batch_neg_loss(q_vec, p_vec, temperature=0.07):
    """
    In-batch negatives: 同一个 batch 内其他样本的 positive 作为 negative
    更高效，不需要显式构造 negative
    """
    # 相似度矩阵
    sim_matrix = torch.mm(q_vec, p_vec.t()) / temperature  # [B, B]
    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
    
    # 对称 loss: q→p 和 p→q
    loss_q2p = F.cross_entropy(sim_matrix, labels)
    loss_p2q = F.cross_entropy(sim_matrix.t(), labels)
    
    return (loss_q2p + loss_p2q) / 2

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        p_ids = batch["p_ids"].to(device)
        p_mask = batch["p_mask"].to(device)
        n_ids = batch["n_ids"].to(device)
        n_mask = batch["n_mask"].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            q_vec = model(q_ids, q_mask)    # [B, D]
            p_vec = model(p_ids, p_mask)    # [B, D]
            n_vec = model(n_ids, n_mask)    # [B, D]
            
            # In-batch negative loss
            loss = in_batch_neg_loss(q_vec, p_vec, cfg.temperature)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * len(q_ids)
        
        # Accuracy: query 的最近邻是否是 positive
        sim_matrix = torch.mm(q_vec, p_vec.t())
        preds = sim_matrix.argmax(dim=1)
        labels = torch.arange(len(q_vec), device=device)
        total_correct += (preds == labels).sum().item()
        total_samples += len(q_vec)
    
    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate_retrieval(model, data, tokenizer, device, max_eval=500):
    """
    检索评估: 对每个 query, 在所有 positive 中找最近邻
    计算 Recall@1, Recall@5, Recall@10
    """
    model.eval()
    
    # 限制评估数量
    eval_data = data[:max_eval]
    
    # 编码所有 query 和 positive
    q_vecs = []
    p_vecs = []
    
    for item in eval_data:
        q_ids = tokenizer.encode(item["query"], cfg.max_seq_len)
        p_ids = tokenizer.encode(item["positive"], cfg.max_seq_len)
        q_mask = [1 if t != 0 else 0 for t in q_ids]
        p_mask = [1 if t != 0 else 0 for t in p_ids]
        
        q_vec = model(
            torch.tensor([q_ids], dtype=torch.long).to(device),
            torch.tensor([q_mask], dtype=torch.long).to(device),
        )
        p_vec = model(
            torch.tensor([p_ids], dtype=torch.long).to(device),
            torch.tensor([p_mask], dtype=torch.long).to(device),
        )
        q_vecs.append(q_vec.cpu())
        p_vecs.append(p_vec.cpu())
    
    q_matrix = torch.cat(q_vecs, dim=0)  # [N, D]
    p_matrix = torch.cat(p_vecs, dim=0)  # [N, D]
    
    # 相似度矩阵
    sim = torch.mm(q_matrix, p_matrix.t())  # [N, N]
    
    # Recall@K
    labels = torch.arange(len(eval_data))
    r1 = r5 = r10 = 0
    
    for k in [1, 5, 10]:
        topk = sim.topk(k, dim=1).indices
        hits = (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
        if k == 1: r1 = hits / len(eval_data)
        elif k == 5: r5 = hits / len(eval_data)
        elif k == 10: r10 = hits / len(eval_data)
    
    # MRR
    ranks = []
    for i in range(len(eval_data)):
        rank = (sim[i] >= sim[i, i]).sum().item()
        ranks.append(1.0 / rank)
    mrr = sum(ranks) / len(ranks)
    
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MRR": mrr}

@torch.no_grad()
def encode_all_memories(model, tokenizer, db_path, device):
    """编码数据库中的所有记忆 (用于后续集成)"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, content FROM memories WHERE content IS NOT NULL AND LENGTH(content) > 10")
    rows = cur.fetchall()
    conn.close()
    
    model.eval()
    results = []
    
    for row in rows:
        ids = tokenizer.encode(row["content"], cfg.max_seq_len)
        mask = [1 if t != 0 else 0 for t in ids]
        vec = model(
            torch.tensor([ids], dtype=torch.long).to(device),
            torch.tensor([mask], dtype=torch.long).to(device),
        )
        results.append({
            "id": row["id"],
            "vec": vec.cpu().numpy().flatten().tolist(),
        })
    
    return results

def main():
    print("=" * 60)
    print("MiniMind Memory Encoder 训练")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 加载数据
    print(f"\n加载数据: {cfg.data_path}")
    with open(cfg.data_path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    print(f"  搜索对: {len(data)} 条")
    
    # 过滤: 去掉 query == positive 的无效对
    data = [d for d in data if d["query"][:80] != d["positive"][:80]]
    print(f"  过滤后: {len(data)} 条")
    
    # 加载 tokenizer
    print(f"\n加载 tokenizer: {cfg.tokenizer_path}")
    tokenizer = CharTokenizer.load(cfg.tokenizer_path)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"  词表大小: {cfg.vocab_size}")
    
    # 划分
    random.seed(42)
    random.shuffle(data)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"\n训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # 数据集
    train_set = TripletDataset(train_data, tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True, drop_last=True)
    
    # 模型
    print("\n构建模型...")
    model = MemoryEncoder(cfg).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.epochs * len(train_loader) - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    
    # 训练
    print(f"\n开始训练 ({cfg.epochs} epochs)...")
    print("-" * 60)
    
    best_r1 = 0
    best_epoch = 0
    
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        elapsed = time.time() - t0
        
        # 每 5 epoch 做一次检索评估
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate_retrieval(model, val_data, tokenizer, device, max_eval=300)
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"R@1: {metrics['R@1']:.3f} R@5: {metrics['R@5']:.3f} R@10: {metrics['R@10']:.3f} MRR: {metrics['MRR']:.3f} | "
                  f"{elapsed:.1f}s")
            
            # 保存最佳模型
            if metrics["R@1"] > best_r1:
                best_r1 = metrics["R@1"]
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
                        "embed_dim": cfg.embed_dim,
                        "vocab_size": cfg.vocab_size,
                        "dropout": cfg.dropout,
                    },
                    "metrics": metrics,
                    "epoch": epoch,
                }, os.path.join(cfg.output_dir, "memory_encoder.pt"))
                print(f"  → 保存最佳模型 (R@1={best_r1:.3f})")
        else:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"{elapsed:.1f}s")
        
        scheduler.step()
    
    # 最终结果
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳 R@1: {best_r1:.3f} (epoch {best_epoch})")
    print(f"模型保存: {cfg.output_dir}/memory_encoder.pt")
    print("=" * 60)
    
    # 最终评估
    checkpoint = torch.load(os.path.join(cfg.output_dir, "memory_encoder.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate_retrieval(model, val_data, tokenizer, device, max_eval=500)
    print(f"\n最终验证结果:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main()
