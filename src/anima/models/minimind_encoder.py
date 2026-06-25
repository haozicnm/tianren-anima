"""
MiniMind Encoder 模型定义（推理用）

从 train_encoder.py 精简，只保留推理需要的部分
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderConfig:
    def __init__(self, hidden_size=384, num_layers=4, num_heads=6, head_dim=64,
                 intermediate_size=768, max_seq_len=128, embed_dim=256,
                 vocab_size=3139, dropout=0.1, **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = dropout


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


class MiniMindEncoder(nn.Module):
    """Transformer Encoder + mean pooling + projection → 256-dim normalized vector"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.embed_dim),
        )
        cos, sin = precompute_freqs_cis(config.head_dim, config.max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def encode(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
            mask = (1.0 - mask) * torch.finfo(torch.float16).min
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, mask)
        x = self.norm(x)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        vec = self.projection(x)
        return F.normalize(vec, p=2, dim=-1)

    def forward(self, input_ids, attention_mask=None):
        return self.encode(input_ids, attention_mask)
