"""
MiniMind Encoder 推理模块

用训练好的 MiniMind Encoder 替代 LM Studio bge-m3
输出 256 维归一化向量，延迟 <1ms
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("anima.models.encoder")

_model = None
_tokenizer = None
_device = None


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    import torch

    # 查找模型文件
    model_dir = Path(__file__).parent.parent.parent.parent / "models"
    model_path = model_dir / "memory_encoder.pt"
    tokenizer_path = model_dir / "tokenizer.json"

    if not model_path.exists():
        # 尝试 hermes home 下的路径
        from hermes_constants import get_hermes_home
        model_dir = get_hermes_home().parent / "dev" / "tianren-anima" / "models"
        model_path = model_dir / "memory_encoder.pt"
        tokenizer_path = model_dir / "tokenizer.json"

    if not model_path.exists():
        raise FileNotFoundError(f"MiniMind encoder not found: {model_path}")

    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]

    # 构建模型
    from anima.models.minimind_encoder import MiniMindEncoder, EncoderConfig
    config = EncoderConfig(**config_dict)
    _model = MiniMindEncoder(config)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_get_device())
    _model.eval()

    # 加载 tokenizer
    with open(tokenizer_path, encoding="utf-8") as f:
        tok_data = json.load(f)
    from anima.models.minimind_encoder import CharTokenizer
    _tokenizer = CharTokenizer(tok_data["max_vocab"])
    _tokenizer.char2id = tok_data["char2id"]
    _tokenizer.id2char = {int(v): k for k, v in tok_data["char2id"].items()}
    _tokenizer.vocab_size = len(_tokenizer.char2id)

    logger.info("MiniMind encoder loaded: %s (%.1fM params, dim=%d)",
                _get_device(), sum(p.numel() for p in _model.parameters()) / 1e6,
                config.embed_dim)


@torch.no_grad()
def encode_text(text: str) -> List[float]:
    """编码单条文本为 256 维归一化向量"""
    import torch

    _load_model()

    max_len = _model.config.max_seq_len
    ids = _tokenizer.encode(text, max_len)
    mask = [1 if t != 0 else 0 for t in ids]

    input_ids = torch.tensor([ids], dtype=torch.long).to(_get_device())
    attention_mask = torch.tensor([mask], dtype=torch.long).to(_get_device())

    vec = _model.encode(input_ids, attention_mask)
    return vec.cpu().numpy().flatten().tolist()


@torch.no_grad()
def encode_batch(texts: List[str]) -> List[List[float]]:
    """批量编码文本"""
    import torch

    _load_model()

    max_len = _model.config.max_seq_len
    all_ids = []
    all_masks = []

    for text in texts:
        ids = _tokenizer.encode(text, max_len)
        mask = [1 if t != 0 else 0 for t in ids]
        all_ids.append(ids)
        all_masks.append(mask)

    input_ids = torch.tensor(all_ids, dtype=torch.long).to(_get_device())
    attention_mask = torch.tensor(all_masks, dtype=torch.long).to(_get_device())

    vecs = _model.encode(input_ids, attention_mask)
    return vecs.cpu().numpy().tolist()


def get_dim() -> int:
    """返回向量维度"""
    _load_model()
    return _model.config.embed_dim
