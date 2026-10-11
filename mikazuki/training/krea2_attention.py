"""Krea 2 attention selection, including GQA and padded-caption training.

The explicit flash mode uses the independent Flash Attention 2 package, including
Windows builds where PyTorch does not ship Flash Attention. It never silently
falls back to a math kernel. Padding is removed from keys/values per example,
preserving every query lane and the original rotary positions.
"""

import torch
from torch.nn import functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_krea2 import Krea2Attention


def attention(query, key, value, mask, mode):
    if mode == "sdpa":
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=mask, enable_gqa=query.shape[2] != key.shape[2],
        ).transpose(1, 2)
    if mode == "flash":
        from .flash_attention import flash_attention

        return flash_attention(query, key, value, mask)
    if mode != "xformers":
        raise ValueError(f"Unknown attention mode: {mode}")
    import xformers.ops as xops

    # xFormers' 5D grouped-query path does not implement backward. Use the 4D
    # path; repeat_interleave remains differentiable for the shared K/V heads.
    repeats = query.shape[2] // key.shape[2]
    if repeats != 1:
        key, value = key.repeat_interleave(repeats, dim=2), value.repeat_interleave(repeats, dim=2)
    if mask is None:
        return xops.memory_efficient_attention(query, key, value)
    if mask.dtype != torch.bool or mask.shape != (query.shape[0], 1, 1, key.shape[1]):
        raise ValueError("Krea 2 attention expects a boolean key-padding mask (B, 1, 1, L)")
    # Rectangular Q x unpadded-K attention exactly matches a key-padding mask;
    # it works for mixed caption lengths and positive/negative CFG branches.
    return torch.cat([
        xops.memory_efficient_attention(query[i:i + 1], key[i:i + 1, mask[i, 0, 0]], value[i:i + 1, mask[i, 0, 0]])
        for i in range(query.shape[0])
    ])


class Krea2TrainingAttention:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        result = attention(q, k, v, attention_mask, self.mode).flatten(2, 3)
        return attn.to_out[0](result * torch.sigmoid(attn.to_gate(hidden_states)))


def configure_attention(model, mode):
    for module in model.modules():
        if isinstance(module, Krea2Attention):
            module.set_processor(Krea2TrainingAttention(mode))


def validate_attention(mode, device):
    if mode == "sdpa":
        return
    # Check both backward and GQA before loading/caching the multi-GB model.
    try:
        with torch.enable_grad(), torch.random.fork_rng(devices=[device.index or 0]):
            q = torch.randn(1, 16, 48, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, 16, 12, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            attention(q, k, v, None, mode).float().square().mean().backward()
            if not all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v)):
                raise RuntimeError("Non-finite attention gradients")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        package = "flash-attn" if mode == "flash" else "xformers"
        raise RuntimeError(f"attn_mode={mode} 在当前环境/GPU 上不可用；请选择 SDPA 或安装匹配的 {package}。原始错误：{error}") from error
