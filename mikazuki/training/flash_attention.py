"""Project integration with the independent Flash Attention 2 package."""

import torch


def flash_attention(query, key, value, mask=None, *, dropout_p=0.0, scale=None):
    """BLHD attention with native GQA gradients and optional key padding.

    Preserve every query position, including padding before suffix/image tokens.
    Removing only masked keys is equivalent to a boolean key-padding mask and
    also handles rectangular SDXL cross attention.
    """
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError) as error:
        raise RuntimeError("Flash Attention 无法加载；请安装与 PyTorch/CUDA/Python 匹配的 flash-attn，或选择 SDPA。") from error
    if query.device.type != "cuda" or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Flash Attention 需要 CUDA 和 FP16/BF16 计算精度。")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Flash Attention 的 Q/K/V 必须具有相同精度。")

    def run(q, k, v):
        return flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=scale, causal=False)

    if mask is None:
        return run(query, key, value)
    if mask.dtype != torch.bool or mask.shape != (query.shape[0], 1, 1, key.shape[1]):
        raise ValueError("Flash Attention expects a boolean key-padding mask (B, 1, 1, K)")
    outputs = []
    for i in range(query.shape[0]):
        valid = mask[i, 0, 0]
        q, k, v = query[i:i + 1], key[i:i + 1, valid], value[i:i + 1, valid]
        # SDPA defines fully masked rows as zero, with zero Q/K/V gradients.
        outputs.append(run(q, k, v) if k.shape[1] else q * 0 + (k.sum() + v.sum()) * 0)
    return torch.cat(outputs)


def validate_flash_attention(device):
    """Exercise the training kernel before expensive model/data loading."""
    if device.type != "cuda":
        raise RuntimeError("Flash Attention 训练需要 CUDA。")
    with torch.enable_grad(), torch.random.fork_rng(devices=[device.index or 0]):
        q = torch.randn(1, 17, 8, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 19, 2, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        flash_attention(q, k, v).float().square().mean().backward()
        if not all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v)):
            raise RuntimeError("Flash Attention returned non-finite training gradients")
