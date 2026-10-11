"""Frozen FP8 storage with BF16 computation and memory-bounded input gradients.

Only base linear weights are quantized. Autograd retains the FP8 weight, never a
BF16 copy for every layer; backward reconstructs one weight at a time. Adapters,
norms and the sensitive input/time/output projections keep their own precision.
"""

import torch
from torch import nn
from torch.nn import functional as F


def dequantize(weight, scale, dtype):
    return weight.to(dtype) * scale.to(dtype)


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, scale, bias):
        dtype = torch.get_autocast_dtype(inputs.device.type) if torch.is_autocast_enabled(inputs.device.type) else inputs.dtype
        ctx.save_for_backward(weight, scale)
        ctx.input_dtype = inputs.dtype
        return F.linear(inputs.to(dtype), dequantize(weight, scale, dtype), None if bias is None else bias.to(dtype))

    @staticmethod
    def backward(ctx, grad_output):
        weight, scale = ctx.saved_tensors
        grad_input = grad_output.matmul(dequantize(weight, scale, grad_output.dtype)).to(ctx.input_dtype)
        return grad_input, None, None, None


class FrozenFP8Linear(nn.Linear):
    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear):
        if linear.weight.requires_grad or (linear.bias is not None and linear.bias.requires_grad):
            raise ValueError("FP8 base weights must be frozen before quantization")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features, result.out_features = linear.in_features, linear.out_features
        # Per-output-channel scaling also preserves rows with very small values.
        source = linear.weight.detach().float()
        scale = source.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / torch.finfo(torch.float8_e4m3fn).max
        result.weight = nn.Parameter((source / scale).clamp(-448, 448).to(torch.float8_e4m3fn), requires_grad=False)
        result.register_buffer("weight_scale", scale)
        result.bias = linear.bias
        return result

    def forward(self, inputs):
        return _FrozenLinear.apply(inputs, self.weight, self.weight_scale, self.bias)


def quantize_frozen_linears(model):
    count = 0
    # Snapshot names: PEFT containers remain intact, only their base_layer is
    # replaced. Adapter initialization happens in BF16/FP32 before this pass.
    for name in [name for name, _ in model.named_modules()]:
        if not name.startswith(("transformer_blocks.", "text_fusion.layerwise_blocks.", "text_fusion.refiner_blocks.")):
            continue
        module = model.get_submodule(name)
        if type(module) is nn.Linear and not module.weight.requires_grad:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), child_name, FrozenFP8Linear.from_linear(module))
            count += 1
    if not count:
        raise ValueError("No frozen Krea 2 linear weights were found for FP8 storage")
    return count
