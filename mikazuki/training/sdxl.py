"""SDXL entry point with project-owned Flash Attention integration.

Uses the existing trainer/data/LoRA implementation without modifying vendored
sd-scripts. Only CrossAttention.forward is replaced; module names, parameters,
LoRA hooks and checkpoint state dictionaries remain unchanged.
"""

import argparse
from types import MethodType

import torch
import sdxl_train_network as upstream
from library import args as args_util
from library.sdxl_original_unet import CrossAttention

from .flash_attention import flash_attention, validate_flash_attention


def _flash_forward(self, hidden_states, context=None, mask=None):
    context = hidden_states if context is None else context.to(hidden_states.dtype)
    q, k, v = (projection(states).unflatten(-1, (self.heads, -1)) for projection, states in (
        (self.to_q, hidden_states), (self.to_k, context), (self.to_v, context)))
    result = flash_attention(q, k, v, mask, scale=self.scale).flatten(2, 3)
    return self.to_out[0](result)


def configure_flash_attention(unet):
    count = 0
    for module in unet.modules():
        if isinstance(module, CrossAttention):
            module.forward = MethodType(_flash_forward, module)
            count += 1
    if not count:
        raise ValueError("No SDXL CrossAttention modules were found")
    return count


class SdxlNetworkTrainer(upstream.SdxlNetworkTrainer):
    def train(self, args):
        mode = args.attn_mode
        if mode is not None:
            args.xformers = mode == "xformers"
            args.mem_eff_attn = False
            args.sdpa = mode in {"sdpa", "flash"}
        if mode == "flash":
            if args.mixed_precision not in {"bf16", "fp16"}:
                raise ValueError("SDXL Flash Attention 训练需要 BF16 或 FP16 混合精度。")
            validate_flash_attention(torch.device("cuda", torch.cuda.current_device()))
        return super().train(args)

    def load_target_model(self, args, weight_dtype, accelerator):
        result = super().load_target_model(args, weight_dtype, accelerator)
        if args.attn_mode == "flash":
            configure_flash_attention(result[-1])
        return result


def setup_parser() -> argparse.ArgumentParser:
    parser = upstream.setup_parser()
    parser.add_argument("--attn_mode", choices=["sdpa", "flash", "xformers"], default=None,
                        help="SDXL attention backend; omitted preserves legacy sdpa/xformers flags")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)
    SdxlNetworkTrainer().train(args)
