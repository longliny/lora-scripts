"""Project Anima entry point: independent Flash preflight and AdaLN precision."""

from types import MethodType

import torch
from torch.nn import functional as F
import anima_train_network as upstream
from library import args as args_util

from .flash_attention import validate_flash_attention


def _modulation_forward(self, inputs):
    # Anima deliberately computes time embeddings/modulation outside autocast.
    # Its loader stores these linears in the base dtype, which otherwise causes
    # FP32-input / BF16-weight matmul errors. Compute only the small modulation
    # projections in FP32; preserve storage, parameter names and LoRA hooks.
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        return F.linear(inputs.float(), self.weight.float(), None if self.bias is None else self.bias.float())


def configure_anima_precision(model):
    count = 0
    for name, module in model.named_modules():
        if "adaln_modulation" in name and isinstance(module, torch.nn.Linear):
            module.forward = MethodType(_modulation_forward, module)
            count += 1
    if not count:
        raise ValueError("No Anima AdaLN modulation projections were found")
    return count


class AnimaNetworkTrainer(upstream.AnimaNetworkTrainer):
    def train(self, args):
        if args.attn_mode == "sdpa":
            args.attn_mode = "torch"
        if args.attn_mode == "flash":
            if args.mixed_precision not in {"bf16", "fp16"}:
                raise ValueError("Anima Flash Attention 训练需要 BF16 或 FP16 混合精度。")
            validate_flash_attention(torch.device("cuda", torch.cuda.current_device()))
        return super().train(args)

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        model, encoders = super().load_unet_lazily(args, weight_dtype, accelerator, text_encoders)
        configure_anima_precision(model)
        return model, encoders


setup_parser = upstream.setup_parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    if args.show_timesteps:
        upstream.anima_train_utils.show_timesteps(args)
    else:
        AnimaNetworkTrainer().train(args)
