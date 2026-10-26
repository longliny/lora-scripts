"""Workbench configuration adapter for pinned sd-scripts v0.12.2 MOSS training."""

import argparse
import math
from pathlib import Path

import toml


def setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MOSS-Image LoRA via sd-scripts v0.12.2")
    p.add_argument("--config_file", help="Training TOML produced by the workbench")
    p.add_argument("--pretrained_model_name_or_path", default="OpenMOSS-Team/MOSS-Image-Base", help="Diffusers directory or Hugging Face repository")
    p.add_argument("--revision", default="main", help="Model revision; resolved to an immutable commit before caching")
    p.add_argument("--local_files_only", action="store_true", help="Use only downloaded model files")
    p.add_argument("--text_encoder", help="Optional Qwen3.5 first shard / ConvRot INT8 checkpoint; auto-detected in a complete model directory")
    p.add_argument("--vae", help="Optional FLUX.2 VAE checkpoint; auto-detected in a complete model directory")
    p.add_argument("--tokenizer_cache_dir", help="Local tokenizer directory for offline training")
    p.add_argument("--dataset_config", help="Dataset TOML")
    p.add_argument("--output_dir", default="output", help="LoRA output directory")
    p.add_argument("--output_name", default="moss-image-lora", help="Output filename without extension")
    p.add_argument("--network_module", choices=["networks.lora_generic"], default="networks.lora_generic", help="Upstream generic LoRA implementation")
    p.add_argument("--network_weights", help="Optional existing LoRA for weight initialization, not optimizer resume")
    p.add_argument("--network_dim", type=int, default=16, help="LoRA rank")
    p.add_argument("--network_alpha", type=float, default=16.0, help="LoRA alpha")
    p.add_argument("--network_dropout", type=float, default=0.0, help="LoRA dropout")
    p.add_argument("--lora_target", choices=["attention", "attention_mlp"], default="attention_mlp", help="Train main DiT attention, optionally including MLP layers")
    p.add_argument("--base_precision", choices=["bf16"], default="bf16", help="BF16 frozen DiT weights; FP32 trainable LoRA")
    p.add_argument("--attn_mode", choices=["sdpa"], default="sdpa", help="PyTorch scaled-dot-product attention with training backward")
    p.add_argument("--mixed_precision", choices=["bf16"], default="bf16", help="MOSS-Image compute precision")
    # store_true defaults stay False so an explicit false in a workbench draft
    # really disables the option when the compiler omits that flag.
    p.add_argument("--gradient_checkpointing", action="store_true", help="Recompute DiT activations to save memory")
    p.add_argument("--cache_latents_to_disk", action="store_true", help="Write upstream latent caches beside dataset images")
    p.add_argument("--cache_text_encoder_outputs_to_disk", action="store_true", help="Write upstream text caches beside dataset images")
    p.add_argument("--convrot_int8", action="store_true", help="Quantize frozen DiT weights with upstream ConvRot INT8")
    p.add_argument("--convrot_int8_text_encoder", action="store_true", help="Quantize frozen Qwen3.5 weights with ConvRot INT8")
    p.add_argument("--blocks_to_swap", type=int, default=0)
    p.add_argument("--block_swap_h2d_only", action="store_true")
    p.add_argument("--block_swap_ring_size", type=int, default=2)
    p.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_epochs", type=int)
    p.add_argument("--max_train_steps", type=int)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--optimizer_type", choices=["AdamW", "AdamW8bit"], default="AdamW8bit")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", choices=["constant", "constant_with_warmup", "linear", "cosine"], default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--timestep_sampling", choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"], default="sigma")
    p.add_argument("--discrete_flow_shift", type=float, default=1.0)
    p.add_argument("--logit_mean", type=float, default=-1.09)
    p.add_argument("--logit_std", type=float, default=0.95)
    p.add_argument("--max_sequence_length", type=int, default=512, help="Formatted caption token cap; supports up to 4096")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--save_every_n_epochs", type=int, default=1)
    p.add_argument("--save_every_n_steps", type=int)
    p.add_argument("--save_state", action="store_true", help="Save upstream Accelerate training states")
    p.add_argument("--resume", help="Upstream training state directory; legacy native states are incompatible")
    p.add_argument("--logging_dir", help="TensorBoard event directory")
    p.add_argument("--log_prefix", default="", help="Event filename prefix")
    p.add_argument("--log_with", choices=["tensorboard"], default="tensorboard")
    p.add_argument("--sample_prompts", help="Workbench sample prompt file")
    p.add_argument("--sample_every_n_epochs", type=int)
    p.add_argument("--sample_every_n_steps", type=int)
    p.add_argument("--sample_at_first", action="store_true")
    p.add_argument("--sample_sampler", choices=["euler"], default="euler")
    return p


def parse_args(argv=None):
    parser = setup_parser()
    initial, _ = parser.parse_known_args(argv)
    if initial.config_file:
        config = toml.load(initial.config_file)
        flat = {}
        for name, value in config.items():
            section = value if isinstance(value, dict) else {name: value}
            duplicate = set(flat) & set(section)
            if duplicate:
                parser.error(f"Duplicate TOML parameters: {sorted(duplicate)}")
            flat.update(section)
        flat, _ = migrate_legacy_values(flat, warn=True)
        actions = {a.dest: a for a in parser._actions}
        unknown = set(flat) - set(actions)
        if unknown:
            parser.error(f"Unknown MOSS-Image parameters: {sorted(unknown)}")
        for name, value in flat.items():
            action = actions[name]
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    parser.error(f"{name} must be a boolean")
            elif action.type is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    parser.error(f"{name} must be an integer")
            elif action.type is float:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    parser.error(f"{name} must be a finite number")
            elif not isinstance(value, str):
                parser.error(f"{name} must be a string")
            if action.choices and value not in action.choices:
                parser.error(f"{name} must be one of {action.choices}")
        parser.set_defaults(**flat)
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args):
    for name in ("network_dim", "network_alpha", "gradient_accumulation_steps", "learning_rate", "logit_std", "discrete_flow_shift", "block_swap_ring_size"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("max_train_epochs", "max_train_steps", "save_every_n_epochs", "save_every_n_steps", "sample_every_n_epochs", "sample_every_n_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("weight_decay", "max_grad_norm", "lr_warmup_steps"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not math.isfinite(args.logit_mean):
        raise ValueError("logit_mean must be finite")
    if not 0 <= args.network_dropout < 1:
        raise ValueError("network_dropout must be in [0, 1)")
    if not 8 <= args.max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must be between 8 and 4096")
    if not args.max_train_epochs and not args.max_train_steps:
        args.max_train_epochs = 10
    elif args.max_train_epochs and args.max_train_steps:
        raise ValueError("Choose either max_train_epochs or max_train_steps")
    import re
    if (not args.output_name or args.output_name in {".", ".."}
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', args.output_name)
            or args.output_name.endswith((" ", "."))
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", args.output_name)):
        raise ValueError("Invalid output_name")
    if args.sample_every_n_epochs and args.sample_every_n_steps:
        raise ValueError("Choose either epoch or step sampling")
    if (args.sample_every_n_epochs or args.sample_every_n_steps or args.sample_at_first) and not args.sample_prompts:
        raise ValueError("Sampling requires sample_prompts")
    if args.resume and not Path(args.resume).is_dir():
        raise ValueError("resume must be a saved MOSS-Image training state directory")
    if args.resume and (Path(args.resume) / 'state.pt').exists():
        raise ValueError('旧版自定义训练状态不兼容 sd-scripts。请新建训练，或使用 network_weights 加载 LoRA。')
    if not 0 <= args.blocks_to_swap <= 40:
        raise ValueError('blocks_to_swap must be between 0 and 40')
    if args.block_swap_h2d_only and (not args.blocks_to_swap or not args.gradient_checkpointing):
        raise ValueError('H2D-only block swap requires blocks_to_swap > 0 and gradient_checkpointing')


def migrate_legacy_values(values, warn=False):
    """Normalize saved pre-migration drafts without silently reusing legacy caches."""
    values = dict(values)
    notices = []
    for name in ('vae_tiling', 'training_cache_dir'):
        if name in values:
            values.pop(name)
            notices.append(f'{name} 已移除：上游使用自身 VAE 和缓存机制，旧缓存不复用。')
    if values.get('network_module') == 'moss.lora':
        values['network_module'] = 'networks.lora_generic'
        notices.append('LoRA 后端已切换为 sd-scripts v0.12.2 networks.lora_generic。')
    if values.get('timestep_sampling') == 'moss':
        values['timestep_sampling'] = 'sigma'
        notices.append('旧 MOSS 时间步采样已迁移为上游 sigma / logit_normal；不再包含 0.95 截断与 1% 均匀混合。')
    if warn:
        import warnings
        for notice in notices:
            warnings.warn(notice, UserWarning, stacklevel=2)
    return values, notices


def main():
    args = parse_args()
    if not args.dataset_config:
        raise ValueError("dataset_config is required")
    from .moss_upstream import train

    train(args)
