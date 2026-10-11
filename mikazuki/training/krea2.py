"""Krea 2 CLI and configuration. Heavy ML imports are deferred until training."""

import argparse
import math
from pathlib import Path

import toml


def setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mikazuki native Krea 2 Raw LoRA training")
    p.add_argument("--config_file", help="Training TOML produced by the workbench")
    p.add_argument("--pretrained_model_name_or_path", default="krea/Krea-2-Raw", help="Diffusers directory or Hugging Face repository")
    p.add_argument("--revision", default="main", help="Model revision; resolved to an immutable commit before caching")
    p.add_argument("--transformer_path", help="Optional original raw.safetensors transformer; other components come from the base model")
    p.add_argument("--local_files_only", action="store_true", help="Use only downloaded model files")
    p.add_argument("--dataset_config", help="Dataset TOML")
    p.add_argument("--output_dir", default="output", help="LoRA output directory")
    p.add_argument("--output_name", default="krea2-lora", help="Output filename without extension")
    p.add_argument("--network_module", choices=["peft.lora"], default="peft.lora", help="Fixed adapter implementation")
    p.add_argument("--network_dim", type=int, default=16, help="LoRA rank")
    p.add_argument("--network_alpha", type=float, default=16.0, help="LoRA alpha")
    p.add_argument("--network_dropout", type=float, default=0.0, help="LoRA dropout")
    p.add_argument("--lora_target", choices=["attention", "attention_mlp"], default="attention_mlp", help="Train main DiT attention, optionally including MLP layers")
    p.add_argument("--base_precision", choices=["nf4", "fp8", "bf16"], default="nf4", help="Frozen DiT weight storage; NF4 or scaled FP8 E4M3, BF16 computation")
    p.add_argument("--attn_mode", choices=["sdpa", "flash", "xformers"], default="sdpa", help="SDPA auto, independent Flash Attention 2, or optional xFormers auto")
    p.add_argument("--mixed_precision", choices=["bf16"], default="bf16", help="Krea 2 compute precision")
    # store_true defaults stay False so an explicit false in a workbench draft
    # really disables the option when the compiler omits that flag.
    p.add_argument("--gradient_checkpointing", action="store_true", help="Recompute DiT activations to save memory")
    p.add_argument("--vae_tiling", action="store_true", help="Tile Qwen-Image VAE encoding/decoding")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_epochs", type=int)
    p.add_argument("--max_train_steps", type=int)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--optimizer_type", choices=["AdamW", "AdamW8bit"], default="AdamW8bit")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", choices=["constant", "constant_with_warmup", "linear", "cosine"], default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--timestep_sampling", choices=["logit_normal", "uniform"], default="logit_normal")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--max_sequence_length", type=int, default=512, help="Text token budget, including the assistant suffix")
    p.add_argument("--training_cache_dir", help="Persistent latent/text cache; default: output_dir/.krea2-cache")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--save_every_n_epochs", type=int, default=1)
    p.add_argument("--save_every_n_steps", type=int)
    p.add_argument("--save_state", action="store_true", help="Save optimizer, scheduler, RNG and exact data cursor with each checkpoint")
    p.add_argument("--resume", help="Local training state directory")
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
        actions = {a.dest: a for a in parser._actions}
        unknown = set(flat) - set(actions)
        if unknown:
            parser.error(f"Unknown Krea 2 parameters: {sorted(unknown)}")
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
    for name in ("network_dim", "network_alpha", "gradient_accumulation_steps", "learning_rate", "logit_std"):
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
    if not 8 <= args.max_sequence_length <= 512:
        raise ValueError("max_sequence_length must be between 8 and 512")
    if not args.max_train_epochs and not args.max_train_steps:
        args.max_train_epochs = 10
    elif args.max_train_epochs and args.max_train_steps:
        raise ValueError("Choose either max_train_epochs or max_train_steps")
    from mikazuki.compiler.compiler import _validate_output_name

    errors = _validate_output_name(args.output_name)
    if errors:
        raise ValueError(errors[0].message)
    if args.sample_every_n_epochs and args.sample_every_n_steps:
        raise ValueError("Choose either epoch or step sampling")
    if (args.sample_every_n_epochs or args.sample_every_n_steps or args.sample_at_first) and not args.sample_prompts:
        raise ValueError("Sampling requires sample_prompts")
    if args.resume and not Path(args.resume).is_dir():
        raise ValueError("resume must be a saved Krea 2 training state directory")


def main():
    args = parse_args()
    if not args.dataset_config:
        raise ValueError("dataset_config is required")
    from .krea2_trainer import train

    train(args)
