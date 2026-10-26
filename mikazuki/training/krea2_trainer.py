"""Project-owned Krea 2 LoRA training, sampling and resumable optimizer loop."""

import json
import math
import re
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline
from safetensors.torch import load_file

from .krea2_data import CachedDataset, model_identity, read_records
from .krea2_model import attach_adapter, cache_latents, cache_text, load_transformer, release_memory


def resolve_model(args):
    source = Path(args.pretrained_model_name_or_path)
    if source.is_file():
        raise ValueError("Krea 2 底模需要完整 Diffusers 目录或仓库 ID；raw.safetensors 请填入 Transformer 权重覆盖项。")
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        patterns = ["model_index.json", "tokenizer/*", "text_encoder/*.json", "text_encoder/*.safetensors", "vae/*.json", "vae/*.safetensors", "scheduler/*.json", "transformer/*.json"]
        if not args.transformer_path:
            patterns.append("transformer/*.safetensors")
        source = Path(snapshot_download(args.pretrained_model_name_or_path, revision=args.revision, local_files_only=args.local_files_only, allow_patterns=patterns))
    config_file = source / "model_index.json"
    if not config_file.is_file():
        raise ValueError(f"Krea 2 Diffusers directory is missing model_index.json: {source}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("_class_name") != "Krea2Pipeline":
        raise ValueError("The base model must be a Krea2Pipeline Diffusers model")
    if config.get("is_distilled", False):
        raise ValueError("Krea 2 LoRA 应在 Raw 底模上训练；Turbo 用于加载训练后的 LoRA 推理。")
    if config.get("patch_size", 2) != 2:
        raise ValueError("This trainer supports Krea 2's 2x2 latent patches")
    for folder in ("transformer", "text_encoder", "vae", "tokenizer", "scheduler"):
        if not (source / folder).is_dir():
            raise ValueError(f"Krea 2 component is missing: {folder}")
    return source.resolve()


def read_sample_specs(path):
    if not path:
        return []
    specs = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = re.split(r"\s+--([nwhsdlg])\s+", line.strip())
        spec = {"prompt": parts[0], "negative_prompt": "", "width": 1024, "height": 1024, "num_inference_steps": 52, "seed": 1337, "guidance_scale": 3.5}
        fields = {"n": ("negative_prompt", str), "w": ("width", int), "h": ("height", int), "s": ("num_inference_steps", int), "d": ("seed", int), "l": ("guidance_scale", float), "g": ("guidance_scale", float)}
        for flag, value in zip(parts[1::2], parts[2::2]):
            name, cast = fields[flag]
            spec[name] = cast(value.strip())
        for name in ("width", "height"):
            if spec[name] < 16 or spec[name] % 16:
                raise ValueError("Krea 2 sample width/height must be positive multiples of 16")
        if spec["num_inference_steps"] <= 0 or not math.isfinite(spec["guidance_scale"]) or spec["guidance_scale"] < 0:
            raise ValueError("Invalid Krea 2 sample steps or guidance scale")
        specs.append(spec)
    return specs


@torch.no_grad()
def sample_images(model_root, model, scheduler_config, specs, args, step, device):
    if not specs:
        return
    output = Path(args.output_dir) / "sample"
    output.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    vae = pipeline = None
    try:
        # Preview RNG must never affect subsequent training noise/dropout.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            model.eval()
            vae = AutoencoderKLQwenImage.from_pretrained(str(model_root), subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
            if args.vae_tiling:
                vae.enable_tiling()
            pipeline = Krea2Pipeline(transformer=model, vae=vae, scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config), text_encoder=None, tokenizer=None, is_distilled=False)
            for index, spec in enumerate(specs):
                positive = load_file(str(spec["prompt_cache"]))["embeddings"]
                negative = load_file(str(spec["negative_prompt_cache"]))["embeddings"]
                # The pipeline shares rotary positions between CFG branches.
                # Disk caches omit padding, so align both lengths and mask the
                # added tokens before passing them to that public API.
                lengths = torch.tensor([len(positive), len(negative)], device=device)
                prompts = torch.nn.utils.rnn.pad_sequence((positive, negative), batch_first=True).to(device)
                masks = torch.arange(prompts.shape[1], device=device)[None, :] < lengths[:, None]
                kwargs = {name: spec[name] for name in ("width", "height", "num_inference_steps", "guidance_scale")}
                result = pipeline(
                    prompt_embeds=prompts[:1], prompt_embeds_mask=masks[:1],
                    negative_prompt_embeds=prompts[1:], negative_prompt_embeds_mask=masks[1:],
                    generator=torch.Generator(device=device).manual_seed(spec["seed"]), **kwargs,
                ).images[0]
                result.save(output / f"{args.output_name}-{step:06d}-{index:02d}.png")
    finally:
        model.train(was_training)
        del pipeline, vae
        release_memory()


def train_cached(args, dataset, transformer, noise_scheduler, accelerator, sampler_callback=None):
    from . import krea2_model
    from .cached_trainer import train_cached as train_cached_native

    return train_cached_native(args, dataset, transformer, noise_scheduler, accelerator,
                               sampler_callback, backend=krea2_model, family="krea2")


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Krea 2 training requires a CUDA GPU")
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    if accelerator.num_processes != 1:
        raise ValueError("Krea 2 当前训练入口支持单 GPU，请只选择一张显卡。")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU must support BF16")
    from .krea2_attention import validate_attention

    validate_attention(args.attn_mode, accelerator.device)
    set_seed(args.seed)
    specs = read_sample_specs(args.sample_prompts)
    cache_dir = Path(args.training_cache_dir or Path(args.output_dir) / ".krea2-cache").resolve()
    # Fail on dataset errors before requesting multi-GB model downloads.
    read_records(args.dataset_config, cache_dir, {})
    model_root = resolve_model(args)
    identity = {"model": model_identity(str(model_root), args.revision), "transformer": model_identity(args.transformer_path, args.revision) if args.transformer_path else None, "sequence_length": args.max_sequence_length, "seed": args.seed}
    records = read_records(args.dataset_config, cache_dir, identity)
    print(f"Krea 2: {len(records)} images; cache={cache_dir}; precision={args.base_precision}; attention={args.attn_mode}", flush=True)
    # Only one component family is on the GPU at each caching stage.
    cache_latents(model_root, records, args, accelerator.device)
    cache_text(model_root, records, specs, cache_dir, identity, args, accelerator.device)
    transformer = load_transformer(model_root, args, accelerator.device)
    # Cache hits and cache misses must initialize the same LoRA and noise stream.
    set_seed(args.seed)
    transformer = attach_adapter(transformer, args)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_root), subfolder="scheduler", local_files_only=True)
    if not scheduler.config.use_dynamic_shifting or scheduler.config.time_shift_type != "exponential":
        raise ValueError("Krea Raw requires an exponential, resolution-dependent flow scheduler")
    callback = lambda model, step: sample_images(model_root, model, scheduler.config, specs, args, step, accelerator.device)
    result = train_cached(args, CachedDataset(records), transformer, scheduler, accelerator, callback if specs else None)
    accelerator.end_training()
    print(json.dumps(result, ensure_ascii=False), flush=True)
