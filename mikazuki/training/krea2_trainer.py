"""Project-owned Krea 2 LoRA training, sampling and resumable optimizer loop."""

import json
import hashlib
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline
from diffusers.optimization import get_scheduler
from safetensors.torch import load_file
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .krea2_data import BucketBatchSampler, CachedDataset, collate_cached, model_identity, read_records
from .krea2_model import attach_adapter, cache_latents, cache_text, flow_loss, load_transformer, release_memory, restore_adapter, save_adapter


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


def make_optimizer(model, args):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer_type == "AdamW8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy_name": numpy_state[0], "numpy_keys": torch.from_numpy(numpy_state[1].astype(np.int64)), "numpy_tail": numpy_state[2:], "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_name"], state["numpy_keys"].numpy().astype(np.uint32), *state["numpy_tail"]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def resume_signature(args, dataset, total_steps, scheduler_config):
    names = ("network_dim", "network_alpha", "network_dropout", "lora_target", "base_precision", "attn_mode", "gradient_accumulation_steps", "learning_rate", "optimizer_type", "weight_decay", "max_grad_norm", "lr_scheduler", "lr_warmup_steps", "seed", "timestep_sampling", "logit_mean", "logit_std")
    return {"version": 1, "dataset": dataset.fingerprint, "steps": total_steps, "args": {name: getattr(args, name) for name in names}, "flow_scheduler": dict(scheduler_config)}


def write_checkpoint(model, optimizer, lr_scheduler, args, name, cursor, signature):
    output = Path(args.output_dir)
    save_adapter(model, output / f"{name}.safetensors")
    if args.save_state:
        folder = output / f"{name}-state"
        folder.mkdir(parents=True, exist_ok=True)
        # Completion marker (state.pt) is replaced last. An interrupted write
        # therefore cannot masquerade as a completed new checkpoint.
        save_adapter(model, folder / "adapter.safetensors")
        adapter_hash = hashlib.sha256((folder / "adapter.safetensors").read_bytes()).hexdigest()
        state = {"signature": signature, "cursor": cursor, "adapter_hash": adapter_hash, "optimizer": optimizer.state_dict(), "scheduler": lr_scheduler.state_dict(), "rng": rng_state()}
        temporary = folder / "state.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(folder / "state.pt")


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
    """Train from disk caches; also used for the real-model small-GPU smoke test."""
    if accelerator.num_processes != 1:
        raise ValueError("Krea 2 当前训练入口支持单 GPU，请在运行设备中只选择一张显卡。")
    sampler = BucketBatchSampler(dataset, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_cached, num_workers=0, pin_memory=accelerator.device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = args.max_train_steps or updates_per_epoch * (args.max_train_epochs or 10)
    epochs = math.ceil(total_steps / updates_per_epoch)
    optimizer = make_optimizer(transformer, args)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=total_steps)
    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)
    raw_model = accelerator.unwrap_model(transformer)
    signature = resume_signature(args, dataset, total_steps, noise_scheduler.config)
    global_step, start_epoch, start_batch = 0, 0, 0
    if args.resume:
        state = torch.load(Path(args.resume) / "state.pt", map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("恢复配置与保存状态不一致：请保持底模、数据、LoRA、优化器、随机种子及总步数一致。")
        if hashlib.sha256((Path(args.resume) / "adapter.safetensors").read_bytes()).hexdigest() != state["adapter_hash"]:
            raise ValueError("训练状态与 LoRA 权重不匹配；检查断点是否在写入时被中断。")
        restore_adapter(raw_model, Path(args.resume) / "adapter.safetensors")
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["scheduler"])
        global_step, start_epoch, start_batch = (state["cursor"][key] for key in ("step", "epoch", "batch"))
        restore_rng(state["rng"])
        del state
    event_root = Path(args.logging_dir or Path(args.output_dir) / "logs")
    # Match the existing job reader's managed child-directory prefix contract.
    writer = SummaryWriter(str(event_root / f"{args.log_prefix}krea2"))
    progress = tqdm(total=total_steps, initial=global_step, desc="steps")
    loss_sum, loss_count, window_loss, window_count = 0.0, 0, 0.0, 0
    step_seconds = []
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    window_started = time.perf_counter()
    cursor = {"step": global_step, "epoch": start_epoch, "batch": start_batch}
    try:
        if args.sample_at_first and sampler_callback and not args.resume:
            sampler_callback(raw_model, global_step)
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, epochs):
            sampler.epoch = epoch
            transformer.train()
            for batch_index, batch in enumerate(loader):
                if global_step >= total_steps:
                    break
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                if window_count == 0:
                    window_started = time.perf_counter()
                with accelerator.accumulate(transformer):
                    with accelerator.autocast():
                        loss = flow_loss(transformer, batch["latents"].to(dtype=torch.bfloat16), batch["embeddings"].to(dtype=torch.bfloat16), batch["mask"], noise_scheduler.config, args)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite Krea 2 loss at step {global_step}")
                    # Accelerate divides by the configured accumulation count.
                    # Correct the final incomplete window instead of weakening it.
                    window_start = batch_index // args.gradient_accumulation_steps * args.gradient_accumulation_steps
                    window_size = min(args.gradient_accumulation_steps, len(loader) - window_start)
                    accelerator.backward(loss * args.gradient_accumulation_steps / window_size)
                    window_loss += loss.detach().float().item()
                    window_count += 1
                    if accelerator.sync_gradients:
                        if args.max_grad_norm:
                            grad_norm = accelerator.clip_grad_norm_([p for p in transformer.parameters() if p.requires_grad], args.max_grad_norm)
                            if not torch.isfinite(grad_norm):
                                raise FloatingPointError("Non-finite Krea 2 gradients")
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients:
                    continue
                global_step += 1
                current_loss = window_loss / window_count
                loss_sum += window_loss
                loss_count += window_count
                window_loss, window_count = 0.0, 0
                writer.add_scalar("loss/current", current_loss, global_step)
                writer.add_scalar("loss/average", loss_sum / loss_count, global_step)
                writer.add_scalar("lr/unet", lr_scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/epoch", epoch + 1, global_step)
                step_seconds.append(time.perf_counter() - window_started)
                writer.add_scalar("train/step_seconds", step_seconds[-1], global_step)
                writer.flush()
                progress.update(1)
                progress.set_postfix(loss=f"{current_loss:.5f}")
                cursor = {"step": global_step, "epoch": epoch, "batch": batch_index + 1}
                if args.save_every_n_steps and global_step % args.save_every_n_steps == 0:
                    write_checkpoint(raw_model, optimizer, lr_scheduler, args, f"{args.output_name}-step{global_step:06d}", cursor, signature)
                if sampler_callback and args.sample_every_n_steps and global_step % args.sample_every_n_steps == 0:
                    sampler_callback(raw_model, global_step)
            completed_epoch = cursor["batch"] == len(loader)
            if completed_epoch:
                cursor = {"step": global_step, "epoch": epoch + 1, "batch": 0}
                if args.save_every_n_epochs and (epoch + 1) % args.save_every_n_epochs == 0:
                    write_checkpoint(raw_model, optimizer, lr_scheduler, args, f"{args.output_name}-epoch{epoch + 1:06d}", cursor, signature)
                if sampler_callback and args.sample_every_n_epochs and (epoch + 1) % args.sample_every_n_epochs == 0:
                    sampler_callback(raw_model, global_step)
            if global_step >= total_steps:
                break
        write_checkpoint(raw_model, optimizer, lr_scheduler, args, args.output_name, cursor, signature)
        result = {"step": global_step, "loss": loss_sum / loss_count if loss_count else None, "step_seconds": step_seconds, "base_precision": args.base_precision, "attn_mode": args.attn_mode, "output": str(Path(args.output_dir) / f"{args.output_name}.safetensors")}
        if accelerator.device.type == "cuda":
            result.update(peak_allocated_gib=torch.cuda.max_memory_allocated(accelerator.device) / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved(accelerator.device) / 2**30)
        (Path(args.output_dir) / f"{args.output_name}-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        progress.close()
        writer.close()


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
