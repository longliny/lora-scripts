"""Shared cached-LoRA optimizer loop, metrics and checkpoint resume (Krea 2 / MOSS)."""

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.optimization import get_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .krea2_data import BucketBatchSampler, collate_cached


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
    return normalize_resume_signature({"version": 1, "dataset": dataset.fingerprint, "steps": total_steps, "args": {name: getattr(args, name) for name in names}, "flow_scheduler": dict(scheduler_config)})


def normalize_resume_signature(signature):
    # Diffusers creates this list from a set, so PYTHONHASHSEED changes its
    # order across processes. Preserve all values and normalize old states too.
    flow = dict(signature["flow_scheduler"])
    if "_use_default_values" in flow:
        flow["_use_default_values"] = sorted(flow["_use_default_values"])
    return {**signature, "flow_scheduler": flow}


def write_checkpoint(model, optimizer, lr_scheduler, args, name, cursor, signature, save_adapter):
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


def train_cached(args, dataset, transformer, noise_scheduler, accelerator, sampler_callback=None, *, backend, family):
    """Train from disk caches; also used for the real-model small-GPU smoke test."""
    if accelerator.num_processes != 1:
        raise ValueError("Cached LoRA 当前训练入口支持单 GPU，请在运行设备中只选择一张显卡。")
    flow_loss, restore_adapter = backend.flow_loss, backend.restore_adapter
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
        if normalize_resume_signature(state["signature"]) != signature:
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
    writer = SummaryWriter(str(event_root / f"{args.log_prefix}{family}"))
    progress = tqdm(total=total_steps, initial=global_step, desc="steps")
    loss_sum, loss_count, window_loss, window_count = 0.0, 0, 0.0, 0
    step_seconds, losses = [], []
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
                        loss = flow_loss(transformer, batch["latents"].to(dtype=getattr(backend, "LATENT_DTYPE", torch.bfloat16)), batch["embeddings"].to(dtype=torch.bfloat16), batch["mask"], noise_scheduler.config, args)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite Cached LoRA loss at step {global_step}")
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
                                raise FloatingPointError("Non-finite Cached LoRA gradients")
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients:
                    continue
                global_step += 1
                current_loss = window_loss / window_count
                losses.append(current_loss)
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
                    write_checkpoint(raw_model, optimizer, lr_scheduler, args, f"{args.output_name}-step{global_step:06d}", cursor, signature, backend.save_adapter)
                if sampler_callback and args.sample_every_n_steps and global_step % args.sample_every_n_steps == 0:
                    sampler_callback(raw_model, global_step)
            completed_epoch = cursor["batch"] == len(loader)
            if completed_epoch:
                cursor = {"step": global_step, "epoch": epoch + 1, "batch": 0}
                if args.save_every_n_epochs and (epoch + 1) % args.save_every_n_epochs == 0:
                    write_checkpoint(raw_model, optimizer, lr_scheduler, args, f"{args.output_name}-epoch{epoch + 1:06d}", cursor, signature, backend.save_adapter)
                if sampler_callback and args.sample_every_n_epochs and (epoch + 1) % args.sample_every_n_epochs == 0:
                    sampler_callback(raw_model, global_step)
            if global_step >= total_steps:
                break
        write_checkpoint(raw_model, optimizer, lr_scheduler, args, args.output_name, cursor, signature, backend.save_adapter)
        result = {"step": global_step, "loss": loss_sum / loss_count if loss_count else None, "step_seconds": step_seconds, "losses": losses, "base_precision": args.base_precision, "attn_mode": args.attn_mode, "output": str(Path(args.output_dir) / f"{args.output_name}.safetensors")}
        if accelerator.device.type == "cuda":
            result.update(peak_allocated_gib=torch.cuda.max_memory_allocated(accelerator.device) / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved(accelerator.device) / 2**30)
        (Path(args.output_dir) / f"{args.output_name}-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        progress.close()
        writer.close()
