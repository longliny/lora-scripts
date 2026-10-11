"""Krea 2 tensor conventions, staged loading and adapter serialization.

Uses public Diffusers/PEFT model APIs; no external trainer is invoked.
"""

import gc
import json
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage, BitsAndBytesConfig, Krea2Pipeline, Krea2Transformer2DModel
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from tqdm.auto import tqdm

from .krea2_data import cache_valid, digest, image_tensor, write_cache


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pack_latents(latents):
    if latents.ndim != 5 or latents.shape[2] != 1:
        raise ValueError("Expected Krea 2 image latents shaped (B, C, 1, H, W)")
    b, c, _, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError("Krea 2 latent dimensions must be divisible by two")
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, h * w // 4, c * 4)


def unpack_latents(tokens, height, width):
    b, _, channels = tokens.shape
    return tokens.reshape(b, height // 2, width // 2, channels // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, channels // 4, 1, height, width)


def normalize_latents(latents, config):
    mean = latents.new_tensor(config.latents_mean).reshape(1, -1, 1, 1, 1)
    std = latents.new_tensor(config.latents_std).reshape(1, -1, 1, 1, 1)
    return (latents - mean) / std


def sample_sigmas(batch_size, image_tokens, scheduler_config, args, device):
    if args.timestep_sampling == "uniform":
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
    else:
        u = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32) * args.logit_std + args.logit_mean)
    # Krea Raw's resolution-dependent exponential flow shift. Unlike reading
    # scheduler.timesteps immediately after construction, this actually applies
    # dynamic shifting to every bucket's training distribution.
    c = scheduler_config
    mu = c.base_shift + (image_tokens - c.base_image_seq_len) * (c.max_shift - c.base_shift) / (c.max_image_seq_len - c.base_image_seq_len)
    shift = math.exp(mu)
    u = u.clamp(1e-5, 1 - 1e-5)
    return shift * u / (1 + (shift - 1) * u)


def flow_loss(transformer, latents, embeddings, mask, scheduler_config, args):
    noise = torch.randn_like(latents)
    sigma = sample_sigmas(len(latents), latents.shape[-2] * latents.shape[-1] // 4, scheduler_config, args, latents.device)
    amount = sigma.reshape(-1, 1, 1, 1, 1).to(latents.dtype)
    noisy = (1 - amount) * latents + amount * noise
    positions = Krea2Pipeline.prepare_position_ids(embeddings.shape[1], latents.shape[-2] // 2, latents.shape[-1] // 2, latents.device)
    prediction = transformer(
        hidden_states=pack_latents(noisy), encoder_hidden_states=embeddings,
        timestep=sigma, position_ids=positions,
        encoder_attention_mask=None if mask.all().item() else mask, return_dict=False,
    )[0]
    return torch.nn.functional.mse_loss(prediction.float(), pack_latents(noise - latents).float())


def load_transformer(model_root: Path, args, device):
    kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if args.base_precision == "nf4":
        kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"": str(device)},
        )
    if args.transformer_path:
        from .krea2_checkpoint import convert_original_checkpoint

        cache_root = Path(args.training_cache_dir or Path(args.output_dir) / ".krea2-cache")
        converted = convert_original_checkpoint(Path(args.transformer_path), model_root, cache_root)
        model = Krea2Transformer2DModel.from_pretrained(str(converted), **kwargs)
    else:
        model = Krea2Transformer2DModel.from_pretrained(str(model_root), subfolder="transformer", **kwargs)
    if args.base_precision == "fp8":
        # Quantize on CPU after adapter initialization; avoid a full BF16 GPU
        # allocation before compression, which would exhaust a 32 GB card.
        model._krea2_target_device = device
    elif args.base_precision != "nf4":
        model.to(device)
    return model


def attach_adapter(model, args):
    model.requires_grad_(False)
    names = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.to_gate"]
    if args.lora_target == "attention_mlp":
        names += ["ff.gate", "ff.up", "ff.down"]
    # Full module names keep text fusion and the timestep/output projections
    # frozen. No fragile substring targeting of unrelated linear layers.
    targets = [name for name, _ in model.named_modules() if name.startswith("transformer_blocks.") and any(name.endswith("." + suffix) for suffix in names)]
    if not targets:
        raise ValueError("The loaded model has no Krea 2 LoRA target layers")
    model.add_adapter(LoraConfig(r=args.network_dim, lora_alpha=args.network_alpha, lora_dropout=args.network_dropout, target_modules=targets, init_lora_weights="gaussian"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if args.base_precision == "fp8":
        from .krea2_precision import quantize_frozen_linears

        count = quantize_frozen_linears(model)
        model.to(getattr(model, "_krea2_target_device", next(model.parameters()).device))
        print(f"Krea 2: {count} frozen linear layers stored in scaled FP8 E4M3; compute=BF16; LoRA=FP32", flush=True)
    from .krea2_attention import configure_attention

    configure_attention(model, args.attn_mode)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()
    return model


def save_adapter(model, path: Path):
    config = model.peft_config["default"].to_dict()
    # PEFT contains sets/enums; Diffusers stores this as lora_adapter_metadata
    # so non-unit alpha/rank scaling survives loading by Krea2Pipeline.
    config = json.loads(json.dumps(config, default=lambda v: sorted(v) if isinstance(v, set) else str(v)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = path.name + ".tmp.safetensors"
    Krea2Pipeline.save_lora_weights(
        str(path.parent), transformer_lora_layers={k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()},
        transformer_lora_adapter_metadata=config, weight_name=temporary_name, safe_serialization=True,
    )
    (path.parent / temporary_name).replace(path)


def restore_adapter(model, path: Path):
    state = load_file(str(path))
    state = {k.removeprefix("transformer."): v for k, v in state.items()}
    expected = get_peft_model_state_dict(model)
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError("Saved adapter does not match this model, target layers or rank")
    set_peft_model_state_dict(model, state, adapter_name="default")


@torch.no_grad()
def cache_latents(model_root, records, args, device):
    pending = {r.latent_path: r for r in records if not cache_valid(r.latent_path, {"latents"})}
    if not pending:
        return
    vae = AutoencoderKLQwenImage.from_pretrained(str(model_root), subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    if args.vae_tiling:
        vae.enable_tiling()
    for path, record in tqdm(pending.items(), desc="Caching Krea 2 latents"):
        pixels = image_tensor(record).to(device=device, dtype=torch.bfloat16)
        generator = torch.Generator(device=device).manual_seed((int(path.stem[:16], 16) + args.seed) % (2**63))
        latent = vae.encode(pixels).latent_dist.sample(generator=generator)
        latent = normalize_latents(latent.float(), vae.config).to(torch.bfloat16)
        if not torch.isfinite(latent).all():
            raise ValueError(f"Non-finite VAE output: {record.path}")
        write_cache(path, {"latents": latent[0]})
    del vae, pixels, latent
    release_memory()


@torch.no_grad()
def cache_text(model_root, records, sample_specs, cache_dir, identity, args, device):
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel

    pending = {r.text_path: r.caption for r in records}
    for spec in sample_specs:
        for key in ("prompt", "negative_prompt"):
            path = cache_dir / "text" / f"{digest(['sample', identity, spec[key]])}.safetensors"
            spec[key + "_cache"] = path
            pending[path] = spec[key]
    pending = {path: caption for path, caption in pending.items() if not cache_valid(path, {"embeddings"})}
    if not pending:
        return
    # Some official snapshots serialize Transformers 5's list form. Our shared
    # Transformers 4 runtime expects a named-token dict under this field. Supply
    # equivalent constructor overrides; never rewrite the user's model assets
    # or change vocabulary IDs (the tokens already exist in tokenizer.json).
    tokenizer_config = json.loads((model_root / "tokenizer" / "tokenizer_config.json").read_text(encoding="utf-8"))
    token_overrides = {}
    if isinstance(tokenizer_config.get("extra_special_tokens"), list):
        token_overrides = {"extra_special_tokens": {}, "additional_special_tokens": list(dict.fromkeys(tokenizer_config.get("additional_special_tokens", []) + tokenizer_config["extra_special_tokens"]))}
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), subfolder="tokenizer", local_files_only=True, **token_overrides)
    encoder_config = json.loads((model_root / "text_encoder" / "config.json").read_text(encoding="utf-8"))
    text_config = encoder_config["text_config"]
    if text_config.get("rope_parameters") and not text_config.get("rope_scaling"):
        rope = dict(text_config["rope_parameters"])
        text_config["rope_theta"] = rope.pop("rope_theta", text_config.get("rope_theta", 5000000.0))
        text_config["rope_scaling"] = rope
    encoder, loading = Qwen3VLModel.from_pretrained(
        str(model_root), subfolder="text_encoder", config=Qwen3VLConfig.from_dict(encoder_config),
        torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa", output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Krea 2 text encoder weights do not match their configuration: {loading}")
    encoder.to(device).eval().requires_grad_(False)
    pipeline_config = json.loads((model_root / "model_index.json").read_text(encoding="utf-8"))
    pipeline = Krea2Pipeline(vae=None, transformer=None, scheduler=None, text_encoder=encoder, tokenizer=tokenizer, text_encoder_select_layers=pipeline_config.get("text_encoder_select_layers"))
    for path, caption in tqdm(pending.items(), desc="Caching Krea 2 captions"):
        embeddings, mask = pipeline.get_text_hidden_states([caption], max_sequence_length=args.max_sequence_length, device=device)
        # Padding is in the MIDDLE, before the assistant suffix. Slice-by-length
        # would discard real suffix tokens and corrupt conditioning.
        embeddings = embeddings[0][mask[0]]
        if not torch.isfinite(embeddings).all():
            raise ValueError(f"Non-finite Qwen3-VL text output for cache {path.name}")
        write_cache(path, {"embeddings": embeddings.to(torch.bfloat16)})
    del pipeline, encoder, tokenizer, embeddings, mask
    release_memory()
