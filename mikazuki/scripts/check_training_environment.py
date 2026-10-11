"""Reproducible dependency, trainer-import and optional CUDA compatibility checks.

Run: python -m mikazuki.scripts.check_training_environment --gpu --output logs/training-environment.json
No pretrained weights are downloaded and no training jobs are submitted.
"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def check_gpu():
    import torch
    import bitsandbytes as bnb
    from mikazuki.training.attention_capabilities import package_available

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    result = {"device": torch.cuda.get_device_name(), "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30, "compute_capability": list(torch.cuda.get_device_capability()), "cuda": torch.version.cuda}
    x = torch.randn(1, 24, 1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for _ in range(5):
        torch.nn.functional.scaled_dot_product_attention(x, x, x).float().square().mean().backward()
        x.grad = None
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for _ in range(20):
        torch.nn.functional.scaled_dot_product_attention(x, x, x).float().square().mean().backward()
        x.grad = None
    torch.cuda.synchronize()
    result["sdpa_forward_backward_ms"] = (time.perf_counter() - tick) * 1000 / 20
    q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for mode, distribution in (("xformers", "xformers"), ("flash", "flash-attn")):
        if not package_available(distribution, "flash_attn" if mode == "flash" else mode):
            result[f"{mode}_backward"] = "unavailable"
            continue
        try:
            if mode == "flash":
                from mikazuki.training.flash_attention import validate_flash_attention
                validate_flash_attention(torch.device("cuda:0"))
            else:
                import xformers.ops
                xformers.ops.memory_efficient_attention(q, q, q).float().square().mean().backward()
                assert q.grad is not None and torch.isfinite(q.grad).all()
            result[f"{mode}_backward"] = "passed"
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            result[f"{mode}_backward"] = "failed"
            result[f"{mode}_error"] = str(error)
    weights = torch.nn.Parameter(torch.randn(8192, device="cuda"))
    optimizer = bnb.optim.AdamW8bit([weights], lr=1e-3)
    before = weights.detach().clone()
    weights.square().mean().backward()
    optimizer.step()
    assert torch.isfinite(weights).all() and not torch.equal(weights, before)
    result["adamw8bit_step"] = "passed"
    # Exercise the project's existing SDXL and FLUX implementations, not just
    # importing their CLI parsers. These are small random-weight components.
    from library.sdxl_original_unet import BasicTransformerBlock
    from library.flux_models import Flux, FluxParams
    block = BasicTransformerBlock(64, 4, 16, 32).to("cuda", dtype=torch.bfloat16)
    block.set_use_sdpa(True)
    hidden = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    block(hidden, context=torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16)).float().square().mean().backward()
    assert torch.isfinite(hidden.grad).all()
    result["sdxl_block_backward"] = "passed"
    if result.get("flash_backward") == "passed":
        from mikazuki.training.sdxl import configure_flash_attention
        configure_flash_attention(block)
        hidden.grad = None
        block(hidden, context=torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16)).float().square().mean().backward()
        assert torch.isfinite(hidden.grad).all()
        result["sdxl_flash_backward"] = "passed"
    from library.anima_models import Attention
    from library.attention import AttentionParams
    for mode in ("torch", "flash") if result.get("flash_backward") == "passed" else ("torch",):
        attention = Attention(128, context_dim=64, n_heads=2, head_dim=64).to("cuda", dtype=torch.bfloat16)
        hidden = torch.randn(1, 17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            attention(hidden, AttentionParams.create_attention_params(mode, False), context=torch.randn(1, 11, 64, device="cuda", dtype=torch.bfloat16)).float().square().mean().backward()
        assert torch.isfinite(hidden.grad).all()
        result[f"anima_{mode}_backward"] = "passed"
    params = FluxParams(in_channels=16, vec_in_dim=16, context_in_dim=32, hidden_size=64, mlp_ratio=2, num_heads=4, depth=1, depth_single_blocks=1, axes_dim=[4, 6, 6], theta=10000, qkv_bias=True, guidance_embed=True)
    flux = Flux(params).to("cuda", dtype=torch.bfloat16)
    image = torch.randn(1, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    prediction = flux(img=image, img_ids=torch.zeros(1, 16, 3, device="cuda"), txt=torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16), txt_ids=torch.zeros(1, 8, 3, device="cuda"), timesteps=torch.tensor([0.5], device="cuda", dtype=torch.bfloat16), y=torch.randn(1, 16, device="cuda", dtype=torch.bfloat16), guidance=torch.ones(1, device="cuda", dtype=torch.bfloat16))
    prediction.float().square().mean().backward()
    assert torch.isfinite(image.grad).all()
    result["flux_model_backward"] = "passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from mikazuki.catalog.inspector import TRAINERS, _extract_parser

    packages = ["torch", "torchvision", "xformers", "flash-attn", "diffusers", "transformers", "accelerate", "peft", "bitsandbytes", "safetensors", "huggingface-hub", "numpy", "protobuf"]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "packages": {}, "trainers": {}, "errors": []}
    for name in packages:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
            if name not in {"xformers", "flash-attn"}:
                report["errors"].append(f"Missing package: {name}")
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report["pip_check"] = check.stdout.strip()
    if check.returncode:
        report["errors"].append(check.stdout + check.stderr)
    for name, trainer in TRAINERS.items():
        try:
            actual = _extract_parser(trainer["module"], root / "scripts" / "dev")
            report["trainers"][name] = {"status": "passed", "parameter_count": len(actual._actions) - 1}
        except Exception as exc:
            report["trainers"][name] = {"status": "failed", "error": str(exc)}
            report["errors"].append(f"{name}: {exc}")
    if args.gpu:
        try:
            report["gpu"] = check_gpu()
            for mode in ("flash", "xformers"):
                if report["gpu"].get(f"{mode}_backward") == "failed":
                    report["errors"].append(f"GPU {mode}: " + report["gpu"][f"{mode}_error"])
        except Exception as exc:
            report["errors"].append(f"GPU: {exc}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
