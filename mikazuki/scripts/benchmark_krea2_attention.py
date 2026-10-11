"""Benchmark actual Krea 2 GQA shapes, forward + backward, with kernel evidence.

Run as a module with --output logs/krea2-attention.json. No model is downloaded.
"""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from mikazuki.training.krea2_attention import attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1337)
    packages = {}
    for name in ("xformers", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"torch": torch.__version__, **packages, "gpu": torch.cuda.get_device_name(), "shape": {"batch": 1, "tokens": args.tokens, "q_heads": 48, "kv_heads": 12, "head_dim": 128}, "compute_dtype": "bfloat16", "forward_and_backward": {}}
    for mode in ("sdpa", "flash", "xformers"):
        if mode == "xformers" and packages["xformers"] is None:
            report["forward_and_backward"][mode] = {"status": "unavailable", "reason": "Optional xformers package is not installed"}
            continue
        tensors = [torch.randn(1, args.tokens, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for heads in (48, 12, 12)]

        def run():
            output = attention(*tensors, None, mode)
            output.float().square().mean().backward()
            if not all(torch.isfinite(t.grad).all() for t in tensors):
                raise FloatingPointError("Non-finite attention gradients")
            for tensor in tensors:
                tensor.grad = None

        try:
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.iterations):
                run()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000 / args.iterations
            peak = torch.cuda.max_memory_allocated() / 2**30
            with profile(activities=[ProfilerActivity.CPU]) as trace:
                run()
                torch.cuda.synchronize()
            report["forward_and_backward"][mode] = {"status": "passed", "milliseconds": elapsed, "peak_allocated_gib": peak, "operators": [entry.key for entry in trace.key_averages() if "attention" in entry.key.lower() or "flash" in entry.key.lower()]}
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            report["forward_and_backward"][mode] = {"status": "failed", "error": str(error)}
        del tensors
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
