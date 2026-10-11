"""Lightweight package checks for the workbench; never initializes CUDA."""

from importlib import metadata, util
from pathlib import Path


def package_available(distribution: str, module: str) -> bool:
    try:
        metadata.version(distribution)
        spec = util.find_spec(module)
        return spec is not None and spec.origin is not None and Path(spec.origin).is_file()
    except (metadata.PackageNotFoundError, ImportError, ValueError):
        return False


def attention_unavailable() -> dict[str, str]:
    missing = {}
    for mode, distribution, module, label in (
        ("xformers", "xformers", "xformers", "xFormers"),
        ("flash", "flash-attn", "flash_attn", "Flash Attention"),
    ):
        if not package_available(distribution, module):
            missing[mode] = f"当前训练环境未安装 {label}；安装匹配版本后刷新页面即可选择。"
    return missing
