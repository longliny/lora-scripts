"""Stream the original Krea checkpoint into reusable Diffusers shards.

Diffusers 0.39 has Krea 2 models but no Krea 2 from_single_file loader.
This project-owned rename/reshape converter keeps memory bounded to one shard.
"""

import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from .krea2_data import digest, file_identity


ROOT_NAMES = {
    "first": "img_in",
    "tmlp.0": "time_embed.linear_1",
    "tmlp.2": "time_embed.linear_2",
    "tproj.1": "time_mod_proj",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1": "txt_in.linear_1",
    "txtmlp.3": "txt_in.linear_2",
    "txtfusion.projector": "text_fusion.projector",
    "last.linear": "final_layer.linear",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
}
BLOCK_NAMES = {
    "attn.wq": "attn.to_q", "attn.wk": "attn.to_k", "attn.wv": "attn.to_v",
    "attn.wo": "attn.to_out.0", "attn.gate": "attn.to_gate",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "prenorm.scale": "norm1.weight", "postnorm.scale": "norm2.weight",
    "mlp.gate": "ff.gate", "mlp.up": "ff.up", "mlp.down": "ff.down",
    "mod.lin": "scale_shift_table",
}


def _rename_prefix(name, mapping):
    for source, target in mapping.items():
        if name == source or name.startswith(source + "."):
            return target + name[len(source):]
    raise ValueError(f"Unrecognized original Krea 2 tensor: {name}")


def native_to_diffusers_name(name):
    match = re.fullmatch(r"(blocks|txtfusion\.layerwise_blocks|txtfusion\.refiner_blocks)\.(\d+)\.(.+)", name)
    if match:
        section, index, suffix = match.groups()
        section = "transformer_blocks" if section == "blocks" else section.replace("txtfusion", "text_fusion")
        return f"{section}.{index}.{_rename_prefix(suffix, BLOCK_NAMES)}"
    return _rename_prefix(name, ROOT_NAMES)


def convert_original_checkpoint(checkpoint: Path, model_root: Path, cache_root: Path, shard_limit=2_000_000_000):
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel

    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() != ".safetensors" or not checkpoint.is_file():
        raise ValueError("Transformer override must be an original raw.safetensors file")
    config = Krea2Transformer2DModel.load_config(str(model_root), subfolder="transformer", local_files_only=True)
    key = digest([1, file_identity(checkpoint), config])
    output = cache_root / "converted-transformers" / key
    index_path = output / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (output / "config.json").is_file() and all((output / shard).is_file() for shard in set(index["weight_map"].values())):
            return output
    with init_empty_weights():
        template = Krea2Transformer2DModel.from_config(config)
    expected = {name: tuple(tensor.shape) for name, tensor in template.state_dict().items()}
    output.mkdir(parents=True, exist_ok=True)
    template.save_config(str(output))
    del template
    shards, shard, shard_bytes, total_bytes = {}, {}, 0, 0
    number = 0

    def flush():
        nonlocal number, shard_bytes
        if not shard:
            return
        number += 1
        filename = f"diffusion_pytorch_model-{number:05d}.safetensors"
        temporary = output / (filename + ".tmp")
        save_file(shard, str(temporary), metadata={"format": "pt"})
        temporary.replace(output / filename)
        shards.update({name: filename for name in shard})
        shard.clear()
        shard_bytes = 0

    print(f"Converting original Krea 2 checkpoint into cached shards: {output}", flush=True)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        seen = set()
        for source_name in handle.keys():
            name = native_to_diffusers_name(source_name)
            if name not in expected or name in seen:
                raise ValueError(f"Unexpected or duplicated Krea 2 tensor: {source_name}")
            tensor = handle.get_tensor(source_name)
            if tuple(tensor.shape) != expected[name]:
                if name.endswith("scale_shift_table"):
                    tensor = tensor.reshape(expected[name])
                else:
                    raise ValueError(f"Krea 2 tensor shape mismatch: {source_name}, expected {expected[name]}")
            shard[name] = tensor.contiguous()
            seen.add(name)
            size = tensor.numel() * tensor.element_size()
            shard_bytes += size
            total_bytes += size
            if shard_bytes >= shard_limit:
                flush()
        if seen != set(expected):
            raise ValueError(f"Krea 2 checkpoint is incomplete: {sorted(set(expected) - seen)[:12]}")
        flush()
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": shards}, indent=2), encoding="utf-8")
    temporary.replace(index_path)
    return output
