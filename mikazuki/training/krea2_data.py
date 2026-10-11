"""Bucketed image/caption data and versioned, bounded-memory caches for Krea 2."""

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import toml
import torch
from PIL import Image, ImageOps
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset, Sampler


CACHE_VERSION = 1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def file_identity(path: Path):
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def model_identity(path: str, revision: str):
    root = Path(path)
    if root.is_file():
        return file_identity(root)
    if root.is_dir():
        # Local changes invalidate the cache without hashing many GB of weights.
        return [file_identity(p) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}]
    return [path, revision]


def write_cache(path: Path, tensors: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temporary))
    temporary.replace(path)


def cache_valid(path: Path, keys) -> bool:
    if not path.is_file():
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            return set(handle.keys()) == set(keys)
    except (OSError, ValueError, SafetensorError):
        return False


def bucket_size(width: int, height: int, options: dict):
    resolution = options.get("resolution", [1024, 1024])
    if isinstance(resolution, int):
        resolution = [resolution, resolution]
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any(not isinstance(n, int) or n < 16 or n % 16 for n in resolution):
        raise ValueError("Krea 2 resolution must be two positive multiples of 16")
    if min(width, height) < 16:
        raise ValueError("Krea 2 images must be at least 16 pixels on each side")
    if not options.get("enable_bucket", True):
        return tuple(resolution)
    step = options.get("bucket_reso_steps", 16)
    low, high = options.get("min_bucket_reso", 256), options.get("max_bucket_reso", 2048)
    if not isinstance(step, int) or step <= 0 or step % 16 or low <= 0 or high < low or low % step or high % step:
        raise ValueError("Bucket bounds and step must align to 16; bounds must align to the step")
    area = resolution[0] * resolution[1]
    if options.get("bucket_no_upscale", True):
        scale = min(1.0, math.sqrt(area / (width * height)), high / max(width, height))
        # As in sd-scripts, no-upscale may produce buckets below the configured
        # minimum. Small images are center-cropped, never enlarged to the minimum.
        return tuple(max(16, int(n * scale) // step * step) for n in (width, height))
    candidates = set()
    for w in range(low, high + 1, step):
        h = min(high, area // w // step * step)
        if h >= low:
            candidates.update({(w, h), (h, w)})
    if not candidates:
        raise ValueError("No bucket fits the requested resolution and bounds")
    return min(candidates, key=lambda size: (abs(math.log((size[0] / size[1]) / (width / height))), -size[0] * size[1], size))


@dataclass
class ImageRecord:
    path: Path
    caption: str
    size: tuple
    repeats: int
    batch_size: int
    latent_path: Path
    text_path: Path


def read_records(config_path: str, cache_dir: Path, cache_identity: dict):
    source = Path(config_path).resolve()
    config = toml.load(source)
    if set(config) - {"general", "datasets"}:
        raise ValueError("Krea 2 dataset supports only general and datasets sections")
    general = config.get("general", {})
    records = []
    common = {"caption_extension", "shuffle_caption", "keep_tokens"}
    dataset_keys = {"resolution", "batch_size", "enable_bucket", "min_bucket_reso", "max_bucket_reso", "bucket_reso_steps", "bucket_no_upscale", "subsets"} | common
    subset_keys = {"image_dir", "num_repeats", "class_tokens"} | common
    if set(general) - common:
        raise ValueError(f"Unsupported Krea 2 dataset options: {sorted(set(general) - common)}")
    for dataset in config.get("datasets", []):
        if set(dataset) - dataset_keys:
            raise ValueError(f"Unsupported Krea 2 dataset options: {sorted(set(dataset) - dataset_keys)}")
        batch_size = dataset.get("batch_size", 1)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Dataset batch_size must be a positive integer")
        for subset in dataset.get("subsets", []):
            if set(subset) - subset_keys:
                raise ValueError(f"Unsupported Krea 2 subset options: {sorted(set(subset) - subset_keys)}")
            options = {**general, **dataset, **subset}
            if options.get("shuffle_caption") or options.get("keep_tokens", 0):
                raise ValueError("Krea 2 caches fixed captions: disable shuffle_caption and keep_tokens")
            root = Path(subset.get("image_dir", ""))
            if not subset.get("image_dir"):
                raise ValueError("Dataset image_dir is required")
            if not root.is_absolute():
                root = source.parent / root
            if not root.is_dir():
                raise ValueError(f"Dataset directory does not exist: {root}")
            repeats = subset.get("num_repeats", 1)
            if not isinstance(repeats, int) or repeats <= 0:
                raise ValueError("Dataset num_repeats must be a positive integer")
            extension = options.get("caption_extension", ".txt")
            if not extension.startswith("."):
                extension = "." + extension
            images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            for path in images:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image)
                    size = bucket_size(*image.size, options)
                caption_file = path.with_suffix(extension)
                caption = caption_file.read_text(encoding="utf-8-sig").strip() if caption_file.is_file() else str(subset.get("class_tokens", ""))
                if not caption:
                    raise ValueError(f"Missing/empty caption: {caption_file}; add a caption or subset class_tokens")
                image_key = digest([CACHE_VERSION, cache_identity, file_identity(path), size])
                text_key = digest([CACHE_VERSION, cache_identity, caption])
                records.append(ImageRecord(path, caption, size, repeats, batch_size, cache_dir / "latents" / f"{image_key}.safetensors", cache_dir / "text" / f"{text_key}.safetensors"))
    if not records:
        raise ValueError("Krea 2 dataset contains no supported images")
    return records


def image_tensor(record: ImageRecord):
    with Image.open(record.path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.fit(image, record.size, method=Image.Resampling.LANCZOS)
        array = np.array(image, dtype=np.float32, copy=True) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)


class CachedDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.indices = [i for i, record in enumerate(records) for _ in range(record.repeats)]
        self.fingerprint = digest([[str(r.latent_path), str(r.text_path), r.repeats, r.batch_size] for r in records])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record = self.records[self.indices[index]]
        latent = load_file(str(record.latent_path))["latents"]
        text = load_file(str(record.text_path))["embeddings"]
        return latent, text


class BucketBatchSampler(Sampler):
    """Every image stays in its aspect bucket; short final batches are retained."""

    def __init__(self, dataset: CachedDataset, seed: int):
        self.groups = defaultdict(list)
        for i, record_index in enumerate(dataset.indices):
            record = dataset.records[record_index]
            self.groups[(record.size, record.batch_size)].append(i)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return sum(math.ceil(len(indices) / key[1]) for key, indices in self.groups.items())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for (_, batch_size), indices in sorted(self.groups.items()):
            indices = indices.copy()
            rng.shuffle(indices)
            batches.extend(indices[i:i + batch_size] for i in range(0, len(indices), batch_size))
        rng.shuffle(batches)
        yield from batches


def collate_cached(batch):
    latents, embeddings = zip(*batch)
    lengths = torch.tensor([len(item) for item in embeddings])
    text = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    mask = torch.arange(text.shape[1])[None, :] < lengths[:, None]
    return {"latents": torch.stack(latents), "embeddings": text, "mask": mask}
