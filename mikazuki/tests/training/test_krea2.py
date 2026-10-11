import copy
import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import toml

from mikazuki.catalog.loader import CatalogService
from mikazuki.compiler.compiler import compile_draft
from mikazuki.compiler.models import TrainDraft
from mikazuki.training.krea2 import parse_args, setup_parser


class Krea2ConfigurationTests(unittest.TestCase):
    def draft(self, **overrides):
        values = dict(trainerId="krea2.lora", name="krea-test", dataset={"root": "train"})
        values.update(overrides)
        return TrainDraft(**values)

    def test_workbench_compiles_into_native_parser_without_foreign_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            result = compile_draft(self.draft(), persist=False, validate_paths=False, base_dir=Path(directory))
            self.assertEqual(result.errors, [])
            self.assertIn("scripts/krea2_train_network.py", result.command)
            path = Path(directory) / "train.toml"
            path.write_text(result.trainConfig, encoding="utf-8")
            args = parse_args(["--config_file", str(path)])
            self.assertEqual(args.pretrained_model_name_or_path, "krea/Krea-2-Raw")
            self.assertEqual(args.base_precision, "nf4")
            self.assertEqual(args.attn_mode, "flash")
            self.assertEqual(args.mixed_precision, "bf16")
            self.assertTrue(args.gradient_checkpointing)
            self.assertTrue(args.vae_tiling)
            self.assertTrue(args.save_state)
            self.assertEqual(args.network_alpha, 16)
            self.assertEqual(toml.loads(result.datasetConfig)["datasets"][0]["bucket_reso_steps"], 16)
            trainer = CatalogService().get_trainer("krea2.lora")
            self.assertEqual(set(trainer.params), {a.dest for a in setup_parser()._actions if a.dest != "help"})

    def test_false_switches_and_cli_overrides_survive_config(self):
        with tempfile.TemporaryDirectory() as directory:
            result = compile_draft(self.draft(params={"gradient_checkpointing": False, "save_state": False, "max_train_steps": 7}), persist=False, validate_paths=False, base_dir=Path(directory))
            self.assertEqual(result.errors, [])
            path = Path(directory) / "train.toml"
            path.write_text(result.trainConfig, encoding="utf-8")
            args = parse_args(["--config_file", str(path), "--learning_rate", "0.0002"])
            self.assertFalse(args.gradient_checkpointing)
            self.assertFalse(args.save_state)
            self.assertIsNone(args.max_train_epochs)
            self.assertEqual(args.max_train_steps, 7)
            self.assertEqual(args.learning_rate, 0.0002)

    @patch("mikazuki.training.attention_capabilities.attention_unavailable", return_value={})
    def test_fp8_and_attention_choices_roundtrip_through_workbench_toml(self, _available):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("sdpa", "flash", "xformers"):
                result = compile_draft(self.draft(params={"base_precision": "fp8", "attn_mode": mode}), persist=False, validate_paths=False, base_dir=Path(directory))
                self.assertEqual(result.errors, [])
                path = Path(directory) / "train.toml"
                path.write_text(result.trainConfig, encoding="utf-8")
                args = parse_args(["--config_file", str(path)])
                self.assertEqual((args.base_precision, args.attn_mode), ("fp8", mode))

    def test_conflicts_fail_before_a_training_job_is_created(self):
        cases = [
            (self.draft(runtime={"gpuIds": ["0", "1"]}), "krea2.single_gpu"),
            (self.draft(dataset={"root": "train", "shuffleCaption": True}), "krea2.cached_captions"),
            (self.draft(dataset={"root": "train", "keepTokens": 1}), "krea2.cached_captions"),
            (self.draft(dataset={"root": "train", "resolution": [1000, 1024]}), "krea2.resolution"),
            (self.draft(modelAssets={"pretrained_model_name_or_path": "raw.safetensors"}), "krea2.model_directory"),
            (self.draft(params={"optimizer_type": "Prodigy"}), "param.invalid"),
        ]
        for draft, code in cases:
            with self.subTest(code=code):
                result = compile_draft(draft, persist=False, validate_paths=False)
                self.assertIn(code, [error.code for error in result.errors])

    def test_toml_rejects_unknown_and_mistyped_parameters(self):
        for content in ('network_dim = 1.5', 'gradient_checkpointing = "false"', 'unknown = 1', 'base_precision = "fp16"', 'attn_mode = "unknown"'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    parse_args(["--config_file", str(path)])


class Krea2TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def test_pack_has_correct_patch_order_and_round_trips_rectangles(self):
        from mikazuki.training.krea2_model import pack_latents, unpack_latents

        t = self.torch
        latent = t.arange(2 * 3 * 4 * 6).reshape(2, 3, 1, 4, 6)
        packed = pack_latents(latent)
        self.assertEqual(tuple(packed.shape), (2, 6, 12))
        t.testing.assert_close(packed[0, 0], latent[0, :, 0, :2, :2].reshape(-1))
        t.testing.assert_close(unpack_latents(packed, 4, 6), latent)

    def test_flow_shift_matches_diffusers_schedule_at_two_resolutions(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from unittest.mock import patch
        from mikazuki.training.krea2_model import sample_sigmas

        t = self.torch
        args = parse_args(["--timestep_sampling", "uniform"])
        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400)
        u = t.tensor([0.9, 0.5, 0.1])
        results = []
        for tokens in (1024, 4096):
            mu = 0.5 + (tokens - 256) * 0.65 / (6400 - 256)
            scheduler.set_timesteps(sigmas=u.numpy(), mu=mu)
            with patch("torch.rand", return_value=u):
                actual = sample_sigmas(3, tokens, scheduler.config, args, "cpu")
            t.testing.assert_close(actual, scheduler.sigmas[:-1])
            results.append(actual)
        self.assertTrue(t.all(results[1] > results[0]))

    def test_vae_channel_normalization(self):
        from types import SimpleNamespace
        from mikazuki.training.krea2_model import normalize_latents

        latent = self.torch.tensor([3.0, 8.0]).reshape(1, 2, 1, 1, 1)
        result = normalize_latents(latent, SimpleNamespace(latents_mean=[1, 2], latents_std=[2, 3]))
        self.torch.testing.assert_close(result.flatten(), self.torch.tensor([1.0, 2.0]))

    def test_bucket_repeats_and_cache_invalidation(self):
        from PIL import Image
        from mikazuki.training.krea2_data import BucketBatchSampler, CachedDataset, bucket_size, read_records

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "images"
            data.mkdir()
            Image.new("RGB", (512, 768)).save(data / "one.png")
            Image.new("RGB", (768, 512)).save(data / "two.png")
            for name in ("one", "two"):
                (data / f"{name}.txt").write_text(f"caption {name}", encoding="utf-8")
            config = root / "dataset.toml"
            config.write_text(toml.dumps({"datasets": [{"resolution": [512, 512], "batch_size": 2, "subsets": [{"image_dir": "images", "num_repeats": 3}]}]}), encoding="utf-8")
            records = read_records(str(config), root / "cache", {"model": "test"})
            dataset = CachedDataset(records)
            batches = list(BucketBatchSampler(dataset, 1))
            self.assertEqual(len(dataset), 6)
            self.assertEqual(sorted(i for batch in batches for i in batch), list(range(6)))
            for batch in batches:
                self.assertEqual(len({records[dataset.indices[i]].size for i in batch}), 1)
            (data / "one.txt").write_text("changed caption", encoding="utf-8")
            changed = read_records(str(config), root / "cache", {"model": "test"})
            self.assertNotEqual(records[0].text_path, changed[0].text_path)
            self.assertEqual(records[0].latent_path, changed[0].latent_path)
            self.assertEqual(bucket_size(240, 160, {"bucket_reso_steps": 16}), (240, 160))

    def test_original_checkpoint_conversion_shards_and_reloads_every_tensor(self):
        from safetensors.torch import save_file
        from diffusers import Krea2Transformer2DModel
        from mikazuki.training.krea2_checkpoint import ROOT_NAMES, BLOCK_NAMES, convert_original_checkpoint, native_to_diffusers_name

        # Known names from the original Krea format, independent of shard I/O.
        self.assertEqual(native_to_diffusers_name("blocks.3.attn.wq.weight"), "transformer_blocks.3.attn.to_q.weight")
        self.assertEqual(native_to_diffusers_name("txtfusion.refiner_blocks.0.prenorm.scale"), "text_fusion.refiner_blocks.0.norm1.weight")
        self.assertEqual(native_to_diffusers_name("last.modulation.lin"), "final_layer.scale_shift_table")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = tiny_transformer()
            model.save_config(root / "model" / "transformer")
            original = {}
            for name, tensor in model.state_dict().items():
                section = next((p for p in ("transformer_blocks", "text_fusion.layerwise_blocks", "text_fusion.refiner_blocks") if name.startswith(p + ".")), None)
                if section:
                    index, suffix = name[len(section) + 1:].split(".", 1)
                    native_section = "blocks" if section == "transformer_blocks" else section.replace("text_fusion", "txtfusion")
                    native = next(k + suffix[len(v):] for k, v in BLOCK_NAMES.items() if suffix == v or suffix.startswith(v + "."))
                    native = f"{native_section}.{index}.{native}"
                else:
                    native = next(k + name[len(v):] for k, v in ROOT_NAMES.items() if name == v or name.startswith(v + "."))
                original[native] = tensor.flatten() if name.endswith("scale_shift_table") else tensor
            source = root / "raw.safetensors"
            save_file(original, str(source))
            converted = convert_original_checkpoint(source, root / "model", root / "cache", shard_limit=4096)
            index = json.loads((converted / "diffusion_pytorch_model.safetensors.index.json").read_text())
            self.assertGreater(len(set(index["weight_map"].values())), 1)
            restored = Krea2Transformer2DModel.from_pretrained(converted, local_files_only=True)
            for name, tensor in model.state_dict().items():
                self.torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
            self.assertEqual(convert_original_checkpoint(source, root / "model", root / "cache"), converted)


def tiny_transformer():
    from diffusers import Krea2Transformer2DModel

    return Krea2Transformer2DModel(in_channels=16, num_layers=1, attention_head_dim=16, num_attention_heads=2, num_key_value_heads=1, intermediate_size=64, timestep_embed_dim=16, text_hidden_dim=16, num_text_layers=2, text_num_attention_heads=2, text_num_key_value_heads=1, text_intermediate_size=32, num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(4, 6, 6))


@unittest.skipUnless(os.environ.get("KREA2_GPU_TESTS") == "1", "Set KREA2_GPU_TESTS=1 for real CUDA forward/backward, NF4 and resume tests")
class Krea2GpuTests(unittest.TestCase):
    def test_full_pipeline_caches_real_encoders_trains_and_samples(self):
        import torch
        from PIL import Image
        from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline
        from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLModel
        from tokenizers import Tokenizer, models, pre_tokenizers
        from mikazuki.training.krea2_trainer import train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_backend = Tokenizer(models.WordLevel({"<unk>": 0, "<pad>": 1, "a": 2, "test": 3}, unk_token="<unk>"))
            tokenizer_backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token="<unk>", pad_token="<pad>")
            text_config = {"vocab_size": 16, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]}}
            vision_config = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 16, "deepstack_visual_indexes": [0]}
            encoder = Qwen3VLModel(Qwen3VLConfig(text_config=text_config, vision_config=vision_config, image_token_id=10, video_token_id=11, vision_start_token_id=12, vision_end_token_id=13))
            vae = AutoencoderKLQwenImage(base_dim=8, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1, latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
            scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400)
            pipeline = Krea2Pipeline(transformer=tiny_transformer(), text_encoder=encoder, tokenizer=tokenizer, vae=vae, scheduler=scheduler, text_encoder_select_layers=[1, 2])
            pipeline.save_pretrained(root / "model")
            # Official Krea snapshots can use Transformers 5's list-shaped
            # extra_special_tokens, while other trainers still require 4.x.
            token_path = root / "model" / "tokenizer" / "tokenizer_config.json"
            token_config = json.loads(token_path.read_text(encoding="utf-8"))
            token_config["extra_special_tokens"] = ["<unk>", "<pad>"]
            token_path.write_text(json.dumps(token_config), encoding="utf-8")
            encoder_path = root / "model" / "text_encoder" / "config.json"
            encoder_config = json.loads(encoder_path.read_text(encoding="utf-8"))
            text_config = encoder_config["text_config"]
            text_config["rope_parameters"] = {**text_config.pop("rope_scaling"), "rope_theta": text_config.pop("rope_theta")}
            encoder_path.write_text(json.dumps(encoder_config), encoding="utf-8")
            del pipeline, encoder, vae
            data = root / "images"
            data.mkdir()
            Image.new("RGB", (48, 32), color=(120, 30, 60)).save(data / "one.png")
            (data / "one.txt").write_text("a test " * 40, encoding="utf-8")
            dataset = root / "dataset.toml"
            dataset.write_text(toml.dumps({"datasets": [{"resolution": [48, 32], "enable_bucket": False, "subsets": [{"image_dir": str(data)}]}]}), encoding="utf-8")
            prompts = root / "prompts.txt"
            # Unequal cached positive/negative lengths exercise CFG padding.
            prompts.write_text("a test " * 40 + "--w 48 --h 32 --s 2 --l 3.5 --d 1\n", encoding="utf-8")
            args = parse_args(["--pretrained_model_name_or_path", str(root / "model"), "--local_files_only", "--dataset_config", str(dataset), "--output_dir", str(root / "output"), "--network_dim", "2", "--network_alpha", "1", "--max_sequence_length", "64", "--max_train_steps", "1", "--gradient_checkpointing", "--vae_tiling", "--base_precision", "nf4", "--optimizer_type", "AdamW8bit", "--sample_prompts", str(prompts), "--sample_every_n_steps", "1"])
            train(args)
            self.assertTrue((root / "output" / "krea2-lora.safetensors").is_file())
            self.assertTrue(list((root / "output" / ".krea2-cache" / "latents").glob("*.safetensors")))
            self.assertTrue(list((root / "output" / ".krea2-cache" / "text").glob("*.safetensors")))
            with Image.open(root / "output" / "sample" / "krea2-lora-000001-00.png") as image:
                self.assertEqual(image.size, (48, 32))

    def test_real_transformer_quantization_adapter_roundtrip_and_exact_resume(self):
        import torch
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline
        from peft import get_peft_model_state_dict
        from mikazuki.training.krea2_data import CachedDataset, ImageRecord, write_cache
        from mikazuki.training.krea2_model import attach_adapter, flow_loss, load_transformer, restore_adapter, save_adapter
        from mikazuki.training.krea2_trainer import train_cached
        from mikazuki.jobs.metrics import _discover_event_files, MetricsSource

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(["--output_dir", str(root / "output"), "--network_dim", "2", "--network_alpha", "1", "--max_train_steps", "3", "--gradient_accumulation_steps", "2", "--save_every_n_steps", "1", "--save_state", "--gradient_checkpointing", "--base_precision", "bf16", "--optimizer_type", "AdamW", "--log_prefix", "test-"])
            scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400)
            set_seed(321)
            base = tiny_transformer()
            base.save_pretrained(root / "model" / "transformer")
            records = []
            for i in range(3):
                latent_path, text_path = root / f"latent{i}.safetensors", root / f"text{i}.safetensors"
                write_cache(latent_path, {"latents": torch.randn(4, 1, 4, 6).bfloat16()})
                write_cache(text_path, {"embeddings": torch.randn(3 + i, 2, 16).bfloat16()})
                records.append(ImageRecord(root / f"{i}.png", "test", (48, 32), 1, 1, latent_path, text_path))
            dataset = CachedDataset(records)
            set_seed(args.seed)
            model = attach_adapter(copy.deepcopy(base).to("cuda", dtype=torch.bfloat16), args)
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            initial_adapter = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2)
            result = train_cached(args, dataset, model, scheduler, accelerator)
            self.assertEqual(result["step"], 3)
            self.assertTrue(math.isfinite(result["loss"]))
            self.assertTrue(any(not torch.equal(initial_adapter[n], p) for n, p in model.named_parameters() if p.requires_grad))
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
            expected = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
            files, _ = _discover_event_files(MetricsSource(root=root / "output" / "logs", prefix="test-"))
            self.assertTrue(files, "The existing job metrics reader must discover native training events")
            # Resume after one optimizer update, including the incomplete final
            # accumulation window in epoch one, then compare the exact weights.
            args.resume = str(root / "output" / "krea2-lora-step000001-state")
            args.output_dir = str(root / "resumed")
            resumed = attach_adapter(copy.deepcopy(base).to("cuda", dtype=torch.bfloat16), args)
            accelerator.free_memory()
            train_cached(args, dataset, resumed, scheduler, accelerator)
            for name, value in get_peft_model_state_dict(resumed).items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            # Public pipeline loading must preserve alpha=1 / rank=2 metadata.
            inference_model = copy.deepcopy(base).to("cuda", dtype=torch.bfloat16)
            pipe = Krea2Pipeline(transformer=inference_model, vae=None, tokenizer=None, text_encoder=None, scheduler=scheduler)
            pipe.load_lora_weights(result["output"])
            adapted = [m for m in inference_model.modules() if hasattr(m, "scaling")]
            self.assertTrue(adapted)
            self.assertTrue(all(m.scaling["default_0"] == 0.5 for m in adapted))
            # Load the actual Diffusers NF4 backend, then verify LoRA gradients.
            args.base_precision = "nf4"
            quantized = attach_adapter(load_transformer(root / "model", args, torch.device("cuda:0")), args)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = flow_loss(quantized, torch.randn(1, 4, 1, 4, 6, device="cuda", dtype=torch.bfloat16), torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.bfloat16), torch.ones(1, 4, device="cuda", dtype=torch.bool), scheduler.config, args)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in quantized.parameters() if p.requires_grad))
            self.assertTrue(all(p.grad is None for p in quantized.parameters() if not p.requires_grad))


if __name__ == "__main__":
    unittest.main()
