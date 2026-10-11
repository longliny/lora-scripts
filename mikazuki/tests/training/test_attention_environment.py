import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mikazuki.catalog.loader import CatalogService
from mikazuki.compiler.compiler import compile_draft
from mikazuki.compiler.models import TrainDraft


class AttentionAvailabilityTests(unittest.TestCase):
    def test_catalog_refreshes_install_state_without_mutating_choices(self):
        catalog = CatalogService()
        with patch("mikazuki.training.attention_capabilities.attention_unavailable", return_value={"xformers": "missing", "flash": "missing flash"}):
            for trainer_id in ("sdxl.lora", "anima.lora", "krea2.lora"):
                trainer = catalog.get_trainer(trainer_id)
                self.assertIn("xformers", trainer.params["attn_mode"].choices)
                self.assertEqual(trainer.params["attn_mode"].unavailableChoices["xformers"], "missing")
                self.assertIn(trainer.params["attn_mode"].effectiveDefault, ("sdpa", "torch"))
            self.assertEqual(catalog.get_trainer("sd.lora").params["xformers"].unavailableReason, "missing")
        with patch("mikazuki.training.attention_capabilities.attention_unavailable", return_value={}):
            self.assertEqual(catalog.get_trainer("krea2.lora").params["attn_mode"].unavailableChoices, {})
            self.assertEqual(catalog.get_trainer("krea2.lora").params["attn_mode"].effectiveDefault, "flash")
            self.assertIsNone(catalog.get_trainer("sd.lora").params["xformers"].unavailableReason)

    @patch("mikazuki.training.attention_capabilities.attention_unavailable", return_value={"xformers": "xFormers missing"})
    def test_uninstalled_backend_rejected_for_selectors_and_legacy_switches(self, _missing):
        for trainer_id, params in (("sdxl.lora", {"attn_mode": "xformers"}), ("anima.lora", {"attn_mode": "xformers"}), ("krea2.lora", {"attn_mode": "xformers"}), ("sd.lora", {"xformers": True}), ("sdxl.lora", {"xformers": True})):
            with self.subTest(trainer=trainer_id, params=params), tempfile.TemporaryDirectory() as directory:
                result = compile_draft(TrainDraft(trainerId=trainer_id, name="test", dataset={"root": "train"}, params=params), persist=True, validate_paths=False, base_dir=Path(directory))
                self.assertIn("param.environment.unavailable", [error.code for error in result.errors])
                self.assertFalse(list(Path(directory).rglob("train.toml")))


@unittest.skipUnless(os.environ.get("KREA2_GPU_TESTS") == "1", "Set KREA2_GPU_TESTS=1 for training GPU checks")
class PriorityTrainingGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "dev"))

    def test_flash_rectangular_gqa_padding_and_gradients(self):
        import torch
        from mikazuki.training.flash_attention import flash_attention
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                tensors = [torch.randn(2, length, heads, 128, device="cuda", dtype=dtype, requires_grad=True) for length, heads in ((37, 8), (19, 2), (19, 2))]
                reference = [x.detach().clone().requires_grad_() for x in tensors]
                mask = torch.ones(2, 1, 1, 19, device="cuda", dtype=torch.bool)
                mask[0, :, :, 3:9] = False
                mask[1] = False
                result = flash_attention(*tensors, mask)
                expected = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in reference), attn_mask=mask, enable_gqa=True).transpose(1, 2)
                torch.testing.assert_close(result, expected, atol=0.015, rtol=0.03)
                gradient = torch.randn_like(result) * 0.01
                result.backward(gradient)
                expected.backward(gradient)
                for tensor, ref in zip(tensors, reference):
                    self.assertIsNotNone(tensor.grad)
                    torch.testing.assert_close(tensor.grad, ref.grad, atol=0.0004, rtol=0.04)

    def test_sdxl_self_cross_attention_gradients_and_lora_update(self):
        import torch
        from library.sdxl_original_unet import BasicTransformerBlock
        from networks.lora import LoRAModule
        from mikazuki.training.sdxl import configure_flash_attention
        reference = BasicTransformerBlock(128, 2, 64, 96).to("cuda", dtype=torch.bfloat16)
        reference.set_use_sdpa(True)
        model = copy.deepcopy(reference)
        original_keys = set(model.state_dict())
        self.assertEqual(configure_flash_attention(model), 2)
        self.assertEqual(set(model.state_dict()), original_keys)
        x = torch.randn(2, 37, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        context = torch.randn(2, 11, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ref_x, ref_c = x.detach().clone().requires_grad_(), context.detach().clone().requires_grad_()
        result, expected = model(x, context), reference(ref_x, ref_c)
        torch.testing.assert_close(result, expected, atol=0.025, rtol=0.03)
        result.float().square().mean().backward()
        expected.float().square().mean().backward()
        for actual, target in ((x.grad, ref_x.grad), (context.grad, ref_c.grad)):
            torch.testing.assert_close(actual, target, atol=0.0001, rtol=0.05)
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        adapter = LoRAModule("test", model.attn1.to_q, lora_dim=4, alpha=4)
        adapter.apply_to()
        adapter.to("cuda", dtype=torch.float32)
        before = adapter.lora_up.weight.detach().clone()
        base = model.attn1.to_q.weight.detach().clone()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = torch.utils.checkpoint.checkpoint(model, x.detach(), context.detach(), use_reentrant=False)
                loss = prediction.float().square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(before, adapter.lora_up.weight))
        self.assertTrue(torch.equal(base, model.attn1.to_q.weight))

    def test_anima_tiny_dit_flash_lora_checkpoint_and_reload(self):
        import torch
        from library.anima_models import Anima
        from networks.lora_anima import create_network
        from safetensors.torch import load_file
        from mikazuki.training.anima import configure_anima_precision
        model = Anima(max_img_h=16, max_img_w=16, max_frames=1, in_channels=4, out_channels=4,
                      patch_spatial=2, patch_temporal=1, concat_padding_mask=False, model_channels=256,
                      num_blocks=2, num_heads=2, crossattn_emb_channels=64, pos_emb_cls="rope3d",
                      use_adaln_lora=True, adaln_lora_dim=8, rope_enable_fps_modulation=False, attn_mode="flash")
        model.requires_grad_(False).to("cuda", dtype=torch.bfloat16)
        configure_anima_precision(model)
        model.enable_gradient_checkpointing()
        network = create_network(1.0, 4, 4, None, [], model)
        network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
        network.to("cuda", dtype=torch.float32)
        initial = {name: value.detach().clone() for name, value in network.named_parameters()}
        frozen = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
        for mode in ("torch", "flash"):
            model.attn_mode = mode
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(torch.randn(1, 4, 1, 8, 8, device="cuda", dtype=torch.bfloat16),
                                   torch.tensor([0.5], device="cuda"), context=torch.randn(1, 11, 64, device="cuda", dtype=torch.bfloat16))
                loss = (prediction.float() - 0.5).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().max() > 0 for p in network.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertTrue(any(not torch.equal(initial[name], value) for name, value in network.named_parameters()))
        self.assertTrue(all(torch.equal(frozen[name], value) for name, value in model.named_parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "anima-lora.safetensors")
            network.save_weights(path, torch.float32, {"ss_base_model_version": "anima"})
            saved = load_file(path, device="cuda")
            network.load_state_dict(saved, strict=True)
            for name, value in network.state_dict().items():
                # The upstream writer serializes the integer alpha buffer in
                # the requested save dtype; loading restores the buffer dtype.
                torch.testing.assert_close(value, saved[name].to(value.dtype))
