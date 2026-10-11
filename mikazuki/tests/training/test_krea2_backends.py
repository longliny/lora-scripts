import copy
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch

from mikazuki.training.krea2 import parse_args
from mikazuki.training.krea2_precision import FrozenFP8Linear, dequantize


class FP8LinearTests(unittest.TestCase):
    def test_frozen_fp8_matches_dequantized_reference_and_retains_no_bf16_weights(self):
        base = torch.nn.Linear(32, 24, bias=True).requires_grad_(False)
        # Cover the zero row and large dynamic range without FP8 overflow.
        base.weight[0].zero_()
        base.weight[1].mul_(100)
        quantized = FrozenFP8Linear.from_linear(base)
        self.assertEqual(quantized.weight.dtype, torch.float8_e4m3fn)
        inputs = torch.randn(2, 3, 32, requires_grad=True)
        reference_input = inputs.detach().clone().requires_grad_()
        weight = dequantize(quantized.weight, quantized.weight_scale, torch.float32)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
            output = quantized(inputs)
        expected = torch.nn.functional.linear(reference_input, weight, base.bias)
        torch.testing.assert_close(output, expected)
        gradient = torch.randn_like(output)
        output.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(inputs.grad, reference_input.grad)
        self.assertIsNone(quantized.weight.grad)
        self.assertEqual([t.dtype for t in saved], [torch.float8_e4m3fn, torch.float32])
        self.assertTrue(torch.isfinite(weight).all())
        with self.assertRaisesRegex(ValueError, "frozen"):
            FrozenFP8Linear.from_linear(torch.nn.Linear(2, 2))


@unittest.skipUnless(os.environ.get("KREA2_GPU_TESTS") == "1", "Set KREA2_GPU_TESTS=1 for CUDA integration")
class Krea2BackendGPUTests(unittest.TestCase):
    def test_flash_and_xformers_gqa_padding_forward_and_backward_match_sdpa(self):
        from mikazuki.training.krea2_attention import attention, validate_attention

        torch.manual_seed(32)
        from mikazuki.training.attention_capabilities import package_available
        modes = ["flash"]
        if package_available("xformers", "xformers"):
            modes.append("xformers")
        for mode in modes:
            validate_attention(mode, torch.device("cuda:0"))
            for masked in (False, True):
                with self.subTest(mode=mode, masked=masked):
                    tensors = [torch.randn(2, 17, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for heads in (8, 2, 2)]
                    reference = [x.detach().clone().requires_grad_() for x in tensors]
                    mask = torch.ones(2, 1, 1, 17, device="cuda", dtype=torch.bool) if masked else None
                    if mask is not None:
                        # Non-contiguous padding before suffix/image tokens.
                        mask[0, :, :, 3:9] = False
                        mask[1, :, :, 5:7] = False
                    result = attention(*tensors, mask, mode)
                    expected = attention(*reference, mask, "sdpa")
                    torch.testing.assert_close(result, expected, atol=0.015, rtol=0.03)
                    gradient = torch.randn_like(result) * 0.01
                    result.backward(gradient)
                    expected.backward(gradient)
                    for value, ref in zip(tensors, reference):
                        torch.testing.assert_close(value.grad, ref.grad, atol=0.0004, rtol=0.04)

    def test_fp8_transformer_lora_update_and_public_adapter_reload(self):
        from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline
        from mikazuki.tests.training.test_krea2 import tiny_transformer
        from mikazuki.training.krea2_model import attach_adapter, flow_loss, load_transformer, save_adapter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = tiny_transformer()
            base.save_pretrained(root / "transformer")
            args = parse_args(["--base_precision", "fp8", "--attn_mode", "flash", "--network_dim", "2", "--network_alpha", "1", "--gradient_checkpointing"])
            model = attach_adapter(load_transformer(root, args, torch.device("cuda:0")), args)
            fp8_weights = {name: p.float().clone() for name, p in model.named_parameters() if p.dtype == torch.float8_e4m3fn}
            self.assertTrue(fp8_weights)
            adapter = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
            self.assertTrue(all(p.dtype == torch.float32 for p in adapter.values()))
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = flow_loss(model, torch.randn(2, 4, 1, 4, 6, device="cuda", dtype=torch.bfloat16), torch.randn(2, 5, 2, 16, device="cuda", dtype=torch.bfloat16), torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 1, 1]], device="cuda", dtype=torch.bool), scheduler.config, args)
                self.assertTrue(math.isfinite(loss.item()))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            self.assertTrue(any(not torch.equal(p, adapter[n]) for n, p in model.named_parameters() if p.requires_grad))
            for name, p in model.named_parameters():
                if name in fp8_weights:
                    self.assertFalse(p.requires_grad)
                    self.assertIsNone(p.grad)
                    torch.testing.assert_close(p.float(), fp8_weights[name], rtol=0, atol=0)
            save_adapter(model, root / "fp8-lora.safetensors")
            pipe = Krea2Pipeline(transformer=copy.deepcopy(base).to("cuda", dtype=torch.bfloat16), vae=None, text_encoder=None, tokenizer=None, scheduler=scheduler)
            pipe.load_lora_weights(str(root / "fp8-lora.safetensors"))
            self.assertTrue(all(m.scaling["default_0"] == 0.5 for m in pipe.transformer.modules() if hasattr(m, "scaling")))
