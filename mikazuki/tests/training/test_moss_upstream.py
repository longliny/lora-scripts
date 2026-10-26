"""Exercise MOSS alongside the other trainers from the same local sd-scripts."""
import sys
from pathlib import Path
import tempfile
import unittest


class MossUpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_root = Path(__file__).resolve().parents[3] / 'scripts' / 'dev'
        sys.path.insert(0, str(cls.source_root))
        import moss_image_train_network
        cls.upstream = moss_image_train_network

    def test_trainers_share_local_library_in_one_process(self):
        import library
        import train_network
        from mikazuki.training import anima, sdxl, krea2
        self.assertEqual(Path(library.__file__).resolve().parent, self.source_root / 'library')
        self.assertEqual(Path(self.upstream.__file__).resolve().parent, self.source_root)
        for trainer in (self.upstream.MossImageNetworkTrainer, anima.AnimaNetworkTrainer, sdxl.SdxlNetworkTrainer):
            self.assertTrue(issubclass(trainer, train_network.NetworkTrainer))
        for module in (self.upstream, anima, sdxl, krea2):
            self.assertIsNotNone(module.setup_parser())

    def config(self, target='attention_mlp'):
        from mikazuki.training.moss_image import parse_args
        from mikazuki.training.moss_upstream import upstream_config
        args = parse_args(['--lora_target', target, '--max_train_steps', '6', '--network_dim', '4', '--network_alpha', '2'])
        return upstream_config(args, {'pretrained_model_name_or_path': 'dit.safetensors',
                                     'text_encoder': 'te.safetensors', 'vae': 'vae.safetensors'})

    def test_native_parser_accepts_all_translated_settings(self):
        config = self.config()
        actions = {a.dest: a for a in self.upstream.setup_parser()._actions}
        self.assertEqual(set(config) - set(actions), set())
        for key, value in config.items():
            if actions[key].choices:
                self.assertIn(value, actions[key].choices)

    def make_network(self, target):
        from library.moss_image_models import MossImage
        from networks.lora_generic import create_network
        model = MossImage(dim=32, ffn_dim=48, num_heads=4, num_layers=2, text_dim=24,
                          freq_dim=16, latent_channels=8, attn_mode='torch')
        model.requires_grad_(False)
        kwargs = dict(item.split('=', 1) for item in self.config(target)['network_args'])
        network = create_network(1, 4, 2, None, [], model, **kwargs)
        network.apply_to([], model, False, True)
        return model, network

    def test_attention_target_excludes_mlp_and_refiners(self):
        _, network = self.make_network('attention')
        self.assertEqual(len(network.unet_loras), 8)
        self.assertTrue(all('_proj_' in m.lora_name and 'ffn' not in m.lora_name for m in network.unet_loras))

    def test_frozen_base_gradient_checkpointing_and_adapter_reload(self):
        import torch
        from safetensors.torch import load_file
        torch.manual_seed(17)
        model, network = self.make_network('attention_mlp')
        self.assertEqual(len(network.unet_loras), 12)
        base = {n: p.detach().clone() for n, p in model.named_parameters()}
        model.enable_gradient_checkpointing()
        model.train()
        network.train()
        x = torch.randn(1, 8, 2, 2, requires_grad=True)
        text = torch.randn(1, 3, 24, requires_grad=True)
        timestep = torch.tensor([500.0])
        mask = torch.ones(1, 3, dtype=torch.int32)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
        loss = model(x, timestep, text, mask).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters()))
        optimizer.step()
        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param, base[name]), name)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'adapter.safetensors')
            network.save_weights(path, torch.float32, {'ss_network_module': 'networks.lora_generic'})
            state = load_file(path)
            self.assertEqual(len(state), 36)
            model.eval()
            network.eval()
            with torch.no_grad():
                expected = model(x.detach(), timestep, text.detach(), mask)
                for param in network.parameters():
                    param.zero_()
                network.load_weights(path)
                actual = model(x.detach(), timestep, text.detach(), mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
