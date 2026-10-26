import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import toml

from mikazuki.catalog.loader import CatalogService
from mikazuki.compiler.compiler import compile_draft
from mikazuki.compiler.models import TrainDraft
from mikazuki.training.moss_image import parse_args, setup_parser


class MossConfigurationTests(unittest.TestCase):
    def draft(self, **kwargs):
        return TrainDraft(**dict(dict(trainerId='moss_image.lora', name='moss-test', dataset={'root': 'train'}), **kwargs))

    def test_ui_toml_matches_parser_and_uses_shared_python(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=False):
            os.environ['MIKAZUKI_MOSS_PYTHON'] = 'obsolete-isolated-python.exe'
            result = compile_draft(self.draft(), persist=False, validate_paths=False, base_dir=Path(directory))
            self.assertEqual(result.errors, [])
            self.assertEqual(sys.executable, result.command[0])
            self.assertIn('scripts/moss_image_train_network.py', result.command)
            path = Path(directory) / 'train.toml'
            path.write_text(result.trainConfig, encoding='utf-8')
            args = parse_args(['--config_file', str(path)])
            self.assertEqual(args.pretrained_model_name_or_path, 'OpenMOSS-Team/MOSS-Image-Base')
            self.assertEqual((args.base_precision, args.attn_mode), ('bf16', 'sdpa'))
            self.assertEqual((args.timestep_sampling, args.logit_mean, args.logit_std), ('sigma', -1.09, 0.95))
            self.assertEqual(args.network_module, 'networks.lora_generic')
            self.assertTrue(args.gradient_checkpointing)
            self.assertTrue(args.save_state)
            self.assertEqual(toml.loads(result.datasetConfig)['datasets'][0]['bucket_reso_steps'], 16)
            self.assertTrue(toml.loads(result.datasetConfig)['general']['enable_multiline_caption'])
            trainer = CatalogService().get_trainer('moss_image.lora')
            self.assertEqual(set(trainer.params), {a.dest for a in setup_parser()._actions if a.dest != 'help'})

    def test_unsupported_settings_fail_in_workbench(self):
        for draft, expected in (
            (self.draft(runtime={'gpuIds': ['0','1']}), 'moss_image.single_gpu'),
            (self.draft(dataset={'root':'train', 'shuffleCaption':True}), 'moss_image.cached_captions'),
            (self.draft(dataset={'root':'train', 'resolution':[1000,1024]}), 'moss_image.resolution'),
            (self.draft(modelAssets={'pretrained_model_name_or_path':'moss.safetensors'}), 'moss_image.component_required'),
            (self.draft(params={'base_precision':'nf4'}), 'param.invalid'),
            (self.draft(params={'blocks_to_swap': 4, 'block_swap_h2d_only': True, 'gradient_checkpointing': False}), 'moss_image.block_swap'),
        ):
            with self.subTest(expected=expected):
                result = compile_draft(draft, persist=False, validate_paths=False)
                self.assertIn(expected, [e.code for e in result.errors])

    def test_false_switches_and_long_caption_cap_survive_toml(self):
        with tempfile.TemporaryDirectory() as directory:
            result = compile_draft(self.draft(params={'gradient_checkpointing':False, 'max_train_steps':6,
                'max_sequence_length':4096}), persist=False, validate_paths=False)
            self.assertEqual(result.errors, [])
            path = Path(directory) / 'train.toml'
            path.write_text(result.trainConfig, encoding='utf-8')
            args = parse_args(['--config_file', str(path), '--learning_rate', '0.0002'])
            self.assertFalse(args.gradient_checkpointing)
            self.assertIsNone(args.max_train_epochs)
            self.assertEqual((args.max_train_steps,args.max_sequence_length,args.learning_rate),(6,4096,0.0002))

    def test_parser_rejects_foreign_flags_and_bad_values(self):
        for content in ('network_dim = 1.5','base_precision = "fp8"','network_train_text_encoder_only = true'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'train.toml'
                path.write_text(content, encoding='utf-8')
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    parse_args(['--config_file', str(path)])
        with self.assertRaises(ValueError):
            parse_args(['--output_name', '../bad'])

    def test_legacy_draft_migration_is_explicit_and_preserves_user_settings(self):
        old = self.draft(params={'network_module': 'moss.lora', 'timestep_sampling': 'moss',
                               'vae_tiling': True, 'training_cache_dir': 'old-cache', 'learning_rate': 0.0002})
        result = compile_draft(old, persist=False, validate_paths=False)
        self.assertEqual(result.errors, [])
        self.assertEqual(len([w for w in result.warnings if w.code == 'moss_image.migrated']), 4)
        config = toml.loads(result.trainConfig)
        flat = {k: v for section in config.values() for k, v in section.items()}
        self.assertEqual(flat['network_module'], 'networks.lora_generic')
        self.assertEqual(flat['timestep_sampling'], 'sigma')
        self.assertEqual(flat['learning_rate'], 0.0002)
        self.assertNotIn('training_cache_dir', flat)
        self.assertNotIn('vae_tiling', flat)
        self.assertEqual(old.params['timestep_sampling'], 'moss')

    def test_model_directory_resolves_to_upstream_components(self):
        from mikazuki.training.moss_upstream import resolve_model_paths, upstream_config
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('transformer/diffusion_pytorch_model-00001-of-00003.safetensors',
                         'text_encoder/model-00001-of-00004.safetensors', 'vae/diffusion_pytorch_model.safetensors'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / 'tokenizer').mkdir()
            args = parse_args(['--pretrained_model_name_or_path', str(root), '--max_train_steps', '6'])
            native = upstream_config(args, resolve_model_paths(args))
            self.assertTrue(native['pretrained_model_name_or_path'].endswith('00001-of-00003.safetensors'))
            self.assertEqual(native['tokenizer_cache_dir'], str(root / 'tokenizer'))
            self.assertEqual(native['attn_mode'], 'torch')
            self.assertTrue(native['network_train_unet_only'])
            self.assertTrue(native['cache_text_encoder_outputs'])
            self.assertEqual(native['weighting_scheme'], 'logit_normal')
            self.assertEqual(native['optimizer_args'], ['weight_decay=0.01'])
            self.assertNotIn('base_precision', native)

    def test_legacy_state_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'state.pt').touch()
            with self.assertRaisesRegex(ValueError, '不兼容'):
                parse_args(['--resume', folder])

    def test_single_file_with_explicit_components_compiles(self):
        result = compile_draft(self.draft(modelAssets={'pretrained_model_name_or_path': 'moss.safetensors',
            'text_encoder': 'qwen.safetensors', 'vae': 'vae.safetensors'}), persist=False, validate_paths=False)
        self.assertEqual(result.errors, [])
