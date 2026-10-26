"""Translate workbench settings; delegate all training/sampling to sd-scripts."""
import json
from pathlib import Path
import time

import toml

def resolve_model_paths(args):
    reference = Path(args.pretrained_model_name_or_path).expanduser()
    if not reference.exists():
        # Hub lookup is deferred until launch; UI preflight never downloads weights.
        from huggingface_hub import snapshot_download
        reference = Path(snapshot_download(args.pretrained_model_name_or_path, revision=args.revision,
                                          local_files_only=args.local_files_only,
                                          allow_patterns=['model_index.json', 'transformer/*', 'text_encoder/*', 'tokenizer/*', 'vae/*']))
    reference = reference.resolve()
    root = reference if reference.is_dir() else None

    def checkpoint(explicit, folder, pattern):
        if explicit:
            path = Path(explicit).expanduser().resolve()
        elif root is not None:
            matches = sorted((root / folder).glob(pattern))
            if not matches:
                raise ValueError(f'MOSS 模型缺少 {folder}/{pattern}')
            path = matches[0]
        else:
            raise ValueError(f'单文件底模必须另行指定 {folder}。')
        if not path.is_file() or path.suffix.lower() != '.safetensors':
            raise ValueError(f'模型权重必须是本地 safetensors 文件：{path}')
        return str(path)

    return {
        'pretrained_model_name_or_path': checkpoint(str(reference) if reference.is_file() else None, 'transformer', 'diffusion_pytorch_model*.safetensors'),
        'text_encoder': checkpoint(args.text_encoder, 'text_encoder', 'model*.safetensors'),
        'vae': checkpoint(args.vae, 'vae', 'diffusion_pytorch_model*.safetensors'),
        'tokenizer_cache_dir': str(Path(args.tokenizer_cache_dir).resolve()) if args.tokenizer_cache_dir else str(root / 'tokenizer') if root is not None else None,
    }


def upstream_config(args, model_paths):
    """Produce explicit native options, suitable for saving and replaying upstream."""
    omitted = {'config_file', 'revision', 'local_files_only', 'base_precision', 'lora_target',
               'max_sequence_length', 'weight_decay'} | set(model_paths)
    config = {key: value for key, value in vars(args).items() if key not in omitted and value is not None}
    config.update({key: value for key, value in model_paths.items() if value is not None})
    config.update(network_train_unet_only=True, cache_latents=True, cache_text_encoder_outputs=True,
                  moss_max_token_length=args.max_sequence_length, attn_mode='torch',
                  weighting_scheme='logit_normal', save_model_as='safetensors', save_precision='float',
                  optimizer_args=[f'weight_decay={args.weight_decay}'], max_data_loader_n_workers=0,
                  save_state_on_train_end=args.save_state, skip_until_initial_step=bool(args.resume))
    # Match the previous implementation's 42 main attention/MLP projections.
    # Explicit patterns exclude time modulation/refiners even if upstream defaults change.
    target = r'layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)' if args.lora_target == 'attention_mlp' else r'layers\.\d+\.proj\.(q|k|v|o)'
    config['network_args'] = ["exclude_patterns=['.*']", f'include_patterns={[target]!r}']
    return config


def train(args):
    import torch
    import moss_image_train_network as upstream
    source_root = Path(upstream.__file__).resolve().parent
    provenance = json.loads((source_root / 'UPSTREAM.json').read_text(encoding='utf-8'))
    config = upstream_config(args, resolve_model_paths(args))
    parser = upstream.setup_parser()
    actions = {action.dest: action for action in parser._actions}
    unknown = sorted(set(config) - set(actions))
    if unknown:
        raise ValueError(f'Unsupported pinned upstream options: {unknown}')
    native = parser.parse_args([])
    for key, value in config.items():
        action = actions[key]
        if action.choices and value not in action.choices:
            raise ValueError(f'Invalid upstream {key}: {value}')
        setattr(native, key, value)
    upstream.args_util.verify_command_line_training_args(native)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    native_path = Path(args.config_file).with_suffix('.upstream.toml') if args.config_file else output / f'{args.output_name}-upstream.toml'
    native_path.write_text(toml.dumps(config), encoding='utf-8')
    print(f'MOSS backend: local sd-scripts {provenance["ref"]} ({provenance["commit"]}) at {source_root}', flush=True)
    print(f'Native config: {native_path}', flush=True)
    print('Sampling: upstream logit_normal; legacy truncated + 1% uniform distribution is not used.', flush=True)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    upstream.MossImageNetworkTrainer().train(native)
    result = {'backend': 'sd-scripts', 'version': provenance['ref'], 'commit': provenance['commit'],
              'source_root': str(source_root),
              'elapsed_seconds': time.perf_counter() - started, 'upstream_config': str(native_path),
              'output': str(output / f'{args.output_name}.safetensors'),
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None}
    (output / f'{args.output_name}-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
