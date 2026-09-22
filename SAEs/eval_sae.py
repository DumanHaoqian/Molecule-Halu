"""Held-out cache evaluation with train-fitted normalization and native SAE inference.

No ChemDFM weights are loaded. Reports original activation units, normalized
units, global feature usage, and Matryoshka prefix quality. BatchTopK inference
uses its saved threshold through sae_models.reconstruct, independent of batching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from activation_store import ActivationStore, denormalize, normalize
from metrics import ReconstructionMetrics
from sae_models import build_sae, reconstruct

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomic artifact write. CLI callers refuse existing output paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path, device: str = 'cpu', *, require_receipt: bool = False) -> tuple:
    """Load a trusted local trainer checkpoint and validate dimensions/normalizer.

    Training checkpoints contain optimizer/RNG Python objects, so this loader is
    for locally produced checkpoints. A trainer receipt is verified if present.
    """
    path = Path(path)
    receipt = path.with_name(path.name + '.receipt.json')
    if require_receipt and not receipt.exists():
        raise ValueError(f'Checkpoint receipt is missing: {receipt}')
    if receipt.exists():
        expected = json.loads(receipt.read_text())
        if sha256_file(path) != expected['sha256']:
            raise ValueError('Checkpoint checksum differs from its trainer receipt')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format_version')
    for key in ('config', 'normalizer', 'sae_state_dict', 'cache_fingerprint'):
        if key not in checkpoint:
            raise ValueError(f'Checkpoint is missing {key}')
    if receipt.exists():
        for key in ('step', 'tokens_seen', 'cache_fingerprint', 'config_fingerprint'):
            if checkpoint.get(key) != expected.get(key):
                raise ValueError(f'Checkpoint {key} differs from its trainer receipt')
    cfg = checkpoint['config']['sae']
    normalizer = dict(checkpoint['normalizer'])
    mean = torch.as_tensor(normalizer['mean'], dtype=torch.float32)
    scale = float(normalizer['scale'])
    if mean.shape != (cfg['d_in'],) or not torch.isfinite(mean).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid checkpoint normalization statistics')
    if int(normalizer.get('n_tokens', 0)) <= 0:
        raise ValueError('Missing training normalization token count')
    normalizer['mean'] = mean
    if 'data_mean' in normalizer:
        data_mean = torch.as_tensor(normalizer['data_mean'], dtype=torch.float32)
        if data_mean.shape != mean.shape or not torch.isfinite(data_mean).all():
            raise ValueError('Invalid saved training data mean')
        normalizer['data_mean'] = data_mean
    sae = build_sae(cfg, device=device)
    sae.load_state_dict(checkpoint['sae_state_dict'], strict=True)
    sae.eval()
    return sae, normalizer, checkpoint


@torch.inference_mode()
def evaluate_checkpoint(checkpoint_path: Path, *, cache_dir: Path | None = None,
                        device: str = 'cpu', batch_size: int = 1024,
                        max_tokens: int | None = None, verify_hashes: bool = True,
                        dense_threshold: float = .1, include_prefixes: bool = True,
                        require_receipt: bool = False) -> dict:
    if batch_size <= 0 or (max_tokens is not None and max_tokens <= 0):
        raise ValueError('Batch size and optional token limit must be positive')
    sae, normalizer, checkpoint = load_checkpoint(checkpoint_path, device=device, require_receipt=require_receipt)
    root = Path(cache_dir or checkpoint['config']['cache_dir'])
    store = ActivationStore(root, verify_hashes=verify_hashes)
    if store.fingerprint != checkpoint['cache_fingerprint']:
        raise ValueError('Evaluation cache fingerprint differs from the training checkpoint')
    if store.d_in != sae.cfg.d_in:
        raise ValueError('Evaluation cache input dimension differs from SAE')
    kwargs = {'dense_threshold': dense_threshold}
    raw_metrics, normalized_metrics = ReconstructionMetrics(**kwargs), ReconstructionMetrics(**kwargs)
    widths = list(getattr(sae.cfg, 'matryoshka_widths', [])) if include_prefixes else []
    prefix_metrics = {int(width): ReconstructionMetrics(**kwargs) for width in widths if width < sae.cfg.d_sae}
    for array in store.iter_batches('validation', batch_size, max_tokens=max_tokens):
        x = torch.from_numpy(array).to(device=device, dtype=torch.float32)
        x_norm = normalize(x, normalizer)
        reconstruction, z = reconstruct(sae, x_norm)
        normalized_metrics.update(x_norm, reconstruction, z)
        raw_metrics.update(x, denormalize(reconstruction, normalizer), z)
        for width, metrics in prefix_metrics.items():
            prefix = z.clone()
            prefix[:, width:] = 0
            recon = sae.decode(prefix)
            metrics.update(x, denormalize(recon, normalizer), prefix[:, :width])
    result = {
        'schema_version': 1, 'evaluation': 'heldout_activation_reconstruction',
        'evaluation_split': 'validation', 'normalization_fit_split': 'train',
        'checkpoint': str(Path(checkpoint_path).resolve()), 'checkpoint_sha256': sha256_file(checkpoint_path),
        'checkpoint_step': checkpoint.get('step'), 'checkpoint_tokens_seen': checkpoint.get('tokens_seen'),
        'architecture': checkpoint['config']['sae']['architecture'],
        'sae_config': checkpoint['config']['sae'], 'cache_dir': str(store.root),
        'cache_fingerprint': store.fingerprint, 'cache_hashes_verified': verify_hashes,
        'cache_validation_tokens': store.n_tokens('validation'),
        'max_tokens': max_tokens, 'batch_size': batch_size,
        'layer': 26, 'block_index': 25, 'activation_site': 'post_block_residual',
        'normalizer': {key: value.tolist() if isinstance(value, torch.Tensor) else value
                       for key, value in normalizer.items()},
        'metrics': raw_metrics.compute(), 'normalized_metrics': normalized_metrics.compute(),
        'matryoshka_prefix_metrics': {str(width): metrics.compute() for width, metrics in prefix_metrics.items()},
        'feature_counts': raw_metrics.feature_counts.tolist(),
        'feature_frequencies': raw_metrics.feature_frequencies.tolist(),
        'metric_definitions': {
            'units': 'metrics and prefix metrics use raw block output; normalized_metrics uses saved train centering/scaling',
            'nmse_fvu': 'sum squared reconstruction error / sum squared deviations from global validation feature mean',
            'fraction_variance_explained': '1 - FVU',
            'explained_variance': '1 - centered residual variance / centered input variance',
            'relative_norm_error': 'mean per-token ||reconstruction-input||_2 / max(||input||_2,1e-12)',
            'cosine_similarity': 'mean token cosine; zero vector pairs contribute 0',
            'dead_features': 'features never observed active over this complete evaluation stream',
            'dense_features': f'fraction of features active on at least {dense_threshold:g} of evaluated tokens',
            'prefixes': 'encode full dictionary once, retain leading features, decode with native scaling and bias; feature usage denominator is prefix width',
            'variance_zero': 'ratios use 1e-12 denominator floor and variance_is_zero=1 when centered variance is at most 1e-12',
        },
        'interpretation': 'Reconstruction and observed feature coverage do not establish hallucination detection quality',
    }
    if hasattr(sae, 'topk_threshold'):
        threshold = float(sae.topk_threshold)
        result['inference_gate'] = {'threshold': threshold, 'mode': 'learned_threshold' if threshold > 0 else 'uncalibrated_per_token_topk_fallback'}
    else:
        result['inference_gate'] = {'mode': 'native_encode'}
    return result


def _project_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (HERE / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=_project_path, required=True)
    parser.add_argument('--cache-dir', type=_project_path)
    parser.add_argument('--output', type=_project_path, required=True)
    parser.add_argument('--device', default='cpu', help='cpu or a visible CUDA device; default is CPU')
    parser.add_argument('--gpus', default='4,5,6,7', help='Authorized physical GPU IDs made visible before CUDA initialization')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-tokens', type=int, help='Fixed first N validation tokens; omitted evaluates the complete held-out cache')
    parser.add_argument('--dense-threshold', type=float, default=.1)
    parser.add_argument('--skip-cache-hashes', action='store_true')
    parser.add_argument('--skip-prefixes', action='store_true')
    parser.add_argument('--wandb-mode', choices=('disabled', 'offline', 'online'), default='disabled')
    parser.add_argument('--wandb-project', default='chemdfm-r-sae')
    parser.add_argument('--wandb-entity')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.device.startswith('cuda'):
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, args.gpus):
            raise ValueError('CUDA_VISIBLE_DEVICES conflicts with --gpus; resolve explicitly')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    result = evaluate_checkpoint(args.checkpoint, cache_dir=args.cache_dir, device=args.device,
                                batch_size=args.batch_size, max_tokens=args.max_tokens,
                                verify_hashes=not args.skip_cache_hashes,
                                dense_threshold=args.dense_threshold, include_prefixes=not args.skip_prefixes,
                                require_receipt=True)
    write_json(args.output, result)
    if args.wandb_mode != 'disabled':
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
                         job_type='evaluate', config={'checkpoint': str(args.checkpoint),
                                                     'checkpoint_sha256': result['checkpoint_sha256'],
                                                     'cache_fingerprint': result['cache_fingerprint']})
        try:
            values = {f'eval/{key}': value for key, value in result['metrics'].items()}
            values.update({f'eval_normalized/{key}': value for key, value in result['normalized_metrics'].items()})
            for width, metrics in result['matryoshka_prefix_metrics'].items():
                values.update({f'eval_prefix_{width}/{key}': value for key, value in metrics.items()})
            run.log(values)
            artifact = wandb.Artifact(f'sae-eval-{result["checkpoint_sha256"][:12]}', type='evaluation')
            artifact.add_file(str(args.output))
            run.log_artifact(artifact)
        finally:
            run.finish()
    print(json.dumps({'output': str(args.output), 'metrics': result['metrics'],
                      'matryoshka_prefix_metrics': result['matryoshka_prefix_metrics']}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
