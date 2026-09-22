"""Optional, expensive held-out causal-LM fidelity evaluation for the Layer 26 SAE.

The forward hook replaces block index 25's post-block hidden state. By default it
patches only eligible assistant *input* token positions, matching SAE training.
Loss is scored at logits[t-1] against assistant token[t], including the first
assistant target whose predictor may be unpatched. ``--patch-scope all`` also
patches prompt/control states and therefore tests a broader intervention.
Teacher-forced full dialogs retain all context. No windows are concatenated and
no generated continuations or reference-answer metadata are used.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def _positions(positions, length: int, device) -> torch.Tensor:
    result = torch.as_tensor(positions, device=device, dtype=torch.long).reshape(-1)
    if result.numel() and ((result < 0).any() or (result >= length).any()):
        raise ValueError('Assistant token position is outside this document')
    if result.unique().numel() != result.numel():
        raise ValueError('Assistant token positions must be unique')
    return result


@torch.no_grad()
def masked_next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor,
                           assistant_positions, *, chunk_size: int = 256) -> tuple[float, int]:
    """Return summed CE and token count; target t is predicted by logits t-1.

    One full document per call avoids padding and multi-document causal leakage.
    Vocabulary promotion to float32 is chunked to bound temporary GPU memory.
    """
    if logits.ndim != 3 or logits.shape[0] != 1 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('Expected one complete document [1,sequence,...]')
    if logits.shape[1] != input_ids.shape[1] or chunk_size < 1:
        raise ValueError('Logit sequence length mismatch or invalid chunk size')
    targets = _positions(assistant_positions, input_ids.shape[1], logits.device)
    targets = targets[targets > 0]
    labels = input_ids.to(logits.device)
    total = 0.
    for part in targets.split(chunk_size):
        if part.numel():
            total += F.cross_entropy(logits[0, part - 1].float(), labels[0, part], reduction='sum').double().item()
    return total, targets.numel()


def reconstruction_hook(sae, normalizer: dict, *, positions=None, ablation: str | None = None):
    """Build a tuple-preserving post-block hook with reversible train normalization.

    ``positions=None`` patches every position. ``ablation='zero'`` means zero in
    raw model coordinates; ``'mean'`` uses the saved training activation mean.
    No validation mean is fitted. The hook does not mutate the original output.
    """
    if ablation not in (None, 'mean', 'zero'):
        raise ValueError('ablation must be mean, zero or None')
    scale = float(normalizer['scale'])
    if not 0 < scale < float('inf'):
        raise ValueError('Invalid saved normalization scale')

    @torch.no_grad()
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError('Expected one post-block hidden-state document [1,tokens,d_in]')
        idx = (torch.arange(hidden.shape[1], device=hidden.device) if positions is None
               else _positions(positions, hidden.shape[1], hidden.device))
        if not idx.numel():
            return output
        mean = torch.as_tensor(normalizer['mean'], device=hidden.device, dtype=torch.float32)
        if mean.shape != (hidden.shape[-1],):
            raise ValueError('SAE normalization dimension differs from LM hidden state')
        selected = hidden[0, idx].float()
        if ablation == 'zero':
            replacement = torch.zeros_like(selected)
        elif ablation == 'mean':
            raw_mean = torch.as_tensor(normalizer.get('data_mean', normalizer['mean']), device=hidden.device, dtype=torch.float32)
            replacement = raw_mean.expand_as(selected)
        else:
            # Keep the SAE local to the hooked block, including an accelerate
            # device_map that distributes transformer blocks across GPUs.
            sae.to(hidden.device)
            from sae_models import reconstruct
            reconstruction, _ = reconstruct(sae, (selected - mean) / scale)
            replacement = reconstruction.float() * scale + mean
        if replacement.shape != selected.shape or not torch.isfinite(replacement).all():
            raise ValueError('Non-finite or incorrectly shaped SAE reconstruction')
        patched = hidden.clone()
        patched[0, idx] = replacement.to(hidden.dtype)
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        if isinstance(output, list):
            return [patched] + output[1:]
        return patched
    return hook


@contextlib.contextmanager
def _installed_hook(block, hook):
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def resolve_block(model, block_index: int = 25):
    """Resolve Qwen/Llama blocks without assuming the whole model is on one GPU."""
    if block_index != 25:
        raise ValueError('This experiment is fixed to Layer 26 / block index 25')
    backbone = getattr(model, 'model', None)
    layers = getattr(backbone, 'layers', None)
    if layers is None or len(layers) <= block_index:
        raise ValueError('Expected ChemDFM-R-14B model.model.layers with at least 26 blocks')
    return layers[block_index]


@torch.inference_mode()
def evaluate_lm(model, block, sae, normalizer: dict, documents: Iterable[dict], *,
                patch_scope: str = 'assistant', baselines: tuple[str, ...] = ('mean', 'zero')) -> dict:
    if patch_scope not in ('assistant', 'all') or not baselines or set(baselines) - {'mean', 'zero'}:
        raise ValueError('Invalid patch scope or ablation baselines')
    model.eval()
    sae.eval()
    device = model.get_input_embeddings().weight.device
    modes = ('clean', 'reconstruction') + tuple(dict.fromkeys(baselines))
    totals = {mode: 0. for mode in modes}
    token_count = document_count = patched_count = 0
    ids_digest = hashlib.sha256()
    for document in documents:
        ids = torch.as_tensor(document['input_ids'], dtype=torch.long, device=device).reshape(1, -1)
        positions = _positions(document['assistant_positions'], ids.shape[1], device)
        n_targets = int((positions > 0).sum().item())
        if not n_targets:
            continue
        patch_positions = positions if patch_scope == 'assistant' else None
        for mode in modes:
            context = contextlib.nullcontext() if mode == 'clean' else _installed_hook(
                block, reconstruction_hook(sae, normalizer, positions=patch_positions,
                                           ablation=mode if mode in baselines else None))
            with context:
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                logits = output.logits if hasattr(output, 'logits') else output[0]
                loss, count = masked_next_token_loss(logits, ids, positions)
            if count != n_targets or not math.isfinite(loss):
                raise ValueError('Invalid or inconsistent loss token accounting')
            totals[mode] += loss
            del output, logits
        token_count += n_targets
        document_count += 1
        patched_count += ids.shape[1] if patch_scope == 'all' else positions.numel()
        ids_digest.update((str(document['id']) + '\n').encode())
    if not token_count:
        raise ValueError('No eligible held-out assistant targets')
    ce = {mode: loss / token_count for mode, loss in totals.items()}
    result = {
        'n_documents': document_count, 'n_scored_tokens': token_count,
        'n_patched_positions': patched_count, 'patch_scope': patch_scope,
        'clean_ce': ce['clean'], 'reconstruction_ce': ce['reconstruction'],
        'delta_ce': ce['reconstruction'] - ce['clean'],
        'evaluated_ids_sha256': ids_digest.hexdigest(),
        'ce_target_scope': 'eligible assistant token t scored from logits[t-1], t>0',
        'patch_position_scope': ('eligible assistant input token states' if patch_scope == 'assistant'
                                 else 'all dialog input token states including prompts and controls'),
    }
    for baseline in baselines:
        denominator = ce[baseline] - ce['clean']
        result[f'{baseline}_ablation_ce'] = ce[baseline]
        result[f'{baseline}_ablation_delta_ce'] = denominator
        result[f'{baseline}_loss_recovery'] = ((ce[baseline] - ce['reconstruction']) / denominator
                                               if abs(denominator) > 1e-12 else None)
        result[f'{baseline}_loss_recovery_denominator_positive'] = denominator > 0
    return result


def validate_lm_provenance(checkpoint: dict, model_path: Path, corpus_root: Path, selection_path: Path) -> dict:
    """Verify the LM/tokenizer, corpus and held-out selection against extraction."""
    from eval_sae import sha256_file
    manifest = checkpoint.get('cache_manifest')
    if not manifest:
        raise ValueError('Checkpoint lacks cache model provenance; re-save it with the current trainer')
    for key, value in (('layer', 26), ('block_index', 25), ('d_in', 5120),
                       ('activation_site', 'post_block_residual')):
        if manifest.get(key) != value:
            raise ValueError(f'Checkpoint cache {key} is incompatible with Layer 26 ChemDFM-R-14B')
    for name, digest in manifest.get('model_files', {}).items():
        if sha256_file(model_path / name) != digest:
            raise ValueError(f'Model/tokenizer identity mismatch: {name}')
    if not manifest.get('model_files'):
        raise ValueError('Checkpoint lacks model/tokenizer fingerprints')
    weights = [{'path': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
               for p in sorted(model_path.glob('*.safetensors'))]
    if not weights or weights != manifest.get('model_weights_identity'):
        raise ValueError('Model weight file identities differ from activation extraction')
    if sha256_file(corpus_root / 'manifest.json') != manifest.get('corpus_manifest_sha256'):
        raise ValueError('Corpus manifest differs from activation extraction')
    if sha256_file(selection_path) != manifest['splits']['validation'].get('selection_sha256'):
        raise ValueError('Held-out selection differs from activation extraction')
    from activation_inputs import read_selection
    corpus_manifest = json.loads((corpus_root / 'manifest.json').read_text())
    output_hashes = {entry['path']: entry['sha256'] for entry in corpus_manifest['outputs']}
    for name in sorted({entry['path'] for entry in read_selection(corpus_root, selection_path)}):
        if name not in output_hashes or sha256_file(corpus_root / name) != output_hashes[name]:
            raise ValueError(f'Validation corpus shard differs from its verified manifest: {name}')
    return {'model_tokenizer_hashes_verified': True, 'corpus_manifest_verified': True,
            'validation_source_shards_verified': True,
            'validation_selection_verified': True,
            'weight_identity_policy': 'file names/sizes/mtime; not cryptographic weight content checksums'}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=HERE.parent / 'chemical_models/ChemDFM-R-14B')
    parser.add_argument('--corpus-root', type=Path, default=HERE / 'data/v1')
    parser.add_argument('--selection', type=Path, default=HERE / 'data/v1/selections/pilot_10m/validation.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-map', default='auto', help='transformers device_map strategy or JSON map; honors CUDA_VISIBLE_DEVICES')
    parser.add_argument('--gpus', default='4,5,6,7', help='Physical GPU IDs; must agree with CUDA_VISIBLE_DEVICES')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--max-memory-gib', type=int, default=36, help='Per visible GPU cap; requires 1 GiB additional free VRAM')
    parser.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
    parser.add_argument('--max-documents', type=int, default=64, help='First N fixed selection documents; 0 evaluates all')
    parser.add_argument('--patch-scope', choices=('assistant', 'all'), default='assistant')
    parser.add_argument('--baselines', choices=('mean', 'zero', 'both'), default='both')
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.device == 'cuda':
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, args.gpus):
            raise ValueError('CUDA_VISIBLE_DEVICES conflicts with --gpus; resolve explicitly')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    from eval_sae import _project_path
    for key in ('checkpoint', 'model', 'corpus_root', 'selection', 'output'):
        setattr(args, key, _project_path(getattr(args, key)))
    # Lazy imports: importing this module never imports transformers or loads weights.
    from activation_inputs import encode_document, iter_selected_documents
    from corpus import LocalTokenizer
    from eval_sae import load_checkpoint, sha256_file, write_json
    from transformers import AutoConfig, AutoModelForCausalLM
    if args.max_documents < 0:
        raise ValueError('--max-documents must be nonnegative')
    if not args.model.is_dir() or args.model.name != 'ChemDFM-R-14B':
        raise ValueError('--model must be the local ChemDFM-R-14B checkpoint directory')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    config = AutoConfig.from_pretrained(str(args.model), local_files_only=True)
    if config.hidden_size != 5120 or config.num_hidden_layers <= 25:
        raise ValueError('Checkpoint is not the expected 5120-wide ChemDFM-R-14B architecture')
    sae, normalizer, checkpoint = load_checkpoint(args.checkpoint, device='cpu', require_receipt=True)
    if len(normalizer['mean']) != config.hidden_size:
        raise ValueError('SAE input width does not match the model')
    provenance = validate_lm_provenance(checkpoint, args.model, args.corpus_root, args.selection)
    tokenizer = LocalTokenizer(args.model)
    def encoded_documents():
        for i, row in enumerate(iter_selected_documents(args.corpus_root, args.selection)):
            if args.max_documents and i >= args.max_documents:
                break
            if row.get('split') != 'validation':
                raise ValueError('Delta LM permits validation records only; train rows are forbidden')
            ids, positions = encode_document(row, tokenizer)
            yield {'id': row['id'], 'input_ids': ids, 'assistant_positions': positions}
    # Materialize only the requested IDs/tokens to validate split/masks before loading weights.
    documents = list(encoded_documents())
    if not documents:
        raise ValueError('Empty validation selection')
    device_map = json.loads(args.device_map) if args.device_map.startswith('{') else args.device_map
    kwargs = dict(local_files_only=True, dtype=getattr(torch, args.dtype), device_map=device_map, attn_implementation='sdpa')
    if args.device == 'cpu':
        kwargs['device_map'] = 'cpu'
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, args.gpus):
            raise ValueError('CUDA_VISIBLE_DEVICES conflicts with --gpus; resolve explicitly')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        if args.max_memory_gib <= 0 or torch.cuda.device_count() != len(args.gpus.split(',')):
            raise ValueError('Invalid GPU memory budget or unavailable requested GPUs')
        for device in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(device)
            if free < (args.max_memory_gib + 1) * 1024 ** 3:
                raise RuntimeError(f'GPU logical {device} has {free / 1024**3:.1f} GiB free; requires {args.max_memory_gib + 1} GiB')
        kwargs['max_memory'] = {i: f'{args.max_memory_gib}GiB' for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(str(args.model), **kwargs)
    if args.device == 'cuda' and any(str(v) in ('cpu', 'disk') for v in getattr(model, 'hf_device_map', {}).values()):
        raise RuntimeError('Model offloaded to CPU/disk; increase available GPU memory')
    baselines = ('mean', 'zero') if args.baselines == 'both' else (args.baselines,)
    result = evaluate_lm(model, resolve_block(model), sae, normalizer, documents,
                         patch_scope=args.patch_scope, baselines=baselines)
    result.update({
        'schema_version': 1, 'evaluation': 'heldout_teacher_forced_delta_lm',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha256_file(args.checkpoint),
        'cache_fingerprint': checkpoint['cache_fingerprint'],
        'model': str(args.model.resolve()), 'model_config_sha256': sha256_file(args.model / 'config.json'),
        'selection': str(args.selection.resolve()), 'selection_sha256': sha256_file(args.selection),
        'normalization_fit_split': 'train', 'evaluation_split': 'validation',
        'layer': 26, 'block_index': 25, 'hook_site': 'post_block_output',
        'device_map': str(getattr(model, 'hf_device_map', device_map)),
        'max_documents': args.max_documents, 'provenance': provenance,
        'interpretation': 'LM fidelity under reconstruction; not a hallucination detection score',
    })
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
