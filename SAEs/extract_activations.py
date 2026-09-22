"""Extract verified assistant post-token residuals; --preflight never loads weights.

All CLI paths are relative to this project, regardless of current working directory.
Extraction caches commit whole documents, permitting restart from the last shard.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np

from activation_inputs import encode_document, iter_selected_documents, read_selection
from corpus import LocalTokenizer
from prepare import dump, file_sha

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent/'chemical_models/ChemDFM-R-14B'


def prefix_config(config, block_index=25):
    """Same causal prefix and original weights; omit unreachable later blocks.

    Capture still happens before the backbone's final norm. Only Qwen2's
    standard (non-dynamic RoPE) configuration is supported for this optimization.
    """
    if config.model_type != 'qwen2' or config.num_hidden_layers <= block_index:
        raise ValueError('Expected Qwen2 with the requested target block')
    if getattr(config,'rope_scaling',None):
        raise ValueError('Prefix loading requires the verified standard RoPE configuration')
    result = copy.deepcopy(config)
    result.num_hidden_layers = block_index+1
    return result


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


class ShardWriter:
    """Bounded float16 shard buffer; manifest is the atomic commit point."""
    def __init__(self, root, split, manifest, d_in, shard_rows=32768):
        self.root, self.split, self.manifest = Path(root), split, manifest
        self.d_in, self.shard_rows = d_in, shard_rows
        self.arrays, self.records, self.rows = [], [], 0
        (self.root/split).mkdir(parents=True, exist_ok=True)

    def add(self, values, metadata):
        values = np.asarray(values, dtype=np.float16)
        if values.ndim != 2 or values.shape[1] != self.d_in or not len(values):
            raise ValueError('Bad activation matrix')
        if not np.isfinite(values).all():
            raise ValueError('Non-finite float16 activations')
        if self.rows and self.rows+len(values) > self.shard_rows:
            self.flush()
        self.records.append({**metadata, 'row_start':self.rows, 'row_end':self.rows+len(values)})
        self.arrays.append(values)
        self.rows += len(values)
        if self.rows >= self.shard_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        stats = self.manifest['splits'][self.split]
        index = len(stats['shards'])
        path = self.root/self.split/f'{index:06d}.npy'
        tmp = path.with_suffix('.npy.tmp')
        # Write into a memmap to avoid duplicating the entire buffered matrix.
        matrix = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=(self.rows,self.d_in))
        offset = 0
        for values in self.arrays:
            matrix[offset:offset+len(values)] = values
            offset += len(values)
        matrix.flush()
        del matrix
        tmp.replace(path)
        meta = path.with_suffix('.jsonl')
        tmp_meta = meta.with_suffix('.jsonl.tmp')
        with tmp_meta.open('w') as f:
            for row in self.records:
                f.write(json.dumps(row,separators=(',',':'))+'\n')
        tmp_meta.replace(meta)
        stats['shards'].append({'path':str(path.relative_to(self.root)), 'n_rows':self.rows,
                               'n_documents':len(self.records), 'sha256':file_sha(path),
                               'metadata_path':str(meta.relative_to(self.root)), 'metadata_sha256':file_sha(meta)})
        stats['n_tokens'] += self.rows
        stats['n_documents'] = stats.get('n_documents',0)+len(self.records)
        dump(self.root/'manifest.json',self.manifest)
        print(json.dumps({'split':self.split,'documents':stats['n_documents'],'tokens':stats['n_tokens'],'shard':index}),flush=True)
        self.arrays, self.records, self.rows = [], [], 0


def extraction_plan(corpus_root, selection, model_path, limit_documents=0, check_hashes=True):
    corpus_root, model_path = Path(corpus_root), Path(model_path)
    complete = json.loads((corpus_root/'COMPLETE.json').read_text())
    if complete['status'] != 'verified':
        raise ValueError('Corpus is not verified')
    for name in ('manifest','verification'):
        if file_sha(corpus_root/f'{name}.json') != complete[f'{name}_sha256']:
            raise ValueError(f'Corpus completion seal mismatch: {name}')
    manifest = json.loads((corpus_root/'manifest.json').read_text())
    cfg = json.loads((model_path/'config.json').read_text())
    if cfg['hidden_size'] != 5120 or cfg['num_hidden_layers'] < 26:
        raise ValueError('Expected ChemDFM-R hidden_size5120, at least26 blocks')
    for name, sha in manifest['model_files'].items():
        if file_sha(model_path/name) != sha:
            raise ValueError(f'Checkpoint/tokenizer fingerprint changed: {name}')
    splits, groups, ids, files = {}, {}, {}, set()
    output_hashes = {r['path']:r['sha256'] for r in manifest['outputs']}
    for split in ('train','validation'):
        selection_path = Path(selection)/f'{split}.jsonl'
        selection_file = (corpus_root/selection_path).resolve()
        if not selection_file.is_relative_to(corpus_root.resolve()):
            raise ValueError('Selection must be inside the verified corpus')
        selection_key = str(selection_file.relative_to(corpus_root.resolve()))
        if selection_key not in output_hashes or file_sha(selection_file) != output_hashes[selection_key]:
            raise ValueError(f'Selection is not sealed by the corpus manifest: {selection_key}')
        refs = read_selection(corpus_root,selection_path)
        if limit_documents:
            refs = refs[:limit_documents]
        if not refs or any(r['split'] != split for r in refs):
            raise ValueError(f'Empty or wrong-split selection: {split}')
        files.update(r['path'] for r in refs)
        groups[split], ids[split] = {r['group_id'] for r in refs}, {r['id'] for r in refs}
        splits[split] = {'selection_path':str(selection_path),'selection_sha256':file_sha(corpus_root/selection_path),
                         'expected_documents':len(refs),'expected_tokens':sum(r['assistant_tokens'] for r in refs)}
    if groups['train'] & groups['validation'] or ids['train'] & ids['validation']:
        raise ValueError('Selection group/document train-validation overlap')
    if check_hashes:
        for name in sorted(files):
            if file_sha(corpus_root/name) != output_hashes[name]:
                raise ValueError(f'Corpus shard changed: {name}')
    # Weight file identities are recorded cheaply; their contents are not rehashed (28GB).
    weights = [{'path':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
               for p in sorted(model_path.glob('*.safetensors'))]
    if not weights:
        raise ValueError('Local model weight files not found')
    return {'format_version':1,'layer':26,'block_index':25,'activation_site':'post_block_residual',
            'timing':'post','d_in':5120,'dtype':'float16','compute_dtype':'bfloat16',
            'attention_implementation':'sdpa','corpus_root':str(corpus_root.resolve()),
            'corpus_manifest_sha256':file_sha(corpus_root/'manifest.json'), 'model_path':str(model_path.resolve()),
            'model_files':manifest['model_files'],'model_weights_identity':weights,
            'weight_identity_policy':'file names/sizes/mtime; not cryptographic weight content checksums',
            'token_policy':'assistant text only, no whitespace/special/control tags; full frozen chat context',
            'limit_documents':limit_documents,'splits':splits}


def select_block(model, block_index=25):
    base = getattr(model,'model',model)
    if not hasattr(base,'layers') or len(base.layers) <= block_index:
        raise ValueError('Expected a Qwen2 model with .layers or .model.layers')
    return base.layers[block_index]


def capture_post_tokens(model, ids, positions, block_index=25):
    import torch
    captured = []
    class LayerCaptured(Exception):
        pass
    def hook(_module,_inputs,output):
        hidden = output[0] if isinstance(output,tuple) else output
        selected = hidden[0,torch.as_tensor(positions,device=hidden.device)].detach()
        captured.append(selected.to(device='cpu',dtype=torch.float16).numpy())
        # Later blocks cannot change this residual, so avoid their computation/logits.
        raise LayerCaptured()
    handle = select_block(model,block_index).register_forward_hook(hook)
    try:
        device = model.get_input_embeddings().weight.device
        with torch.inference_mode():
            try:
                input_ids = torch.tensor([ids],dtype=torch.long,device=device)
                model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=False)
            except LayerCaptured:
                pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('Layer hook did not execute exactly once')
    return captured[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',default='data/v1')
    p.add_argument('--selection',default='selections/pilot_10m')
    p.add_argument('--model',default=str(DEFAULT_MODEL))
    p.add_argument('--output',default='activations/layer26/pilot_10m')
    p.add_argument('--gpus',default='4,5,6,7',help='Physical visible GPU IDs; set before importing torch')
    p.add_argument('--max-memory-gib',type=int,default=36,help='Per visible GPU cap; require this much free VRAM plus1GiB')
    p.add_argument('--load-through-target',action='store_true',help='Load unchanged BF16 embedding and first26 blocks only; identical target-layer computation with less VRAM')
    p.add_argument('--shard-rows',type=int,default=32768)
    p.add_argument('--limit-documents',type=int,default=0,help='Smoke extraction, first N documents per split; 0=full')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight',action='store_true',help='Validate input and report storage needs without loading torch/model')
    args = p.parse_args()
    if args.limit_documents < 0 or args.shard_rows <= 0 or args.max_memory_gib <= 0:
        p.error('Invalid non-positive budget')
    corpus,model_path,output = project_path(args.corpus),project_path(args.model),project_path(args.output)
    plan = extraction_plan(corpus,args.selection,model_path,args.limit_documents)
    plan['load_through_target'] = args.load_through_target
    plan['loaded_num_hidden_layers'] = 26 if args.load_through_target else json.loads((model_path/'config.json').read_text())['num_hidden_layers']
    total_tokens = sum(s['expected_tokens'] for s in plan['splits'].values())
    needed = total_tokens*5120*2
    disk_path = output
    while not disk_path.exists():disk_path=disk_path.parent
    print(json.dumps({'plan':plan,'raw_activation_gib':needed/1024**3,
                      'disk_free_gib':shutil.disk_usage(disk_path).free/1024**3,
                      'model_loaded':False},ensure_ascii=False,indent=2),flush=True)
    if args.preflight:
        return
    config_identity = {**plan,'splits':plan['splits'].copy(),'shard_rows':args.shard_rows}
    if output.exists():
        if not args.resume:
            raise FileExistsError(f'{output} exists; use --resume or choose another output')
        manifest = json.loads((output/'manifest.json').read_text())
        if manifest['extraction_config'] != config_identity:
            raise ValueError('Resume extraction configuration changed')
        for stats in manifest['splits'].values():
            for shard in stats['shards']:
                for name,sha in [('path','sha256'),('metadata_path','metadata_sha256')]:
                    if file_sha(output/shard[name]) != shard[sha]:
                        raise ValueError(f'Cached shard changed: {shard[name]}')
        if manifest['status'] == 'complete':
            print('Cache already complete and verified.');return
    else:
        manifest = {**plan,'status':'extracting','created_utc':datetime.now(timezone.utc).isoformat(),
                    'extraction_config':config_identity,
                    'splits':{s:{**v,'n_tokens':0,'n_documents':0,'shards':[]} for s,v in plan['splits'].items()}}
    remaining = needed-sum(s['n_tokens'] for s in manifest['splits'].values())*5120*2
    if shutil.disk_usage(disk_path).free < remaining*1.1+2*1024**3:
        raise RuntimeError('Insufficient space for activation cache and metadata margin')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,args.gpus):
        raise ValueError('CUDA_VISIBLE_DEVICES conflicts with --gpus; resolve explicitly')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from transformers import AutoConfig, AutoModel
    if torch.cuda.device_count() != len(args.gpus.split(',')):
        raise RuntimeError('Requested CUDA devices are unavailable')
    for device in range(torch.cuda.device_count()):
        free,_ = torch.cuda.mem_get_info(device)
        if free < (args.max_memory_gib+1)*1024**3:
            raise RuntimeError(f'GPU logical {device} has {free/1024**3:.1f}GiB free; extraction budget needs {args.max_memory_gib+1}GiB. Other processes were not changed.')
    output.mkdir(parents=True,exist_ok=True)
    dump(output/'manifest.json',manifest)
    model_config = AutoConfig.from_pretrained(model_path,local_files_only=True)
    if args.load_through_target:
        model_config = prefix_config(model_config)
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    model = AutoModel.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,
        config=model_config,
        device_map='auto',max_memory={i:f'{args.max_memory_gib}GiB' for i in range(torch.cuda.device_count())},
        attn_implementation='sdpa').eval()
    if any(str(v) in ('cpu','disk') for v in getattr(model,'hf_device_map',{}).values()):
        raise RuntimeError('Model offloaded to CPU/disk; increase GPU memory before extracting')
    manifest['runtime_device_map'] = {k:str(v) for k,v in getattr(model,'hf_device_map',{}).items()}
    print(json.dumps({'model_loaded':True,'loaded_blocks':len(getattr(model,'model',model).layers),
                      'device_map':manifest['runtime_device_map']}),flush=True)
    tokenizer = LocalTokenizer(model_path)
    started = time.monotonic()
    for split,stats in manifest['splits'].items():
        writer = ShardWriter(output,split,manifest,5120,args.shard_rows)
        start = stats['n_documents']
        for selection_index,row in enumerate(iter_selected_documents(corpus,stats['selection_path'],start=start,
            limit=stats['expected_documents']-start),start=start):
            ids,positions = encode_document(row,tokenizer)
            values = capture_post_tokens(model,ids,positions)
            writer.add(values,{'id':row['id'],'group_id':row['group_id'],'source':row['source'],'task':row['task'],
                'selection_index':selection_index,'positions':positions.tolist(),'token_ids':[ids[i] for i in positions]})
        writer.flush()
        if stats['n_documents'] != stats['expected_documents'] or stats['n_tokens'] != stats['expected_tokens']:
            raise ValueError('Extraction count does not match frozen selection')
    manifest['status'] = 'complete'
    manifest['extraction_seconds'] = time.monotonic()-started
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['gpu_memory'] = [{'physical_gpu':gpu,'peak_allocated_gib':torch.cuda.max_memory_allocated(i)/1024**3,
        'peak_reserved_gib':torch.cuda.max_memory_reserved(i)/1024**3} for i,gpu in enumerate(args.gpus.split(','))]
    dump(output/'manifest.json',manifest)
    dump(output/'COMPLETE.json',{'status':'complete','manifest_sha256':file_sha(output/'manifest.json')})


if __name__ == '__main__':
    main()
