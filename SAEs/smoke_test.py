"""Persistent CPU-only integration check; synthetic numbers are NOT research results."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch

from prepare import dump, file_sha
from sae_models import ARCHITECTURES
from train_sae import resolve_config, run_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/smoke_cpu')
    p.add_argument('--wandb-mode',choices=('offline','disabled'),default='offline')
    args = p.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError('Choose a fresh smoke output directory')
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    dictionary = rng.normal(size=(5,16))
    cache = root/'synthetic_cache'
    cache.mkdir(parents=True)
    manifest = {'format_version':1,'status':'complete','synthetic':True,
                'layer':26,'block_index':25,'activation_site':'post_block_residual',
                'dtype':'float16','d_in':16,'splits':{}}
    for split,n in [('train',512),('validation',128)]:
        x = (rng.normal(size=(n,5)) @ dictionary + rng.normal(size=(n,16))*.1).astype(np.float16)
        path = cache/f'{split}.npy'
        np.save(path,x)
        manifest['splits'][split] = {'n_tokens':n,'shards':[{'path':path.name,'n_rows':n,'sha256':file_sha(path)}]}
    dump(cache/'manifest.json',manifest)
    dump(cache/'COMPLETE.json',{'status':'complete','manifest_sha256':file_sha(cache/'manifest.json')})
    report = {'synthetic_only':True,'device':'cpu','wandb_mode':args.wandb_mode,'architectures':{}}
    for arch in ARCHITECTURES:
        model = {'architecture':arch,'d_in':16,'d_sae':64,'k':4}
        if arch == 'matryoshka':model['matryoshka_widths']=[16,32,64]
        if arch == 'jumprelu':model.update(l0_coefficient=.01,l0_warm_up_steps=4)
        if arch == 'sparsemax_attention':model.update(key_dim=8,preselect_k=16,l0_coefficient=.01)
        config = resolve_config({'cache_dir':str(cache),'output_dir':str(root/arch),'device':'cpu','seed':42,
            'sae':model,'training':{'batch_size':32,'max_tokens':512,'lr':.01,'warmup_steps':2,
             'log_every':2,'checkpoint_every':8,'validate_every':8,'dead_feature_tokens':256,'chunk_rows':64},
            'normalization':{'sample_tokens':256},'evaluation':{'batch_size':32,'max_tokens':128},
            'wandb':{'mode':args.wandb_mode,'project':'chemdfm-sae-cpu-smoke','name':f'synthetic-{arch}'}})
        report['architectures'][arch] = run_training(config)
        dump(root/'summary.json',report)
    (root/'README.md').write_text('# CPU integration smoke checks\n\nAll activations here are synthetic (d_in=16). '
        'These runs test software integration and W&B offline logging. They are not ChemDFM scientific training or quality results.\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
