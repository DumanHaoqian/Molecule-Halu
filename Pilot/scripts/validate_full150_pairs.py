#!/usr/bin/env python3
"""Reexecute every selected edit and reconstruct all labels before evaluation."""
import json
from pathlib import Path
import sys
from rdkit import rdBase
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from molhallulens.modules.agent_generation.orchestrator import load_origins,write_json
from molhallulens.modules.agent_generation.full_pipeline import project_row
from molhallulens.modules.agent_generation.full_reference import execute_full_plan
from molhallulens.modules.agent_generation.full_renderer import render_full_reasoning
from molhallulens.modules.agent_generation.chemistry_tools import compare_molecules
from molhallulens.modules.agent_generation.labels import label_tokens
from transformers import AutoTokenizer
from summarize_full150_exp import validate_generation


def main():
    base=ROOT/'Experiments/agent_corruption_v18/full150'
    validate_generation(base,ROOT/'Dataset/raw_benchmark_data/mol_edit')
    tokenizers={name:AutoTokenizer.from_pretrained(base/'tokenizers'/name,local_files_only=True,use_fast=True) for name in ('ChemDFM-R-14B','Chem-R-8B')}
    summaries=[]
    for original in load_origins(ROOT):
        row=project_row(original);origin=row['origin_id'];directory=base/'origins'/origin
        r=json.loads((directory/'accepted.json').read_text());reference=json.loads((directory/'reference.json').read_text());plan=r['plan']
        with rdBase.BlockLogs():
            clean=execute_full_plan(row['indexed_smiles'],reference['edit_plan'])
            altered=execute_full_plan(row['indexed_smiles'],plan['edit_plan'])
            assert compare_molecules(clean['product_smiles'],row['gt_smiles'])['equivalent'],origin
            assert altered==plan['execution'],origin
            rendered=render_full_reasoning(row,{**reference,'execution':clean},plan['edit_plan'],altered,tokenizers=tokenizers)
        for name in ('nodes','roots'):
            assert rendered[name]==plan[name],(origin,name)
        assert rendered['H_render']==plan['render'],origin
        assert rendered['N_render']['text']==r['N_visible']==row['N_visible'],origin
        for name,tok in tokenizers.items():
            rebuilt=label_tokens(plan['render']['text'],r['annotations'],tok)
            saved=r['token_labels'][name]
            # Snapshot location differs; every token ID, spelling, character
            # offset, semantic label and provenance value must still match.
            assert all(t['tokenizer']==str(ROOT.parent/'chemical_models'/name) for t in saved),(origin,name)
            assert [{k:v for k,v in t.items() if k!='tokenizer'} for t in rebuilt]==[{k:v for k,v in t.items() if k!='tokenizer'} for t in saved],(origin,name)
        assert rendered['checks']['changed_node_count']>=6 and min(rendered['checks']['annotated_tokens'].values())>=40,origin
        summaries.append({'origin_id':origin,'nodes':rendered['checks']['changed_node_count'],'tokens':rendered['checks']['annotated_tokens'],'roots':len(plan['roots']),'N_projection_removed_lines':len(row['projection_removed_lines']),'agent_clean_status':r['review_status']['clean_reference'].get('status'),'agent_conditional_status':r['review_status']['conditional_fidelity'].get('status')})
    write_json(base/'release_validation.json',{'status':'pass','origins':150,'validation':'Full reexecution, GT equality, immutable question, exact original-N projection, exact H reconstruction, semantic nodes, standalone tokenizer label reconstruction, minimum density and complete-answer exclusion. Reviews diagnostic.','rows':summaries})
    print('Full150 reexecution and label reconstruction passed for all 150 origins.',flush=True)

if __name__=='__main__':main()
