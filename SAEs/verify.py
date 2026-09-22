"""Independent full-output integrity/leakage/accounting verification and release receipt."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from corpus import LocalTokenizer,blocked_reasons,digest,iter_rows,norm,render_and_count,ANSWER_CUE,molecule_keys
from prepare import file_sha,dump

def verify(out,model=None):
    out=Path(out).resolve();manifest=json.loads((out/'manifest.json').read_text());stats=json.loads((out/'stats.json').read_text())
    model=model or manifest['config']['model'];tok=LocalTokenizer(model)
    block=json.loads((out/'audit/pilot_exclusion.json').read_text())
    for entry in block['fingerprints']:
        assert file_sha(entry['path'])==entry['sha256'],('Pilot file changed',entry['path'])
    block={k:set(block[k]) for k in ('questions','cids','molecules')}
    ids=set();hashes={};group_splits={};key_splits={};counts=Counter();tokens=Counter();main={};checked_tokens=0
    source_stats=Counter();pool_count=0;task_counts=Counter();task_tokens=Counter()
    expected_inputs_by_ref={}
    for source in manifest['sources']:
        if not source['path'].startswith('Downloads/Train/Chemcot/'):continue
        for index,raw in enumerate(iter_rows(Path(manifest['config']['source_root'])/source['path'])):
            meta=raw.get('meta') or {}
            if isinstance(meta,str):
                try:meta=json.loads(meta)
                except json.JSONDecodeError:continue
            inputs=[meta.get('molecule'),meta.get('rxn_smiles')]
            reactants=meta.get('reactants') or []
            inputs.extend([reactants] if isinstance(reactants,str) else reactants)
            keys={k for value in inputs if isinstance(value,str) for k in molecule_keys(value)}
            expected_inputs_by_ref[(source['path'],index)]=keys
    for name,sha in manifest['model_files'].items():assert file_sha(Path(model)/name)==sha,('model changed',name)
    for entry in manifest['outputs']:
        path=out/entry['path'];assert file_sha(path)==entry['sha256'],('checksum',path)
        if entry['path'].startswith('selections/'):continue
        for index,row in enumerate(iter_rows(path)):
            rid=row['id'];assert rid not in ids,('duplicate id',rid);ids.add(rid)
            assert not blocked_reasons(row,block),('Pilot overlap',rid)
            assert not row['quality_flags'],('quality flags',rid)
            assert not ANSWER_CUE.search('\n'.join(m['content'] for m in row['messages'])),('answer hint',rid)
            if row['source']=='chemcot':
                expected=expected_inputs_by_ref[(row['source_ref']['path'],row['source_ref']['row_index'])]
                assert expected <= set(row['molecule_keys']),('missing source input identity',rid)
            assert all(m['content'].strip() and '<mol>' not in m['content'] for m in row['messages'])
            normalized=[{'role':m['role'],'content':norm(m['content'])} for m in row['messages']]
            assert digest(normalized)==row['content_sha256'],('content changed',rid)
            pool=row['tier']=='prompt_pool';key=('pool' if pool else 'corpus',row['content_sha256'])
            assert key not in hashes,('content duplicate',rid);hashes[key]=rid
            source_stats[row['source']]+=1;counts[(row['tier'],row['split'],row['source'])]+=1
            task_counts[(row['source'],row['task'],row['split'])]+=1
            if pool:
                assert row['split']=='candidate' and all(m['role']=='user' for m in row['messages']);pool_count+=1
                continue
            assert 'reference_for_evaluation_only' not in row
            assert row['messages'][0]['role']=='user' and row['messages'][-1]['role']=='assistant'
            assert all(m['role']==('user' if i%2==0 else 'assistant') for i,m in enumerate(row['messages']))
            assert 16<=row['tokenization']['assistant_tokens']<=row['tokenization']['sequence_tokens']<=manifest['config']['max_length']
            split=row['split'];group=row['group_id']
            assert group_splits.get(group,split)==split;group_splits[group]=split
            for key in row['group_keys']:
                assert key_splits.get(key,split)==split,('cross-split identity',key)
                key_splits[key]=split
            tokens[(row['tier'],split,row['source'])]+=row['tokenization']['assistant_tokens']
            task_tokens[(row['source'],row['task'],split)]+=row['tokenization']['assistant_tokens']
            if index<2:
                assert render_and_count(row['messages'],tok)==row['tokenization'],('token count mismatch',rid)
                checked_tokens+=1
            if row['tier']=='main':main[rid]={'id':rid,'group_id':group,'source':row['source'],'task':row['task'],'split':split,'path':entry['path'],'line_index':index,'assistant_tokens':row['tokenization']['assistant_tokens'],'sequence_tokens':row['tokenization']['sequence_tokens']}
    excluded=Counter();reason_counts=Counter();excluded_ids=set()
    for row in iter_rows(out/'audit/excluded.jsonl'):
        assert row['id'] not in ids and row['id'] not in excluded_ids
        excluded_ids.add(row['id']);excluded[row['source']]+=1;reason_counts.update(row['reasons'])
        if 'duplicate_of' in row:assert row['duplicate_of'] in ids
    expected_inputs=Counter()
    for j in stats['job_counts']:expected_inputs[j['source']]+=j['counts']['input']
    assert expected_inputs==source_stats+excluded,('source accounting',expected_inputs,source_stats+excluded)
    for s in stats['tables']:
        key=(s['tier'],s['split'],s['source']);assert counts[key]==s['records']
        if s['tier']!='prompt_pool':assert tokens[key]==s['assistant_tokens']
    selections={};selection_ids=set()
    for split in ('train','validation'):
        selected=Counter();records=Counter()
        for row in iter_rows(out/'selections/pilot_10m'/f'{split}.jsonl'):
            assert row==main[row['id']],('selection reference',row['id'])
            assert row['split']==split and row['id'] not in selection_ids
            selection_ids.add(row['id']);selected[row['source']]+=row['assistant_tokens'];records[row['source']]+=1
        expected=stats['selection'][split]
        assert sum(selected.values())==expected['assistant_tokens'] and sum(records.values())==expected['records']
        for source,s in expected['sources'].items():
            assert selected[source]==s['actual_tokens'] and records[source]==s['records']
        selections[split]={'records':sum(records.values()),'assistant_tokens':sum(selected.values())}
    manifest['status']='verified'
    manifest['code_files']={name:file_sha(Path(__file__).parent/name) for name in ('corpus.py','prepare.py','verify.py')}
    manifest['verification_code_sha256']=file_sha(__file__)
    manifest['code_receipt_note']='build_code_files preserves original build hashes; code_files records pipeline at most recent full verification. pre_relocation_code_files preserves pre-migration verification code.'
    dump(out/'manifest.json',manifest)
    receipt={'status':'passed','records':len(ids),'main_records':len(main),'prompt_candidates':pool_count,'excluded_records':sum(excluded.values()),'source_inputs':dict(expected_inputs),'source_kept':dict(source_stats),'source_excluded':dict(excluded),'exclusion_reasons':dict(reason_counts),'output_sha256_verified':len(manifest['outputs']),'tokenization_recomputed_records':checked_tokens,'checks':['source accounting','all unique IDs','all normalized-content hashes','no duplicate dialogs/prompts','no Pilot normalized-question/CID/connectivity matches','no train/validation group/identity-key overlap','role and reference separation','sampled contextual token counts','all selection references/counts/splits','all output shard SHA-256'],'selection':selections,'limits':['No scaffold or semantic-near-duplicate isolation.','Molecule identity compares supplied source metadata using RDKit whole/largest-component connectivity; does not extract every molecule mentioned in natural-language answers.','Answer-cue matching is heuristic; external descriptions/reasoning are not verified factual ground truth.','Prompt candidates have no SAE split; generated answers must be regrouped before use.'],'manifest_sha256':file_sha(out/'manifest.json')}
    receipt['verification_code_sha256']=file_sha(__file__)
    dump(out/'verification.json',receipt)
    dump(out/'task_counts.json',[{'source':k[0],'task':k[1],'split':k[2],'records':v,'assistant_tokens':task_tokens[k]} for k,v in sorted(task_counts.items())])
    readme(out,stats,receipt)
    dump(out/'COMPLETE.json',{'status':'verified','manifest_sha256':file_sha(out/'manifest.json'),'verification_sha256':file_sha(out/'verification.json')})
    print(json.dumps(receipt,ensure_ascii=False,indent=2));return receipt

def readme(out,stats,receipt):
    lines=['# SAE chemistry text corpus v1','','面向 ChemDFM-R-14B 第 26 个 block（零索引 25）的 post-token 激活获取。已完成文本准备和验证；没有生成新回答、抽取激活或训练 SAE。源文件和 Pilot 标注保持原样。','','## 目录','','```text','main/{train,validation}/          ChemCoT + Mol-LLaMA，主要训练语料','supplement/{train,validation}/    PubChem 原始描述 + ChEBI-20，可选补充','prompt_pool/                     OpenMolIns + MoleculeQA，仅题目候选','selections/pilot_10m/             下一步激活获取的固定记录索引','audit/                           Pilot 排除表、逐条排除原因','config.json / manifest.json      参数、来源与模型/代码指纹','stats.json / verification.json   全量统计和独立验证结果','COMPLETE.json                    构建和验证完成标记','```','','## 实际规模','','| 层级 | split | 来源 | 记录数 | 有效 assistant tokens |','|---|---|---|---:|---:|']
    for r in stats['tables']:lines.append(f'| {r["tier"]} | {r["split"]} | {r["source"]} | {r["records"]:,} | {r.get("assistant_tokens",0):,} |')
    lines += ['','候选池的 0 tokens 表示未作为 SAE 激活样本计数，不表示题目为空。token 数使用完整聊天上下文实际分词，排除纯空白、聊天控制符以及 `<think>/<answer>` 标签；不是之前报告的抽样估计。没有添加伪造 CoT 标签，也没有把解释文本认定为已验证的正确推理。','','## 使用','','建议首先使用 `selections/pilot_10m/train.jsonl` 和 `validation.jsonl`。每条索引包含语料 shard 路径、从 0 开始的行号、记录 ID、分子组和 assistant token 数。训练预算约 10M tokens：30% ChemCoT、70% Mol-LLaMA；验证预算约 0.5M，按完整记录采样，允许小幅超出且不重复采样。', '']
    for split,s in stats['selection'].items():
        lines.append(f'- {split}: {s["records"]:,} 条，{s["assistant_tokens"]:,} 个有效 assistant tokens。')
        for source,detail in s['sources'].items():
            if detail['shortfall']:
                lines.append(f'  - {source} 可用独立样本不足该来源配额，短缺 {detail["shortfall"]:,} tokens；已取该来源全部可用样本，没有重复采样，也未从训练组补入。因此实际总量和来源比例与预算不同。')
    lines += ['','读取消息仅使用记录的 `messages` 字段。通过本地 checkpoint 的聊天模板渲染；`tokenization.assistant_char_spans` 是该渲染文本中的字符区间。将分词 offsets 投影到这些区间并排除控制符，抽第 26 层在当前 token 位置的输出。不要把 provenance、分子键、统计、参考答案或质量信息序列化进模型输入。','', '**上下文上限为 16,384 tokens；少于 16 个有效 assistant tokens 的文档也被隔离。** 超长文档未截断，原文仍在源文件中，排除日志可定位。多轮对话整体保留，每个 assistant 回答只统计一次。各源原有 system 文本不直接沿用（含多模态/选择题指令）；统一使用 checkpoint 聊天模板默认 system。','', '**按分子组划分 98%/2%**：同分子 connectivity/CID、相同完整问题上下文和较长相同回答通过连通分组保持同 split。忽略立体化学，并对盐型使用整体/最大组分键，属于保守隔离。PubChem、ChEBI 与主语料共同分组。实际条数比例不保证正好 98/2。','', '**Pilot 全部 600 道题排除**：匹配规范化文本、caption CID、输入/参考分子的 RDKit connectivity。所有留存行已重新检查 0 次这些规则定义下的命中；不意味着已排除所有语义近重复或相同骨架。','', '**题目池尚未用于训练**：只有 user 消息，答案在 `reference_for_evaluation_only` 中单独保存，不可传给模型；生成 ChemDFM CoT 后，需要与现有 train/validation 分组重新匹配，再计数和采样，不能直接把全部候选加进 train。','','## 排除与验证','','```json',json.dumps({'excluded_records':receipt['excluded_records'],'reasons':receipt['exclusion_reasons']},ensure_ascii=False,indent=2),'```','','包含多个原因的同一条记录只计为一条排除，原因计数不应相加当作排除总数。`answer_cue` 是保守隔离规则，不等于已证明幻觉/泄漏。重复副本与较小 OpenMolIns 版本没有再次读入。未启用下载新语料或生成新回答。','','完整性验证：来源条数守恒、全局 ID/文本去重、Pilot 排除、跨 split 分组隔离、所有 shard 哈希、抽样 token 重算和全部 selection 引用。见 `verification.json`。','','## 复现','','```bash','cd /mnt_nas1/haoqian/Data/Molecule','SAEs/.venv/bin/python -m unittest discover -s SAEs/tests -v','SAEs/.venv/bin/python SAEs/prepare.py --out SAEs/data/v1_rebuilt --workers 8','SAEs/.venv/bin/python SAEs/verify.py SAEs/data/v1_rebuilt','```','','构建拒绝覆盖已有目录。默认只使用 CPU；原始数据文件、tokenizer、配置、代码 SHA-256 均可追溯。`_build/` 为本次生成的中间文件，不是额外独立语料。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--model');args=p.parse_args();verify(args.out,args.model)
