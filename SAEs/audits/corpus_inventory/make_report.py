import json, collections, math
from pathlib import Path

D=Path(__file__).resolve().parent
p=json.loads((D/'profile.json').read_text()); c=json.loads((D/'chemcot_selected.json').read_text()); o=json.loads((D/'overlap.json').read_text())
groups={k:[r for r in p['results'] if r['kind']==k] for k in ['mollama','pubchem','openmol','moleculeqa','chebi','processed']}
assert len(p['results'])==35 and len(c['results'])==16, 'Audit not complete'

def total(rs,k='second'):return sum(x['tokens'][k]['estimated_total'] for x in rs)
def n(rs):return sum(x['rows'] for x in rs)
def m(x):return f'{x/1e6:.2f}M'
pub=groups['pubchem'][0]; xl=next(x for x in groups['openmol'] if '/xlarge/' in x['path'])
lines=[
'# ChemDFM-R Layer 26 SAE 语料检查',
'',
'检查日期：2026-09-21。源目录 `/home/haoqian/Data/past/Molecule`；只读检查，未抽取 activations、未训练、未下载额外语料。',
'',
'**结论：现有语料足以启动小到中等规模的 SAE 探索实验。优先解决去重、评估集隔离和 CoT 分布匹配；不需要为了凑文本量立即下载大型通用语料。是否足以让某个 SAE 收敛，仍需结合字典宽度、稀疏度和验证曲线判断。**',
'',
'## 统计口径',
'',
'样本数为逐文件全量解析的精确计数。token 为本地 ChemDFM-R-14B tokenizer 对固定种子 20260921 的 reservoir 样本估算：一般每文件 1,024 条；ChemCoT 字段回退版 512 条，小文件全量。M=百万。统计不含 system prompt/chat-template 的固定开销，未做长度截断；未扣除全局重复、评估重合与质量过滤，所以不能当作最终可训练 activation 数。每文件采样误差和长度分位数见 profile.json / chemcot_selected.json。',
'',
'## 候选数据',
'',
'| 数据 | 本地记录数 | 回答/推理 token 估计 | 用途与限制 |',
'|---|---:|---:|---|',
f'| ChemCoT | {n(c["results"]):,} | {m(total(c["results"]))} | 推理语料优先；raw_cot → struct_cot → cot_result 选择一个视图 |',
f'| Mol-LLaMA-Instruct 四类 | {n(groups["mollama"]):,} | {m(total(groups["mollama"]))} | 自然语言化学描述/问答；不是 ChemDFM 自生成 CoT |',
f'| PubChem 分子描述 | {pub["rows"]:,} | 原始描述 {m(total([pub],"first"))}；扩写 {m(total([pub]))} | 两个替代视图，与 Mol-LLaMA 有分子重合；不能都当独立新增数据 |',
f'| OpenMolIns xlarge | {xl["rows"]:,} | 答案 SMILES {m(total([xl]))}；含题目 {m(total([xl],"combined"))} | 无 CoT；主要用作 ChemDFM 自生成推理的题目池 |',
f'| MoleculeQA train | {n(groups["moleculeqa"]):,} | {m(total(groups["moleculeqa"]))} | 多项选择短答案；适合作为生成解释的题目池 |',
f'| ChEBI-20 train | {n(groups["chebi"]):,} | {m(total(groups["chebi"]))} | 分子描述补充，需与 Pilot 隔离 |',
f'| 已整理 Datasets/Train | {n(groups["processed"]):,} | 不另加总 | 上述原料的变换/子集，含大量非文本 world-model 字段 |',
'',
f'Mol-LLaMA 四类共 {p["mollama_unique_cids"]:,} 个不同 CID、{sum(x["turns"] for x in groups["mollama"]):,} 个问答轮次；含用户输入约 {m(total(groups["mollama"],"combined"))} tokens。PubChem 与这四类共享 {pub.get("cids_shared_with_mollama",0):,} 个 CID，共享分子不等于全文重复。',
'',
'主要入口：', '',
'- `Downloads/Train/Chemcot/`',
'- `Molecule/Baselines/Mol-LLaMA/data/Mol-LLaMA-Instruct/`',
'- `Downloads/Train/OpenMolIns/xlarge/train.csv`',
'- `Molecule/Baselines/Mol-LLaMA/data/moleculeqa/train.json`',
'- `Molecule/Baselines/Mol-LLaMA/data/ChEBI-20/train.txt`',
'',
'## 已证实的数据风险',
'',
f'1. **不同任务的 CoT 字段不同。** 13,597 条使用 raw_cot，6,484 条需要 struct_cot，3,142 条需要 cot_result。只读 raw_cot 会漏掉 9,626 条；不能把两个推理视图拼起来重复计数。反应 major-product 的 2,788 条只有 1,164 个不同的题目+选定推理文本对。',
f'2. **答案提示。** 选定推理中 {sum(x["answer_cue_rows"] for x in c["results"]):,} 条命中 ground truth / reference answer / correct answer is；这是需要复核的文本信号，不代表这些全部存在泄漏。已有样例明确提到已知真值。建议隔离或在只给问题、不给参考答案的条件下重新生成。',
'3. **重复副本。** 已对 ChemCoT add、OpenMolIns xlarge、Mol-LLaMA comprehensive 的原料/Construction 副本验证 SHA-256 相同。其他同名副本尚未逐一做字节级验证。OpenMolIns 的 light/small/medium/large 文本对全部包含在 xlarge 中，只保留 xlarge 即可。',
'4. **模态与字段。** Mol-LLaMA 的 `<mol>` 需要替换为实际 SMILES；PubChem 的 4.42 GB 包含原子和坐标，不能视为 4.42 GB 文本。加工版的 latent slots、subgraphs 和目标分子字段不能整体序列化为提示；有些 molecule 字段是目标结构，误用会把答案放进输入。',
'5. **评估集重合。** OpenMolIns xlarge 与 Pilot 有 10 道完全相同的题目文本，其中 4 道属于 test；Mol-LLaMA 与 Pilot caption 任务共享 42 个 CID，涉及 43 个 task/question 组合（30 train、8 validation、5 test）。ChEBI-20 train 另命中 1 道 Pilot train 题目。这里是按规范化题目文本/CID 的保守检查，未进行全量 canonical-SMILES 或语义近重复检索，实际重合可能更多。',
'6. **训练集与测试集用途。** 优先训练来源，并排除 Pilot 全部 600 道题及对应分子/反应组；至少严格隔离其 validation/test。原目录 Testset 和外部 benchmark 的评估 split 不作为 SAE 默认训练池。',
'',
'## 是否需要补充',
'',
'**数量上：第一版不需要先补充外部数据。分布上：建议优先补充 ChemDFM-R 自己生成的化学推理。** Mol-LLaMA 的 GPT-4o 描述、ChemCoT 的其他模型推理通过 teacher forcing 仍能用于学习 ChemDFM 的激活字典，但不能等同于 ChemDFM 自身 CoT 的分布。',
'',
'建议在不重合的新题目上按当前 12 个任务分层，用 ChemDFM-R 生成 2–5 万条新回答；假设每条保留约 1,000 个 CoT/回答 token，即新增约 20–50M tokens（规划假设，不是已测量结果）。不需要幻觉标签，也不必只保留正确回答；需要排除破损文本、异常循环和无上下文输出，保留正常生成中的错误模式。应按 token 配比，避免某类长 CoT 独占训练。',
'',
'若需要补领域，再考虑：', '',
'- **更新 ChemCoTDataset 的缺失子任务**：官方 2026-01-19 更新记录加入逆合成、机理等；本地 16 文件中没有 retro / mech_sel / nepp。只补独立训练语料，先与我们的评估题去重。[官方数据卡](https://huggingface.co/datasets/IDEA-AI4S/ChemCoTDataset)',
'- **ChemPile 的 chemistry reasoning / 教育文本小子集**：用于弥补通用化学解释覆盖，不必下载全量。其不同 config 有相同题目的多种模板，不可直接累加。官方卡片的总 token 统计与文件/行规模不易相符，不据此规划容量，应下载选定 split 后重新测量。[官方数据卡](https://huggingface.co/datasets/jablonkagroup/chempile-reasoning)',
'',
'## 激活获取建议',
'',
'沿用前一实验 ChemDFM-R-14B：第 26 个 transformer block 的输出（零索引 25），post-token 表示，维度 5,120。冻结语言模型，以真实输入/对话作为上下文；主要采样 assistant 的 CoT/回答 token，并保留 region、source、task、token offset。padding 和模板控制符不作为主要 SAE 样本。不得将标签、标注原因或 reference 内容额外加入模型实际不可见的提示。',
'',
'先用 10–20M 个去重后有效 tokens 做流程和小字典试验，再向 50–100M 扩展；这是实验预算建议，非已证实的最低需求。另按分子/题目组留出 SAE 验证语料。用重建误差/解释方差、稀疏度、dead-feature 比例、特征稳定性及独立 Pilot 验证表现判断是否需要更多数据；最终测试集不用于选择数据配比或超参数。',
'',
'按 BF16/FP16 2 bytes 缓存：1M tokens = 10.24 GB；10M = 102.4 GB；50M = 512 GB；100M = 1.024 TB（均为十进制、仅激活张量，不含索引等开销）。磁盘容量已复核更正：先前 df -h 曾返回可用 643G；用户反馈后再次检查同一挂载点 /mnt_nas1，当前为可用 9.8T、使用率 87%，df -B1 给出的可用空间为 10,674,734,628,864 bytes（约 9.71 TiB / 10.67 TB）。两次读数变化原因未查明。按当前容量，100M tokens 的 1.024 TB 激活缓存可以容纳；之前“放不下”的判断不再适用。仍建议分片缓存以便恢复和管理，但当前容量不要求必须采用流式方案。',
'',
'SAE 需求取决于字典宽度和稀疏度，并应多维度评估，而不是设一个通用 token 数门槛。参考：[Gao et al., Scaling and evaluating sparse autoencoders](https://arxiv.org/html/2406.04093v1)',
'',
'## 逐文件数据',
'',
'ChemCoT 表使用字段回退后的推理；PubChem second 是 enriched_description；processed 表仅为字段诊断，缺少对话 history/分子上下文，不可直接视为完整输入。',
'',
'| 路径（相对源目录） | 行数 | 文件内不同文本对 | second 字段 token 估计 |',
'|---|---:|---:|---:|',
]
for r in c['results']+[r for r in p['results'] if r['kind']!='chemcot']:
    lines.append(f'| `{r["path"]}` | {r["rows"]:,} | {r["unique_text_pairs"]:,} | {m(total([r]))} |')
lines += ['', 'OpenMolIns 与 xlarge 的文本对重合：', '', '```json', json.dumps(p['openmol_overlap'],ensure_ascii=False,indent=2),'```','','复核文件：`inventory.json`、`profile.json`、`chemcot_selected.json`、`overlap.json`。复现代码：`audit.py`、`check_chemcot_fallback.py`、`check_overlap.py`；可交互查看 `AUDIT.ipynb`。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n')

cells=[{'cell_type':'markdown','metadata':{},'source':['# Layer 26 SAE 语料审计\n','只读检查源目录；token 是固定种子 reservoir 估计。完整方法和限制见 REPORT.md。\n','以下代码嵌入实际执行的脚本，末尾展示已保存结果。重跑会覆盖本审计目录的统计文件，不改动原始语料。']}]
for name in ['audit.py','check_chemcot_fallback.py','check_overlap.py']:
    cells.append({'cell_type':'markdown','metadata':{},'source':[f'## {name}']})
    # Include exact script as inspectable text; rerun via subprocess so __file__ and imports resolve correctly.
    cells.append({'cell_type':'markdown','metadata':{},'source':['```python\n'+(D/name).read_text()+'\n```']})
cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['# Optional reproducible rerun (CPU only):\n','import subprocess, sys\n',f'audit_dir = {str(D)!r}\n','# for script in ("audit.py", "check_chemcot_fallback.py", "check_overlap.py"):\n','#     subprocess.run([sys.executable, f"{audit_dir}/{script}"], check=True, cwd=audit_dir)\n']})
summary=[{'kind':r['kind'],'path':r['path'],'rows':r['rows'],'estimated_second_tokens':r['tokens']['second']['estimated_total']} for r in c['results']+[x for x in p['results'] if x['kind']!='chemcot']]
cells.append({'cell_type':'code','execution_count':1,'metadata':{},'outputs':[{'output_type':'display_data','data':{'application/json':summary,'text/plain':[json.dumps(summary,ensure_ascii=False,indent=2)]},'metadata':{}}],'source':['import json\n','from pathlib import Path\n','from IPython.display import display, JSON\n','root = Path(audit_dir)\n','p = json.loads((root / "profile.json").read_text())\n','c = json.loads((root / "chemcot_selected.json").read_text())\n','rows = c["results"] + [r for r in p["results"] if r["kind"] != "chemcot"]\n','display(JSON([{"kind": r["kind"], "path": r["path"], "rows": r["rows"], "estimated_second_tokens": r["tokens"]["second"]["estimated_total"]} for r in rows]))\n']})
(D/'AUDIT.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}},'nbformat':4,'nbformat_minor':4},ensure_ascii=False,indent=2))
print('Wrote REPORT.md and AUDIT.ipynb')
