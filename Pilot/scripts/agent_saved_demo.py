"""Read-only display of saved Agent pairs and their exact annotation sidecars."""
from __future__ import annotations

import difflib
import html
import json
from pathlib import Path
import re

import gradio as gr

CSS = """
mark.agent-root { background: #fecaca; color: #172033; border-radius: 2px; }
mark.agent-propagated { background: #fef08a; color: #172033; border-radius: 2px; }
mark.agent-root.agent-propagated { background: #fed7aa; }
.agent-trace pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; }
"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'无法读取已保存的配套文件 {path}: {error}') from error


def is_agent_dataset(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                return 'detector_input' in record and 'serialized' not in record
    return False


def _comparison(left, right, left_label):
    a, b = [], []
    for tag, lo, hi, x, y in difflib.SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
        first, second = html.escape(left[lo:hi]), html.escape(right[x:y])
        a.append(first if tag=='equal' else '<mark class="diff-before">'+first+'</mark>')
        b.append(second if tag=='equal' else '<mark class="diff-after">'+second+'</mark>')
    return ('<section class="step-compare"><div class="compare-grid"><article><h4>'
            +html.escape(left_label)+'</h4><pre>'+''.join(a)
            +'</pre></article><article><h4>新版 H</h4><pre>'+''.join(b)
            +'</pre></article></div></section>')


def marked_trace(text, spans):
    """Render union intervals; overlapping root/propagated labels stay visible."""
    boundaries = sorted({0,len(text),*(s[k] for s in spans for k in ['start','end'])})
    pieces = []
    for start,end in zip(boundaries,boundaries[1:]):
        value = html.escape(text[start:end])
        active = [s for s in spans if s['start']<end and s['end']>start]
        if active:
            classes = sorted({'agent-root' if s['kind']=='root_error' else 'agent-propagated' for s in active})
            title = html.escape('; '.join(s['node_id']+' | '+s['kind']+' | '+','.join(s['root_ids']) for s in active),quote=True)
            value = '<mark class="'+' '.join(classes)+'" title="'+title+'">'+value+'</mark>'
        pieces.append(value)
    return '<section class="step-compare agent-trace"><pre>'+''.join(pieces)+'</pre></section>'


class AgentSavedDemo:
    def __init__(self,path):
        self.path = Path(path)
        grouped = {}
        with self.path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip(): continue
                pair = json.loads(line)
                label = pair['variant_label']; variants = grouped.setdefault(pair['pair_id'],{})
                _require(label in {'H','N'} and label not in variants,'Unknown or duplicate H/N variant')
                variants[label] = pair
        _require(bool(grouped),'No saved Agent pairs')
        self.records = {}
        for pair_id,pairs in grouped.items():
            _require(set(pairs)=={'H','N'},'Incomplete Agent H/N pair: '+pair_id)
            h,n = pairs['H'],pairs['N']; origin = h['origin_id']
            _require(bool(re.fullmatch(r'[A-Za-z0-9_.-]+',origin)) and origin not in {'.','..'},'Invalid origin identity')
            directory = self.path.parent/'origins'/origin
            saved = _read(directory/'accepted.json'); row = _read(directory/'input.json')['row']
            _require(saved.get('status')=='accepted' and saved['pair_id']==pair_id
                     and saved['origin_id']==n['origin_id']==row['origin_id']==origin,'Saved pair identity mismatch')
            saved_pairs = saved['pairs']
            _require(len(saved_pairs)==2 and {p['variant_label']:p for p in saved_pairs}==pairs,
                     'Dataset and annotation sidecar pairs differ')
            for field,raw in [('instruction','instruction'),('indexed_smiles','indexed_smiles'),('final_answer','gt_smiles')]:
                _require(h['detector_input'][field]==n['detector_input'][field]==row[raw], 'Original question/GT mismatch')
            nt,ht = n['detector_input']['reasoning_chain'],h['detector_input']['reasoning_chain']
            _require(nt==saved['N_visible'] and ht==saved['plan']['render']['text'],'Saved trace mismatch')
            _require(saved.get('N_source_visible',row['N_visible'])==row['N_visible'],'Original N mismatch')
            nodes,spans = saved['plan']['nodes'],saved['annotations']
            _require(bool(spans) and spans==saved['plan']['render']['spans'],'Saved annotations mismatch')
            for span in spans:
                lo,hi = span['start'],span['end']
                _require(type(lo) is int and type(hi) is int and 0<=lo<hi<=len(ht)
                         and ht[lo:hi]==span['text'],'Saved annotation offset/text mismatch')
                node = nodes.get(span['node_id'])
                _require(node is not None and span['kind'] in {'root_error','propagated_error'}
                         and span['kind']==node['kind'] and span['root_ids']==node['root_ids']
                         and node['changed'],'Saved annotation node/root mismatch')
            for name,tokens in saved['token_labels'].items():
                for token in tokens:
                    lo,hi = token['start'],token['end']
                    _require(0<=lo<=hi<=len(ht) and token['text']==ht[lo:hi]
                             and bool(token['labels']) and set(token['labels'])<={'root_error','propagated_error','unchanged'},
                             'Saved token offsets/labels mismatch: '+name)
            self.records[pair_id] = {'saved':saved,'row':row,'N':nt,'H':ht}

    def outputs(self,pair_id):
        record = self.records[pair_id]; saved,row = record['saved'],record['row']
        nodes,spans = saved['plan']['nodes'],saved['annotations']
        status = (f"**{saved['origin_id']}** · `{saved['protocol']}` · `{saved.get('style','')}` · "
                  f"{saved.get('acceptance_scope','未记录验收范围')}\n\n"
                  f"{len(saved['plan']['roots'])} 个根错误 / {len({s['node_id'] for s in spans})} 个错误节点 / "
                  f"{len(spans)} 个标注区间。只读回放；本次模型/API 请求 0。")
        question = row['instruction']+'\n\n原始分子（indexed SMILES）：\n'+row['indexed_smiles']
        node_rows = [[key,node['kind'],', '.join(node['parents']),', '.join(node['root_ids']),
                      str(node['before']),str(node['after']),node['changed']] for key,node in nodes.items()]
        span_rows = [[s['node_id'],s['kind'],', '.join(s['root_ids']),s['start'],s['end'],s['text']] for s in spans]
        token_rows = []
        for name,tokens in saved['token_labels'].items():
            errors = sum(any(label!='unchanged' for label in t['labels']) for t in tokens)
            roots = sum('root_error' in t['labels'] for t in tokens)
            propagated = sum('propagated_error' in t['labels'] for t in tokens)
            token_rows.append([name,len(tokens),errors,roots,propagated,round(100*errors/len(tokens),2) if tokens else 0])
        audit = {k:saved.get(k) for k in ['protocol','style','acceptance_scope','error_counts','density_report','actual_api_calls','review_status','release_contract']}
        audit['original_N_source'] = 'input.json → row.N_visible；原始 N 已移除历史完整答案子句'
        audit['paired_N_source'] = 'pairs.jsonl → N.detector_input.reasoning_chain'
        audit['annotation_source'] = 'accepted.json → annotations；不是字符 diff 自动推断'
        audit['token_scope'] = '已保存的独立 H 分词结果；不是完整聊天提示词。边界 token 可同时有多个标签。'
        return (status,question,_comparison(row['N_visible'],record['H'],'原始 N'),
                _comparison(record['N'],record['H'],'实验配对 N'),marked_trace(record['H'],spans),
                node_rows,span_rows,token_rows,audit)


def build_agent_demo(path):
    viewer = AgentSavedDemo(path); ids = tuple(viewer.records); first = ids[0]
    initial = viewer.outputs(first)
    choices = [(r['saved']['origin_id']+' · '+r['saved'].get('style',''),key) for key,r in viewer.records.items()]
    with gr.Blocks(title='MolHalluLens · Agent H/N 对比') as app:
        gr.Markdown(f'# Agent H/N 对比\n已加载 **{len(ids)} 对**，来源 `{Path(path)}`。')
        gr.Markdown('原始 N 与实验配对 N 分开展示；文字差异不等于幻觉标签。此页面只读取已有数据，不生成样本、不调用模型。')
        with gr.Row():
            previous = gr.Button('← 上一个',interactive=False,scale=0)
            selection = gr.Dropdown(choices=choices,value=first,label='Origin / Pair',scale=5)
            following = gr.Button('下一个 →',interactive=len(ids)>1,scale=0)
        position = gr.Markdown(f'第 1 / {len(ids)} 题')
        status = gr.Markdown(initial[0])
        question = gr.Textbox(initial[1],label='原始 instruction 与分子（只读）',interactive=False,lines=4)
        gr.Markdown('## 原始 N → 新版 H\n红色为删除/替换前，绿色为新增/替换后。新增连接说明也会显示为文字差异，不代表整段都是错误。')
        original = gr.HTML(initial[2])
        with gr.Tabs():
            with gr.Tab('实验配对 N vs H'):
                gr.Markdown('这里的 N 是实际实验输入，可能比原始 N 增加连接说明。')
                paired = gr.HTML(initial[3])
            with gr.Tab('精确错误标签'):
                gr.Markdown('红色：根错误；黄色：传播错误；橙色：重叠。悬停查看节点与根 ID。未标色不表示已经证明正确。')
                marked = gr.HTML(initial[4])
                spans = gr.Dataframe(initial[6],headers=['节点','类型','根 ID','开始','结束','H 文本'],interactive=False,wrap=True)
                gr.Markdown('下表使用已保存的两个模型 tokenizer，范围是独立 H；根/传播计数可能重叠，不能直接相加。')
                tokens = gr.Dataframe(initial[7],headers=['模型','总 tokens','错误 tokens','根 tokens','传播 tokens','错误比例 %'],interactive=False)
            with gr.Tab('错误传播与审计'):
                nodes = gr.Dataframe(initial[5],headers=['节点','类型','父节点','根 ID','N 值','H 值','是否改变'],interactive=False,wrap=True)
                audit = gr.JSON(initial[8],label='已保存的诊断范围与标签说明')
        outputs = [status,question,original,paired,marked,nodes,spans,tokens,audit]

        def navigate(pair_id,offset):
            index = ids.index(pair_id) if pair_id in ids else 0
            index = max(0,min(len(ids)-1,index+offset))
            return ids[index],gr.update(interactive=index>0),gr.update(interactive=index<len(ids)-1),f'第 {index+1} / {len(ids)} 题'

        def load(pair_id):
            try:
                _,prev,nxt,counter = navigate(pair_id,0)
                return (*viewer.outputs(pair_id),prev,nxt,counter)
            except Exception as error:
                return (f'加载失败：{error}','','','','',[],[],[],None,
                        gr.update(interactive=False),gr.update(interactive=False),'请选择有效题目')

        previous.click(lambda value:navigate(value,-1),selection,[selection,previous,following,position],api_name='previous_agent_pair',queue=False)
        following.click(lambda value:navigate(value,1),selection,[selection,previous,following,position],api_name='next_agent_pair',queue=False)
        selection.change(load,selection,[*outputs,previous,following,position],api_name='select_agent_pair',
                         concurrency_limit=1,trigger_mode='always_last')
    return app
