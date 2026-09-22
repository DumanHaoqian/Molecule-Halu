"""Pure corpus normalization, identity and token accounting helpers (no GPU use)."""
import csv
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
CONTROL = re.compile(r'</?(?:think|answer)>|<\|[^<>]*\|>')
ANSWER_CUE = re.compile(r'ground[\s_-]*truth|\bsecret(?:ly)?\b|\bpretend\b|(?:reference|correct|expected|given|provided) answer|answer.{0,20}(?:provided|given|supplied)|target molecule.{0,40}(?:provided|given|supplied|mentioned)|align with the target molecule',re.I)

def digest(value):
    if not isinstance(value,str):value=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(value.encode()).hexdigest()

def norm(text):return re.sub(r'\s+',' ',str(text)).strip()

def iter_rows(path):
    path=Path(path)
    if path.suffix=='.jsonl':
        with path.open() as f:
            for line in f:
                if line.strip():yield json.loads(line)
    elif path.suffix in ('.csv','.txt'):
        with path.open(newline='') as f:yield from csv.DictReader(f,delimiter='\t' if path.suffix=='.txt' else ',')
    else:
        decoder=json.JSONDecoder()
        with path.open() as f:
            buf='';pos=0;eof=False;started=False
            while True:
                if len(buf)-pos<65536 and not eof:
                    chunk=f.read(1048576);buf=buf[pos:]+chunk;pos=0;eof=not chunk
                while pos<len(buf) and buf[pos].isspace():pos+=1
                if not started:
                    if pos==len(buf):
                        if eof:raise ValueError('Empty JSON array')
                        continue
                    if buf[pos]!='[':raise ValueError('Expected JSON array')
                    pos+=1;started=True;continue
                while pos<len(buf) and buf[pos] in ' \r\n\t,':pos+=1
                if pos==len(buf):
                    if eof:raise ValueError('Unterminated JSON array')
                    continue
                if buf[pos]==']':return
                try:r,pos=decoder.raw_decode(buf,pos)
                except json.JSONDecodeError:
                    if eof:raise
                    chunk=f.read(1048576);buf=buf[pos:]+chunk;pos=0;eof=not chunk;continue
                yield r

@lru_cache(maxsize=50000)
def molecule_keys(smiles):
    if not isinstance(smiles,str) or not smiles.strip():return []
    # Reaction sides are complete molecular mixtures; stereochemistry is deliberately ignored.
    if '>' in smiles:
        return sorted({k for side in smiles.split('>') if side for k in molecule_keys(side)})
    # Source metadata sometimes carries the sentence-ending period from the prompt.
    mol=Chem.MolFromSmiles(smiles.strip().rstrip('.'))
    if mol is None or mol.GetNumAtoms()==0:return []
    keys={Chem.MolToSmiles(mol,isomericSmiles=False)}
    fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True)
    if len(fragments)>1:
        biggest=max(fragments,key=lambda m:(m.GetNumHeavyAtoms(),Chem.MolToSmiles(m,isomericSmiles=False)))
        keys.add(Chem.MolToSmiles(biggest,isomericSmiles=False))
    return sorted(keys)

def flatten(value):
    if isinstance(value,dict):return '\n'.join(str(k)+': '+flatten(v) for k,v in value.items())
    if isinstance(value,list):return '\n'.join(flatten(v) for v in value)
    return str(value)

def select_cot(raw):
    for key in ('raw_cot','struct_cot','cot_result'):
        text=raw.get(key)
        if text and str(text).strip():
            text=str(text).strip()
            if key!='raw_cot':
                text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
                try:text=flatten(json.loads(text))
                except json.JSONDecodeError:pass
            return key,text
    return 'missing',''

def metadata_molecules(raw):
    meta=raw.get('meta') or {}
    if isinstance(meta,str):
        try:meta=json.loads(meta)
        except json.JSONDecodeError:meta={}
    result=[]
    for key in ('molecule','reference','reactants','reagents','rxn_smiles','gt'):
        value=meta.get(key)
        if isinstance(value,str):result.append(value)
        elif isinstance(value,list):result.extend(str(v) for v in value)
    if isinstance(raw.get('gt'),str):result.append(raw['gt'])
    return result

def normalize(source,task,raw,path,index):
    messages=[];smiles=[];flags=[];field=None;cid=None;reference=None
    if source=='chemcot':
        field,answer=select_cot(raw)
        messages=[{'role':'user','content':str(raw.get('query') or '').strip()},{'role':'assistant','content':answer}]
        smiles=metadata_molecules(raw)
        meta=raw.get('meta') or {}
        if isinstance(meta,str):
            try:meta=json.loads(meta)
            except json.JSONDecodeError:meta={};flags.append('malformed_molecule_metadata')
        inputs=[]
        if meta.get('molecule'):inputs.append(str(meta['molecule']))
        if meta.get('rxn_smiles'):inputs.extend(s for s in str(meta['rxn_smiles']).split('>') if s)
        reactants=meta.get('reactants') or []
        if isinstance(reactants,str):reactants=[reactants]
        inputs.extend(str(s) for s in reactants if s)
        if any(not molecule_keys(s) for s in inputs):flags.append('invalid_input_molecule')
    elif source in ('mollama','moleculeqa'):
        cid=str(raw['cid']) if raw.get('cid') is not None else None
        smi=str(raw.get('smiles') or '').strip();smiles=[smi]
        if not molecule_keys(smi):flags.append('invalid_input_molecule')
        # Standard text-only system prompt is supplied by the model template; discard multimodal system instructions.
        for turn in raw.get('conversations',[]):
            user=str(turn.get('user') or '').replace('<mol>','SMILES: '+smi).strip()
            if not messages and smi not in user:user+='\nMolecule SMILES: '+smi
            messages.append({'role':'user','content':user})
            if source=='mollama':messages.append({'role':'assistant','content':str(turn.get('assistant') or '').strip()})
            else:
                reference=turn.get('assistant');break
        field='conversations'
    elif source in ('pubchem','chebi'):
        cid=str(raw.get('cid',raw.get('CID',''))) or None
        smi=str(raw.get('smiles',raw.get('SMILES','')));smiles=[smi]
        if not molecule_keys(smi):flags.append('invalid_input_molecule')
        messages=[{'role':'user','content':'Describe the chemical structure and properties of the molecule.\nMolecule SMILES: '+smi},
                  {'role':'assistant','content':str(raw.get('description') or '').strip()}]
        field='description'
    elif source=='openmolins':
        task=str(raw['SubTask']);prompt=str(raw['Instruction']).strip()
        reference=str(raw.get('molecule') or '');smiles=[reference]
        # Source molecule is embedded in editing/optimization prompts. Parse candidate whitespace tokens strictly as whole SMILES.
        for token in prompt.split():
            token=token.rstrip('.,;:')
            if len(token)>=5 and molecule_keys(token):smiles.append(token)
        messages=[{'role':'user','content':prompt}];field='Instruction'
    else:raise ValueError(source)
    if not messages or any(not m['content'] for m in messages):flags.append('empty_message')
    text='\n'.join(m['content'] for m in messages)
    if '<mol>' in text:flags.append('unresolved_molecule')
    if '<|im_' in text:flags.append('embedded_chat_control')
    if ANSWER_CUE.search(text):flags.append('answer_cue')
    molecules=sorted({k for s in smiles for k in molecule_keys(s)})
    if source=='chemcot' and not molecules:flags.append('missing_molecule_identity')
    normalized_messages=[{'role':m['role'],'content':norm(m['content'])} for m in messages]
    prompt='\n'.join(m['content'] for m in normalized_messages if m['role']=='user')
    answers='\n'.join(m['content'] for m in normalized_messages if m['role']=='assistant')
    keys=['mol:'+k for k in molecules]
    if cid:keys.append('cid:'+cid)
    if prompt:keys.append('prompt:'+digest(prompt))
    if len(answers)>=80:keys.append('assistant:'+digest(answers))
    result={'id':digest([source,path,index])[:24],'source':source,'task':task,
            'source_ref':{'path':path,'row_index':index,'original_id':str(raw.get('id',raw.get('cid',raw.get('CID',index)))),'text_field':field},
            'cid':cid,'molecule_keys':molecules,'group_keys':sorted(set(keys)),
            'messages':messages,'content_sha256':digest(normalized_messages),'quality_flags':sorted(set(flags)),
            'origin':'external_text_teacher_forcing' if source not in ('openmolins','moleculeqa') else 'prompt_only_candidate'}
    if reference is not None:result['reference_for_evaluation_only']=reference
    return result

def blocked_reasons(row,block):
    reasons=[]
    if any(norm(m['content']) in block['questions'] for m in row['messages']):reasons.append('pilot_question')
    if row.get('cid') in block['cids']:reasons.append('pilot_cid')
    if set(row['molecule_keys']) & block['molecules']:reasons.append('pilot_molecule')
    return reasons

def group_records(rows,seed=20260921):
    parent=list(range(len(rows)));seen={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        for key in row['group_keys']:
            if key in seen:
                a,b=find(i),find(seen[key]);parent[max(a,b)]=min(a,b)
            else:seen[key]=i
    group_ids={}
    for i,row in enumerate(rows):
        root=find(i)
        group_ids[root]=min(group_ids.get(root,row['id']),row['id'])
    result={}
    for i,row in enumerate(rows):
        gid=digest(['group',group_ids[find(i)]])[:24]
        split='validation' if int(digest([seed,gid])[:12],16)/16**12<.02 else 'train'
        result[row['id']]=(gid,split)
    return result

class LocalTokenizer:
    """No model weights or torch import; render the checkpoint's actual Jinja template."""
    def __init__(self,path):
        from tokenizers import Tokenizer
        from jinja2 import Environment, StrictUndefined
        self.backend=Tokenizer.from_file(str(Path(path)/'tokenizer.json'))
        self.template=Environment(trim_blocks=True,lstrip_blocks=True).from_string((Path(path)/'chat_template.jinja').read_text())
        self.all_special_ids=[t['id'] for t in json.loads((Path(path)/'tokenizer.json').read_text())['added_tokens'] if t['special']]
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        assert not tokenize
        return self.template.render(messages=messages,add_generation_prompt=add_generation_prompt)
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        e=self.backend.encode(text,add_special_tokens=add_special_tokens)
        return {'input_ids':e.ids,'offset_mapping':e.offsets}

def render_and_count(messages,tok):
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    spans=[]
    for i,m in enumerate(messages):
        if m['role']!='assistant':continue
        prefix=tok.apply_chat_template(messages[:i],tokenize=False,add_generation_prompt=True)
        start=len(prefix);end=start+len(m['content'])
        if text[start:end]!=m['content']:raise ValueError('Chat template/content mismatch')
        spans.append([start,end])
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    controls=[(m.start(),m.end()) for m in CONTROL.finditer(text)]
    special=set(tok.all_special_ids);count=0
    for token,(a,b) in zip(encoded['input_ids'],encoded['offset_mapping']):
        if token in special or a==b or not text[a:b].strip():continue
        if not any(s<=a<b<=e for s,e in spans):continue
        if any(a<e and s<b for s,e in controls):continue
        count+=1
    return {'sequence_tokens':len(encoded['input_ids']),'assistant_tokens':count,'assistant_char_spans':spans,'rendered_sha256':digest(text)}
