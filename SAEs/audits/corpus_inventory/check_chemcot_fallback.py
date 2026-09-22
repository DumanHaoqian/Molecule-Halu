"""Recover task-dependent CoT fields; choose one view per record, never concatenate duplicates."""
import json,re
from collections import Counter
import audit

def flatten(x):
    if isinstance(x,dict): return '\n'.join(str(k)+': '+flatten(v) for k,v in x.items())
    if isinstance(x,list): return '\n'.join(flatten(v) for v in x)
    return str(x)

def select(r):
    for key in ('raw_cot','struct_cot','cot_result'):
        s=r.get(key)
        if s and str(s).strip():
            s=str(s).strip()
            if key!='raw_cot':
                s=re.sub(r'^```(?:json)?\s*|\s*```$','',s)
                try: s=flatten(json.loads(s))
                except json.JSONDecodeError: pass
            return key,s
    return 'missing',''

audit.texts=lambda r,k:(str(r.get('query') or ''),select(r)[1])
audit.N_SAMPLE=512
result=[]
for p in sorted((audit.ROOT/'Downloads/Train/Chemcot').glob('*/*.json')):
    r,_,_=audit.audit(p,'chemcot')
    r['selected_fields']=dict(Counter(select(x)[0] for x in audit.rows(p)))
    result.append(r)
    (audit.OUT/'chemcot_selected.json').write_text(json.dumps({'method':'raw_cot preferred, otherwise flattened struct_cot, otherwise cot_result stripped of code fences and flattened if JSON parses; no reference/meta/gt injected. Exact counts; 512-record seeded reservoir per file for token estimates.','results':result},ensure_ascii=False,indent=2))
