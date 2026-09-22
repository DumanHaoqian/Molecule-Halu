"""Conservative exact-match overlap checks; no claim of exhaustive chemical deduplication."""
import hashlib,json,re
from pathlib import Path
from audit import ROOT,OUT,rows

def norm(s): return re.sub(r'\s+',' ',str(s)).strip()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

pilot=[json.loads(s) for s in (OUT.parents[2]/'Pilot_v2/experiments/post_token_layer26/prepared/records.jsonl').open()]
questions={norm(r['input_text']):(r['task'],r['question_id'],r['split']) for r in pilot}
caption_ids={r['question_id'] for r in pilot if r['task'] in ('cap2mol','mol2cap')}
result={'method':'Exact whitespace-normalized question matches; CID matches only for Pilot cap2mol/mol2cap, where question IDs are PubChem IDs. No canonical-molecule or semantic near-duplicate checks performed. CID overlap means shared molecules, not necessarily identical questions.','pilot_unique_questions':len(questions),'pilot_caption_cids':len(caption_ids),'copies':[],'sources':[]}
base=ROOT/'Molecule/Baselines/Mol-LLaMA/data'
jobs=[(p,'query') for p in sorted((ROOT/'Downloads/Train/Chemcot').glob('*/*.json'))]
jobs += [(p,'Instruction') for p in sorted((ROOT/'Downloads/Train/OpenMolIns').glob('*/train.csv'))]
jobs += [(p,None) for p in sorted((base/'Mol-LLaMA-Instruct').glob('*.json')) if p.name!='pubchem-molecules.json']
jobs += [(base/'ChEBI-20/train.txt','description')]
for p,field in jobs:
    hits=set(); cid_hits=set(); cue_examples=[]
    for r in rows(p):
        if field and norm(r.get(field,'')) in questions: hits.add(questions[norm(r[field])])
        cid=str(r.get('cid',r.get('CID','')))
        if cid in caption_ids:cid_hits.add(cid)
        if field=='query' and len(cue_examples)<2:
            s=r.get('raw_cot') or ''; m=re.search(r'ground[ -]truth|reference answer|correct answer is',s,re.I)
            if m:cue_examples.append({'id':r.get('id'),'excerpt':s[max(0,m.start()-80):m.end()+160]})
    result['sources'].append({'path':str(p.relative_to(ROOT)),'exact_pilot_question_matches':sorted(hits),'shared_caption_cids':sorted(cid_hits),'answer_cue_examples':cue_examples})
for rel_a,rel_b in [
 ('Downloads/Train/Chemcot/mol_edit/add.json','Construction/Datasets/Chemcot/mol_edit/add.json'),
 ('Downloads/Train/OpenMolIns/xlarge/train.csv','Construction/TOMG-Bench/data/OpenMolIns/xlarge/train.csv'),
 ('Molecule/Baselines/Mol-LLaMA/data/Mol-LLaMA-Instruct/comprehensive_conversations.json','Construction/Datasets/Mol-LLaMA-Instruct/comprehensive_conversations.json')]:
    a,b=ROOT/rel_a,ROOT/rel_b
    if a.exists() and b.exists():
        ha,hb=sha(a),sha(b);result['copies'].append({'a':rel_a,'b':rel_b,'sha256_a':ha,'sha256_b':hb,'identical':ha==hb})
(OUT/'overlap.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'copy_checks':result['copies'],'sources_with_overlap':[{k:v for k,v in r.items() if k!='answer_cue_examples'} for r in result['sources'] if r['exact_pilot_question_matches'] or r['shared_caption_cids']]},ensure_ascii=False))
