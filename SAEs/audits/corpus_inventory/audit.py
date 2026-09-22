"""Read-only corpus census; token totals are seeded reservoir estimates, not exact counts."""
import csv, hashlib, json, os, random, re, statistics
from collections import Counter
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from tokenizers import Tokenizer

ROOT = Path('/home/haoqian/Data/past/Molecule')
OUT = Path(__file__).resolve().parent
TOK = Tokenizer.from_file('/mnt_nas1/haoqian/Data/Molecule/chemical_models/ChemDFM-R-14B/tokenizer.json')
N_SAMPLE = 1024

def array_rows(path):
    decoder = json.JSONDecoder()
    with open(path) as f:
        buf, pos, eof = '', 0, False
        while True:
            if len(buf) - pos < 65536 and not eof:
                chunk = f.read(1048576)
                buf, pos, eof = buf[pos:] + chunk, 0, not chunk
            while pos < len(buf) and buf[pos] in ' \t\r\n[,':
                pos += 1
            if pos == len(buf):
                if eof: return
                continue
            if buf[pos] == ']': return
            try:
                row, pos = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof: raise
                chunk = f.read(1048576)
                buf, pos, eof = buf[pos:] + chunk, 0, not chunk
                continue
            yield row

def rows(path):
    if path.suffix == '.json': yield from array_rows(path)
    elif path.suffix == '.jsonl':
        with open(path) as f:
            for line in f:
                if line.strip(): yield json.loads(line)
    else:
        with open(path) as f: yield from csv.DictReader(f, delimiter='\t' if path.suffix == '.txt' else ',')

def texts(r, kind):
    if kind == 'chemcot': return str(r.get('query') or ''), str(r.get('raw_cot') or '')
    if kind in ('mollama', 'moleculeqa'):
        qa = r.get('conversations', [])
        # Preserve dialog order; replace unavailable multimodal input with explicit SMILES.
        smi = str(r.get('smiles') or '')
        full = '\n'.join(str(t.get(k) or '').replace('<mol>', 'Molecule SMILES: '+smi) for t in qa for k in ('user','assistant'))
        ans = '\n'.join(str(t.get('assistant') or '') for t in qa)
        return full, ans
    if kind == 'pubchem': return str(r.get('description') or ''), str(r.get('enriched_description') or '')
    if kind == 'openmol': return str(r.get('Instruction') or ''), str(r.get('molecule') or '')
    if kind == 'chebi': return str(r.get('SMILES') or ''), str(r.get('description') or '')
    if kind == 'processed':
        inp, target = r.get('input') or {}, r.get('targets') or {}
        return str(inp.get('instruction') or ''), str(target.get('text') or '')
    raise ValueError(kind)

def estimate(lengths, n):
    mean = statistics.mean(lengths)
    se = statistics.stdev(lengths) / len(lengths)**.5 * max(0,1-len(lengths)/n)**.5 if len(lengths)>1 else 0
    ordered = sorted(lengths)
    return {'mean':mean, 'estimated_total':round(mean*n), 'approx_95pct_margin':round(1.96*se*n), 'sample_p50':ordered[len(ordered)//2], 'sample_p95':ordered[min(len(ordered)-1,int(.95*len(ordered)))]}

def audit(path,kind):
    rng, sample = random.Random(20260921), []
    seen, cids, tasks = set(), set(), Counter()
    chars = [0,0]; empty = [0,0]; markers=0; placeholders=0; turns=0
    for n,r in enumerate(rows(path),1):
        a,b = texts(r,kind)
        digest=hashlib.sha256((a+'\x00'+b).encode()).digest()[:16]
        seen.add(digest)
        for i,t in enumerate((a,b)):
            chars[i]+=len(t); empty[i]+=not bool(t.strip())
        if r.get('cid') is not None: cids.add(str(r['cid']))
        if kind in ('mollama','moleculeqa'):
            turns += len(r.get('conversations',[]))
            placeholders += any('<mol>' in str(t.get('user','')) for t in r.get('conversations',[]))
        tasks[str(r.get('SubTask') or r.get('subtask') or r.get('task_type') or r.get('category') or kind)] += 1
        if kind=='chemcot': markers += bool(re.search(r'ground[ -]truth|reference answer|correct answer is', b, re.I))
        if len(sample)<N_SAMPLE: sample.append((a,b))
        else:
            j=rng.randrange(n)
            if j<N_SAMPLE: sample[j]=(a,b)
    metrics={}
    for name, seq in [('first',[a for a,b in sample]),('second',[b for a,b in sample]),('combined',[(a+'\n'+b) if kind not in ('mollama','moleculeqa') else a for a,b in sample])]:
        lens=[len(x.ids) for x in TOK.encode_batch(seq,add_special_tokens=False)]
        metrics[name]=estimate(lens,n)
    result={'path':str(path.relative_to(ROOT)), 'kind':kind, 'bytes':path.stat().st_size,'rows':n,'unique_text_pairs':len(seen),'unique_cids':len(cids),'turns':turns,'empty_fields':empty,'text_characters':chars,'tasks':dict(tasks),'answer_cue_rows':markers,'mol_placeholder_rows':placeholders,'sample_n':len(sample),'tokens':metrics}
    print(json.dumps({k:result[k] for k in ('path','rows','unique_text_pairs','answer_cue_rows')})+' tokens='+str(metrics['combined']['estimated_total']),flush=True)
    return result,cids,seen

def main():
    jobs=[]
    jobs += [(p,'chemcot') for p in sorted((ROOT/'Downloads/Train/Chemcot').glob('*/*.json'))]
    base=ROOT/'Molecule/Baselines/Mol-LLaMA/data'
    jobs += [(base/'Mol-LLaMA-Instruct'/name,'mollama') for name in ['detailed_structural_descriptions.json','structure2chemical_features_relationships.json','structure2biological_features_relationships.json','comprehensive_conversations.json']]
    jobs += [(base/'Mol-LLaMA-Instruct/pubchem-molecules.json','pubchem')]
    jobs += [(ROOT/'Downloads/Train/OpenMolIns'/size/'train.csv','openmol') for size in ['xlarge','large','medium','small','light']]
    jobs += [(base/'moleculeqa/train.json','moleculeqa'),(base/'ChEBI-20/train.txt','chebi')]
    jobs += [(p,'processed') for p in sorted((ROOT/'Datasets/Train').glob('*/*.jsonl'))]
    results=[]; all_cids=set(); openmol_xlarge=set(); intersections=[]
    for p,k in jobs:
        r,cids,seen=audit(p,k); results.append(r)
        if k=='mollama': all_cids.update(cids)
        if k=='pubchem': r['cids_shared_with_mollama']=len(cids & all_cids)
        if k=='openmol':
            if p.parent.name=='xlarge': openmol_xlarge=seen
            else: intersections.append({'size':p.parent.name,'unique_pairs':len(seen),'shared_with_xlarge':len(seen&openmol_xlarge)})
        (OUT/'profile.json').write_text(json.dumps({'method':'Exact rows and text-pair hashes; seeded reservoir token estimates. No system/chat-template tokens. PubChem first=original, second=enriched (alternative text views); MolLLaMA first=full dialog, second=assistant-only; other first=prompt, second=answer. Processed diagnostic only: excludes history/molecule input and is not a ready extraction recipe.','results':results,'mollama_unique_cids':len(all_cids),'openmol_overlap':intersections},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
