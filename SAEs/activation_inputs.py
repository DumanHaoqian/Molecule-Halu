"""Read frozen corpus selections and reproduce eligible post-token positions."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

from corpus import CONTROL, digest, render_and_count


def encode_document(row, tokenizer):
    text = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=False)
    frozen = row['tokenization']
    if render_and_count(row['messages'], tokenizer) != frozen:
        raise ValueError(f"Frozen tokenization mismatch for {row.get('id', '<unknown>')}")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    controls = [(m.start(), m.end()) for m in CONTROL.finditer(text)]
    special = set(tokenizer.all_special_ids)
    spans = frozen['assistant_char_spans']
    positions = [i for i, (token, (a,b)) in enumerate(zip(encoded['input_ids'], encoded['offset_mapping']))
                 if token not in special and a < b and text[a:b].strip()
                 and any(s <= a < b <= e for s,e in spans)
                 and not any(a < e and s < b for s,e in controls)]
    if len(positions) != frozen['assistant_tokens']:
        raise ValueError('Eligible token count changed')
    return encoded['input_ids'], np.asarray(positions, dtype=np.int64)


def read_selection(corpus_root, selection_path):
    root = Path(corpus_root).resolve()
    path = Path(selection_path)
    if not path.is_absolute():
        path = root/path
    with path.open() as f:
        refs = [json.loads(line) for line in f if line.strip()]
    if len({r['id'] for r in refs}) != len(refs):
        raise ValueError('Repeated document in selection')
    for r in refs:
        p = (root/r['path']).resolve()
        if not p.is_relative_to(root) or r['line_index'] < 0:
            raise ValueError('Invalid corpus reference')
    return refs


def iter_selected_documents(corpus_root, selection_path, start=0, limit=None):
    """Preserve selection order with byte offsets, without retaining whole corpora.

    Each referenced JSONL is scanned once for offsets. Content is read on demand;
    only a few integers per selected document are retained in memory.
    """
    root = Path(corpus_root).resolve()
    refs = read_selection(root, selection_path)
    refs = refs[start:] if limit is None else refs[start:start+limit]
    wanted = defaultdict(set)
    for r in refs:
        wanted[r['path']].add(r['line_index'])
    offsets = {}
    for name, lines in wanted.items():
        with (root/name).open('rb') as f:
            i = 0
            while lines:
                offset = f.tell()
                line = f.readline()
                if not line:
                    raise ValueError(f'Missing lines in {name}: {sorted(lines)[:5]}')
                if i in lines:
                    offsets[(name, i)] = offset
                    lines.remove(i)
                i += 1
    for r in refs:
        with (root/r['path']).open('rb') as f:
            f.seek(offsets[(r['path'], r['line_index'])])
            row = json.loads(f.readline())
        for key in ('id','group_id','source','split'):
            if row.get(key) != r.get(key):
                raise ValueError(f'Selection {key} mismatch: {r["id"]}')
        for key in ('assistant_tokens','sequence_tokens'):
            if row['tokenization'][key] != r[key]:
                raise ValueError(f'Selection {key} mismatch: {r["id"]}')
        yield row
