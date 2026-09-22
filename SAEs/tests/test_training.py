"""Input selection and extraction integrity, independent of actual GPU weights."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activation_inputs import encode_document, iter_selected_documents
from corpus import LocalTokenizer, render_and_count

class InputTests(unittest.TestCase):
    def test_document_encoding_matches_frozen_mask(self):
        model=Path(__file__).resolve().parents[2]/'chemical_models/ChemDFM-R-14B'
        tok=LocalTokenizer(model)
        messages=[{'role':'user','content':'What is water?'},{'role':'assistant','content':'<think>It contains oxygen and hydrogen.</think><answer>O</answer>'}]
        row={'messages':messages,'tokenization':render_and_count(messages,tok)}
        ids,positions=encode_document(row,tok)
        self.assertEqual(len(ids),row['tokenization']['sequence_tokens'])
        self.assertEqual(len(positions),row['tokenization']['assistant_tokens'])
        self.assertTrue(np.all(np.diff(positions)>0))
        self.assertLess(int(positions[-1]),len(ids))
        row['tokenization']['assistant_tokens']+=1
        with self.assertRaises(ValueError):encode_document(row,tok)

    def test_selection_order_and_identity_are_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rows=[{'id':str(i),'group_id':str(i),'source':'x','split':'train','tokenization':{'assistant_tokens':1,'sequence_tokens':3}} for i in range(3)]
            (root/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            refs=[{**{k:rows[i][k] for k in ('id','group_id','source','split')},'path':'rows.jsonl','line_index':i,'assistant_tokens':1,'sequence_tokens':3} for i in [2,0]]
            p=root/'selected.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in refs))
            self.assertEqual([r['id'] for r in iter_selected_documents(root,p)],['2','0'])
            refs[0]['id']='wrong';p.write_text(''.join(json.dumps(r)+'\n' for r in refs))
            with self.assertRaises(ValueError):list(iter_selected_documents(root,p))

    def test_shard_writer_recovers_completed_documents_only(self):
        from extract_activations import ShardWriter
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest={'splits':{'train':{'n_tokens':0,'n_documents':0,'shards':[]}}}
            writer=ShardWriter(root,'train',manifest,d_in=3,shard_rows=4)
            writer.add(np.ones((3,3)),{'id':'a','selection_index':0,'positions':[1,2,3]})
            writer.add(np.ones((2,3))*2,{'id':'b','selection_index':1,'positions':[2,3]})
            self.assertEqual(manifest['splits']['train']['n_documents'],1)
            # Unflushed b must be re-extracted after interruption.
            resumed=ShardWriter(root,'train',manifest,d_in=3,shard_rows=4)
            resumed.add(np.ones((2,3))*2,{'id':'b','selection_index':1,'positions':[2,3]});resumed.flush()
            self.assertEqual(manifest['splits']['train']['n_tokens'],5)
            a=np.concatenate([np.load(root/s['path']) for s in manifest['splits']['train']['shards']])
            self.assertEqual(a.shape,(5,3));self.assertEqual(a[-1,0],2)

    def test_capture_is_current_token_at_block_index_25(self):
        import torch
        from extract_activations import capture_post_tokens
        class Block(torch.nn.Module):
            def __init__(self, increment):
                super().__init__();self.increment=increment
            def forward(self,x):return (x+self.increment,)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.emb=torch.nn.Embedding(10,3)
                with torch.no_grad():self.emb.weight.copy_(torch.arange(10).view(-1,1).expand(-1,3))
                self.layers=torch.nn.ModuleList(Block(i+1) for i in range(30))
            def get_input_embeddings(self):return self.emb
            def forward(self,input_ids,attention_mask,use_cache):
                x=self.emb(input_ids)
                for b in self.layers:x=b(x)[0]
                raise AssertionError('Must stop after target block')
        model=Toy();values=capture_post_tokens(model,[2,3,7],[0,2])
        np.testing.assert_array_equal(values[:,0],[353,358])
        self.assertEqual(len(model.layers[25]._forward_hooks),0)

    def test_preflight_rejects_modified_frozen_selection(self):
        from extract_activations import extraction_plan
        from prepare import dump,file_sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'model';model.mkdir()
            dump(model/'config.json',{'hidden_size':5120,'num_hidden_layers':48})
            (model/'model.safetensors').write_bytes(b'toy weight identity, never loaded')
            (root/'selected').mkdir()
            entries=[]
            for split in ('train','validation'):
                p=root/f'{split}.jsonl';p.write_text('{}\n')
                ref={'id':split,'group_id':split,'source':'toy','split':split,'path':p.name,'line_index':0,'assistant_tokens':1,'sequence_tokens':2}
                q=root/'selected'/f'{split}.jsonl';q.write_text(json.dumps(ref)+'\n')
                entries.extend({'path':str(x.relative_to(root)),'sha256':file_sha(x)} for x in (p,q))
            dump(root/'manifest.json',{'model_files':{'config.json':file_sha(model/'config.json')},'outputs':entries})
            dump(root/'verification.json',{'status':'passed'})
            dump(root/'COMPLETE.json',{'status':'verified','manifest_sha256':file_sha(root/'manifest.json'),'verification_sha256':file_sha(root/'verification.json')})
            plan=extraction_plan(root,'selected',model)
            self.assertEqual(plan['splits']['train']['expected_tokens'],1)
            with (root/'selected/train.jsonl').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'not sealed'):extraction_plan(root,'selected',model)

if __name__=='__main__':unittest.main()
