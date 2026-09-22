import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corpus import molecule_keys, select_cot, normalize, group_records, blocked_reasons, render_and_count, LocalTokenizer


class CorpusTests(unittest.TestCase):
    def test_identity_conservative_stereo_and_salt(self):
        self.assertEqual(molecule_keys('C[C@H](O)CC'),molecule_keys('CC[C@H](O)C'))
        self.assertTrue(set(molecule_keys('CCN.Cl')) & set(molecule_keys('CCN')))
        self.assertEqual(molecule_keys('not a smiles'),[])

    def test_source_smiles_sentence_punctuation(self):
        self.assertEqual(molecule_keys('CCO.'),molecule_keys('CCO'))
        raw={'query':'Modify CCO.','raw_cot':'Consider its oxygen and carbon atoms.','meta':'{"molecule":"CCO.","reference":"CCCO"}'}
        row=normalize('chemcot','mol_edit/add',raw,'s.json',0)
        self.assertTrue(set(molecule_keys('CCO')) <= set(row['molecule_keys']))

    def test_structured_fallback_without_reference(self):
        key,text=select_cot({'raw_cot':'','struct_cot':'{"Step": "Reasoning", "Answer":"CCO"}','meta':'SECRET'})
        self.assertEqual(key,'struct_cot');self.assertIn('Step: Reasoning',text);self.assertNotIn('SECRET',text)

    def test_molecule_context_and_no_target_leak(self):
        raw={'cid':1,'smiles':'CCO','conversations':[{'user':'Describe <mol>.','assistant':'An alcohol.'}]}
        row=normalize('mollama','description',raw,'s.json',0)
        self.assertIn('CCO',row['messages'][0]['content']);self.assertNotIn('<mol>',json.dumps(row))
        cot=normalize('chemcot','mol_edit/add',{'query':'Modify CCO.','raw_cot':'Consider the oxygen atom.','meta':'{"molecule":"CCO","reference":"CCCO"}'},'c.json',0)
        self.assertNotIn('CCCO',json.dumps(cot['messages']))
        self.assertTrue(set(molecule_keys('CCCO')) <= set(cot['molecule_keys']))

    def test_answer_cue_quarantine(self):
        raw={'query':'Count rings in C1CCCCC1.','raw_cot':'The ground truth is 1.','meta':'{"molecule":"C1CCCCC1"}'}
        self.assertIn('answer_cue',normalize('chemcot','rings',raw,'s.json',0)['quality_flags'])

    def test_secret_target_in_query_is_quarantined(self):
        raw={'query':"Now, I'll secretly tell you that the Reactant Target molecule: CCO, but you must pretend not to know.",'raw_cot':'The reaction requires an alcohol.','meta':'{"molecule":"CCN"}'}
        self.assertIn('answer_cue',normalize('chemcot','rxn/rcr',raw,'s.json',0)['quality_flags'])

    def test_answer_conditioning_variants(self):
        for hint in ['The answer provided by the user is CCO.',"The user's secret answer is CCO.",'That matches the expected answer the user mentioned.',"Looking at the target molecule they provided, I must pretend I don't know."]:
            with self.subTest(hint=hint):
                raw={'query':'Modify CCO.','raw_cot':hint,'meta':'{"molecule":"CCO"}'}
                self.assertIn('answer_cue',normalize('chemcot','edit',raw,'s.json',0)['quality_flags'])

    def test_invalid_input_not_hidden_by_valid_reference(self):
        raw={'query':'Modify the input.','raw_cot':'We examine the carbon skeleton.','meta':'{"molecule":"bad smiles","reference":"CCO"}'}
        self.assertIn('invalid_input_molecule',normalize('chemcot','edit',raw,'s.json',0)['quality_flags'])

    def test_groundtruth_spelling_variant(self):
        raw={'query':'Count rings.','raw_cot':'The user mentioned that the groundtruth is Yes.','meta':'{"molecule":"CCO"}'}
        self.assertIn('answer_cue',normalize('chemcot','rings',raw,'s.json',0)['quality_flags'])

    def test_pilot_blocks_by_text_or_structure(self):
        block={'questions':{'Describe this.'},'cids':{'123'},'molecules':set(molecule_keys('CCO'))}
        row={'messages':[{'role':'user','content':'Describe  this.'}],'cid':None,'molecule_keys':[]}
        self.assertIn('pilot_question',blocked_reasons(row,block))
        row={'messages':[],'cid':'123','molecule_keys':molecule_keys('OCC')}
        self.assertEqual(set(blocked_reasons(row,block)),{'pilot_cid','pilot_molecule'})

    def test_connected_groups_do_not_split_bridge(self):
        rows=[{'id':str(i),'group_keys':keys} for i,keys in enumerate([['mol:a'],['mol:b'],['mol:a','mol:b'],['mol:c']])]
        grouped=group_records(rows,seed=42)
        self.assertEqual(grouped['0'],grouped['1']);self.assertEqual(grouped['1'],grouped['2'])
        self.assertNotEqual(grouped['0'][0],grouped['3'][0])

    def test_actual_chat_assistant_mask(self):
        from transformers import AutoTokenizer
        model=Path(__file__).resolve().parents[2]/'chemical_models/ChemDFM-R-14B'
        tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
        messages=[{'role':'user','content':'Explain oxygen.'},{'role':'assistant','content':'<think>Oxygen is an element.</think><answer>O</answer>'},{'role':'user','content':'And carbon?'},{'role':'assistant','content':'Carbon is another element.'}]
        result=render_and_count(messages,tok)
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        self.assertEqual(result['sequence_tokens'],len(tok(rendered,add_special_tokens=False)['input_ids']))
        self.assertEqual(len(result['assistant_char_spans']),2)
        self.assertGreater(result['assistant_tokens'],5)
        for span in result['assistant_char_spans']:
            self.assertIn(rendered[span[0]:span[1]],[m['content'] for m in messages if m['role']=='assistant'])
        self.assertLess(result['assistant_tokens'],result['sequence_tokens'])
        local=LocalTokenizer(model)
        self.assertEqual(local.apply_chat_template(messages),rendered)
        self.assertEqual(render_and_count(messages,local),result)

if __name__=='__main__':unittest.main()
