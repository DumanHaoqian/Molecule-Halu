import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

class PrefixLoadingTests(unittest.TestCase):
    def test_prefix_matches_full_qwen_post_block_before_final_norm(self):
        import torch
        from transformers import Qwen2Config,Qwen2Model
        from extract_activations import prefix_config
        torch.set_num_threads(2);torch.manual_seed(123)
        config=Qwen2Config(vocab_size=40,hidden_size=32,intermediate_size=48,num_hidden_layers=30,
                          num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
        short=prefix_config(config)
        self.assertEqual(config.num_hidden_layers,30)
        self.assertEqual(short.num_hidden_layers,26)
        full=Qwen2Model(config).eval();prefix=Qwen2Model(short).eval()
        prefix.load_state_dict({k:v for k,v in full.state_dict().items() if k in prefix.state_dict()},strict=True)
        captured=[]
        def save(_,__,out):captured.append((out[0] if isinstance(out,tuple) else out).detach().clone())
        h1=full.layers[25].register_forward_hook(save);h2=prefix.layers[25].register_forward_hook(save)
        ids=torch.tensor([[1,4,5,6,7]])
        with torch.inference_mode():full(ids,use_cache=False);prefix(ids,use_cache=False)
        h1.remove();h2.remove()
        self.assertTrue(torch.equal(captured[0],captured[1]))
        self.assertGreater((captured[0]-prefix(ids).last_hidden_state).abs().max().item(),0)

if __name__=='__main__':unittest.main()
