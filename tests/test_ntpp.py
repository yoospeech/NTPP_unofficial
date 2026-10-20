"""CPU regression checks; no checkpoints, datasets, or codec downloads required."""
import copy
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache
from ntpp.model import LlamaConfig, LlamaForCausalLM, PhiConfig, PhiForCausalLM
from ntpp.train.data import split_aligned_examples


def tiny(family, depth=4, stage="ntpp", backend="sdpa", embedding="one_hot"):
    C, M = (LlamaConfig, LlamaForCausalLM) if family == "llama" else (PhiConfig, PhiForCausalLM)
    extra = {"partial_rotary_factor": 0.5} if family == "phi" else {}
    c = C(vocab_size=32, hidden_size=32, intermediate_size=64,
          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
          rvq_depth=depth, ntpp_stage=stage, channel_embedding_type=embedding, **extra)
    c._attn_implementation = backend
    return M(c)


class NTPPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_data_boundaries(self):
        rows = [[1]*24, [2]*16]
        out = split_aligned_examples({"input_ids": rows, "attention_mask": [[1]*len(r) for r in rows]}, 16, 8)
        self.assertEqual(out["input_ids"], [[1]*16, [2]*16])
        with self.assertRaises(ValueError):
            split_aligned_examples({"input_ids": [[1]*17]}, 16, 8)
        with self.assertRaises(ValueError):
            split_aligned_examples({"input_ids": rows}, 17, 8)

    def test_waveform_loading_without_torchcodec(self):
        import numpy as np
        import soundfile as sf
        from pathlib import Path
        from ntpp.utils.moss_audio import MossAudioTokenEncoder
        encoder = MossAudioTokenEncoder.__new__(MossAudioTokenEncoder)
        encoder.sample_rate = 16000
        with tempfile.TemporaryDirectory() as directory:
            stereo = Path(directory) / "stereo.wav"
            mono = Path(directory) / "mono.wav"
            samples = np.stack([np.ones(800)*0.1, np.ones(800)*0.3], axis=1)
            sf.write(stereo, samples, 8000)
            sf.write(mono, samples[:, 0], 8000)
            loaded = encoder._load_waveform(stereo)
            self.assertEqual(tuple(loaded.shape), (2, 1600))
            self.assertLess(loaded[0].mean(), loaded[1].mean())
            self.assertEqual(tuple(encoder._load_waveform(mono, require_two=False).shape), (1, 1600))
            with self.assertRaises(ValueError):
                encoder._load_waveform(mono)

    def test_checkpointing_forward_and_gradients(self):
        for family in ("llama", "phi"):
            for stage in ("pretrain", "ntpp"):
                for backend in ("eager", "sdpa"):
                    for embedding in ("one_hot", "learned"):
                        with self.subTest(family=family, stage=stage, backend=backend, embedding=embedding):
                            a = tiny(family, stage=stage, backend=backend, embedding=embedding).train()
                            b = copy.deepcopy(a)
                            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                            x = torch.randint(0,32,(2,24))
                            ya = a(x, labels=x, use_cache=False)
                            yb = b(x, labels=x, use_cache=False)
                            torch.testing.assert_close(ya.logits, yb.logits)
                            ya.loss.backward(); yb.loss.backward()
                            for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
                                if p.grad is not None:
                                    torch.testing.assert_close(p.grad, q.grad, msg=name)

    def test_cache_and_mask(self):
        for family in ("llama", "phi"):
            for depth in (1,4):
                with self.subTest(family=family, depth=depth), torch.no_grad():
                    m=tiny(family,depth).eval(); stride=2*depth
                    x=torch.randint(0,32,(1,3*stride))
                    full=m(x,use_cache=False).logits
                    cache=DynamicCache(); pieces=[]
                    for i in range(x.shape[1]):
                        y=m(x[:,i:i+1],past_key_values=cache,use_cache=True)
                        cache=y.past_key_values; pieces.append(y.logits)
                    torch.testing.assert_close(full,torch.cat(pieces,1),atol=1e-6,rtol=1e-5)
                    # Change A in frame 2: B in frame 2 must be unaffected at all layers.
                    changed=x.clone(); changed[:,stride:stride+depth]=(changed[:,stride:stride+depth]+1)%32
                    out=m(changed,use_cache=False).logits
                    torch.testing.assert_close(full[:,stride+depth:2*stride],out[:,stride+depth:2*stride])
                    # Alter a future depth and future frames: earlier depth cannot change.
                    changed=x.clone(); changed[:,stride+1:]=(changed[:,stride+1:]+1)%32
                    out=m(changed,use_cache=False).logits
                    torch.testing.assert_close(full[:,:stride+1],out[:,:stride+1])

    def test_generation_matches_full_prefix_reference(self):
        for family in ("llama", "phi"):
            for depth in (1,4):
                for stage in ("pretrain", "ntpp"):
                    with self.subTest(family=family,depth=depth,stage=stage), torch.no_grad():
                        m=tiny(family,depth,stage).eval()
                        channels=2 if stage=="ntpp" else 1; stride=channels*depth
                        prompt=torch.randint(0,32,(2,2*stride))
                        ref=prompt.clone()
                        for frame in range(2):
                            start=ref.shape[1]
                            for c in range(channels):
                                for d in range(depth):
                                    source=start-stride+(c+1)*depth-1 if d==0 else ref.shape[1]-1
                                    logits=m(ref,use_cache=False).logits[:,source]
                                    ref=torch.cat([ref,logits.argmax(-1,keepdim=True)],1)
                        actual=m.generate(prompt,max_new_tokens=2*stride)
                        self.assertTrue(torch.equal(ref,actual))

    def test_padding_loss_and_stage_checkpoint(self):
        for family in ("llama", "phi"):
            m=tiny(family).eval(); x=torch.randint(0,32,(2,24)); labels=x.clone()
            labels[1,16:]=-100
            loss=m(x,labels=labels,use_cache=False).loss
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(m(x,labels=torch.full_like(x,-100),use_cache=False).loss.item(),0)
            with self.assertRaises(ValueError):m(x[:,:17],labels=x[:,:17])
            a=tiny(family,stage="pretrain")
            with tempfile.TemporaryDirectory() as path:
                a.save_pretrained(path)
                b=type(a).from_pretrained(path,attn_implementation="sdpa")
                b.config.ntpp_stage="ntpp"
                self.assertEqual(a.state_dict().keys(),b.state_dict().keys())
                for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name])
                self.assertTrue(torch.isfinite(b(x,labels=x,use_cache=False).loss))

    def test_depth_vocabulary_and_one_hot(self):
        for family in ("llama","phi"):
            m=tiny(family).eval()
            emb=m.model.channel_emb
            torch.testing.assert_close(emb(torch.tensor([0,1])),torch.eye(emb.channel_embedding.embedding_dim)[:2])
            allowed=[[d*8+i for i in range(8)] for d in range(4)]
            m.config.audio_depth_token_ids=allowed
            x=torch.randint(0,32,(1,16))
            generated=m.generate(x,max_new_tokens=16,do_sample=True,top_k=3,top_p=0.9)[:,16:]
            for i,token in enumerate(generated[0].tolist()):self.assertIn(token,allowed[i%4])

    def test_streaming_matches_reference_and_chunk_sizes(self):
        from ntpp.inference.streaming import NTPPStreamingGenerator
        class Tokenizer:
            unk_token_id=-1
            def convert_tokens_to_ids(self, token):
                q,code=token.removeprefix('<moss_q').removesuffix('>').split('_')
                return int(q)*8+int(code)
        codec=SimpleNamespace(num_codebooks=4,codebook_size=8)
        prompt=torch.randint(0,8,(2,4,2)); user=torch.randint(0,8,(4,3))
        for family in ('llama','phi'):
            m=tiny(family).eval()
            g=NTPPStreamingGenerator(m,Tokenizer(),codec,chunk_frames=1,temperature=0,device='cpu')
            a=torch.cat(list(g.stream(user,prompt)),dim=1)
            g.chunk_frames=2
            b=torch.cat(list(g.stream(user,prompt)),dim=1)
            self.assertTrue(torch.equal(a,b))
            prefix=torch.cat([g.codes_to_ids(prompt[0]),g.codes_to_ids(prompt[1])],dim=0).T.reshape(1,-1)
            reference=[]
            with torch.no_grad():
                for frame in range(user.shape[1]):
                    first=m(prefix,use_cache=False).logits[0,-1]
                    prefix=torch.cat([prefix,g.codes_to_ids(user[:,frame:frame+1]).T],dim=1)
                    ids=[]
                    for d in range(4):
                        logits=first if d==0 else m(prefix,use_cache=False).logits[0,-1]
                        token=g._sample_depth(logits,d);ids.append(token)
                        prefix=torch.cat([prefix,token.reshape(1,1)],dim=1)
                    reference.append(torch.stack(ids))
            self.assertTrue(torch.equal(a,g.ids_to_codes(torch.stack(reference,dim=1))))

if __name__ == '__main__':unittest.main()
