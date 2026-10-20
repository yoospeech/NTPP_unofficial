import json, os, subprocess, sys, tempfile, unittest, importlib.util
from pathlib import Path
from tokenizers import Tokenizer, models, processors
from transformers import PreTrainedTokenizerFast
from ntpp.model import LlamaConfig,LlamaForCausalLM,PhiConfig,PhiForCausalLM

@unittest.skipUnless(importlib.util.find_spec("datasets"), "Install datasets for CLI integration tests")
class TrainingCLI(unittest.TestCase):
    def test_preprocessing_and_evaluation_without_training(self):
        with tempfile.TemporaryDirectory(prefix='ntpp-cli-') as directory:
            root=Path(directory)
            for family,C,M in [('llama',LlamaConfig,LlamaForCausalLM),('phi',PhiConfig,PhiForCausalLM)]:
                path=root/family
                tokens=['[PAD]','[UNK]','[BOS]','<sosp>','<eosp>']+[f'<{i}>' for i in range(32)]
                t=Tokenizer(models.WordLevel({s:i for i,s in enumerate(tokens)},unk_token='[UNK]'))
                t.post_processor=processors.TemplateProcessing(single='[BOS] $A',special_tokens=[('[BOS]',2)])
                tok=PreTrainedTokenizerFast(tokenizer_object=t,unk_token='[UNK]',pad_token='[PAD]',bos_token='[BOS]',model_max_length=32)
                tok.add_special_tokens({'additional_special_tokens':tokens[3:]})
                tok.save_pretrained(path)
                c=C(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,rvq_depth=4,ntpp_stage='pretrain')
                M(c).save_pretrained(path)
                data=root/f'{family}.jsonl'
                data.write_text('\n'.join(json.dumps({'text':''.join(f'<{j%32}>' for j in range(n))}) for n in [16,24,32]))
                for stage in ['pretrain','sft']:
                    result=subprocess.run([sys.executable,'-m',f'ntpp.train.{stage}','--model_name_or_path',str(path),'--train_file',str(data),'--validation_file',str(data),'--output_dir',str(root/f'{family}-{stage}-out'),'--do_eval','--use_cpu','True','--block_size','16','--per_device_eval_batch_size','2','--report_to','none','--overwrite_cache','True'],capture_output=True,text=True)
                    if result.returncode:
                        self.fail(result.stdout + result.stderr)
                    self.assertIn('eval_loss', result.stdout)
