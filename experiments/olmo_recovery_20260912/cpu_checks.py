"""CPU contract tests for both supported architectures and exact PEFT resume."""
import importlib.util,tempfile
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Olmo2Config,Olmo2ForCausalLM
from .runtime import configure,step,trainable_sha
from .train import order,rng_state,restore_rng,save_bundle
from scripts.experiments.cross_audit.tables import install_static,tensor_sha

def tiny(kind):
 common=dict(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32,rope_theta=500000,pad_token_id=0,eos_token_id=2)
 return LlamaForCausalLM(LlamaConfig(**common)) if kind=='llama' else Olmo2ForCausalLM(Olmo2Config(**common))
def batch():return np.arange(3,20),{'input_ids':[3,4,5,6,7,2],'target_start':3,'kl_positions':[0,1], 'prompt_ids':[3,4,5], 'generation_budget':3, 'task':'instruction'}
def exercise(kind):
 torch.manual_seed(7);model=tiny(kind);native=model.model.rotary_emb.inv_freq.detach().numpy().astype(np.float32);has_peft=bool(importlib.util.find_spec('peft'))
 wrapper=None
 if has_peft:model,wrapper=configure(model,4);student=(native*.9).astype(np.float32);install_static(model,student,1.07);model.train()
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,betas=(.9,.95),weight_decay=0.);cpt,row=batch();weights={'cpt':1.,'sft':1.,'replay':.25,'kl':.25 if wrapper else 0.}
 record=step(model,wrapper,opt,cpt,row,row,native_table=native,weights=weights,chunk_size=3,amp=False)
 assert record['cpt_prediction_tokens']==16 and record['sft_prediction_tokens']==3
 assert all(record['module_grad_norms'][m]>0 for m in ('q_proj','v_proj','gate_proj','up_proj','down_proj'))
 if wrapper:
  assert record['native_teacher_kl']>=0 and tensor_sha(model.model.rotary_emb.inv_freq.detach().numpy())==tensor_sha(student)
  assert float(model.model.rotary_emb.attention_scaling)==1.07 and model.training
 return record,has_peft
def exact_peft_resume(kind):
 if not importlib.util.find_spec('peft'):return 'SKIP_PEFT_UNAVAILABLE'
 cpt,row=batch();weights={'cpt':1.,'sft':1.,'replay':.25,'kl':.25}
 torch.manual_seed(31);base=tiny(kind);native=base.model.rotary_emb.inv_freq.detach().numpy().astype(np.float32);model,wrapper=configure(base,4);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
 step(model,wrapper,opt,cpt,row,row,native_table=native,weights=weights,chunk_size=3,amp=False)
 with tempfile.TemporaryDirectory() as td:
  path=Path(td)/'resume';save_bundle(path,wrapper,opt,{'arm':'test','micro_sequences':1,'cpt_tokens':16,'contract_hash':'x','gradient_accumulation':1})
  step(model,wrapper,opt,cpt+1,row,row,native_table=native,weights=weights,chunk_size=3,amp=False);expected=trainable_sha(model)
  torch.manual_seed(31);fresh=tiny(kind)
  from peft import PeftModel
  restored_wrapper=PeftModel.from_pretrained(fresh,path,is_trainable=True);restored=restored_wrapper.get_base_model();restored_opt=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-3);saved=torch.load(path/'training.pt',map_location='cpu',weights_only=False);restored_opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
  step(restored,restored_wrapper,restored_opt,cpt+1,row,row,native_table=native,weights=weights,chunk_size=3,amp=False)
  assert trainable_sha(restored)==expected
 return 'PASS'
def run():
 result={}
 for kind in ('llama','olmo2'):
  record,peft=exercise(kind);result[kind]={'graph':'PASS','peft':peft,'resume':exact_peft_resume(kind),'objective':record['weighted_objective']}
 assert [order(i,7,19) for i in range(7)]!=[order(i,7,20) for i in range(7)]
 assert [order(i,7,19) for i in range(3,14)]==[order(i,7,19) for i in range(14)][3:]
 return result
if __name__=='__main__':print(run())
