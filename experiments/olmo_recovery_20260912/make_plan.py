#!/usr/bin/env python3
"""Create a locked three-arm plan from actual Llama-3 or OLMo-2 metadata."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from experiments.evq_recovery.tables import anchored_cosh
from scripts.lib.rope.schedules import evq_cosh_inv_freq
from scripts.experiments.cross_audit.tables import native_table,tensor_sha,transform

OLD_OLMO_EVQ_SHA='917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607'
PROFILES={'llama':dict(native=8192,layers=32,scale=2,contracts=('llama_midpoint_tau_sqrt2',)),'olmo2':dict(native=4096,layers=16,scale=4,contracts=('olmo_old_evq_exact','olmo_fixed_support_tau2'))}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cosh-contract',choices=['llama_midpoint_tau_sqrt2','olmo_old_evq_exact','olmo_fixed_support_tau2'],required=True);p.add_argument('--cosh-table',type=Path);a=p.parse_args()
 cfg=json.loads((a.model/'config.json').read_text());kind=cfg['model_type'];profile=PROFILES.get(kind);dim=cfg['hidden_size']//cfg['num_attention_heads'];base=cfg.get('rope_theta',cfg.get('rope_parameters',{}).get('rope_theta'))
 if profile is None or dim!=128 or base!=500000 or cfg['max_position_embeddings']!=profile['native'] or cfg['num_hidden_layers']!=profile['layers'] or a.cosh_contract not in profile['contracts']:raise ValueError('actual model config and explicit recovery contract disagree')
 native=native_table(dim,base).astype(np.float32)
 if kind=='llama':cosh=evq_cosh_inv_freq(dim,1.414,base,midpoint=True,dtype=torch.float64).float().numpy();cosh_meta={'method':'paper midpoint Cosh','tau':1.414,'identity_scope':'legacy Llama construction; endpoints differ from Native'}
 elif a.cosh_contract=='olmo_fixed_support_tau2':
  z=anchored_cosh(dim//2,2.);logs=-np.log(native.astype(np.float64));cosh=np.exp(-(logs[0]+(logs[-1]-logs[0])*z)).astype(np.float32);cosh[[0,-1]]=native[[0,-1]];cosh_meta={'method':'fixed-native-support anchored Cosh','tau':2.,'identity_scope':'new OLMo matched recovery contract; not old Llama replay'}
 else:
  if a.cosh_table is None or not a.cosh_table.is_file():raise ValueError('old OLMo EVQ table is missing; recover or reconstruct it and verify the registered tensor SHA')
  cosh=np.load(a.cosh_table,allow_pickle=False).astype(np.float32)
  if tensor_sha(cosh)!=OLD_OLMO_EVQ_SHA:raise ValueError('old OLMo EVQ table SHA mismatch')
  cosh_meta={'method':'old OLMo EVQ tensor with registered byte identity','tensor_sha256':OLD_OLMO_EVQ_SHA,'identity_scope':'tensor SHA matches failure lineage; this alone does not establish original-file recovery'}
 yarn,gain,meta=transform(native,dim=dim,base=base,reference_length=profile['native'],scale=profile['scale'],method='yarn')
 tables={'Native':dict(values_float32=native.tolist(),tensor_sha256=tensor_sha(native),gain=1.,construction={'method':'identity'}),'Cosh':dict(values_float32=cosh.tolist(),tensor_sha256=tensor_sha(cosh),gain=1.,construction=cosh_meta),'OfficialYaRN':dict(values_float32=yarn.tolist(),tensor_sha256=tensor_sha(yarn),gain=gain,construction=meta)}
 index=a.model/'model.safetensors.index.json'
 expected_weights=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.is_file() else ['model.safetensors']
 weights=[a.model/name for name in expected_weights if (a.model/name).is_file()];weights_present=len(weights)==len(expected_weights)
 if weights_present:
  for path in weights:
   with safe_open(path,framework='pt',device='cpu') as handle:
    if not list(handle.keys()):raise ValueError('empty safetensors shard: '+path.name)
 status='CPU_PLAN_GPU_NOT_RUN' if weights_present else 'MODEL_MISSING_CPU_PLAN_ONLY';root=Path(__file__).resolve().parents[2]
 metadata=[x for x in a.model.iterdir() if x.is_file() and x.suffix in ('.json','.txt')]
 code=[Path(__file__),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('train.py'),Path(__file__).with_name('probe.py')]
 plan=dict(status=status,model_path=str(a.model.resolve()),model_weights_present=weights_present,model_identity={'expected_weight_shards':expected_weights,'metadata':{x.name:sha(x) for x in metadata},'weights':{x.name:sha(x) for x in weights} if weights_present else 'PENDING_HASH_AT_ACQUISITION'},code_identity={str(x.relative_to(root)):sha(x) for x in code},model_type=kind,cosh_contract=a.cosh_contract,native_length=profile['native'],train_length=16384,gradient_accumulation=1,effective_cpt_tokens_per_update=16384,arms=list(tables),rank=32,alpha=32,dropout=0.,modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],seed=20260912,learning_rate=2e-5,betas=[.9,.95],weight_decay=0.,max_grad_norm=1.,schedule_cpt_tokens=134217728,checkpoint_cpt_tokens=[8388608,33554432,67108864,134217728],resume_every_updates=64,first_endpoint_cpt_tokens=33554432,loss_weights={'cpt':1.,'sft':1.,'replay':.25,'kl':.25},loss_note='Each CE/KL term independently mean-normalized; teacher KL coefficient 0.25 is locked identically for all arms.',data_manifest=str(a.data_manifest.resolve()),data_manifest_sha256=sha(a.data_manifest),hardware='RTX 5090 sm120; BF16 base, Flash SDPA, activation checkpointing, chunked lm_head CE; no 4-bit')
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'tables.json').write_text(json.dumps(tables,indent=2)+'\n');(a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
