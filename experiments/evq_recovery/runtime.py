"""Actual model path: static FP32 RoPE, attention+FFN LoRA, chunked dense CE."""
from __future__ import annotations

import math
from pathlib import Path
import torch

from scripts.experiments.cross_audit.training import causal_loss
from scripts.experiments.cross_audit.tables import install_static, verify_static

MODULES = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')


def configure(model, regime='lora', rank=32):
    if regime=='full':
        for parameter in model.parameters():parameter.requires_grad_(True)
        return model.float(), None
    if regime!='lora':raise ValueError('unknown update regime')
    from peft import LoraConfig, get_peft_model
    wrapper=get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0.,
        target_modules=list(MODULES),bias='none',task_type='CAUSAL_LM'))
    base=wrapper.get_base_model()
    names=[name for name,p in base.named_parameters() if p.requires_grad]
    if not names or any('lora_' not in name for name in names):raise ValueError('unplanned trainable parameters')
    for module in MODULES:
        if not any(f'.{module}.' in name for name in names):raise ValueError(f'missing adaptation module {module}')
    return base,wrapper


def tensors(row,device):
    ids=torch.tensor([row['input_ids']],dtype=torch.long,device=device)
    labels=ids.clone()
    labels[:,:row['target_start']]=-100
    return ids,labels


def training_step(model,optimizer,cpt,sft,native,*,sft_weight=1.,native_weight=.25,chunk_size=128,amp=False):
    optimizer.zero_grad(set_to_none=True)
    device=next(model.parameters()).device
    records={}
    batches=[('cpt',torch.tensor([cpt.tolist()],dtype=torch.long,device=device),None,1.),
             ('sft',*tensors(sft,device),sft_weight),
             ('native',*tensors(native,device),native_weight)]
    for label,ids,targets,weight in batches:
        if targets is None:targets=ids
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
            loss,count=causal_loss(model,ids,targets,chunk_size=chunk_size)
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite {label} loss')
        (weight*loss).backward()
        records[label+'_ce']=float(loss.detach())
        records[label+'_prediction_tokens']=count
        records[label+'_input_tokens']=ids.shape[1]-1
    named=list(model.named_parameters())
    gradients={module:sum(float(p.grad.detach().float().square().sum()) for name,p in named
                         if p.grad is not None and f'.{module}.' in name)**.5 for module in MODULES}
    if any(not math.isfinite(v) for v in gradients.values()):raise RuntimeError('nonfinite module gradient')
    records['module_grad_norms']=gradients
    norm=torch.nn.utils.clip_grad_norm_([p for _,p in named if p.requires_grad],1.,error_if_nonfinite=True)
    records['unclipped_gradient_norm']=float(norm)
    optimizer.step()
    return records


def load(plan,root,arm,checkpoint=None,training=False,regime='lora'):
    import json
    import numpy as np
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; preparation does not load full model weights')
    table=json.loads((Path(root)/'tables.json').read_text())[arm]
    model_path=plan['model_path']
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
        dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    wrapper=None
    if checkpoint:
        manifest=json.loads((Path(checkpoint)/'state.json').read_text())
        if manifest['arm']!=arm or manifest['regime']!=regime:raise ValueError('checkpoint arm/regime mismatch')
        from .acquire import file_hash
        contract_hash=file_hash(Path(root)/'plan.json')+file_hash(Path(root)/'tables.json')+file_hash(Path(root)/'data/data_manifest.json')
        if manifest['contract_hash']!=contract_hash:raise ValueError('checkpoint data/plan/table mismatch')
        if regime=='lora':
            from peft import PeftModel
            wrapper=PeftModel.from_pretrained(model,checkpoint,is_trainable=training)
            model=wrapper.get_base_model()
        else:
            del model
            torch.cuda.empty_cache()
            model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,
                dtype=torch.float32 if training else torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
    elif training:
        torch.manual_seed(plan['seed'])
        torch.cuda.manual_seed_all(plan['seed'])
        model,wrapper=configure(model,regime,plan['rank'])
    install_static(model,np.asarray(table['values'],dtype=np.float32),table['gain'])
    if training:
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model.train()
    else:model.eval()
    return model,wrapper,tokenizer,table
