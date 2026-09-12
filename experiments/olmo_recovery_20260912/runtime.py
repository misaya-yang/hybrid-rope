"""Shared Llama-3 recovery mechanics: all-linear LoRA and normalized losses."""
from __future__ import annotations
import contextlib
import hashlib
import math
import random
import numpy as np
import torch
from scripts.experiments.cross_audit.tables import install_static, verify_static
from scripts.experiments.cross_audit.training import causal_loss, native_kl

MODULES=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')

def seed_all(seed):
 random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed))
 if torch.cuda.is_available():torch.cuda.manual_seed_all(int(seed))

def trainable_sha(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().float().contiguous().numpy().tobytes())
 return h.hexdigest()

def validate_cuda():
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required')
 name=torch.cuda.get_device_name();cap=torch.cuda.get_device_capability();arch=torch.cuda.get_arch_list()
 if cap!=(12,0) or '5090' not in name.upper() or not any(x.startswith('sm_120') for x in arch):raise RuntimeError(f'requires RTX5090 sm120 and compatible cubin: {name} {cap} {arch}')
 torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
 return {'gpu':name,'capability':list(cap),'architectures':arch,'torch':torch.__version__,'cuda':torch.version.cuda}

def configure(model,rank=32):
    from peft import LoraConfig,get_peft_model
    wrapper=get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0.,
        target_modules=list(MODULES),bias='none',task_type='CAUSAL_LM'))
    base=wrapper.get_base_model(); names=[n for n,p in base.named_parameters() if p.requires_grad]
    if not names or any('lora_' not in n for n in names):raise ValueError('unexpected trainable parameters')
    for module in MODULES:
        if not any(f'.{module}.' in n for n in names):raise ValueError('missing '+module)
    return base,wrapper

def supervised(row,device):
    ids=torch.tensor([row['input_ids']],device=device);labels=ids.clone();labels[:,:row['target_start']]=-100
    return ids,labels

def frozen_native_teacher(model,wrapper,replay,native_table,cache=None):
    """Build Native/no-adapter teacher targets without retaining teacher logits.

    Instruction replay follows the frozen base model's own complete greedy
    trajectory.  Text replay stays on the original contiguous token stream.
    ``positions[p]`` always denotes the hidden/logit row predicting
    ``kl_ids[p+1]``.
    """
    device=next(model.parameters()).device
    if cache is None:
        cache=getattr(model,'_frozen_native_teacher_cache',None)
        if cache is None:
            cache={};setattr(model,'_frozen_native_teacher_cache',cache)
    is_text=replay.get('task')=='text'
    prompt=list(replay['input_ids'] if is_text else replay.get('prompt_ids',[]))
    if len(prompt)<2:
        raise ValueError('Native teacher needs at least two real prompt/text tokens')
    budget=0 if is_text else int(replay.get('generation_budget',0))
    if not is_text and budget<=0:
        raise ValueError('instruction replay needs a positive frozen generation budget')
    prompt_bytes=np.asarray(prompt,dtype='<i8').tobytes()
    key=(str(replay.get('id',replay.get('source_id',''))),hashlib.sha256(prompt_bytes).hexdigest(),budget,is_text)
    current=model.model.rotary_emb.inv_freq.detach().cpu().numpy().copy();gain=float(model.model.rotary_emb.attention_scaling)
    was_training=model.training
    try:
        model.eval();install_static(model,native_table,1.)
        with torch.no_grad(), wrapper.disable_adapter():
            cache_hit=key in cache
            if cache_hit:
                teacher_ids=list(cache[key])
            elif is_text:
                teacher_ids=prompt
                cache[key]=list(teacher_ids)
            else:
                prompt_tensor=torch.tensor([prompt],dtype=torch.long,device=device)
                generated=model.generate(input_ids=prompt_tensor,attention_mask=torch.ones_like(prompt_tensor),
                    do_sample=False,num_beams=1,max_new_tokens=budget,use_cache=True)[0].tolist()
                if generated[:len(prompt)]!=prompt:
                    raise RuntimeError('teacher generation did not preserve the prompt prefix')
                teacher_ids=generated;cache[key]=list(teacher_ids)
            kl_ids=torch.tensor([teacher_ids],dtype=torch.long,device=device)
            if is_text:
                positions=list(map(int,replay.get('kl_positions',[])))
                if not positions:
                    count=min(128,len(teacher_ids)-1)
                    positions=np.linspace(0,len(teacher_ids)-2,num=count,dtype=np.int64).tolist()
            else:
                answer_positions=list(range(len(prompt)-1,len(teacher_ids)-1))
                if not answer_positions:
                    raise RuntimeError('Native teacher generated no answer-side token')
                if len(answer_positions)>128:
                    selected=np.linspace(0,len(answer_positions)-1,num=128,dtype=np.int64)
                    positions=[answer_positions[int(i)] for i in selected]
                else:
                    positions=answer_positions
            positions=torch.as_tensor(positions,dtype=torch.long,device=device)
            if positions.numel()==0 or positions.min()<0 or positions.max()>=kl_ids.shape[1]-1:
                raise ValueError('teacher KL positions must predict existing next tokens')
            hidden=model.model(input_ids=kl_ids,use_cache=False).last_hidden_state[0,positions]
            probs=torch.nn.functional.linear(hidden,model.lm_head.weight).float().softmax(-1).detach()
            eos=model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            new_tokens=teacher_ids[len(prompt):] if not is_text else []
            metadata=dict(kind='contiguous_text' if is_text else 'greedy_instruction_trajectory',
                prompt_tokens=len(prompt),teacher_tokens=len(teacher_ids),generated_tokens=len(new_tokens),
                terminated_eos=bool(new_tokens and new_tokens[-1] in eos),
                hit_cap=bool(not is_text and len(new_tokens)==budget and (not new_tokens or new_tokens[-1] not in eos)),
                kl_positions=len(positions),cache_hit=cache_hit)
    finally:
        install_static(model,current,gain);model.train(was_training)
    return kl_ids,positions,probs,metadata

def step(model,wrapper,optimizer,cpt,sft,replay,*,native_table,weights,chunk_size=128,amp=True,
         zero_grad=True,do_step=True,gradient_scale=1.,teacher_cache=None):
    if zero_grad:optimizer.zero_grad(set_to_none=True)
    device=next(model.parameters()).device;out={};terms=[]
    cpt=torch.as_tensor(cpt,dtype=torch.long,device=device).unsqueeze(0)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):loss,n=causal_loss(model,cpt,cpt,chunk_size=chunk_size)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite cpt loss')
    objective=weights['cpt']*float(loss.detach());(loss*weights['cpt']*gradient_scale).backward();out.update(cpt_ce=float(loss.detach()),cpt_prediction_tokens=n);del loss,cpt
    ids,labels=supervised(sft,device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):loss,n=causal_loss(model,ids,labels,chunk_size=chunk_size)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite sft loss')
    objective+=weights['sft']*float(loss.detach());(loss*weights['sft']*gradient_scale).backward();out.update(sft_ce=float(loss.detach()),sft_prediction_tokens=n);del loss,labels
    ids,labels=supervised(replay,device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):loss,n=causal_loss(model,ids,labels,chunk_size=chunk_size)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite replay loss')
    objective+=weights['replay']*float(loss.detach());(loss*weights['replay']*gradient_scale).backward();out.update(replay_ce=float(loss.detach()),replay_prediction_tokens=n);del loss,labels
    if weights.get('kl',0)>0:
        kl_ids,positions,probs,teacher=frozen_native_teacher(model,wrapper,replay,native_table,teacher_cache)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):kl,nk=native_kl(model,kl_ids,positions,probs)
        if not torch.isfinite(kl):raise RuntimeError('nonfinite native KL')
        objective+=weights['kl']*float(kl.detach());(kl*weights['kl']*gradient_scale).backward();out.update(native_teacher_kl=float(kl.detach()),native_kl_positions=nk,native_teacher=teacher);del kl,probs,kl_ids,positions
    named=list(model.named_parameters());grads={m:sum(float(p.grad.float().square().sum()) for n,p in named if p.grad is not None and f'.{m}.' in n)**.5 for m in MODULES}
    if any(not math.isfinite(x) for x in grads.values()):raise RuntimeError('nonfinite module gradient')
    norm=torch.nn.utils.clip_grad_norm_([p for _,p in named if p.requires_grad],1.,error_if_nonfinite=True) if do_step else torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for _,p in named if p.requires_grad and p.grad is not None]))
    if do_step:optimizer.step()
    out.update(module_grad_norms=grads,gradient_norm=float(norm),weighted_objective=objective,optimizer_stepped=do_step)
    return out

def load_model(plan,table,training=True,checkpoint=None):
    from transformers import AutoModelForCausalLM
    seed_all(plan['seed'])
    model=AutoModelForCausalLM.from_pretrained(plan['model_path'],local_files_only=True,dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='sdpa')
    if checkpoint:
        from peft import PeftModel
        wrapper=PeftModel.from_pretrained(model,checkpoint,is_trainable=training);model=wrapper.get_base_model()
    else:model,wrapper=configure(model,plan['rank'])
    install_static(model,np.asarray(table['values_float32'],dtype=np.float32),table['gain']);verify_static(model,np.asarray(table['values_float32'],dtype=np.float32),table['gain'])
    if training:
        model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
    return model,wrapper
