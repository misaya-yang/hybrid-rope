"""Shared Llama-3 recovery mechanics: all-linear LoRA and normalized losses."""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from scripts.experiments.cross_audit.tables import install_static
from scripts.experiments.cross_audit.training import causal_loss, native_kl

MODULES=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
BLACKWELL_CAUSAL_ATTENTION='blackwell_causal_flash'
BLACKWELL_CHUNKED_CAUSAL_ATTENTION='blackwell_chunked_causal_flash'
PHI3_SLIDING_FLEX_ATTENTION='phi3_sliding_flex_v1'
_phi3_flex=None

def set_static_in_place(model,values,gain):
    """Update the default RoPE buffers without changing their tensor identity."""
    rotary=model.model.rotary_emb
    values=np.asarray(values,dtype=np.float32)
    current=rotary.inv_freq
    if current.dtype!=torch.float32 or tuple(current.shape)!=tuple(values.shape):
        raise ValueError('in-place rotary table shape/dtype mismatch')
    replacement=torch.as_tensor(values,device=current.device,dtype=torch.float32)
    with torch.no_grad():
        current.copy_(replacement)
        original=getattr(rotary,'original_inv_freq',None)
        if original is not None:
            if original.dtype!=torch.float32 or original.shape!=current.shape:
                raise ValueError('original rotary buffer shape/dtype mismatch')
            original.copy_(replacement)
    rotary.attention_scaling=float(gain)
    return rotary

def blackwell_causal_flash(module,query,key,value,attention_mask,dropout=0.0,scaling=None,**kwargs):
    """Flash-only full-sequence causal attention for unpadded MHA or GQA rows."""
    if attention_mask is not None:
        raise RuntimeError('blackwell causal training path requires an unpadded full sequence')
    if query.shape[-2]!=key.shape[-2] or key.shape[-2]!=value.shape[-2]:
        raise RuntimeError('blackwell causal training path forbids cached or unequal sequence lengths')
    if key.shape!=value.shape or query.shape[0]!=key.shape[0] or query.shape[-1]!=key.shape[-1]:
        raise RuntimeError('blackwell causal training path received incompatible Q/K/V shapes')
    groups=int(getattr(module,'num_key_value_groups',query.shape[1]//key.shape[1]))
    if groups<1 or query.shape[1]!=key.shape[1]*groups:
        raise RuntimeError('blackwell causal training path received an invalid GQA head layout')
    output=torch.nn.functional.scaled_dot_product_attention(
        query,key,value,attn_mask=None,dropout_p=dropout,scale=scaling,is_causal=True,
        enable_gqa=groups>1)
    return output.transpose(1,2).contiguous(),None

def register_blackwell_attention(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(BLACKWELL_CAUSAL_ATTENTION,blackwell_causal_flash)
    model.config._attn_implementation=BLACKWELL_CAUSAL_ATTENTION

def blackwell_chunked_causal_flash(module,query,key,value,attention_mask,dropout=0.0,scaling=None,**kwargs):
    """Flash causal attention aligned to the lower-right of an accumulated KV cache."""
    if key.shape!=value.shape or query.shape[0]!=key.shape[0] or query.shape[-1]!=key.shape[-1]:
        raise RuntimeError('blackwell chunked path received incompatible Q/K/V shapes')
    if query.shape[0]!=1:
        raise RuntimeError('blackwell chunked path is restricted to one unpadded sequence')
    if query.shape[-2]>key.shape[-2]:
        raise RuntimeError('blackwell chunked path received more query than key positions')
    groups=int(getattr(module,'num_key_value_groups',query.shape[1]//key.shape[1]))
    if groups<1 or query.shape[1]!=key.shape[1]*groups:
        raise RuntimeError('blackwell chunked path received an invalid GQA head layout')
    from torch.nn.attention.bias import causal_lower_right
    bias=causal_lower_right(query.shape[-2],key.shape[-2])
    output=torch.nn.functional.scaled_dot_product_attention(
        query,key,value,attn_mask=bias,dropout_p=dropout,scale=scaling,is_causal=False,
        enable_gqa=groups>1)
    return output.transpose(1,2).contiguous(),None

def register_blackwell_chunked_attention(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(BLACKWELL_CHUNKED_CAUSAL_ATTENTION,blackwell_chunked_causal_flash)
    model.config._attn_implementation=BLACKWELL_CHUNKED_CAUSAL_ATTENTION

def phi3_sliding_visibility(q_len,key_len,window,device=None):
    """Exact Phi-3 causal window: key > query-window, lower-right aligned."""
    if q_len<=0 or key_len<q_len or window<=0:raise ValueError('invalid Phi sliding geometry')
    offset=key_len-q_len
    q=torch.arange(offset,key_len,device=device)[:,None]
    k=torch.arange(key_len,device=device)[None,:]
    return (k<=q)&(k>q-window)

def phi3_pad_attention_head(query,key,value):
    """Zero-pad non-power-of-two Phi heads for the FlexAttention Triton kernel."""
    head_dim=query.shape[-1]
    if key.shape[-1]!=head_dim or value.shape[-1]!=head_dim:
        raise RuntimeError('Phi sliding FlexAttention received incompatible Q/K/V')
    padded_dim=1<<(head_dim-1).bit_length()
    if padded_dim==head_dim:return query,key,value,head_dim
    pad=(0,padded_dim-head_dim)
    return (
        torch.nn.functional.pad(query,pad),
        torch.nn.functional.pad(key,pad),
        torch.nn.functional.pad(value,pad),
        head_dim,
    )

def phi3_sliding_flex(module,query,key,value,attention_mask,dropout=0.0,scaling=None,
                      sliding_window=None,**kwargs):
    """Preserve Phi-3's configured local causal operator with sparse FlexAttention."""
    del attention_mask,kwargs
    global _phi3_flex
    if query.shape[0]!=1 or key.shape[0]!=1 or value.shape[0]!=1:
        raise RuntimeError('Phi sliding FlexAttention requires unpadded batch one')
    if dropout:
        raise RuntimeError('Phi sliding FlexAttention is frozen-inference only')
    window=int(getattr(module,'_phi3_sliding_window',0) or sliding_window or 0)
    if window<=0:raise RuntimeError('Phi sliding window is missing')
    if key.shape!=value.shape or query.shape[-1]!=key.shape[-1]:
        raise RuntimeError('Phi sliding FlexAttention received incompatible Q/K/V')
    groups=query.shape[1]//key.shape[1]
    if groups<1 or query.shape[1]!=key.shape[1]*groups:
        raise RuntimeError('Phi sliding FlexAttention received invalid head mapping')
    from torch.nn.attention.flex_attention import flex_attention
    from experiments.nongeometric_screen.distance_operator import masks
    if _phi3_flex is None:_phi3_flex=torch.compile(flex_attention,dynamic=True)
    q_len,key_len=query.shape[-2],key.shape[-2]
    query,key,value,output_dim=phi3_pad_attention_head(query,key,value)
    local_mask,_=masks(q_len,key_len,window-1,query.device)
    output=_phi3_flex(
        query,key,value,block_mask=local_mask,scale=scaling,enable_gqa=groups>1,
        kernel_options={'FORCE_USE_FLEX_ATTENTION':q_len>1},
    )
    output=output[...,:output_dim]
    return output.transpose(1,2).contiguous(),None

def register_phi3_sliding_flex_attention(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if getattr(model.config,'model_type',None)!='phi3':
        raise ValueError('Phi sliding FlexAttention requires a Phi-3 model')
    window=int(getattr(model.config,'sliding_window',0) or 0)
    if window<=0:raise ValueError('Phi-3 checkpoint lacks a sliding window')
    for layer in model.model.layers:
        layer.self_attn._phi3_sliding_window=window
    ALL_ATTENTION_FUNCTIONS.register(PHI3_SLIDING_FLEX_ATTENTION,phi3_sliding_flex)
    model.config._attn_implementation=PHI3_SLIDING_FLEX_ATTENTION

def seed_all(seed):
 random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed))
 if torch.cuda.is_available():torch.cuda.manual_seed_all(int(seed))

def validate_cuda():
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required')
 name=torch.cuda.get_device_name();cap=torch.cuda.get_device_capability();arch=torch.cuda.get_arch_list()
 if cap<(8,0):raise RuntimeError(f'requires a BF16 CUDA GPU with Flash SDPA support: {name} {cap} {arch}')
 torch.set_float32_matmul_precision('high')
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
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
    disk_path=getattr(model,'_frozen_native_teacher_cache_path',None)
    if disk_path and not getattr(model,'_frozen_native_teacher_disk_loaded',False):
        disk_path=Path(disk_path)
        if disk_path.is_file():
            with disk_path.open() as stream:
                for line in stream:
                    if not line.strip():continue
                    row=json.loads(line);cache[tuple(row['key'])]=list(row['teacher_ids'])
        model._frozen_native_teacher_disk_loaded=True
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
    was_training=model.training;attention_implementation=model.config._attn_implementation
    setter=set_static_in_place if getattr(model,'_inplace_rope_table_updates',False) else install_static
    try:
        model.eval();model.config._attn_implementation='sdpa';setter(model,native_table,1.)
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
            if not cache_hit and disk_path:
                disk_path=Path(disk_path);disk_path.parent.mkdir(parents=True,exist_ok=True)
                with disk_path.open('a') as stream:
                    stream.write(json.dumps({'key':list(key),'teacher_ids':teacher_ids})+'\n')
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
        setter(model,current,gain);model.config._attn_implementation=attention_implementation;model.train(was_training)
    return kl_ids,positions,probs,metadata

def step(model,wrapper,optimizer,cpt,sft,replay,*,native_table,weights,chunk_size=128,amp=True,
         zero_grad=True,do_step=True,gradient_scale=1.,teacher_cache=None):
    if zero_grad:optimizer.zero_grad(set_to_none=True)
    device=next(model.parameters()).device;out={};terms=[]
    persistent_checkpointing=bool(getattr(model,'_activation_checkpointing',False))
    if persistent_checkpointing and not model.is_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    cpt=torch.as_tensor(cpt,dtype=torch.long,device=device).unsqueeze(0)
    backbone=model.__dict__.get('_compiled_cpt_backbone')
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):loss,n=causal_loss(model,cpt,cpt,chunk_size=chunk_size,backbone=backbone)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite cpt loss')
    objective=weights['cpt']*float(loss.detach());(loss*weights['cpt']*gradient_scale).backward();out.update(cpt_ce=float(loss.detach()),cpt_prediction_tokens=n);del loss,cpt
    variable_checkpointing=bool(not persistent_checkpointing and backbone is not None and model.__dict__.get('_variable_checkpointing',True))
    if variable_checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
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
    named=list(model.named_parameters());module_squares=[]
    for module in MODULES:
        parts=[p.grad.float().square().sum() for n,p in named if p.grad is not None and f'.{module}.' in n]
        module_squares.append(torch.stack(parts).sum() if parts else torch.zeros((),device=device))
    module_values=torch.sqrt(torch.stack(module_squares)).detach().cpu().tolist();grads=dict(zip(MODULES,map(float,module_values)))
    if any(not math.isfinite(x) for x in grads.values()):raise RuntimeError('nonfinite module gradient')
    norm=torch.nn.utils.clip_grad_norm_([p for _,p in named if p.requires_grad],1.,error_if_nonfinite=True) if do_step else torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for _,p in named if p.requires_grad and p.grad is not None]))
    if do_step:optimizer.step()
    if variable_checkpointing:model.gradient_checkpointing_disable()
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
    values=np.asarray(table['values_float32'],dtype=np.float32)
    install_static(model,values,table['gain'])
    rotary=model.model.rotary_emb
    if rotary.inv_freq.shape!=(len(values),) or not torch.isfinite(rotary.inv_freq).all() or float(rotary.attention_scaling)!=float(table['gain']):
        raise RuntimeError('runtime table shape/value/gain invalid')
    if training:
        model.config.use_cache=False
        checkpointing=os.environ.get('OLMO_ACTIVATION_CHECKPOINTING','0')=='1'
        if checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:model.gradient_checkpointing_disable()
        model.enable_input_require_grads();model.train()
        model._activation_checkpointing=checkpointing
        compile_mode=os.environ.get('OLMO_COMPILE_MODE','max-autotune-no-cudagraphs')
        if compile_mode.lower() not in ('0','off','none'):
            register_blackwell_attention(model)
            compiled=torch.compile(model.model,mode=compile_mode,dynamic=False,fullgraph=False)
            model.__dict__['_compiled_cpt_backbone']=compiled
            model.__dict__['_variable_checkpointing']=os.environ.get('OLMO_VARIABLE_CHECKPOINTING','1')=='1'
        model.__dict__['_compile_mode']=compile_mode
    return model,wrapper
