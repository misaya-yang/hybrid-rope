"""Explicit Qwen2 layer/group allocations and a joint-normalized dual-frequency kernel."""
from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward, apply_rotary_pos_emb


def embeddings(hidden, positions, table):
    freq = torch.tensor(table['values_float32'], device=hidden.device, dtype=torch.float32)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        phase = (freq[None,:,None].expand(positions.shape[0],-1,1) @ positions[:,None,:].float()).transpose(1,2)
        phase = torch.cat((phase,phase),-1)
        cos=phase.cos()*table['gain'];sin=phase.sin()*table['gain']
    return cos.to(hidden.dtype),sin.to(hidden.dtype)


def install(model, spec, tables):
    kind=spec['operator']
    if kind not in ('layer','group','dual_frequency','distance'):
        raise ValueError('unknown explicit operator: '+kind)
    context={}
    def position_hook(module,args,kwargs):
        context['positions']=args[1] if len(args)>1 else kwargs['position_ids']
    handle=model.model.rotary_emb.register_forward_pre_hook(position_hook,with_kwargs=True)
    originals=[]

    def make_forward(original,layer_idx):
        def forward(module,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
            positions=context['positions']
            if kind=='layer':
                return original(hidden_states,embeddings(hidden_states,positions,spec['replacement']),attention_mask,
                    past_key_values=past_key_values,**kwargs)
            if kind=='distance':
                from .distance_operator import forward_distance
                return forward_distance(module,hidden_states,positions,attention_mask,past_key_values,spec,tables,**kwargs)
            input_shape=hidden_states.shape[:-1]
            shape=(*input_shape,-1,module.head_dim)
            q=module.q_proj(hidden_states).view(shape).transpose(1,2)
            k=module.k_proj(hidden_states).view(shape).transpose(1,2)
            v=module.v_proj(hidden_states).view(shape).transpose(1,2)
            cos,sin=position_embeddings
            qr,kr=apply_rotary_pos_emb(q,k,cos,sin)
            if kind=='group':
                rc,rs=embeddings(hidden_states,positions,spec['replacement'])
                rq,rk=apply_rotary_pos_emb(q,k,rc,rs)
                group=spec['group'];group_size=qr.shape[1]//kr.shape[1]
                qr[:,group*group_size:(group+1)*group_size]=rq[:,group*group_size:(group+1)*group_size]
                kr[:,group:group+1]=rk[:,group:group+1]
            else:
                rc,rs=embeddings(hidden_states,positions,spec.get('second_table',tables['MrProBM']))
                rq,rk=apply_rotary_pos_emb(q,k,rc,rs)
                indices=torch.tensor(list(range(24,40))+list(range(88,104)),device=q.device)
                # Put the exact power-of-two weight on Q only. This is the same
                # bilinear half/half kernel without two BF16 sqrt(.5) roundings.
                qr[...,indices]*=.5
                qr=torch.cat((qr,rq[...,indices]*.5),-1)
                kr=torch.cat((kr,rk[...,indices]),-1)
            if past_key_values is not None:
                kr,v=past_key_values.update(kr,v,module.layer_idx)
            interface=ALL_ATTENTION_FUNCTIONS.get_interface(module.config._attn_implementation,eager_attention_forward)
            # Fused SDPA requires equal Q/K/V head dimensions. V padding is
            # temporary; the persistent cache stores its original 128 channels.
            value=F.pad(v,(0,qr.shape[-1]-v.shape[-1])) if qr.shape[-1]!=v.shape[-1] else v
            out,weights=interface(module,qr,kr,value,attention_mask,dropout=0.,scaling=module.scaling,
                sliding_window=module.sliding_window,**kwargs)
            out=out[...,:module.head_dim].reshape(*input_shape,-1).contiguous()
            return module.o_proj(out),weights
        return forward

    for layer_idx,layer in enumerate(model.model.layers):
        if kind in ('layer','group') and layer_idx!=spec['layer']:
            continue
        module=layer.self_attn
        originals.append((module,module.forward))
        module.forward=types.MethodType(make_forward(module.forward,layer_idx),module)
    def restore():
        for module,forward in originals:module.forward=forward
        handle.remove()
    return restore
