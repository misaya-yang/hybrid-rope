"""Matched full/LoRA mechanics; all loss terms expose their prediction counts.

Loss helpers operate on the backbone plus selected/chunked LM-head positions,
so a full 16K x vocabulary activation is never retained. CPU tests compare both
values and gradients against dense shifted CE and forward KL.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def causal_loss(model, ids, labels, *, chunk_size=128):
    """labels align with input tokens; hidden[i] predicts labels[i+1]."""
    h=model.model(input_ids=ids[:,:-1],use_cache=False).last_hidden_state
    target=labels[:,1:]
    mask=target!=-100
    n=int(mask.sum())
    if not n:raise ValueError('no supervised next-token positions')
    selected=h[mask];targets=target[mask]
    loss=h.sum()*0
    def chunk_ce(hidden,weight,gold):
        return F.cross_entropy(F.linear(hidden,weight).float(),gold,reduction='sum')
    for begin in range(0,n,chunk_size):
        args=(selected[begin:begin+chunk_size],model.lm_head.weight,targets[begin:begin+chunk_size])
        loss=loss+checkpoint(chunk_ce,*args,use_reentrant=False)/n
    return loss,n


def native_kl(model,ids,positions,teacher_probs):
    """positions are zero-based hidden positions predicting input[position+1]."""
    if positions.numel()==0 or positions.min()<0 or positions.max()>=ids.shape[1]:
        raise ValueError('invalid Native prediction positions')
    h=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,positions]
    logits=F.linear(h,model.lm_head.weight).float()
    p=teacher_probs.to(device=logits.device,dtype=torch.float32)
    if p.shape!=logits.shape or not torch.isfinite(p).all() or (p<0).any():
        raise ValueError('invalid teacher distribution')
    if not torch.allclose(p.sum(-1),torch.ones_like(p[:,0]),atol=1e-5,rtol=1e-5):
        raise ValueError('teacher probabilities not normalized; do not repair silently')
    kl=F.kl_div(logits.log_softmax(-1),p,reduction='none').sum(-1).mean()
    return kl,len(positions)


def configure_training(model,regime,rank=16):
    if regime=='full':
        for p in model.parameters():p.requires_grad_(True)
    elif regime=='lora':
        from peft import LoraConfig,get_peft_model
        wrapped=get_peft_model(model,LoraConfig(r=rank,lora_alpha=32,lora_dropout=0.,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            bias='none',task_type='CAUSAL_LM'))
        # Return underlying model with injected LoRA modules, plus wrapper for save.
        return wrapped.get_base_model(),wrapped
    else:raise ValueError('unknown training regime')
    return model,None


def update(model,optimizer,cpt,sft,replay,*,kl_weight=1.,cpt_weight=1.,sft_weight=1.):
    optimizer.zero_grad(set_to_none=True)
    loss_c,nc=causal_loss(model,cpt,cpt)
    (cpt_weight*loss_c).backward()
    sft_count=sum(int((labels[:,1:]!=-100).sum()) for _,labels in sft)
    if not sft_count:raise ValueError('empty supervised answer budget')
    ls=0.
    for ids,labels in sft:
        loss,n=causal_loss(model,ids,labels)
        (sft_weight*loss*n/sft_count).backward();ls+=float(loss.detach())*n/sft_count
    ids,positions,probs=replay
    loss_k,nk=native_kl(model,ids,positions,probs)
    (kl_weight*loss_k).backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not all(torch.isfinite(x) for x in (loss_c,loss_k,norm)) or not torch.isfinite(torch.tensor(ls)):
        raise RuntimeError('nonfinite training step')
    optimizer.step()
    return dict(cpt_ce=float(loss_c.detach()),sft_ce=ls,native_kl=float(loss_k.detach()),
        cpt_prediction_tokens=nc,answer_prediction_tokens=sft_count,native_prediction_positions=nk,
        cpt_input_tokens=cpt.numel()-cpt.shape[0],sft_input_tokens=sum(x.numel()-x.shape[0] for x,_ in sft),
        replay_input_tokens=ids.numel(),grad_norm=float(norm))
