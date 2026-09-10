"""Continuous relative-distance clocks via two sparse masks and one partition.

Uses PyTorch FlexAttention's fused block-sparse kernel, not a dense T x T mask.
Raw K is cached once. Rotations are reconstructed under the fixed window rule.
"""
import math

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention


_flex=None
_masks={}


def _pack(mask):
    rows,cols=mask.shape
    count=mask.sum(-1).to(torch.int32)
    indices=torch.arange(cols,device=mask.device)[None].expand(rows,-1)
    # BlockMask's transpose reconstructs the block-grid width from the final
    # indices dimension. Keep all block columns even when each row is sparse.
    indices=torch.where(mask,indices,cols).sort(-1).values
    indices=torch.where(indices==cols,0,indices).to(torch.int32)
    return count[None,None].contiguous(),indices[None,None].contiguous()


def masks(q_len,k_len,window,device):
    key=(q_len,k_len,window,str(device))
    if key in _masks:return _masks[key]
    block=128;offset=k_len-q_len
    qleft=torch.arange(0,q_len,block,device=device)+offset
    qright=(qleft+block-1).clamp_max(k_len-1)
    kleft=torch.arange(0,k_len,block,device=device)
    kright=(kleft+block-1).clamp_max(k_len-1)
    local_any=(kright[None]>=qleft[:,None]-window)&(kleft[None]<=qright[:,None])
    local_full=(kleft[None]>=qright[:,None]-window)&(kright[None]<=qleft[:,None])
    far_any=kleft[None]<qright[:,None]-window
    far_full=kright[None]<qleft[:,None]-window
    offset_tensor=torch.tensor(offset,device=device)
    def local_mod(b,h,q,k):
        delta=q+offset_tensor-k
        return (delta>=0)&(delta<=window)
    def far_mod(b,h,q,k):
        return q+offset_tensor-k>window
    results=[]
    for any_mask,full_mask,fn in ((local_any,local_full,local_mod),(far_any,far_full,far_mod)):
        partial=any_mask&~full_mask
        pn,pi=_pack(partial);fnn,fi=_pack(full_mask)
        results.append(BlockMask.from_kv_blocks(pn,pi,full_kv_num_blocks=fnn,full_kv_indices=fi,
            BLOCK_SIZE=block,mask_mod=fn,seq_lengths=(q_len,k_len)))
    if len(_masks)>300:_masks.clear()
    _masks[key]=tuple(results)
    return tuple(results)


def rotate(raw,positions,freq,gain,extra=None):
    phase=positions.float()[:,None]*freq[None]
    if extra is not None:phase=phase+extra[None]
    cos=(phase.cos()*gain).to(raw.dtype);sin=(phase.sin()*gain).to(raw.dtype)
    a,b=raw[...,:64],raw[...,64:]
    return torch.cat((a*cos-b*sin,b*cos+a*sin),-1)


def attention(q,k,v,window,native,mr,gain,scale):
    global _flex
    if _flex is None:_flex=torch.compile(flex_attention,dynamic=True)
    q_len,k_len=q.shape[-2],k.shape[-2]
    qp=torch.arange(k_len-q_len,k_len,device=q.device);kp=torch.arange(k_len,device=q.device)
    local_mask,far_mask=masks(q_len,k_len,window,q.device)
    ql=rotate(q,qp,native,gain);kl=rotate(k,kp,native,gain)
    qr=rotate(q,qp,mr,gain,window*(native-mr));kr=rotate(k,kp,mr,gain)
    # The decoding template packs GQA groups into its query tile: 97 queries
    # times 8 heads pads to 1024 rows and exceeds consumer-GPU shared memory.
    # Use the general template for multi-token queries and decoding for Q=1.
    options={'FORCE_USE_FLEX_ATTENTION':q_len>1}
    ol,ll=_flex(ql,kl,v,block_mask=local_mask,scale=scale,enable_gqa=True,return_lse=True,kernel_options=options)
    of,lf=_flex(qr,kr,v,block_mask=far_mask,scale=scale,enable_gqa=True,return_lse=True,kernel_options=options)
    # Empty far rows are a mathematically zero numerator, not NaN * zero.
    of=torch.where(torch.isfinite(lf)[...,None],of,0)
    denominator=torch.logaddexp(ll,lf)
    out=ol.float()*(ll-denominator).exp()[...,None]+of.float()*(lf-denominator).exp()[...,None]
    return out.to(v.dtype)


def forward_distance(module,hidden,positions,mask,cache,spec,tables,**kwargs):
    shape=(*hidden.shape[:-1],-1,128)
    if hidden.shape[0]!=1:raise ValueError('distance pilot currently requires unpadded batch one')
    q=module.q_proj(hidden).view(shape).transpose(1,2)
    k=module.k_proj(hidden).view(shape).transpose(1,2)
    v=module.v_proj(hidden).view(shape).transpose(1,2)
    if cache is not None:k,v=cache.update(k,v,module.layer_idx)
    native=torch.tensor(tables['Native']['values_float32'],device=q.device)
    mr=torch.tensor(spec['table']['values_float32'],device=q.device)
    if int(positions[0,0])!=k.shape[-2]-q.shape[-2] or int(positions[0,-1])!=k.shape[-2]-1:
        raise ValueError('distance pilot expects contiguous original position IDs')
    out=attention(q,k,v,spec['window'],native,mr,spec['table']['gain'],module.scaling)
    out=out.transpose(1,2).contiguous().reshape(*hidden.shape[:-1],-1)
    return module.o_proj(out),None
