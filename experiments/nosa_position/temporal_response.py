"""E03 decode-time convex-response intervals with live exact refreshes.

Prefill follows B0 exactly. The first single-query step initializes the cache;
subsequent live single-query steps attempt reuse. No future Q prediction.
"""
import math

import torch

from .bounded_int8 import BoundedInt8Selector
from .exact_probe import ExactBlockSelector
from .runtime import select_with_scores


class TemporalResponseSelector(ExactBlockSelector):
    def __init__(self, mode='e03_temporal', **kwargs):
        super().__init__('exact_mass', **kwargs)
        self.responses = {}

    def _refresh(self, context, state, flags):
        h, length, d = context.k.shape
        size = context.settings.block_size
        a = context.q[:, 0].float().reshape(h,-1,d)/math.sqrt(d)
        for head in range(h):
            ids = torch.where(flags[head])[0]
            if ids.numel() == 0:continue
            positions = ids[:,None]*size+torch.arange(size,device=a.device)
            valid = positions < length
            positions = positions.clamp_max(length-1)
            keys = context.k[head,positions].float()
            bias = context.cis[head,positions].float().masked_fill(~valid,-torch.inf)
            logits = torch.einsum('gd,btd->gbt',a[head],keys)+bias[None]
            f = logits.logsumexp(-1)
            mean = torch.einsum('gbt,btd->gbd',logits.softmax(-1),keys)
            state['f'][head,:,ids] = f
            state['mean'][head,:,ids] = mean
            state['anchor'][head,:,ids] = a[head,:,None]
            state['minimum'][head,ids] = keys.masked_fill(~valid[...,None],torch.inf).amin(1)
            state['maximum'][head,ids] = keys.masked_fill(~valid[...,None],-torch.inf).amax(1)
            self.metrics['e03_refresh_blocks'] = self.metrics.get('e03_refresh_blocks',0)+ids.numel()
            self.metrics['e03_refresh_raw_k_elements'] = self.metrics.get('e03_refresh_raw_k_elements',0)+keys.numel()
            self.metrics['e03_gradient_elements'] = self.metrics.get('e03_gradient_elements',0)+mean.numel()

    def _bounds(self, context, state):
        h, _, d = context.k.shape
        a = context.q[:,0].float().reshape(h,-1,d)/math.sqrt(d)
        difference = a[:,:,None]-state['anchor']
        lower = state['f']+(difference*state['mean']).sum(-1)
        width = (difference.abs()*(state['maximum']-state['minimum'])[:,None]).sum(-1)
        radius = width.square()/16
        return (lower+radius)[:,:,None],radius[:,:,None]

    @torch.no_grad()
    def __call__(self, context):
        if context.q.shape[1]>1:
            self.responses.pop(context.layer_idx,None)
            self.metrics['e03_prefill_exact_queries'] = self.metrics.get('e03_prefill_exact_queries',0)+context.q.shape[1]
            return super().__call__(context)
        self.metrics['calls']+=1
        h,length,d=context.k.shape
        size=context.settings.block_size
        blocks=math.ceil(length/size)
        if blocks<=context.settings.topk:
            return select_with_scores(context,context.q.new_empty(0))
        group=context.q.shape[0]//h
        old=self.responses.get(context.layer_idx)
        done=old['f'].shape[-1] if old else 0
        if done>blocks:old=None;done=0
        if blocks>done:
            fresh={'f':torch.zeros((h,group,blocks-done),device=context.q.device),
                   'mean':torch.zeros((h,group,blocks-done,d),device=context.q.device),
                   'anchor':torch.zeros((h,group,blocks-done,d),device=context.q.device),
                   'minimum':torch.zeros((h,blocks-done,d),device=context.q.device),
                   'maximum':torch.zeros((h,blocks-done,d),device=context.q.device)}
            old={name:torch.cat((old[name],t),1 if name in ('minimum','maximum') else 2) for name,t in fresh.items()} if old else fresh
            self.responses[context.layer_idx]=old
        flags=torch.zeros((h,blocks),device=context.q.device,dtype=torch.bool)
        flags[:,done:]=True
        flags[:,-1]=True  # current block gains tokens and is never reused stale
        self._refresh(context,old,flags)
        center,radius=self._bounds(context,old)
        score,certified,ambiguous=BoundedInt8Selector._decision(self,context,center,radius)
        self.metrics['e03_decode_states']=self.metrics.get('e03_decode_states',0)+h
        self.metrics['e03_available_blocks']=self.metrics.get('e03_available_blocks',0)+h*blocks
        if not bool(certified.all()):
            refresh=(ambiguous & ~certified[...,None])[:,0] & ~flags
            self._refresh(context,old,refresh)
            center,radius=self._bounds(context,old)
            score,certified,_=BoundedInt8Selector._decision(self,context,center,radius)
        if not bool(certified.all()):
            self._refresh(context,old,(~certified[:,0,None]).expand(-1,blocks))
            center,_=self._bounds(context,old)
            score=center.softmax(-1).sum(1)
            self.metrics['e03_full_refresh_states']=self.metrics.get('e03_full_refresh_states',0)+int((~certified).sum())
        self.metrics['max_metadata_bytes']=sum(t.nbytes for entry in self.responses.values() for t in entry.values())
        return select_with_scores(context,score)
