"""E07 redistribute the fixed total QK slots; retain B0 CIS-only fillers.

Logical selected blocks stay fixed in total. The existing padded reader's
extra slots are explicitly counted, so this is not a ragged-kernel speed claim.
"""
import math

import torch

from .exact_probe import ExactBlockSelector
from .runtime import mandatory_blocks, select_with_scores, _stable_topk


class GroupBudgetSelector(ExactBlockSelector):
    def __init__(self, mode='e07_group_budget', **kwargs):
        super().__init__('exact_mass', **kwargs)

    @torch.no_grad()
    def __call__(self, context):
        self.metrics['calls']+=1
        h,length,_=context.k.shape
        queries=context.q.shape[1]
        blocks=math.ceil(length/context.settings.block_size)
        if blocks<=context.settings.topk:
            return select_with_scores(context,context.q.new_empty(0))
        score=self.logmass(context).softmax(-1).sum(1)
        baseline=select_with_scores(context,score)
        visible=torch.arange(blocks,device=score.device)[None]<=context.query_positions[:,None]//context.settings.block_size
        mandatory=mandatory_blocks(context,blocks)
        ranked=score.masked_fill(mandatory,torch.inf).masked_fill(~visible,-torch.inf)
        count=min(context.settings.init_blocks+context.settings.local_blocks+context.settings.select_blocks,blocks)
        qk=_stable_topk(ranked,count)
        qk_mask=torch.zeros_like(score,dtype=torch.bool).scatter_(-1,qk,True)
        b0_mask=torch.zeros_like(score,dtype=torch.long).scatter_add_(-1,baseline.clamp_min(0),(baseline>=0).long())>0
        fixed=b0_mask & (~qk_mask | mandatory)
        available=visible & ~fixed
        value=score.masked_fill(~available,-torch.inf)
        best=value.argmax(-1,keepdim=True)
        minimum=torch.zeros_like(fixed).scatter_(-1,best,True)&available
        remaining=(baseline>=0).sum((0,2))-fixed.sum((0,2))-minimum.sum((0,2))
        flat=value.masked_fill(minimum,-torch.inf).permute(1,0,2).reshape(queries,-1)
        order=torch.argsort(flat,dim=-1,descending=True,stable=True)
        keep=torch.arange(h*blocks,device=score.device)[None]<remaining[:,None]
        chosen=torch.zeros_like(flat,dtype=torch.bool).scatter_(1,order,keep)
        chosen=chosen.reshape(queries,h,blocks).permute(1,0,2)|fixed|minimum
        chosen &= visible
        counts=chosen.sum(-1)
        width=int(counts.max())
        ids=torch.arange(blocks,device=score.device).expand_as(score).masked_fill(~chosen,blocks).sort(-1).values[...,:width]
        self.metrics['e07_logical_selected_blocks']=self.metrics.get('e07_logical_selected_blocks',0)+int(counts.sum())
        self.metrics['e07_b0_selected_blocks']=self.metrics.get('e07_b0_selected_blocks',0)+int((baseline>=0).sum())
        self.metrics['e07_padded_reader_slots']=self.metrics.get('e07_padded_reader_slots',0)+h*queries*width
        self.metrics['e07_max_group_quota']=max(self.metrics.get('e07_max_group_quota',0),width)
        self.metrics['e07_min_group_quota']=min(self.metrics.get('e07_min_group_quota',context.settings.topk),int(counts.min()))
        return ids.masked_fill(ids==blocks,-1)
