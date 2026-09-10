"""PC2 coarse candidates, full second-order scores only within original topk.

The shortlist width equals the existing reader block budget, not a tuned new
parameter. Restrict the QK nomination stage to this shortlist after recomputing
its head logmass; retain coarse outside mass only for GQA normalization. Original
CIS filling and reader remain unchanged. This is approximate coarse-to-fine
routing, not certified exact routing or a claim of novel shortlist selection.
"""
import argparse
import ast
import json
import fcntl
import hashlib
import math
from pathlib import Path
import sys

import torch
from . import run as base
from .runtime import mandatory_blocks,select_with_scores
from .selector_controls import BlockSummarySelector


@torch.no_grad()
def refine_candidate_scores(context,coarse):
    q,k,cis,s=context.q,context.k,context.cis,context.settings
    h,t,d=k.shape; g=q.shape[0]//h; queries=q.shape[1]; blocks=coarse.shape[-1]
    width=min(s.topk,blocks)
    ranked=coarse.softmax(-1).sum(1).masked_fill(mandatory_blocks(context,blocks)[None],torch.inf)
    pool=ranked.argsort(dim=-1,descending=True,stable=True)[...,:width]
    updated=coarse.clone()
    heads=torch.arange(h,device=q.device)[:,None,None,None]
    offsets=torch.arange(s.block_size,device=q.device)
    query=q.reshape(h,g,queries,d).permute(0,2,1,3).float()/math.sqrt(d)
    raw_scores=0
    with torch.autocast(device_type=q.device.type,enabled=False):
        for begin in range(0,queries,s.attention_query_chunk_size):
            end=min(queries,begin+s.attention_query_chunk_size);n=end-begin
            chosen=pool[:,begin:end]
            indices=chosen[...,None]*s.block_size+offsets
            valid=(indices<t)&(indices<=context.query_positions[None,begin:end,None,None])
            keys=k[heads,indices.clamp_max(t-1)].float()
            logits=query[:,begin:end] @ keys.reshape(h,n,width*s.block_size,d).transpose(-1,-2)
            logits=logits.reshape(h,n,g,width,s.block_size)
            biases=cis[heads,indices.clamp_max(t-1)].float().masked_fill(~valid,-torch.inf)
            any_valid=valid.any(-1,keepdim=True)
            weights=torch.where(any_valid,biases,torch.zeros_like(biases)).softmax(-1)*valid
            mean=(weights[:,:,None]*logits).sum(-1)
            variance=(weights[:,:,None]*(logits-mean[...,None]).square()).sum(-1)
            full=biases.logsumexp(-1)[:,:,None]+mean+0.5*variance
            current=chosen==context.query_positions[None,begin:end,None]//s.block_size
            exact_current=(logits+biases[:,:,None]).logsumexp(-1)
            values=torch.where(current[:,:,None],exact_current,full).permute(0,2,1,3)
            updated[:,:,begin:end].scatter_(-1,chosen[:,None].expand(-1,g,-1,-1),values)
            raw_scores+=h*g*n*width*s.block_size
    allowed=torch.zeros_like(ranked,dtype=torch.bool).scatter_(-1,pool,True)
    # Do not let unrefined overestimates jump back into a supposedly refined
    # candidate list. Outside estimates affect head denominators, not eligibility.
    grouped=updated.softmax(-1).sum(1).masked_fill(~allowed,-torch.inf)
    return grouped,updated,pool,{'refinement_raw_key_scores':raw_scores,
        'dense_equivalent_raw_key_scores':q.shape[0]*queries*t,
        'gathered_key_elements':h*queries*width*s.block_size*d}


class CandidateRefinementSelector(BlockSummarySelector):
    def __init__(self,mode='pc2_refine_full',**kwargs):
        self.refine=mode=='pc2_refine_full'
        super().__init__('pc2' if self.refine else mode,**kwargs)
        if self.refine:
            self.metrics.update(refinement_raw_key_scores=0,dense_equivalent_raw_key_scores=0,gathered_key_elements=0)
    @torch.no_grad()
    def __call__(self,context):
        if not self.refine:return super().__call__(context)
        self.metrics['calls']+=1
        if math.ceil(context.k.shape[1]/context.settings.block_size)<=context.settings.topk:
            if int(context.query_positions[0])==0:self.cache.pop(context.layer_idx,None)
            return select_with_scores(context,context.q.new_empty(0))
        coarse=super().logmass(context)
        grouped,_,_,metrics=refine_candidate_scores(context,coarse)
        for k,v in metrics.items():self.metrics[k]=self.metrics.get(k,0)+v
        return select_with_scores(context,grouped)


def scientific_ast(source_text):
    names={'refine_candidate_scores','CandidateRefinementSelector'}
    nodes=[node for node in ast.parse(source_text).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes}!=names:raise ValueError('missing refinement computation definitions')
    return ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)


def reuse_refined_rows(original,source,target,row_ids,arms,*,kind,row_keys=None):
    # Keep all existing model/data/dtype/reader/allocation checks in the shared
    # loader, then explicitly support this additional operator with its own
    # computation identity. Wrapper-only edits do not invalidate operator reuse.
    rows=original(source,target,row_ids,arms,kind=kind,row_keys=row_keys)
    if kind!='pc2' or 'pc2_refine_full' not in arms:return rows
    source=Path(source)
    snapshot=source/'code_snapshot'/'candidate_refinement.py'
    if not snapshot.is_file() or scientific_ast(snapshot.read_text())!=scientific_ast(Path(__file__).read_text()):
        raise ValueError('cannot reuse changed candidate refinement computation')
    seen={}
    for line in (source/'generations.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['row_id'] not in row_ids or row['selector']!='pc2_refine_full':continue
        key=row['row_id']
        if key in seen and seen[key]!=row['generated_token_ids']:raise ValueError('conflicting refined generations')
        if key in seen:continue
        seen[key]=row['generated_token_ids']
        rows.append({**row,'reused_candidate':True,'reused_from_run':str(source.resolve())})
    if not seen:raise ValueError('requested refined source contains no matching generations; refuse silent rerun')
    return rows


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--queue-root',default='/root/autodl-tmp/position_overnight_20260909')
    options,remaining=parser.parse_known_args()
    if not options.execute:
        print({'status':'DRY_RUN','runner_argv':remaining,'candidate_pool':'existing topk budget','no_certified_exact_claim':True});return
    from experiments.position_overnight import reuse
    original_reuse=reuse.candidate_rows
    original_factory,original_modes,original_hashes,original_argv=base.BlockSummarySelector,base.MODES,base.source_hashes,sys.argv
    def hashes():return {**original_hashes(),Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    root=Path(options.queue_root)
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'STOP').exists():raise RuntimeError('global STOP')
        try:
            reuse.candidate_rows=lambda *args,**kwargs:reuse_refined_rows(original_reuse,*args,**kwargs)
            base.BlockSummarySelector=CandidateRefinementSelector
            base.MODES=(*original_modes,'pc2_refine_full');base.source_hashes=hashes
            sys.argv=[str(Path(__file__)),*remaining];base.main()
        finally:
            reuse.candidate_rows=original_reuse
            base.BlockSummarySelector,base.MODES,base.source_hashes,sys.argv=original_factory,original_modes,original_hashes,original_argv

if __name__=='__main__':main()
