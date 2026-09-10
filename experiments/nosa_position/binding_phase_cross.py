"""Cross history routing versus final-prompt/answer routing on two fixed DEV.

The model's normal prefill chunks and reader remain unchanged. At the one
chunk containing the final prompt token, evaluate both selectors on the same
unmodified context and combine their per-query chosen sets. No inherited final
logits, answer tokens, alternate positions, or external state is injected.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import torch
from . import run as base
from .selector_controls import BlockSummarySelector
from .runtime import native_select
from .full_covariance_probe import FullCovarianceSelector

ROWS=('ruler_dev_niah_multiquery_16384_001','ruler_dev_niah_multiquery_16384_003')
MODES={'full_to_cobs':('full_covariance','cobs_rank2'),'cobs_to_full':('cobs_rank2','full_covariance'),'cobs_to_cobs':('cobs_rank2','cobs_rank2'),'pc2_to_full':('pc2','full_covariance'),'pc2_to_pc2':('pc2','pc2'),'native_to_full':('native','full_covariance'),'native_to_native':('native','native')}


def make_selector(name):
    if name=='native':return native_select
    return FullCovarianceSelector(name) if name=='full_covariance' else BlockSummarySelector(name)


class PhaseCrossSelector(BlockSummarySelector):
    def __init__(self,mode=None,*,history=None,answer=None):
        super().__init__('pc2')
        self.history_name,self.answer_name=MODES[mode] if mode is not None else (history,answer)
        self.history=make_selector(self.history_name)
        self.answer=self.history if self.answer_name==self.history_name else make_selector(self.answer_name)
        self.threshold=None
        self.metrics.update(history_queries=0,answer_queries=0,crossing_contexts=0)
    @torch.no_grad()
    def __call__(self,context):
        if self.threshold is None:raise RuntimeError('final prompt boundary must be set before prefill')
        early=context.query_positions<self.threshold
        self.metrics['calls']+=1
        self.metrics['history_queries']+=int(early.sum())
        self.metrics['answer_queries']+=int((~early).sum())
        if self.history is self.answer:return self.history(context)
        if bool(early.all()):return self.history(context)
        if not bool(early.any()):return self.answer(context)
        self.metrics['crossing_contexts']+=1
        left,right=self.history(context),self.answer(context)
        return torch.where(early[None,:,None],left,right)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--reference-run',required=True)
    parser.add_argument('--queue-root',default='/root/autodl-tmp/position_overnight_20260909')
    args,remaining=parser.parse_known_args()
    locations=argparse.ArgumentParser(add_help=False)
    locations.add_argument('--data',required=True)
    loc,_=locations.parse_known_args(remaining)
    reference=json.loads((Path(args.reference_run)/'contract.json').read_text())
    if hashlib.sha256(Path(loc.data).read_bytes()).hexdigest()!=reference['data_sha256']:raise ValueError('data identity')
    for name in ('run.py','runtime.py','selector_controls.py','full_covariance_probe.py','exact_probe.py'):
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=reference['source_hashes'][name]:raise ValueError('reference computation '+name)
    rows={tuple(r['prompt_ids']):r['row_id'] for r in map(json.loads,Path(loc.data).read_text().splitlines()) if r['row_id'] in ROWS}
    if not args.execute:print({'status':'DRY_RUN','row_ids':ROWS,'phase':'history through prompt[-2], other selector from prompt[-1]'});return
    old_generate,old_factory,old_modes,old_hashes,old_argv=base.generate,base.BlockSummarySelector,base.MODES,base.source_hashes,sys.argv
    def generate(model,ids,max_new_tokens,eos_ids,chunk_size,device):
        if tuple(ids[0].tolist()) not in rows:raise ValueError('only two fixed binding examples')
        if chunk_size!=reference['chunk_size'] or model.settings.attention_query_chunk_size!=reference['attention_query_chunk_size'] or model.settings.topk!=reference['topk'] or model.settings.select_blocks!=reference['select_blocks']:raise ValueError('reference reader/quota/chunk mismatch')
        model.selector.threshold=ids.numel()-1
        tokens,timing=old_generate(model,ids,max_new_tokens,eos_ids,chunk_size,device)
        return tokens,{**timing,'history_selector':model.selector.history_name,
            'answer_selector':model.selector.answer_name,'switch_query_position':ids.numel()-1,
            'final_prompt_scored_by_answer_selector':True,'prompt_chunks_unchanged':True}
    def hashes():
        files=(Path(__file__),Path(__file__).with_name('full_covariance_probe.py'),Path(__file__).with_name('exact_probe.py'))
        return {**old_hashes(),**{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    root=Path(args.queue_root)
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'STOP').exists():raise RuntimeError('STOP')
        try:
            base.generate,base.BlockSummarySelector,base.MODES,base.source_hashes=generate,PhaseCrossSelector,(*old_modes,*MODES),hashes
            sys.argv=[str(Path(__file__)),*remaining];base.main()
        finally:
            base.generate,base.BlockSummarySelector,base.MODES,base.source_hashes,sys.argv=old_generate,old_factory,old_modes,old_hashes,old_argv

if __name__=='__main__':main()
