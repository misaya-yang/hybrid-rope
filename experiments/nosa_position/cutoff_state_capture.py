"""Two fixed DEV replays with read-only, per-block cutoff state capture.

Full-covariance trajectory is unchanged and must reproduce prior raw tokens.
Save per-head logmass and final supports for pair/rank1/COBS/full/exact, plus
original Q/K/CIS only on full-vs-rank1 differing support blocks at last prompt.
No scores produced by this observer are returned to the model's selector.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import torch

from . import run as base
from .full_covariance_probe import full_covariance_logmass, main as full_cov_main
from .exact_probe import ExactBlockSelector
from .selector_controls import BlockSummarySelector
from .runtime import mandatory_blocks,select_with_scores

ROWS=('ruler_dev_niah_multiquery_16384_000','ruler_dev_niah_single_1_16384_000')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-run',required=True)
    args, remaining=parser.parse_known_args()
    locations=argparse.ArgumentParser(add_help=False)
    locations.add_argument('--data',required=True);locations.add_argument('--output',required=True)
    paths,_=locations.parse_known_args(remaining)
    out,old=Path(paths.output),Path(args.reference_run)
    contract=json.loads((old/'contract.json').read_text())
    if hashlib.sha256(Path(paths.data).read_bytes()).hexdigest()!=contract['data_sha256']:
        raise ValueError('reference data identity mismatch')
    for name in ('runtime.py','selector_controls.py','run.py','full_covariance_probe.py','exact_probe.py'):
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=contract['source_hashes'][name]:
            raise ValueError('reference runtime identity mismatch '+name)
    rows=[json.loads(l) for l in Path(paths.data).read_text().splitlines()]
    by_tokens={tuple(r['prompt_ids']):r for r in rows if r['row_id'] in ROWS}
    refs={r['row_id']:r for r in [json.loads(l) for l in (old/'generations.jsonl').read_text().splitlines()] if r['selector']=='full_covariance'}
    original_generate,original_hashes=base.generate,base.source_hashes
    def source_hashes():
        return {**original_hashes(),Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def generate(model,ids,max_new_tokens,eos_ids,chunk_size,device):
        row=by_tokens.get(tuple(ids[0].tolist()))
        if row is None: raise ValueError('only two fixed DEV inputs allowed')
        if model.selector.mode!='full_covariance' or chunk_size!=contract['chunk_size'] or model.settings.topk!=contract['topk'] or model.settings.select_blocks!=contract['select_blocks'] or model.settings.attention_query_chunk_size!=contract['attention_query_chunk_size']:
            raise ValueError('reference selector/allocation/chunk identity')
        rid=row['row_id']; targets={ids.numel()-1,ids.numel()}
        previous=model.trace_callback; receipts=[]
        def observe(context,selected):
            if previous is not None: previous(context,selected)
            for pos in targets:
                match=(context.query_positions==pos).nonzero(as_tuple=True)[0]
                if not match.numel(): continue
                torch.cuda.synchronize(); started=time.perf_counter();i=int(match[0])
                point=replace(context,q=context.q[:,i:i+1],query_positions=context.query_positions[i:i+1])
                scores={name:BlockSummarySelector(name).logmass(point) for name in ('pc2','pc2_rank1','cobs_rank2')}
                scores['full_covariance']=full_covariance_logmass(point)
                scores['exact_mass']=ExactBlockSelector('exact_mass').logmass(point)
                grouped={name:x.softmax(-1).sum(1) for name,x in scores.items()}
                supports={name:select_with_scores(point,x) for name,x in grouped.items()}
                # A different single-Q GEMM shape may differ by a rounding bit;
                # save actual live support separately instead of asserting it away.
                actual=selected[:,i:i+1].clone()
                blocks=scores['full_covariance'].shape[-1]
                protected=mandatory_blocks(point,blocks)
                raw=[]
                if pos==ids.numel()-1:
                    for head in range(point.k.shape[0]):
                        left=set(supports['full_covariance'][head,0].tolist())
                        right=set(supports['pc2_rank1'][head,0].tolist())
                        ids_changed=sorted((left^right)-{-1})
                        for block in ids_changed:
                            lo,hi=block*point.settings.block_size,min((block+1)*point.settings.block_size,point.k.shape[1])
                            raw.append({'kv_head':head,'block':block,'keys':point.k[head,lo:hi].cpu(),
                                'cis':point.cis[head,lo:hi].cpu()})
                saved={'row_id':rid,'layer_idx':point.layer_idx,'query_position':pos,
                    'query_kind':'last_prompt' if pos==ids.numel()-1 else 'first_ingested_answer',
                    'settings':asdict(point.settings),'query':point.q.cpu(),'prefix_length':point.k.shape[1],
                    'logmass':{k:v.cpu() for k,v in scores.items()},'group_scores':{k:v.cpu() for k,v in grouped.items()},
                    'supports':{k:v.cpu() for k,v in supports.items()},'actual_live_support':actual.cpu(),
                    'mandatory':protected.cpu(),'changed_raw_blocks':raw,
                    'raw_scope':'last-prompt full/rank1 support symmetric difference only; no V needed for score audit'}
                filename=f'{rid}.layer{point.layer_idx:02d}.pos{pos}.pt'
                torch.save(saved,out/filename)
                torch.cuda.synchronize()
                receipts.append({'row_id':rid,'layer':point.layer_idx,'position':pos,'file':filename,
                    'observer_seconds':time.perf_counter()-started,'raw_block_count':len(raw),
                    'sliced_full_support_matches_live':torch.equal(actual,supports['full_covariance'])})
        model.trace_callback=observe
        try:
            tokens,timing=original_generate(model,ids,max_new_tokens,eos_ids,chunk_size,device)
        finally:
            model.trace_callback=previous
        same=tokens==refs[rid]['generated_token_ids']
        with (out/'capture_receipts.jsonl').open('a') as stream:
            for receipt in receipts: stream.write(json.dumps(receipt)+'\n')
        base.write_json(out/(rid+'.replay.json'),{'row_id':rid,'reference_run':str(old),
            'raw_tokens_exactly_reproduced':same,'state_count':len(receipts),
            'observer_seconds':sum(x['observer_seconds'] for x in receipts),'generated_token_ids':tokens})
        if not same: raise RuntimeError('replay differs; do not treat as original trajectory')
        return tokens,{**timing,'timing_includes_diagnostics':True,'raw_tokens_exactly_reproduced':same,
            'observer_seconds':sum(x['observer_seconds'] for x in receipts)}
    try:
        base.generate,base.source_hashes=generate,source_hashes
        full_cov_main(remaining)
    finally:
        base.generate,base.source_hashes=original_generate,original_hashes

if __name__=='__main__':main()
