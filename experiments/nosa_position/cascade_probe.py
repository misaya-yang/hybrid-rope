"""Bounded same-state cascade decomposition on B0 trajectories; no oracle deployment."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from .cascade import candidate_ids, exact_candidates, replace_candidate_mass, final_selection
from .exact_probe import ExactBlockSelector
from .projected_distribution import ProjectedDistributionSelector
from .runtime import AttentionSettings, NosaReferenceForCausalLM, mandatory_blocks, select_with_scores, _stable_topk
from .run import generate, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in Path(args.data).read_text().splitlines()]
    ids=['ruler_dev_niah_multikey_1_16384_000','ruler_dev_niah_multikey_1_16384_001','ruler_dev_niah_multiquery_16384_000','ruler_dev_niah_multiquery_16384_001','pc2ten_occurrence_dev_000','pc2ten_occurrence_dev_006','broad_qasper_dev_000','broad_qasper_dev_001']
    rows=[next(r for r in rows if r['row_id']==rid) for rid in ids]
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.9)
    status={'status':'LOADING','pid':os.getpid(),'completed_rows':0,'total_rows':len(rows)};write_json(out/'status.json',status)
    write_json(out/'contract.json',{'data_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),'row_ids':ids,'width_multipliers':[2,4], 'state_scope':'last prompt token and first eight decode query states, all layers, B0 trajectory','selection':'original mandatory/QK33/CIS64; mask outside only after GQA normalization','source_hashes':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['cascade.py','cascade_probe.py','projected_distribution.py','exact_probe.py','runtime.py']}})
    exact=ExactBlockSelector('exact_mass');current=None;coarse=None;records=[];tensors=[]
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize();return result,time.perf_counter()-start
    @torch.no_grad()
    def observe(ctx,selected):
        pos=int(ctx.query_positions[-1]);last=current['input_tokens']-1
        if not last<=pos<=last+8:return
        c=replace(ctx,q=ctx.q[:,-1:],query_positions=ctx.query_positions[-1:])
        ex,te=timed(lambda:exact.logmass(c))
        ap,build=timed(lambda:coarse.logmass(c))
        ap,tc=timed(lambda:coarse.logmass(c))  # cached descriptor scoring; construction separately charged
        b0=select_with_scores(c,ex.softmax(-1).sum(1));m=mandatory_blocks(c,ex.shape[-1])
        slots=min(33,ex.shape[-1]);qk=_stable_topk(ex.softmax(-1).sum(1).masked_fill(m,torch.inf),slots)
        for factor in (2,4):
            (ci,mask),tr=timed(lambda:candidate_ids(c,ap,factor))
            (ce,reads),tx=timed(lambda:exact_candidates(c,ci))
            def finish():
                mixed=replace_candidate_mass(ap,ci,ce)
                return mixed,final_selection(c,mixed,mask)
            (mixed,chosen),tf=timed(finish)
            # Oracle is exclusively diagnostic: exact numerator AND full exact denominator.
            oracle=select_with_scores(c,ex.softmax(-1).sum(1).masked_fill(~mask,-torch.inf))
            conditional=final_selection(c,ex.masked_fill(~mask[:,None],-torch.inf),mask)
            target=mask.gather(-1,qk)
            r={'row_id':current['row_id'],'layer':ctx.layer_idx,'position':pos,'phase':'prompt' if pos==last else 'decode','factor':factor,'qk_covered_fraction':float(target.float().mean()),'all_qk_covered':bool(target.all()),'oracle_final_equal':bool((oracle==b0).all()),'mixed_final_equal':bool((chosen==b0).all()),'candidate_only_final_equal':bool((conditional==b0).all()),'relative_denominator_error_max':float((mixed.logsumexp(-1)-ex.logsumexp(-1)).expm1().abs().max()),'candidate_count':int((ci>=0).sum()),'visible_blocks':ex.shape[-1]*ex.shape[0],'raw_k_gather_elements':reads,'exact_seconds':te,'coarse_build_and_score_seconds':build,'warm_coarse_seconds':tc,'candidate_route_seconds':tr,'candidate_exact_seconds':tx,'mix_and_final_route_seconds':tf,'warm_cascade_seconds':tc+tr+tx+tf}
            records.append(r)
            with (out/'states.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        tensors.append({'row_id':current['row_id'],'layer':ctx.layer_idx,'position':pos,'exact_logmass':ex.cpu(),'e04_logmass':ap.cpu(),'mandatory':m.cpu(),'B0_selected':b0.cpu()})
    model=NosaReferenceForCausalLM.from_pretrained(args.model,device='cuda',dtype=torch.bfloat16,selector=exact,settings=AttentionSettings(topk=64,select_blocks=16,attention_query_chunk_size=1024),trace_callback=observe)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True);eos=tok.eos_token_id
    try:
        for current in rows:
            coarse=ProjectedDistributionSelector('e04_empirical')
            status.update(status='RUNNING',row_id=current['row_id']);write_json(out/'status.json',status)
            generated,timing=generate(model,torch.tensor([current['prompt_ids']],device='cuda'),9,{2,73440},1024,'cuda')
            with (out/'trajectories.jsonl').open('a') as f:f.write(json.dumps({'row_id':current['row_id'],'generated_token_ids':generated,'output_text':tok.decode(generated),'timing':timing})+'\n')
            status['completed_rows']+=1;write_json(out/'status.json',status)
        torch.save(tensors,out/'same_state_scores.pt')
        summary={}
        for factor in (2,4):
            rr=[r for r in records if r['factor']==factor]
            summary[str(factor)]={k:sum(r[k] for r in rr)/len(rr) for k in ['qk_covered_fraction','all_qk_covered','oracle_final_equal','mixed_final_equal','candidate_only_final_equal']}
            summary[str(factor)].update(states=len(rr),warm_cost_ratio=sum(r['warm_cascade_seconds'] for r in rr)/sum(r['exact_seconds'] for r in rr),candidate_fraction=sum(r['candidate_count'] for r in rr)/sum(r['visible_blocks'] for r in rr))
        write_json(out/'summary.json',summary);status.update(status='COMPLETE');write_json(out/'status.json',status)
    except Exception as e:
        status.update(status='FAILED',error=repr(e));write_json(out/'status.json',status);raise

if __name__=='__main__':main()
