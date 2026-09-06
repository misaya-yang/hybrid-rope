#!/usr/bin/env python3
"""Failure-conditioned EOS diagnosis; never rescues a failed registered budget."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import load_runtime,guard_resources,retention_rows,rows,sha,write_json,greedy
from scripts.lib.rope.generation_contract import kl_argmax_radius


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','retention-manifest','native-result','candidate-result','table','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gain',type=float,default=1.102585782722872)
    p.add_argument('--authorized',action='store_true');p.add_argument('--max-seconds',type=float,default=900)
    p.add_argument('--recover-logits',type=Path,help='reuse preserved prefix logits; rerun only missing extended decodes')
    a=p.parse_args();a.data=None;a.adapter=None;a.seed=42;a.min_headroom_gib=1.
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    native={r['asset_sha256']:r for r in rows(a.native_result/'examples.jsonl')}
    candidate={r['asset_sha256']:r for r in rows(a.candidate_result/'examples.jsonl')}
    selected=[]
    for row in sorted(retention_rows(a.retention_manifest,'selection'),key=lambda r:r['asset_sha256']):
        key=row['asset_sha256']
        if (row['task']=='gov_report' and native[key]['ended_with_eos'] and not candidate[key]['ended_with_eos']
                and len(row['input_ids'])+1024<=4096): selected.append(row)
        if len(selected)==4: break
    if not selected: raise ValueError('no eligible failure-conditioned Native-length cases')
    import numpy as np
    import torch
    start=time.monotonic();vectors={};records=[]
    if a.recover_logits:
        archive=np.load(a.recover_logits/'terminal_logits.npz',allow_pickle=False)
        vectors={(name.split('_',1)[0],name.split('_',1)[1]):archive[name] for name in archive.files}
    table,gain=a.table,a.gain
    for label in ('N','Z'):
        if a.recover_logits and label=='N': continue
        a.table=None if label=='N' else table;a.gain=1. if label=='N' else gain
        model,tok,identity=load_runtime(a)
        write_json(a.output/(label+'_runtime.json'),identity)
        for row in selected:
            guard_resources(model,identity,start,a);key=row['asset_sha256']
            gold=native[key]['generated_token_ids'];ids=[*row['input_ids'],*gold[:-1]]
            if a.recover_logits: logits=vectors[(label,key)]
            else:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,return_dict=True).last_hidden_state
                    logits=model.lm_head(hidden[:,-1:]).float()[0,0].cpu().numpy().astype(np.float64)
            vectors[(label,key)]=logits
            if label=='Z':
                out=greedy(model,row['input_ids'],tok.eos_token_id,1024)
                ended=bool(out and out[-1]==tok.eos_token_id)
                original=vectors[('N',key)]
                logp=original-original.max();logp-=np.log(np.exp(logp).sum())
                logq=logits-logits.max();logq-=np.log(np.exp(logq).sum())
                prob=np.exp(logp);kl=float(np.sum(prob*(logp-logq)));radius=kl_argmax_radius(prob)
                margin=lambda x:float(x[tok.eos_token_id]-np.max(np.delete(x,tok.eos_token_id)))
                record={'asset_sha256':key,'source_group':row['group'],'input_tokens':len(row['input_ids']),
                        'registered_budget':512,'diagnostic_budget':1024,'generated_ids_1024':out,
                        'ended_by_1024':ended,'new_tokens_1024':len(out),'Native_EOS_margin_at_Native_terminal_prefix':margin(original),
                        'Z_EOS_margin_at_Native_terminal_prefix':margin(logits),'teacher_top1_is_EOS':int(original.argmax())==tok.eos_token_id,
                        'terminal_prefix_KL':kl,'terminal_prefix_argmax_radius':float(radius),'conditional_argmax_certificate':bool(kl<radius)}
                records.append(record)
                with (a.output/'rows.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
        del model;gc.collect();torch.cuda.empty_cache()
    np.savez(a.output/'terminal_logits.npz',**{label+'_'+key:value for (label,key),value in vectors.items()})
    write_json(a.output/'diagnosis.json',{'status':'COMPLETED_FAILURE_CONDITIONED_EOS_DIAGNOSIS','rows':records,
               'logits_sha256':sha(a.output/'terminal_logits.npz'),'script_sha256':sha(__file__),
               'recovered_prefix_logits_sha256':sha(a.recover_logits/'terminal_logits.npz') if a.recover_logits else None,
               'scope':'Selected registered-budget failures with enough Native generation reserve. Not population performance or a revised gate.',
               'seconds':time.monotonic()-start})
    print(json.dumps([{k:v for k,v in r.items() if k!='generated_ids_1024'} for r in records],indent=2))

if __name__=='__main__': main()
