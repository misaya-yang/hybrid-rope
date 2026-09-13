#!/usr/bin/env python3
"""Append-only research job definitions; execute one selected job in an idle GPU slot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def jobs(root, python):
    base=root/'rope_fast_5090_20260912';e2=base/'e2_prepared_final_20260912'
    model=root/'olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct'
    sources=root/'olmo_recovery_20260912/sources';prefix='experiments.rope_fast_5090_20260912.'
    result=[]
    for job,module,count in [('p1','p1_factorial',384),('source','source_counterfactual',768),('layers','layer_group_probe',360)]:
        prepared=base/('supplement_'+module+'_prepared');out=base/('supplement_'+module+'_run')
        run=[python,'-m',prefix+module,'run','--prepared',str(prepared),'--out',str(out),'--execute']
        score=[python,'-m',prefix+module,'score','--prepared',str(prepared),'--out',str(out/'summary.json')]
        if job=='p1':
            run+=['--model',str(model),'--e2-prepared',str(e2),'--sources',str(sources)]
            score+=['--runs',str(out)]
        else:
            score+=['--run',str(out)]
        if job=='layers':
            run+=['--model',str(model),'--reuse-e2',str(base/'e2_run_final_20260912_port24904'),'--e2-prepared',str(e2)]
        result.append({'job':job,'module':module,'prepared':str(prepared),'new_generations':count,
                       'run_argv':run,'score_argv':score})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp'))
    parser.add_argument('--job',choices=['p1','source','layers']);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();queue=jobs(args.data_root,sys.executable)
    payload={'status':'APPEND_QUEUE_NOT_AUTOSTARTED','priority':'existing healthy work and LoRA remain first',
             'asset_identity_policy':'user_attested_clone/no_sha_validation','gpu_jobs':queue,
             'new_generations_with_E2_reuse':1512,'without_E2_reuse':1632}
    if args.output:args.output.write_text(json.dumps(payload,indent=2)+'\n')
    if not args.execute:print(json.dumps(payload,indent=2));return
    if not args.job:parser.error('--execute requires one --job selected by the existing execution owner')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(line.strip().isdigit() for line in active.splitlines()):
        raise RuntimeError('GPU has an active job; leave it running and queue this job afterward')
    job=next(item for item in queue if item['job']==args.job)
    if not (Path(job['prepared'])/'manifest.json').is_file():raise FileNotFoundError(job['prepared'])
    cwd=Path(__file__).resolve().parents[2]
    subprocess.run(job['run_argv'],cwd=cwd,check=True)
    subprocess.run(job['score_argv'],cwd=cwd,check=True)


if __name__=='__main__':main()
