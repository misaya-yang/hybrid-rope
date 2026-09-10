"""Finite shared-DEV queue; reuse matching completed rows and record every job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run import write_json


ACCEPTED={
 'exact_mass':['pc2_exact_mass_full_dev_v1','pc2_ten_v1_natural_E02_E08_dev_v1'],
 'e02_nonlinear':['pc2_ten_v1_E02_retrieval_dev_v1','pc2_ten_v1_E02_retrieval_remaining4_v1','pc2_ten_v1_natural_E02_E08_dev_v1'],
 'e08_tail_value':['pc2_ten_v1_E08_retrieval_dev_v1','pc2_ten_v1_natural_E02_E08_dev_v1'],
 'e01_int8':['pc2_ten_v1_E01_common48_v2'],
 'e04_empirical':['pc2_ten_v1_E04_retrieval_dev_v1'],
 'e04_second':['pc2_ten_v1_E04_retrieval_dev_v1']}
ORDER=['exact_mass','e02_nonlinear','e08_tail_value','e04_empirical','e04_second',
       'e03_temporal','e07_group_budget','e05_two_component','e05_contiguous','e05_rank4',
       'e09_local','e10_cutoff','e06_residual_sampling']


def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def read_rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--wait-pid',type=int)
    p.add_argument('--tag',default='v1')
    p.add_argument('--modes',nargs='+',help='Optional separate finite lane, e.g. E01 remaining common rows')
    p.add_argument('--memory-fraction',type=float,default=0.52)
    p.add_argument('--chunk-size',type=int,default=2048)
    args=p.parse_args();root=Path(args.root);stage=root/'prepared/pc2_ten_v1'
    common=stage/'common48/rows.jsonl';panel=read_rows(common);byid={r['row_id']:r for r in panel}
    sources={}
    for path in (common,root/'data/pc2/rows.jsonl',root/'prepared/broad_panel_20260909/data/pc2.jsonl'):
        sources[hashlib.sha256(path.read_bytes()).hexdigest()]={r['row_id']:r for r in read_rows(path)}
    queue_dir=stage/('queue_'+args.tag);queue_dir.mkdir(exist_ok=True)
    state={'status':'WAITING','pid':os.getpid(),'wait_pid':args.wait_pid,'stages':[],
           'common_sha256':hashlib.sha256(common.read_bytes()).hexdigest(),
           'parallel_lane':'E01 corrected full48 executes independently; not duplicated in this queue'}
    write_json(queue_dir/'status.json',state)
    if args.wait_pid:
        while live(args.wait_pid):time.sleep(1)
    env={**os.environ,'PC2_QUERY_BASIS':str(stage/'calibration_v2/query_basis.pt'),
         'PC2_CUTOFF_MODEL':str(stage/'cutoff_fit_v1/cutoff_model.pt')}
    # All stages are finite. A failed stage is retained as failed; independent
    # following stages proceed, while the owner repairs the concrete fault.
    def execute(mode,ids,suffix='',seed=None):
        output=root/'runs'/f'pc2_ten_v1_common48_{mode}{suffix}_{args.tag}'
        rowfile=queue_dir/f'{mode}{suffix}_rows.json';write_json(rowfile,ids)
        entry={'mode':mode,'row_ids':ids,'output':str(output),'status':'STARTING','seed':seed}
        state['stages'].append(entry);state['status']='RUNNING';write_json(queue_dir/'status.json',state)
        launch=f'import torch; torch.cuda.set_per_process_memory_fraction({args.memory_fraction}); from experiments.nosa_position.ten_run import main; main()'
        command=[sys.executable,'-u','-c',launch,
            '--model','/root/autodl-tmp/NOSA-1B','--data',str(common),'--output',str(output),
            '--selectors',mode,'--split','dev','--row-ids',str(rowfile),'--topk','64','--select-blocks','16',
            '--chunk-size',str(args.chunk_size),'--attention-query-chunk-size',str(args.chunk_size)]
        job_env={**env,'PC2_SAMPLE_SEED':str(seed or 20260910)}
        started=time.time()
        with (queue_dir/f'{mode}{suffix}.log').open('a') as log:
            job=subprocess.Popen(command,cwd=stage/'code',env=job_env,stdout=log,stderr=subprocess.STDOUT)
            entry.update(status='RUNNING',pid=job.pid,command=command);write_json(queue_dir/'status.json',state)
            result=job.wait()
        receipt=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
        entry.update(status='COMPLETE' if result==0 and receipt.get('status')=='COMPLETE' else 'FAILED',
                     returncode=result,seconds=time.time()-started,receipt=receipt)
        write_json(queue_dir/'status.json',state)
        print(json.dumps(entry),flush=True)
    order=args.modes or ORDER
    for mode in order:
        done={}
        accepted=ACCEPTED.get(mode,[])+[f'pc2_ten_v1_common48_{mode}_{args.tag}']
        for name in accepted:
            folder=root/'runs'/name
            if not (folder/'generations.jsonl').exists():continue
            contract=json.loads((folder/'contract.json').read_text())
            origin=sources.get(contract.get('data_sha256'),{})
            if contract.get('topk')!=64 or contract.get('select_blocks')!=16:
                continue
            for row in read_rows(folder/'generations.jsonl'):
                rid=row['row_id']
                if row['selector']!=mode or rid not in byid or rid not in origin:continue
                a,b=origin[rid],byid[rid]
                if any(a.get(k)!=b.get(k) for k in ('prompt_ids','max_new_tokens','score_contract','expected','references')):
                    raise ValueError(f'cannot reuse changed input/scoring contract: {rid}')
                done[rid]={'run':name,'record':row}
        write_json(queue_dir/f'{mode}_reused.json',done)
        ids=[r['row_id'] for r in panel if r['row_id'] not in done]
        if ids:execute(mode,ids)
    probe=[]
    for family in ('retrieval','occurrence','multi_hop','natural_qa'):
        probe.extend([r['row_id'] for r in panel if r['ten_family']==family][:2])
    if 'e06_residual_sampling' in order:
        for seed in (20260911,20260912):execute('e06_residual_sampling',probe,'_seed'+str(seed),seed)
    state['status']='COMPLETE' if all(s['status']=='COMPLETE' for s in state['stages']) else 'COMPLETE_WITH_FAILURES'
    write_json(queue_dir/'status.json',state)


if __name__=='__main__':main()
