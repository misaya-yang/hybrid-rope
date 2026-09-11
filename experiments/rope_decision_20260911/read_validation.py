"""Paired equal-task readout, restricted to complete identified arms."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def contrast(a,b,cases,cap):
    cells=[]
    values=[]
    for task in sorted({c['task'] for c in cases.values() if c['length_cap']==cap}):
        keys=[k for k,c in cases.items() if c['length_cap']==cap and c['task']==task]
        d=np.array([a[k]['correct']-b[k]['correct'] for k in keys])
        if len(d)<2:raise ValueError('at least two paired samples per task')
        cells.append(dict(task=task,n=len(d),delta=float(d.mean()),variance=float(d.var(ddof=1)/len(d))))
        values.extend(d.tolist())
    mean=float(np.mean([c['delta'] for c in cells]))
    total=sum(c['variance'] for c in cells)
    se=math.sqrt(total)/len(cells)
    denom=sum(c['variance']**2/(c['n']-1) for c in cells)
    df=total**2/denom if denom else None
    radius=float(student_t.ppf(.975,df))*se if df else 0.
    return dict(delta_pp=100*mean,se_pp=100*se,ci95_pp=[100*(mean-radius),100*(mean+radius)],
                approximate_df=df,wins=sum(d>0 for d in values),losses=sum(d<0 for d in values),
                ties=sum(d==0 for d in values),task_cells=cells)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    plan=json.loads((args.root/'validation_plan.json').read_text())
    cases={c['row_id']:c for c in plan['cases']}
    arms={}
    for dirname in ('validation_reference','validation_methods','validation_longbridge'):
        for path in (args.root/dirname/'generations').glob('*.json'):
            r=json.loads(path.read_text())
            c=cases[r['row_id']]
            expected=hashlib.sha256(np.asarray(c['prompt_ids'],dtype='<i8').tobytes()).hexdigest()
            if r['prompt_sha256']!=expected:raise ValueError('paired prompt mismatch')
            group=arms.setdefault(r['label'],{})
            if r['row_id'] in group:raise ValueError('duplicate output row')
            group[r['row_id']]=r
    complete={k:v for k,v in arms.items() if v.keys()==cases.keys()}
    result=dict(counts={k:len(v) for k,v in arms.items()},complete_arms=sorted(complete),scores={},contrasts={})
    for name,rows in complete.items():
        result['scores'][name]={}
        for cap in (4096,16384):
            tasks={task:np.mean([rows[k]['correct'] for k,c in cases.items() if c['length_cap']==cap and c['task']==task]) for task in sorted({c['task'] for c in cases.values() if c['length_cap']==cap})}
            result['scores'][name][cap]=dict(macro=float(np.mean(list(tasks.values()))),tasks=tasks)
        for baseline in ('bm','b4wide','gain_calibrated','yarn_index','mrpro'):
            if baseline in complete and baseline!=name:
                result['contrasts'][name+'-minus-'+baseline]={cap:contrast(rows,complete[baseline],cases,cap) for cap in (4096,16384)}
    result['scope']='Shared 72-row fresh evaluation of frozen candidates; equal-task means, paired within-task SE and approximate Welch t intervals. No full-benchmark or universal-optimality claim.'
    text=json.dumps(result,indent=2)
    if args.out:args.out.write_text(text)
    print(text)


if __name__=='__main__':main()
