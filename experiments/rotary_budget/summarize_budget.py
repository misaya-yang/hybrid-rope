"""Strict paired summary. Missing runs remain missing, never filled by old results."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

ARMS=('G32','E32','G16','E16','P16','U16')
CONDITIONS=((512,'intact'),(2048,'intact'),(4096,'intact'),(8192,'intact'),(4096,'remote_replaced'))

def summarize(paths):
    cells=defaultdict(dict);target_identity={};tokens=set()
    for p in paths:
        for line in Path(p).read_text().splitlines():
            row=json.loads(line);arm=row['arm'];seed=row['training_seed'];doc=row['document_index']
            if arm not in ARMS or seed not in (42,137,256):raise ValueError('Unregistered arm or seed')
            condition=(row['context_length'],row['condition'])
            if condition not in CONDITIONS:raise ValueError('Unregistered condition')
            key=(seed,arm,*condition)
            if doc in cells[key]:raise ValueError('Duplicate evaluation row')
            if row['target_token_count']!=256 or not np.isfinite(row['sum_nll']):raise ValueError('Invalid target loss')
            identity=(row['source_document_id'],row['target_sha256'],row['target_start'],row['target_end'])
            if doc in target_identity and target_identity[doc]!=identity:raise ValueError('Targets or document identity differ')
            target_identity[doc]=identity;tokens.add(row['checkpoint_tokens'])
            cells[key][doc]=row
    if len(tokens)>1:raise ValueError('Training budgets differ')
    if any(set(v)!=set(range(512)) for v in cells.values()):raise ValueError('Incomplete document cell')
    seeds=sorted({k[0] for k in cells})
    complete_seeds=[s for s in seeds if all((s,a,*c) in cells for a in ARMS for c in CONDITIONS)]
    means=[];contrasts=[]
    for key,values in sorted(cells.items()):
        losses=[values[i]['sum_nll']/256 for i in range(512)]
        mean=float(np.mean(losses));item=dict(seed=key[0],arm=key[1],length=key[2],condition=key[3],nll=mean,ppl=math.exp(mean))
        if key[2]==2048:
            if any(v.get('full_window_target_count')!=2048 for v in values.values()):raise ValueError('Missing full native window')
            item['full_window_nll']=float(np.mean([v['full_window_sum_nll']/2048 for v in values.values()]))
        means.append(item)
    lookup={(r['seed'],r['arm'],r['length'],r['condition']):r['nll'] for r in means}
    for seed in complete_seeds:
        for length in (512,2048,4096,8192):
            l={a:lookup[(seed,a,length,'intact')] for a in ARMS}
            d16=l['G16']-l['E16'];d32=l['G32']-l['E32'];loss=l['G16']-l['G32']
            contrasts.append(dict(seed=seed,length=length,D16=d16,D32=d32,J=d16-d32,H=l['E16']-l['G32'],
              P=l['P16']-l['E16'],U=l['U16']-l['E16'],geo_budget_loss=loss,
              recovered_fraction=d16/loss if loss>=.02 else None))
        c={a:lookup[(seed,a,4096,'remote_replaced')]-lookup[(seed,a,4096,'intact')] for a in ARMS}
        contrasts.append(dict(seed=seed,length=4096,condition='context_effect',C_A=c,I_context=c['E16']-c['G16']))
    aggregate={}
    for length in (512,2048,4096,8192):
        rows=[r for r in contrasts if r['length']==length and 'D16' in r]
        if not rows:continue
        aggregate[str(length)]={}
        for metric in ('D16','D32','J','H','P','U','geo_budget_loss'):
            v=np.array([r[metric] for r in rows]);n=len(v);mean=float(v.mean());sd=float(v.std(ddof=1)) if n>1 else None
            ci=[mean-4.302652729911275*sd/math.sqrt(3),mean+4.302652729911275*sd/math.sqrt(3)] if n==3 else None
            aggregate[str(length)][metric]={'mean':mean,'std':sd,'n_training_seeds':n,'t95_df2':ci}
    return dict(status='COMPLETE' if complete_seeds==[42,137,256] else 'PARTIAL',
      complete_seeds=complete_seeds,training_tokens=next(iter(tokens),None),cell_count=len(cells),means=means,
      per_seed_contrasts=contrasts,aggregate=aggregate,interval_assumption='paired_training_seed_differences_approximately_normal; n=3,df=2',
      result_boundary='No equivalence inferred from nonsignificance; no absent result imputed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=summarize(a.inputs);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    if r['means']:
        with open(a.output.with_suffix('.csv'),'w') as f:
            writer=csv.DictWriter(f,fieldnames=['seed','arm','length','condition','nll','ppl','full_window_nll']);writer.writeheader();writer.writerows(r['means'])
    print(json.dumps({'status':r['status'],'complete_seeds':r['complete_seeds']}))
