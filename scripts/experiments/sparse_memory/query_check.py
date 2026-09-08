"""Same-context counterfactual query with a different correct answer, fixed in data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import ENTITY
from .model import Config, Model
from .run import acquire_gpu, setup, autocast, load_data, atomic, sha


def main():
    p=argparse.ArgumentParser()
    for name in ['data','checkpoint','out']: p.add_argument('--'+name,required=True)
    p.add_argument('--split', default='development', choices=['development','test'])
    a=p.parse_args(); lock=acquire_gpu(a.data); setup(0,'cuda')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if not saved['final']: raise ValueError('nonfinal checkpoint')
    model=Model(Config(**saved['config']),saved['arm']).cuda(); model.load_state_dict(saved['model']);model.eval()
    x,y,identity=load_data(a.data,a.split,'cuda')
    meta=json.loads((Path(a.data)/(a.split+'.json')).read_text())['metadata']
    entry=identity['metadata_sha256']
    if sha(Path(a.data)/(a.split+'.json')) != entry: raise ValueError('metadata drift')
    selected=[(i,m['alternate_query']) for i,m in enumerate(meta) if m['alternate_query']]
    ids=torch.tensor([i*3 for i,_ in selected],device='cuda')
    original=x[ids]; alternate=original.clone()
    alternate[:,-3]=torch.tensor([ENTITY+alt[0] for _,alt in selected],device='cuda')
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(ids),64):
            with autocast('cuda'):
                p0=model(original[start:start+64]).argmax(-1).cpu().tolist()
                p1=model(alternate[start:start+64]).argmax(-1).cpu().tolist()
            for j,(u,v) in enumerate(zip(p0,p1)):
                i,alt=selected[start+j]
                gold=int(y[3*i]); other=int(alt[1])
                rows.append(dict(pair_id=i,target0=gold,target1=other,prediction0=u,prediction1=v,
                    both_correct=(u==gold and v==other)))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=dict(query_pair_em=float(np.mean([r['both_correct'] for r in rows])), n=len(rows),
        prediction_changes=float(np.mean([r['prediction0'] != r['prediction1'] for r in rows])),
        original_em=float(np.mean([r['prediction0'] == r['target0'] for r in rows])),
        alternate_em=float(np.mean([r['prediction1'] == r['target1'] for r in rows])),
        excluded_no_different_answer=len(meta)-len(rows), checkpoint_sha256=sha(a.checkpoint),
        metadata_sha256=entry, data=identity)
    atomic(out/'result.json',result);print(json.dumps(result))


if __name__ == '__main__': main()
