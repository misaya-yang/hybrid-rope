"""Prepare a small, explicit replay/calibration contract from observed cases."""
import argparse
import hashlib
import json
from pathlib import Path

from .tables import build_tables


def read(path):
    return {r['row_id']:r for r in map(json.loads,Path(path).read_text().splitlines())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    old=Path('/root/autodl-tmp/phase1_20260910')
    prepared=Path('/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02')
    panel=read('/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl')
    bm=read(old/'holdout180/beta_b1p0.jsonl')
    b4=read(old/'holdout180/wide_b4p0.jsonl')
    tables=build_tables(500000.,4096)
    archived=json.loads((prepared/'tables.json').read_text())
    tables['bm_archive']=archived['MrProBM']
    tables['mrpro_archive']=archived['MrPro']
    cases=[]
    # Minimize full-output replay cost within each observed whole-row conflict.
    # These are development cases, not random held-out samples or performance estimates.
    for role,cap,source,a,b in [('short_repair',4096,'bm',bm,b4),
                              ('long_preserve',16384,'b4wide',b4,bm)]:
        keys=[k for k in panel if panel[k]['length_cap']==cap and a[k]['correct']>=.999 and b[k]['correct']<.999]
        keys.sort(key=lambda k:(len(a[k]['output_text']),k))
        unique={}
        for k in keys:
            digest=hashlib.sha256(json.dumps(panel[k]['prompt_ids']).encode()).hexdigest()
            unique.setdefault(digest,k)
        keys=list(unique.values())
        for k in keys[:2]:
            cases.append(dict(panel[k],role=role,target_arm=source,
                              archived_bm_score=bm[k]['correct'],archived_b4_score=b4[k]['correct']))
    plan=dict(model='/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct',
              theta=500000.,window=4096,scale=4.,tables=tables,cases=cases,
              source='Observed OLMo 180-row conflicts, shortest successful output first.',
              scope='Task calibration and execution diagnosis only; these cases cannot estimate generalization.',
              initial_arm='b4wide',generation='stock generation_config.json; greedy; original row max_new_tokens')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(path=str(args.out),cases=[{k:c[k] for k in ('row_id','role','max_new_tokens')} for c in cases],
                          plan_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
