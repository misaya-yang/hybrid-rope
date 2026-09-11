"""Freeze a development-selected candidate before opening fresh task scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    root=args.root
    plan=json.loads((root/'olmo_plan.json').read_text())
    cases=plan['cases']
    initial=np.array([json.loads((root/'probe_01/generations'/f'b4wide__{c["row_id"]}.json').read_text())['correct'] for c in cases])
    nu=np.array(plan['tables']['b4wide']['values_float32'])
    radii=np.append(np.minimum(.05,.25/(max(len(c['prompt_ids']) for c in cases)*math.log(4)*nu)),.05)
    eligible=[]
    for run in ('calibration_01','calibration_02'):
        if json.loads((root/run/'status.json').read_text())['status']=='RUNNING':
            raise ValueError('finish development calibration before freezing')
        for path in (root/run).glob('candidate_*.json'):
            candidate=json.loads(path.read_text())
            scores=np.array(candidate['event']['scores'])
            if np.all(scores>=initial) and np.any(scores>initial):
                eligible.append((float(scores.sum()),float(np.linalg.norm(np.array(candidate['parameters'])/radii)),str(path),candidate))
    if not eligible:raise ValueError('no task-Pareto repair to promote')
    # Actual task improvement first; among identical outcomes prefer the least
    # movement, not additional margin gains that have not improved task scores.
    eligible.sort(key=lambda x:(-x[0],x[1],x[2]))
    _,movement,path,candidate=eligible[0]
    gain_history=json.loads((root/'gain_only_01/history.json').read_text())
    gain_step=min((r for r in gain_history if r.get('accepted')),key=lambda r:r['merit'])['iteration']
    gain_path=root/'gain_only_01'/f'candidate_{gain_step:02}.json'
    gain=json.loads(gain_path.read_text())
    tables=plan['tables']
    tables['decision_calibrated']=candidate['table']
    tables['gain_calibrated']=gain['table']
    fresh=root/'fresh_72'
    identity=json.loads((fresh/'identity_check.json').read_text())
    assert identity['new_unique']==72 and identity['overlap_with_old']==0
    rows=[dict(json.loads(line),role='evaluation') for line in (fresh/'screen.jsonl').read_text().splitlines()]
    plan.update(cases=rows,tables=tables,scope='72 fresh unique RULER prompts, equal task weights within length; no calibration on these rows.',
                selection=dict(candidate_path=path,scaled_movement=movement,development_scores=candidate['event']['scores'],
                               rule='Task-Pareto repair, best total development score then minimum scaled movement.',
                               gain_control_path=str(gain_path),fresh_identity=identity))
    out=root/'validation_plan.json'
    if out.exists():raise ValueError('validation plan already frozen')
    out.write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(plan=str(out),plan_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),selection=plan['selection']),indent=2))


if __name__=='__main__':main()
