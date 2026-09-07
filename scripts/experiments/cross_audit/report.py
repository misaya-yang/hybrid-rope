"""Generate a receipt-based operator report; never invent absent experiment results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .contracts import read_rows,sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    rows=[];curves=[];text=['# Cross-audit execution report','','Generated from receipts. Completion is execution status, not scientific validity.','']
    for job in plan['jobs']:
        state=Path(plan['state_dir'])/f'{job["id"]}.json'
        status=json.loads(state.read_text()) if state.exists() else {'status':'NOT_STARTED'}
        receipt=Path(job['output'])/'manifest.json'
        value=json.loads(receipt.read_text()) if receipt.exists() else None
        rows.append(dict(job=job['id'],status=status['status'],manifest_sha256=sha_file(receipt) if value else None,
                         receipt=value,blocked_reason=job.get('blocked_reason')))
        text.append(f'- {job["id"]}: {status["status"]}'+(f' — {job["blocked_reason"]}' if job.get('blocked_reason') else ''))
        steps=Path(job['output'])/'steps.jsonl'
        if steps.exists():
            for r in read_rows(steps):
                curves.append(dict(job=job['id'],step=r['step'],input_tokens=r['cumulative_input_tokens'],
                    prediction_tokens=r['cumulative_prediction_tokens'],cpt_ce=r['cpt_ce'],sft_ce=r['sft_ce'],
                    native_kl=r['native_kl'],step_seconds=r['step_seconds'],save_seconds=r.get('save_seconds',0),
                    peak_allocated_bytes=r['peak_allocated_bytes']))
    write_json(out/'execution_receipts.json',dict(plan_sha256=sha_file(a.plan),jobs=rows))
    if curves:
        with (out/'paired_training_curves.csv').open('x') as f:
            w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    text+=['','## Interpretation limits','',
        'Keep compact/near/far/deleted controls, Native strata, raw generation/EOS and NLL separate. ',
        'Compare equal completed token milestones, not equal step labels unless token receipts match. ',
        'No automatic method ranking, significance claim, checkpoint selection or new launch. ',
        'New scratch overlays use seeds 137/256; historical seed42 results remain separate. ',
        'External confirmation and conditional E4/E5 are not completed evidence.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
