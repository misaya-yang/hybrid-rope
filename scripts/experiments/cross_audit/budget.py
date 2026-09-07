"""Cost a common training budget from completed probes, without launching it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .contracts import read_rows,sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probes',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--per-run-hours',type=float,default=5.)
    a=p.parse_args();facts={};bounds=[]
    if not 0<a.per_run_hours<=5:raise ValueError('common per-run allowance must fit the reserved five-hour repeat budget')
    for name in a.probes:
        root=Path(name);m=json.loads((root/'manifest.json').read_text())
        if m['status']!='COMPLETE' or m['mode']!='DISCARDABLE_COST_PROBE' or m['regime']!='full':
            raise ValueError('require completed full-parameter cost probes')
        rows=list(read_rows(root/'steps.jsonl'))
        if len(rows)!=4:raise ValueError('incomplete four-update probe')
        # Cover both physical lengths; do not use only the faster shape.
        seconds=max(r['step_seconds'] for r in rows[2:])*1.25
        save=max(r.get('save_seconds',0) for r in rows)
        bound=int((a.per_run_hours*3600-4*save)/seconds)//4*4
        if bound<4:raise ValueError('no complete matched budget fits')
        facts[m['arm']]=dict(conservative_seconds_per_update=seconds,
            measured_final_save_seconds=save,max_steps=bound,probe_sha256=sha_file(root/'manifest.json'))
        bounds.append(bound)
    if set(facts) not in ({'Z'},{'YaRN','MrPro','Z'}):
        raise ValueError('require own Z probe, or the preserved historical three-arm probes')
    write_json(a.out,dict(status='BUDGET_PROPOSAL_NOT_AUTHORIZATION',full_steps=min(bounds),
        per_run_hours=a.per_run_hours,probes=facts,
        limits='Four steps give a conservative starting estimate, not guaranteed sustained throughput; hard wall limits still apply. Freeze paired token counts and source order before launch.'))


if __name__=='__main__':main()
