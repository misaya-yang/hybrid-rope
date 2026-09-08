"""Verify and summarize complete cross-checkpoint BM/MrPro paired runs."""
import argparse
import json
from pathlib import Path

from scripts.analysis.summarize_olmo_fast_screen import load_arm, compare, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--models',nargs='+',default=['qwen3'])
    args=p.parse_args();root=args.root
    report=dict(status='COMPLETE_PAIRED_ARMS_ONLY',models={},
        scope='User-selected Qwen2.5-3B-Instruct, six-task RULER subset, fixed S4 formula. Not the full 13-task published score.')
    prepared=[]
    for key in args.models:
        folder=root/f'prepared_{key}_01'
        manifest=json.loads((folder/'manifest.json').read_text())
        prepared.append(manifest)
        for name,h in manifest['prepared_files'].items():
            if sha(folder/name)!=h:raise ValueError('input preparation drift: '+key+'/'+name)
        run=f'run_{key}_01'
        if not all((root/run/(name+'.json')).exists() for name in ('MrPro','MrProBM')):continue
        phase=json.loads((root/run/'phase.json').read_text())
        if phase['attempts'][-1]['status']!='COMPLETE':raise ValueError('phase not complete')
        baseline,b=load_arm(root,run,'MrPro');candidate,c=load_arm(root,run,'MrProBM')
        inputs={r['row_id']:r for r in map(json.loads,(folder/'screen.jsonl').read_text().splitlines())}
        for rows in (baseline,candidate):
            if [r['row_id'] for r in rows]!=manifest['row_order']:raise ValueError('run input order differs')
            for r in rows:
                if r['prompt_sha256']!=inputs[r['row_id']]['prompt_sha256']:raise ValueError('prompt differs')
        report['models'][key]=dict(model_id=manifest['model_id'],revision=manifest['revision'],
            parameters=manifest['actual_parameters'],weight_files_sha256=manifest['weight_files_sha256'],
            static_scale=manifest['static_scale'],native_length=manifest['native_length'],
            prepared_manifest_sha256=sha(folder/'manifest.json'),prompt_collection_sha256=manifest['prompt_collection_sha256'],
            runtime=json.loads((root/run/'runtime.json').read_text()),
            phase_seconds=phase['elapsed_wall_seconds'],arms=dict(MrPro=b,MrProBM=c),
            comparison=compare(candidate,baseline))
    report['shared_prompt_collection']=len({m['prompt_collection_sha256'] for m in prepared})==1
    report['shared_decoding']=len({m['prepared_files']['generation_config.json'] for m in prepared})==1
    if not report['shared_prompt_collection'] or not report['shared_decoding']:
        raise ValueError('cross-checkpoint input/decoder contract differs')
    report['completed_models']=len(report['models'])
    cancelled=root/'run_qwen15_01/phase.json'
    if cancelled.exists():
        state=json.loads(cancelled.read_text())
        if state['attempts'][-1]['status']=='STOPPED':
            raw=root/'run_qwen15_01/MrPro.jsonl'
            report['user_cancelled_1p5b']=dict(status='STOPPED',complete_comparison=False,
                completed_baseline_rows=len(raw.read_text().splitlines()) if raw.exists() else 0,
                phase_seconds=state['elapsed_wall_seconds'],reason='User switched directly to the official-paper 3B checkpoint.')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    for key,m in report['models'].items():
        print(key,m['comparison']['status'],m['comparison']['macro_delta_by_length'],
              m['comparison']['paired_by_length'])


if __name__=='__main__':main()
