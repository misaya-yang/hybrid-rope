"""Collect scoped phase receipts without exporting checkpoints or other projects."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();root=Path(args.root)
    runs=[]
    for identity_path in sorted((root/'runs').glob('*/identity.json')):
        identity=json.loads(identity_path.read_text());run_dir=identity_path.parent
        result_path=run_dir/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {'status':'RUNNING_OR_INTERRUPTED'}
        runs.append(dict(name=run_dir.name,arm=identity['args']['arm'],seed=identity['args']['seed'],
            training={k:identity['args'][k] for k in ('steps','batch','lr','eval_every','log_every','device') if k in identity['args']},
            software={'torch':identity['torch'],'gpu':identity['gpu']},
            model=identity['model'],initial_state_sha256=identity['initial_state_sha256'],
            train=identity['train'],development=identity['development'],code=identity['code'],
            result=result))
    evaluations={}
    for path in sorted(root.glob('*/result.json')):
        evaluations[path.parent.name]=json.loads(path.read_text())
    jobs=[]
    for path in sorted((root/'supervision').glob('*.json')):
        row=json.loads(path.read_text())
        # Retain evidence identities, omit private command/path fields in summary.
        jobs.append({k:row[k] for k in ('id','status','start_unix','end_unix','exit_code','plan_sha256') if k in row})
    total=sum(r['result'].get('wall_seconds',0) for r in runs)
    summary=dict(status='COLLECTED',runs=runs,evaluations=evaluations,jobs=jobs,
        recorded_training_wall_seconds=total,
        supervised_job_wall_seconds=sum(j.get('end_unix',j.get('start_unix',0))-j.get('start_unix',0) for j in jobs),
        limits='training wall time excludes preparation/probes/evaluation/cloud billing; each model/data regime remains separate')
    summary_path=Path(args.out);summary_path.write_text(json.dumps(summary,indent=2))
    archive=summary_path.with_suffix('.tar.gz')
    allowed=[]
    for folder in ['runs','supervision']:
        allowed.extend(f for f in (root/folder).rglob('*') if f.is_file() and f.suffix in ('.json','.jsonl','.log'))
    for folder in root.iterdir():
        if folder.is_dir() and folder.name.startswith(('eval_','test_','query_')):
            allowed.extend(f for f in folder.rglob('*') if f.is_file() and f.suffix in ('.json','.jsonl'))
    allowed.extend(f for f in root.iterdir() if f.is_file() and f.suffix in ('.json','.log'))
    for folder in ['data_v1','data_selective']:
        for name in ['manifest.json','development.json','test.json']:
            f=root/folder/name
            if f.exists(): allowed.append(f)
    with tarfile.open(archive,'w:gz') as tar:
        for f in sorted(set(allowed)):
            if f.resolve() == summary_path.resolve(): continue
            tar.add(f,arcname=str(f.relative_to(root)))
    print(json.dumps(dict(runs=len(runs),training_wall_seconds=total,receipt_archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(set(allowed)))))


if __name__ == '__main__': main()
