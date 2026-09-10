"""Run the already-pinned upstream RULER generators for a finite recovery panel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .acquire import file_hash
from .data import write_json

TASKS=('niah_single_1','niah_multikey_3','vt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nltk-data',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    receipt=[]
    for split,seed in [('dev',202609101),('test',202609102)]:
        records=[]
        for length in (4096,8192,16384,32768):
            for task in TASKS:
                stage=a.out/'ruler_sources'/f'{split}_{length}_{task}'
                argv=[sys.executable,'-m','scripts.experiments.scale_transport.ruler_full_prepare',
                    '--upstream',str(a.upstream),'--model',str(a.model),'--out',str(stage),
                    '--task',task,'--length',str(length),'--count','24','--seed',str(seed)]
                if not (stage/'manifest.json').exists():
                    # Failed stage artifacts are not silently overwritten.
                    if stage.exists():raise FileExistsError(f'incomplete generator stage needs inspection: {stage}')
                    subprocess.run(argv,check=True,env={**os.environ,'NLTK_DATA':str(a.nltk_data)})
                manifest=json.loads((stage/'manifest.json').read_text())
                if file_hash(stage/'rows.jsonl')!=manifest['rows_sha256']:raise ValueError('RULER input drift')
                for line in (stage/'rows.jsonl').read_text().splitlines():
                    r=json.loads(line)
                    # Upstream multi-key rows do not all fill the nominal cap.
                    # Keep every row and classify by actual prompt+decode reserve.
                    actual_bucket=next(n for n in (4096,8192,16384,32768)
                                       if r['input_tokens']+r['budget']<=n)
                    records.append(dict(id=r['row_id'],source_id=r['row_id'],split=split,task=task,
                        prompt_ids=r['ids'],references=r['references'],generation_budget=r['budget'],
                        input_tokens=r['input_tokens'],length_bucket=actual_bucket,
                        requested_length_cap=length,prompt_sha256=r['prompt_sha256'],
                        ids_sha256=r['ids_sha256'],metric='official answer-item recall, whole-response exact secondary'))
                receipt.append(dict(split=split,length=length,task=task,rows=24,
                                    manifest_sha256=file_hash(stage/'manifest.json')))
        with (a.out/f'ruler_{split}.jsonl').open('w') as f:
            for row in records:f.write(json.dumps(row)+'\n')
    manifest=dict(rows_per_split=288,tasks=TASKS,lengths=[4096,8192,16384,32768],
        upstream='c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a',stages=receipt)
    manifest['files']={}
    for split in ('dev','test'):
        path=a.out/f'ruler_{split}.jsonl'
        manifest['files'][path.name]=dict(bytes=path.stat().st_size,sha256=file_hash(path))
    write_json(a.out/'ruler_manifest.json',manifest)


if __name__=='__main__':main()
