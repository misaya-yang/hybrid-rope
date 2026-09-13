#!/usr/bin/env python3
"""Download raw public recovery data on the server; never tokenize or load models.

Use --execute for the user-authorized raw downloads. Two network workers keep
memory bounded on the 0.5-core preparation instance. Existing RULER/NIAH source
data are reused in place. Run from the repository root with python -m.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import urllib.request

from experiments.evq_recovery.acquire import (
    LONGALIGN_REV, PG_BUCKET, download, list_books,
)


def save(path, value):
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ruler-upstream', type=Path, required=True)
    parser.add_argument('--hf-endpoint', default='https://hf-mirror.com')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    summary = dict(status='DOWNLOAD_PLAN_ONLY', pg19_books={'train':128,'validation':24,'test':24},
                   long_instruction='zai-org/LongAlign-10k', short_replay='databricks/databricks-dolly-15k',
                   evaluation='QASPER dev/test + existing official RULER/NIAH sources',
                   tokenization=False, model_loading=False, gpu_execution=False)
    if not args.execute:
        print(json.dumps(summary, indent=2)); return
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / 'acquisition.json'
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    metadata_path = root / 'dolly_source.json'
    if metadata_path.exists():
        dolly = json.loads(metadata_path.read_text())
    else:
        url = args.hf_endpoint.rstrip('/') + '/api/datasets/databricks/databricks-dolly-15k'
        request = urllib.request.Request(url, headers={'User-Agent':'RoPE-data-preparation'})
        with urllib.request.urlopen(request, timeout=60) as response:
            info = json.load(response)
        if 'databricks-dolly-15k.jsonl' not in {x['rfilename'] for x in info['siblings']}:
            raise ValueError('Dolly source filename changed')
        dolly = {'repo':'databricks/databricks-dolly-15k','revision':info['sha'],
                 'license':info.get('cardData',{}).get('license'), 'metadata_url':url}
        save(metadata_path, dolly)
    selection_path = root / 'pg19_books.json'
    if selection_path.exists():
        books = json.loads(selection_path.read_text())
    else:
        books = []
        for split, count in [('train',128),('validation',24),('test',24)]:
            books.extend({**row,'split':split} for row in list_books(split,count))
        save(selection_path, books)
    ruler = args.ruler_upstream.resolve()
    ruler_files = ['PaulGrahamEssays.json','english_words.json','hotpotqa.json','squad.json']
    ruler_receipt = []
    for name in ruler_files:
        path = ruler / 'scripts/data/synthetic/json' / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'existing RULER source missing: {path}')
        ruler_receipt.append({'path':str(path),'bytes':path.stat().st_size,'action':'REUSED_IN_PLACE'})
    state = {**summary, 'status':'DOWNLOADING_RAW_SOURCES','started_at':time.time(),
             'dolly_revision':dolly['revision'],'longalign_revision':LONGALIGN_REV,
             'pg19_selection':'first 128/24/24 >=180000-byte books in fixed official train/validation/test listings',
             'ruler':{'revision_label':'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a',
                      'root':str(ruler),'files':ruler_receipt,
                      'niah_note':'NIAH instances are generated from these official sources after resources are available; no tokenization now'},
             'files':[], 'failures':[]}
    save(receipt_path, state)
    jobs = [(PG_BUCKET+b['key'],root/'pg19'/b['key'],b['etag'],b['bytes'],None) for b in books]
    endpoint = args.hf_endpoint.rstrip('/')
    jobs += [
        (f'{endpoint}/datasets/zai-org/LongAlign-10k/resolve/{LONGALIGN_REV}/long.jsonl',root/'longalign.jsonl',None,None,LONGALIGN_REV),
        (f'{endpoint}/datasets/databricks/databricks-dolly-15k/resolve/{dolly["revision"]}/databricks-dolly-15k.jsonl',root/'dolly.jsonl',None,None,dolly['revision']),
        ('https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz',root/'qasper-train-dev.tgz',None,None,None),
        ('https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz',root/'qasper-test.tgz',None,None,None),
    ]
    # Start the instruction/evaluation archives early while downloading the books.
    jobs = jobs[-4:] + jobs[:-4]
    old = {row['path']:row for row in previous.get('files',[])}
    def fetch(job):
        url,path,md5,size,revision = job
        cached = old.get(str(path))
        if cached and path.is_file() and path.stat().st_size == cached['bytes']:
            return {**cached,'action':'REUSED_RECORDED_DOWNLOAD'}
        item = download(url,path,md5=md5,size=size)
        if revision: item['revision'] = revision
        item['action'] = 'DOWNLOADED_RAW'
        return item
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(fetch,job):str(job[1]) for job in jobs}
        for future in as_completed(pending):
            try:
                item = future.result(); state['files'].append(item)
                print(json.dumps({'complete':len(state['files']),'total':len(jobs),'file':item['path'],'bytes':item['bytes']}),flush=True)
            except Exception as error:
                failure = {'path':pending[future],'error':repr(error)}
                state['failures'].append(failure); print(json.dumps(failure),flush=True)
            save(receipt_path,state)
    state['status'] = 'RAW_SOURCES_READY' if not state['failures'] else 'RAW_SOURCES_PARTIAL'
    state['downloaded_or_reused_bytes'] = sum(x['bytes'] for x in state['files'])
    state['finished_at'] = time.time(); save(receipt_path,state)
    print(json.dumps({'status':state['status'],'files':len(state['files']),
                      'bytes':state['downloaded_or_reused_bytes'],'failures':state['failures']}),flush=True)
    if state['failures']: raise SystemExit(1)


if __name__ == '__main__':
    main()
