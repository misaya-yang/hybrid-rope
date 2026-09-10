"""Paired real-document 64K/128K NLL; cache the missing MrPro baseline once."""
import json
import time

import numpy as np
import torch

from .worker import read_rows, save, sha


def run(worker,job):
    prepared=worker.root/'long_inputs'
    manifest=json.loads((prepared/'manifest.json').read_text())
    docs=[];counts={}
    for doc in manifest['docs']:
        dataset=doc['source']['dataset'];counts.setdefault(dataset,0)
        if counts[dataset]<job.get('docs_per_dataset',2):docs.append(doc);counts[dataset]+=1
    folder=worker.root/'long_nll';folder.mkdir(exist_ok=True)
    result={}
    for name in job['methods']:
        spec={'table':worker.tables['MrPro']} if name=='MrPro' else json.loads((worker.root/'results'/name/'contract.json').read_text())['spec']
        contract=folder/(name+'_contract.json')
        if contract.exists() and json.loads(contract.read_text())!=spec:raise ValueError('long-eval method changed')
        save(contract,spec);worker.apply(spec)
        raw=folder/(name+'.jsonl');existing={(r['doc'],r['length']):r for r in read_rows(raw)}
        for length in job.get('lengths',[65536,131072]):
            for doc in docs:
                key=(doc['file'],length)
                if key in existing:
                    if existing[key]['input_sha256']!=doc['sha256']:raise ValueError('long input drift')
                    continue
                if sha(prepared/doc['file'])!=doc['sha256']:raise ValueError('long source tokens changed')
                data=np.load(prepared/doc['file'])
                ids=torch.tensor(data[:length].astype(np.int64),device='cuda')[None]
                target=torch.tensor(data[length-511:length+1].astype(np.int64),device='cuda')
                started=time.monotonic()
                with torch.inference_mode():
                    logits=worker.model(ids,use_cache=False,logits_to_keep=512).logits[0].float()
                    nll=torch.nn.functional.cross_entropy(logits,target,reduction='none')
                r=dict(doc=doc['file'],dataset=doc['source']['dataset'],length=length,input_sha256=doc['sha256'],
                    nll=nll.mean().item(),token_nll=nll.tolist(),target_ids=target.tolist(),elapsed_seconds=time.monotonic()-started)
                with raw.open('a') as stream:stream.write(json.dumps(r)+'\n')
                existing[key]=r
                save(worker.root/'live.json',dict(job=job['id'],phase='long_nll',method=name,doc=doc['file'],length=length))
                del ids,target,logits,nll
        cells={}
        for dataset in counts:
            for length in job.get('lengths',[65536,131072]):
                vals=[r['nll'] for r in existing.values() if r['dataset']==dataset and r['length']==length]
                cells[f'{dataset}/{length}']=dict(n=len(vals),mean_nll=float(np.mean(vals)))
        result[name]=cells
        save(folder/(name+'_summary.json'),dict(status='COMPLETE',cells=cells,raw_sha256=sha(raw),source_sha256=sha(__file__),
            scope='Tail 512 next-token NLL following one real contiguous document prefix; not full-document PPL'))
    return dict(status='COMPLETE',results=result)
