"""Read actual prepared arrays/rows; no model weights are loaded into memory."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .acquire import file_hash
from .data import write_json
from .tables import digest_array


def validate(root,verify_model=False):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    manifest=json.loads((root/'data/data_manifest.json').read_text())
    if manifest['status']!='CPU_DATA_READY' or 'sft_source_screen' not in manifest:
        raise ValueError('preparation/source-overlap screen incomplete')
    required=['cpt_train.npy','lm_validation.npy','lm_test.npy','ruler_manifest.json']
    required += [f'{kind}_{split}.jsonl' for kind in ('sft','native','qa','ruler') for split in ('dev','test')]
    required += ['sft_train.jsonl','native_train.jsonl']
    for name in required:
        entry=manifest['files'][name]
        if file_hash(root/'data'/name)!=entry['sha256']:raise ValueError(f'input hash changed: {name}')
    training=np.load(root/'data/cpt_train.npy',mmap_mode='r')
    if training.dtype!=np.int32 or training.shape[1]!=16385:raise ValueError('wrong CPT array')
    counts={};source_sets=defaultdict(set)
    for name in required:
        if not name.endswith('.jsonl'):continue
        local_ids=set();cells=Counter();lengths=[];targets=0
        with (root/'data'/name).open() as f:
            for line in f:
                row=json.loads(line)
                if row['id'] in local_ids:raise ValueError(f'duplicate row ID: {name}')
                local_ids.add(row['id'])
                source_sets[name].add(row['source_id'])
                ids=row.get('input_ids',row.get('prompt_ids'))
                if not ids or min(ids)<0 or max(ids)>=100352:raise ValueError('token outside model vocabulary')
                lengths.append(len(ids));cells[str(row.get('length_bucket',row.get('task','unknown')))]+=1
                if name.startswith(('sft_','native_')):
                    start=row['target_start']
                    if not 0<start<len(ids):raise ValueError('invalid supervised mask')
                    if row.get('task')!='text' and ids[-1]!=manifest['eos_token_id']:
                        raise ValueError('missing native assistant EOS')
                    if name.startswith('sft_') and (len(ids)>16384 or row['prompt_tokens']<=4096):
                        raise ValueError('SFT is not intact OOD-length data')
                    targets+=len(ids)-start
        if not local_ids:raise ValueError(f'empty prepared split: {name}')
        counts[name]=dict(rows=len(local_ids),source_groups=len(source_sets[name]),cells=dict(cells),
                          min_max_tokens=[min(lengths),max(lengths)],supervised_targets=targets)
    for kind in ('sft','native'):
        if source_sets[f'{kind}_train.jsonl']&(source_sets[f'{kind}_dev.jsonl']|source_sets[f'{kind}_test.jsonl']):
            raise ValueError(f'{kind} source split overlap')
    for kind in ('sft','native','qa','ruler'):
        if source_sets[f'{kind}_dev.jsonl']&source_sets[f'{kind}_test.jsonl']:
            raise ValueError(f'{kind} dev/test source overlap')
    tables=json.loads((root/'tables.json').read_text())
    for name,t in tables.items():
        x=np.asarray(t['values'],dtype=np.float32)
        if digest_array(x)!=t['sha256'] or not np.all(x[:-1]>x[1:]):raise ValueError(f'{name}: grid drift')
    model_check={'path':plan['model_path'],'weight_sha256_verified':False}
    if verify_model:
        weight=Path(plan['model_path'])/'model.safetensors'
        if file_hash(weight)!=plan['model_weight_sha256']:raise ValueError('cached model weights do not match selected revision')
        model_check.update(weight_sha256_verified=True,bytes=weight.stat().st_size)
    packages={}
    for package in ('torch','transformers','peft','numpy'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    return dict(status='CPU_PREPARATION_VERIFIED_GPU_EXECUTION_PENDING',model=model_check,
                input_files_verified=len(required),data=counts,cpt_windows=len(training),
                cpt_unique_prediction_tokens=int(len(training)*16384),packages=packages,
                GPU_training_executed=False,GPU_memory_and_speed_verified=False,
                limitation='Actual GPU smoke, baseline capability and trained outcomes remain unmeasured.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--verify-model',action='store_true')
    a=p.parse_args();result=validate(a.root,a.verify_model)
    write_json(a.root/'readiness.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
