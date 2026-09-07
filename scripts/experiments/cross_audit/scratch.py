"""E1 existing-checkpoint overlays, using the unchanged checkpoint owner's loader.

The legacy repository is an explicit read-only dependency, never a fallback
training implementation. NLL on the exposed 32 anchors is a paired diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from .contracts import sha_file, write_json
from .runtime import cuda_runtime, versions
from .tables import tensor_sha, transform


def owner_module(repo):
    sys.path.append(str(Path(repo).resolve()))
    return importlib.import_module('rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment')


MODEL_SHA='9851f86ea0056be976feefd21b8dfb57c7555d932f8f238fa7aa4072fd39661f'


def load_saved(owner,directory):
    """New evaluation loader: validate checkpoint/sidecar/config rather than
    demand that new evaluator/tests equal the historical training bundle hash.
    The architecture source is separately pinned and Native NLL is replayed.
    """
    meta=json.loads((directory/'train_meta.json').read_text())
    saved=np.load(directory/'inv_freq.npy',allow_pickle=False)
    payload=torch.load(directory/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    if not isinstance(payload.get('metadata'),dict) or not isinstance(payload.get('model'),dict):
        raise ValueError('checkpoint lacks state/provenance')
    for key,value in payload['metadata'].items():
        if meta.get(key)!=value:raise ValueError('embedded checkpoint metadata differs from sidecar')
    state=payload['model']
    for key,value in state.items():
        if key.endswith('inv_freq') and not np.array_equal(value.float().numpy(),saved):
            raise ValueError('checkpoint/sidecar rotary mismatch')
    # mmap + direct CUDA construction avoids a duplicate CPU model allocation.
    with torch.device('cuda'):
        model=owner.GPT(meta['model_config'],torch.from_numpy(saved.copy()).to('cuda'))
    model.load_state_dict(state,strict=True)
    if sum(p.numel() for p in model.parameters())!=meta['parameter_count']:
        raise ValueError('scratch parameter-count mismatch')
    return model,meta


def preflight(args):
    owner=owner_module(args.legacy_repo)
    model_path=Path(args.legacy_repo)/'experiments/native_rope_evq_150m/model.py'
    if sha_file(model_path)!=MODEL_SHA:raise ValueError('scratch architecture source drift')
    root=Path(args.root); manifest=json.loads((root/'data/data_manifest.json').read_text())
    for name in ('validation','anchors'):
        entry=manifest[name]
        if sha_file(entry['path'])!=entry['sha256']:raise ValueError(f'{name} hash drift')
    found=[];missing=[]
    for seed in (137,256):
        for arm in ('fmrope_base256','anchored_cosh_tau4_fmrope_range'):
            p=root/f'seed_{seed}/runs/{arm}'
            if not (p/'model.pt').is_file():missing.append([seed,arm]);continue
            meta=json.loads((p/'train_meta.json').read_text())
            if sha_file(p/'model.pt')!=meta['checkpoint_sha256']:raise ValueError('scratch weight drift')
            table=np.load(p/'inv_freq.npy',allow_pickle=False)
            if tensor_sha(table)!=meta['training_inv_freq_sha256']:raise ValueError('scratch table drift')
            found.append(dict(seed=seed,arm=arm,checkpoint_sha256=meta['checkpoint_sha256'],
                              historical_training_code_sha256=meta['code_sha256']))
    return dict(found=found,missing=missing,owner_code_sha256=owner.code_fingerprint(),
                anchors_sha256=manifest['anchors']['sha256'],status='ASSET_PREFLIGHT_ONLY',
                model_source_sha256=MODEL_SHA,
                code_note='current aggregate includes changed evaluator/tests; architecture pinned; Native NLL replay required')


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    assets=preflight(args);write_json(out/'assets.json',assets)
    available={(r['seed'],r['arm']) for r in assets['found']}
    if any((s,a) not in available for s in args.seeds for a in ('fmrope_base256','anchored_cosh_tau4_fmrope_range')):
        raise ValueError('requested seed lacks complete primary checkpoint pair')
    hardware=cuda_runtime();owner=owner_module(args.legacy_repo)
    manifest=json.loads((Path(args.root)/'data/data_manifest.json').read_text())
    validation=np.load(manifest['validation']['path'],mmap_mode='r')
    anchors=np.load(manifest['anchors']['path'],allow_pickle=False)
    start=time.monotonic();records=[]
    with (out/'existing_checkpoint_overlay.csv').open('x') as f:
        writer=None
        for seed in args.seeds:
            historical=json.loads((Path(args.root)/f'seed_{seed}/evaluation/results.json').read_text())
            native_records={(r['arm'],r['length'],r['anchor']):r for r in historical['records']
                if r['condition'] in ('fixed_train_base','fixed_train_range')}
            for arm in ('fmrope_base256','anchored_cosh_tau4_fmrope_range'):
                model,meta=load_saved(owner,Path(args.root)/f'seed_{seed}/runs/{arm}')
                model=model.to('cuda').eval()
                source=owner._model_inv_freq(model).detach().cpu().float().numpy()
                for policy in ('fixed_s4','target_s'):
                    for method in ('identity','yarn','mrpro'):
                        if policy=='target_s' and method=='identity':continue  # identical work already recorded
                        for length in (256,512,1024,2048):
                            factor=1 if method=='identity' else 4 if policy=='fixed_s4' else length/256
                            table,gain,identity=transform(source,dim=64,base=256.,reference_length=256,scale=factor,method=method)
                            owner.set_runtime_rope(model,torch.from_numpy(table),length=length,mscale=gain)
                            inputs,targets=owner.build_eval_windows(validation,anchors,length=length)
                            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                                for i,(x,y) in enumerate(zip(inputs,targets)):
                                    logits=model(torch.tensor(x[None],device='cuda'))
                                    if hasattr(logits,'logits'):logits=logits.logits
                                    nll=F.cross_entropy(logits[0].float(),torch.tensor(y,device='cuda'),reduction='none')
                                    if not torch.isfinite(nll).all():raise RuntimeError('nonfinite scratch NLL')
                                    r=dict(seed=seed,training_arm=arm,policy=policy,method=method,scale=factor,length=length,
                                        anchor=int(anchors[i]),full_nll=float(nll.mean()),tail128_nll=float(nll[-128:].mean()),
                                        prediction_tokens=len(y),table_sha256=tensor_sha(table),amplitude=gain,
                                        checkpoint_sha256=meta['checkpoint_sha256'],
                                        target_sha256=__import__('hashlib').sha256(np.asarray(y,dtype='<i8').tobytes()).hexdigest())
                                    if method=='identity':
                                        old=native_records[(arm,length,int(anchors[i]))]
                                        tail_sha=__import__('hashlib').sha256(np.asarray(y[-128:],dtype='<i8').tobytes()).hexdigest()
                                        if tail_sha!=old['tail_target_sha256']:
                                            raise ValueError('historical anchor target drift')
                                        if max(abs(r['full_nll']-old['full_nll']),abs(r['tail128_nll']-old['tail_nll']))>.01:
                                            raise ValueError('Native replay differs by >0.01 nat; stop overlays and inspect backend/loader')
                                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader()
                                    writer.writerow(r);f.flush();records.append(r)
                            print(json.dumps(dict(seed=seed,arm=arm,policy=policy,method=method,length=length)),flush=True)
                del model;torch.cuda.empty_cache()
    write_json(out/'manifest.json',dict(status='COMPLETE',hardware=hardware,software=versions(),
        rows=len(records),wall_seconds=time.monotonic()-start,csv_sha256=sha_file(out/'existing_checkpoint_overlay.csv'),
        limits='32 historical anchors; training seeds and anchors are separate uncertainty units; no generation capability claim'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('root','legacy-repo','out'):p.add_argument('--'+a,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[137,256]);p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if a.preflight_only:write_json(a.out,preflight(a))
    else:run(a)


if __name__=='__main__':main()
