"""CPU validation and exact matched token accounting before any GPU purchase."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from .contracts import read_rows,sha_file,write_json
from .runtime import verify_prepared
from .teacher import pool_rows,prediction_positions
from .train import paired_sft


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('prepared','cpt-dir','pool','out'):p.add_argument('--'+a,required=True)
    a=p.parse_args();root=Path(a.prepared);assets=verify_prepared(root)
    cpt=Path(a.cpt_dir);cm=json.loads((cpt/'manifest.json').read_text())
    facts={}
    for split,path in [('train',cpt/'train_2048x16385.npy'),('validation',cpt/'validation.npy')]:
        digest=sha_file(path)
        if digest!=cm[split]['sha256']:raise ValueError(f'{split} CPT differs from original manifest')
        data=np.load(path,mmap_mode='r',allow_pickle=False)
        if list(data.shape)!=cm[split]['shape']:raise ValueError('CPT shape drift')
        if data.min()<0 or data.max()>=100352:raise ValueError('CPT token id outside model vocabulary')
        facts[split]=dict(path=str(path.resolve()),sha256=digest,shape=list(data.shape))
    native,pool=pool_rows(a.pool,assets['model'],'train')
    validation,_=pool_rows(a.pool,assets['model'],'validation')
    if {r['source_id'] for r in native}&{r['source_id'] for r in validation}:
        raise ValueError('Native source overlap')
    counts=Counter();cache_bytes=0
    for r in native:
        ids,positions=prediction_positions(r)
        counts[r['group']]+=1;cache_bytes+=len(positions)*100352*4
        if r['group']!='text' and positions[0]!=len(r['prompt_ids'])-1:
            raise ValueError('Native first answer prediction is off by one')
    pairs=paired_sft(list(read_rows(root/'sft_rows.jsonl')),137)
    steps=[];cumulative=0
    for step,pair in enumerate(pairs,1):
        length=8192 if step%2 else 16384
        tokens=length+sum(len(r['prompt_ids'])+len(r['target_ids'])-1 for r in pair)
        cumulative+=tokens
        steps.append(dict(step=step,input_tokens=tokens,cumulative_input_tokens=cumulative,
                          answer_prediction_tokens=sum(len(r['target_ids']) for r in pair)))
    write_json(a.out,dict(status='CPU_INPUTS_VALIDATED',cpt=facts,
        native_manifest_sha256=sha_file(a.pool),native_train_strata=dict(counts),
        native_train_validation_source_overlap=0,teacher_cache_float32_bytes=cache_bytes,
        sft_pairs=len(pairs),first_sft_cycle_tokens=steps,
        notes=['CPT is the existing 33.55M-token v1 corpus, not a completed 500M v2 corpus',
               'Repeated epochs are exposure, not new unique documents/tokens',
               'Teacher KL retains valid original hidden indices including the post-answer distribution; labeled NLL excludes positions without next-token labels',
               'Per-source provenance from original manifests is preserved; corpus-wide semantic deduplication unverified']))


if __name__=='__main__':main()
