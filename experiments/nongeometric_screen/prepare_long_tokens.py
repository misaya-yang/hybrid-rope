"""Tokenize previously selected real documents with the pinned Qwen tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main(source,out,model):
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'sources.json').read_text())
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    kept=[];counts={'pg19':0,'proofpile':0};skipped=[]
    for doc in manifest['docs']:
        if counts[doc['dataset']]>=8:continue
        raw=(source/doc['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=doc['sha256']:raise ValueError('source changed')
        ids=tok.encode(raw.decode('utf-8',errors='ignore'),add_special_tokens=False)
        if len(ids)<131073:
            skipped.append(dict(file=doc['file'],tokens=len(ids)));continue
        name=doc['file'][:-4]+'.npy'
        np.save(out/name,np.asarray(ids[:131073],dtype=np.int32),allow_pickle=False)
        kept.append(dict(source=doc,file=name,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),source_tokens=len(ids)))
        counts[doc['dataset']]+=1
    if min(counts.values())<2:raise ValueError('need two genuine long documents from each source')
    receipt=dict(status='READY',docs=kept,counts=counts,skipped_short=skipped,lengths=[65536,131072],
        tail_tokens=512,source_manifest_sha256=hashlib.sha256((source/'sources.json').read_bytes()).hexdigest(),
        tokenizer_sha256=hashlib.sha256((Path(model)/'tokenizer.json').read_bytes()).hexdigest(),
        policy='Contiguous prefix of one held-out original document; no repeated or concatenated filler')
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='READY',counts=counts,skipped=len(skipped))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--model',required=True)
    a=p.parse_args();main(a.source,a.out,a.model)
