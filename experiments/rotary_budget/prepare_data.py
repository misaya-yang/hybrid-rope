"""Rebuild the original exact token stream and document-preserving validation."""
import argparse, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
import pyarrow.parquet as pq
from tokenizers import Tokenizer

REV='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
SHARDS={'000':'b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871',
        '004':'33557ddd87a07a4ae6fcaf7a4789c7b484e5cc0c273ca12a65b74200e6d8748b'}
TOKENS=499974144

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def get(root,key):
    p=root/f'{key}_00000.parquet'
    if p.exists():
        if sha(p)!=SHARDS[key]: raise ValueError(f'Existing shard hash mismatch: {p}')
        return p
    url=f'https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/{REV}/sample/10BT/{key}_00000.parquet'
    for attempt in range(4):
        try:
            with requests.get(url,stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                with open(str(p)+'.partial','wb') as f:
                    for b in r.iter_content(8<<20):f.write(b)
            if sha(str(p)+'.partial')!=SHARDS[key]:raise ValueError('Download hash mismatch')
            os.replace(str(p)+'.partial',p)
            print(json.dumps({'downloaded':key}),flush=True)
            return p
        except Exception:
            if attempt==3:raise
            time.sleep(2)

def batches(path,tok):
    offset=0
    for b in pq.ParquetFile(path).iter_batches(batch_size=1024,columns=['text']):
        texts=b.column(0).to_pylist()
        encoded=tok.encode_batch(texts,add_special_tokens=False)
        yield offset,encoded
        offset+=len(encoded)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    if (a.root/'manifest.json').exists():raise FileExistsError('Prepared manifest already exists')
    with ThreadPoolExecutor(2) as pool:paths=list(pool.map(lambda k:get(a.root,k),SHARDS))
    tokenizer_file=a.tokenizer/'tokenizer.json'
    if sha(tokenizer_file)!='c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624':
        raise ValueError('Tokenizer identity changed')
    tok=Tokenizer.from_file(str(tokenizer_file))
    target=a.root/'train.npy';partial=a.root/'train.partial.npy'
    out=np.lib.format.open_memmap(partial,mode='w+',dtype=np.uint16,shape=(TOKENS//2048,2048)).reshape(-1)
    count=0;h=hashlib.sha256()
    for offset,encoded in batches(paths[0],tok):
        for row in encoded:
            ids=np.asarray(row.ids[:TOKENS-count],dtype=np.int64)
            if ids.size and (ids.min()<0 or ids.max()>=50304):raise ValueError('Token outside vocabulary')
            out[count:count+len(ids)]=ids
            h.update(ids.astype('<i8').tobytes());count+=len(ids)
            if count==TOKENS:break
        if count==TOKENS:break
    if count!=TOKENS:raise ValueError('Source too short')
    if h.hexdigest()!='66ee82396750d2c2fe9ab0a678092383a46ad28290983d091bd83895d5f83e60':
        raise ValueError('Rebuilt token prefix differs from original exact-range stream')
    out.flush();del out;os.replace(partial,target)
    docs=[];row_ids=[];content_hashes=set()
    for offset,encoded in batches(paths[1],tok):
        for i,row in enumerate(encoded):
            if len(row.ids)<8193:continue
            ids=np.asarray(row.ids[:8193],dtype=np.uint16)
            digest=hashlib.sha256(ids.tobytes()).hexdigest()
            if digest in content_hashes:continue
            content_hashes.add(digest);docs.append(ids);row_ids.append(offset+i)
            if len(docs)==512:break
        if len(docs)==512:break
    if len(docs)!=512:raise ValueError(f'Only {len(docs)} eligible unique long documents')
    np.save(a.root/'validation.npy',np.stack(docs))
    manifest={'status':'READY','data_revision':REV,'train':{'path':str(target),'tokens':TOKENS,
      'sha256':sha(target),'semantic_int64_prefix_sha256':h.hexdigest(),'storage':'uint16_lossless','packing':'no separators'},
      'validation':{'path':str(a.root/'validation.npy'),'sha256':sha(a.root/'validation.npy'),'document_ids':row_ids,
      'count':512,'anchor_end':8193,'source_shard':'004','derangement':list(range(1,512))+[0]},
      'tokenizer_sha256':sha(tokenizer_file),'source_shards':SHARDS}
    (a.root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'READY','documents':512}),flush=True)
if __name__=='__main__':main()
