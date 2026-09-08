"""Pinned official Qwen3.5-0.8B download with Git/LFS identity checks."""
import hashlib,json,os,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
root=Path(sys.argv[1]);meta=json.loads(Path(sys.argv[2]).read_text());root.mkdir(parents=True,exist_ok=True)
revision=meta['sha'];repo='Qwen/Qwen3.5-0.8B'
files=[x for x in meta['siblings'] if x['rfilename'].endswith(('.json','.jinja','.txt','.safetensors'))]
def fetch(item):
    name=item['rfilename'];p=root/name
    if not p.exists():
        url=f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}'
        with requests.get(url,stream=True,timeout=(30,120)) as r:
            r.raise_for_status()
            with open(str(p)+'.partial','wb') as f:
                for b in r.iter_content(8<<20):f.write(b)
        os.replace(str(p)+'.partial',p)
    h=hashlib.sha256();git=hashlib.sha1(f'blob {p.stat().st_size}\0'.encode())
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b);git.update(b)
    if item.get('lfs'):
        if h.hexdigest()!=item['lfs']['sha256']:raise ValueError(f'LFS mismatch {name}')
    elif git.hexdigest()!=item['blobId']:raise ValueError(f'Git blob mismatch {name}')
    print(json.dumps({'file':name,'status':'VERIFIED'}),flush=True)
    return name,h.hexdigest()
(root.parent/'download_status.json').write_text(json.dumps({'status':'RUNNING','pid':os.getpid(),'revision':revision}))
try:
    with ThreadPoolExecutor(3) as pool:hashes=dict(pool.map(fetch,files))
    (root.parent/'download_status.json').write_text(json.dumps({'status':'COMPLETE','revision':revision,'repo':repo,'files':hashes},indent=2))
except Exception as e:
    (root.parent/'download_status.json').write_text(json.dumps({'status':'FAILED','error':str(e)}));raise
