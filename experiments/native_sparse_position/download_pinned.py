"""Resumable pinned public model download, bounded ranges and final Git/LFS hashes."""
import argparse,concurrent.futures,hashlib,json,os,threading,time
from pathlib import Path
import requests

def digest(path,algorithm='sha256',prefix=b''):
    h=hashlib.new(algorithm);h.update(prefix)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    meta=json.loads(a.metadata.read_text());revision=meta['sha'];status=a.output.parent/'download_status.json'
    base={'repo':a.repo,'revision':revision,'pid':os.getpid(),'metadata_sha256':digest(a.metadata)}
    status.write_text(json.dumps({**base,'status':'RUNNING'}));verified={};start=time.monotonic()
    try:
        for item in meta['siblings']:
            name=item['rfilename']
            if not name.endswith(('.json','.py','.model','.safetensors','.jinja','.txt')):continue
            target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);size=item['size']
            url=f'https://hf-mirror.com/{a.repo}/resolve/{revision}/{name}'
            if not target.exists():
                partial=target.with_name(target.name+'.partial');receipt=target.with_name(target.name+'.ranges.jsonl')
                if size<(8<<20):
                    r=requests.get(url,timeout=(20,60));r.raise_for_status()
                    if len(r.content)!=size:raise ValueError(f'Short file {name}')
                    partial.write_bytes(r.content)
                else:
                    completed={tuple(json.loads(x)['range']) for x in receipt.read_text().splitlines()} if receipt.exists() else set()
                    if completed and (not partial.exists() or partial.stat().st_size!=size):raise ValueError('Resume file identity missing')
                    fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,size);lock=threading.Lock()
                    ranges=[(lo,min(lo+(8<<20),size)-1) for lo in range(0,size,8<<20)]
                    todo=[b for b in ranges if b not in completed]
                    def fetch(bounds):
                        lo,hi=bounds
                        for attempt in range(4):
                            try:
                                with requests.get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(15,40)) as r:
                                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}':raise ValueError('Range identity mismatch')
                                    pos=lo
                                    for b in r.iter_content(1<<20):
                                        if pos+len(b)>hi+1 or os.pwrite(fd,b,pos)!=len(b):raise ValueError('Range write mismatch')
                                        pos+=len(b)
                                    if pos!=hi+1:raise ValueError('Short range')
                                with lock:
                                    with receipt.open('a') as f:f.write(json.dumps({'range':bounds})+'\n')
                                    completed.add(bounds)
                                    if len(completed)%32==0:print(json.dumps({'file':name,'ranges_complete':len(completed),'total_ranges':len(ranges),'seconds':time.monotonic()-start}),flush=True)
                                return
                            except Exception:
                                if attempt==3:raise
                    try:
                        with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:list(pool.map(fetch,todo))
                        os.fsync(fd)
                    finally:os.close(fd)
                actual=digest(partial)
                if item.get('lfs'):
                    if actual!=item['lfs']['sha256']:raise ValueError(f'LFS mismatch {name}')
                elif digest(partial,'sha1',f'blob {size}\0'.encode())!=item['blobId']:raise ValueError(f'Git blob mismatch {name}')
                os.replace(partial,target)
            actual=digest(target)
            if target.stat().st_size!=size:raise ValueError('Existing size mismatch')
            if item.get('lfs'):
                if actual!=item['lfs']['sha256']:raise ValueError('Existing LFS mismatch')
            elif digest(target,'sha1',f'blob {size}\0'.encode())!=item['blobId']:raise ValueError('Existing Git blob mismatch')
            verified[name]=actual;print(json.dumps({'file':name,'status':'VERIFIED'}),flush=True)
            status.write_text(json.dumps({**base,'status':'RUNNING','verified_files':verified},indent=2))
        status.write_text(json.dumps({**base,'status':'COMPLETE','files':verified,'seconds':time.monotonic()-start},indent=2))
    except Exception as e:
        status.write_text(json.dumps({**base,'status':'FAILED','error':str(e),'verified_files':verified},indent=2));raise
if __name__=='__main__':main()
