"""Finish the slow single weight file with validated parallel HTTP ranges."""
import concurrent.futures,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import requests
root=Path(sys.argv[1]);pid=int((root/'download.pid').read_text())
name='model.safetensors-00001-of-00001.safetensors';size=1746942600
expected='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
p=root/'model'/name;part=Path(str(p)+'.partial')
if p.exists():raise RuntimeError('Weight already complete; use original verifier')
cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
if b'download_model.py' not in cmd:raise RuntimeError('Downloader process identity mismatch')
os.kill(pid,signal.SIGTERM)
for _ in range(20):
    proc=Path(f'/proc/{pid}')
    if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
    time.sleep(.1)
else:raise RuntimeError('Downloader did not stop')
offset=part.stat().st_size
fd=os.open(part,os.O_RDWR);os.ftruncate(fd,size)
url='https://hf-mirror.com/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17/'+name
chunk=64<<20
ranges=[(start,min(start+chunk,size)-1) for start in range(offset,size,chunk)]
def fetch(bounds):
    lo,hi=bounds
    for attempt in range(3):
        pos=lo
        try:
            with requests.get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(20,90)) as r:
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}':
                    raise ValueError('Incorrect range response')
                for b in r.iter_content(1<<20):
                    if pos+len(b)>hi+1:raise ValueError('Range overflow')
                    written=os.pwrite(fd,b,pos)
                    if written!=len(b):raise IOError('Short positional write')
                    pos+=len(b)
            if pos!=hi+1:raise ValueError('Short range')
            print(json.dumps({'range_complete':[lo,hi]}),flush=True)
            return
        except Exception:
            if attempt==2:raise
            time.sleep(1)
(root/'download_status.json').write_text(json.dumps({'status':'RUNNING','pid':os.getpid(),'resumed_at_byte':offset,'workers':8}))
try:
    with concurrent.futures.ThreadPoolExecutor(8) as ex:list(ex.map(fetch,ranges))
    os.fsync(fd);os.close(fd)
    h=hashlib.sha256()
    with part.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    if h.hexdigest()!=expected:raise ValueError('Final weight SHA mismatch')
    os.replace(part,p)
    subprocess.run(['/root/miniconda3/bin/python','/root/autodl-tmp/download_model.py',str(root/'model'),'/root/autodl-tmp/qwen35_08_metadata.json'],check=True)
except Exception as e:
    (root/'download_status.json').write_text(json.dumps({'status':'FAILED','error':str(e),'resumed_at_byte':offset}));raise
