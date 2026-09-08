"""Resume only uncompleted straggler ranges from the range receipt log."""
import concurrent.futures,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import requests
r=Path(sys.argv[1]);p=r/'model/model.safetensors-00001-of-00001.safetensors';part=Path(str(p)+'.partial')
if p.exists():sys.exit(0)
pid=int((r/'range_download.pid').read_text())
if b'finish_download.py' not in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError('Process identity changed')
os.kill(pid,signal.SIGTERM)
for _ in range(30):
 proc=Path(f'/proc/{pid}')
 if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise RuntimeError('Range downloader still live')
completed={tuple(json.loads(x)['range_complete']) for x in (r/'range_download.log').read_text().splitlines() if 'range_complete' in x}
size=1746942600;initial=469762048;block=64<<20
missing=[(lo,min(lo+block,size)-1) for lo in range(initial,size,block) if (lo,min(lo+block,size)-1) not in completed]
ranges=[(lo,min(lo+(8<<20)-1,hi)) for start,hi in missing for lo in range(start,hi+1,8<<20)]
fd=os.open(part,os.O_RDWR)
url='https://hf-mirror.com/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17/'+p.name
print(json.dumps({'missing_original_ranges':missing,'subranges':len(ranges)}),flush=True)
def fetch(bounds):
 lo,hi=bounds
 for attempt in range(3):
  pos=lo
  try:
   with requests.get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(15,30)) as res:
    if res.status_code!=206 or res.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}':raise ValueError('range mismatch')
    for b in res.iter_content(1<<20):
     if pos+len(b)>hi+1 or os.pwrite(fd,b,pos)!=len(b):raise ValueError('write size')
     pos+=len(b)
   if pos!=hi+1:raise ValueError('short range')
   print(json.dumps({'subrange_complete':[lo,hi]}),flush=True);return
  except Exception:
   if attempt==2:raise
(r/'download_status.json').write_text(json.dumps({'status':'RUNNING','pid':os.getpid(),'missing_ranges':missing}))
with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(fetch,ranges))
os.fsync(fd);os.close(fd)
h=hashlib.sha256()
with part.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
if h.hexdigest()!='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696':raise ValueError('Final weight SHA mismatch')
os.replace(part,p)
subprocess.run(['/root/miniconda3/bin/python','/root/autodl-tmp/download_model.py',str(r/'model'),'/root/autodl-tmp/qwen35_08_metadata.json'],check=True)
