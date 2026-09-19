import concurrent.futures,json,time,traceback
from pathlib import Path
import requests
ROOT=Path('/root/autodl-tmp/mrrope_official_20260919')
DEST=Path('/root/autodl-tmp/models/Llama-3.1-8B-Instruct'); DEST.mkdir(parents=True,exist_ok=True)
API='https://modelscope.cn/api/v1/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/repo'
status=ROOT/'model_download_status.json'
def state(value):
 p=status.with_suffix('.tmp'); p.write_text(json.dumps(value,indent=2)+'\n'); p.replace(status)
def fetch(f):
 name=f['Path']; target=DEST/name; temp=target.with_suffix(target.suffix+'.partial'); size=f['Size']
 if target.exists() and target.stat().st_size==size: return name
 for attempt in range(6):
  try:
   start=temp.stat().st_size if temp.exists() else 0
   if start==size: temp.replace(target); return name
   headers={'Range':f'bytes={start}-'} if start else {}
   with requests.get(API,params={'Revision':f['Revision'],'FilePath':name},headers=headers,stream=True,timeout=(20,90)) as r:
    r.raise_for_status()
    if start and r.status_code!=206: start=0
    if r.status_code==206 and not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):
     raise RuntimeError('Unexpected response range')
    with temp.open('ab' if start else 'wb') as out:
     for chunk in r.iter_content(4*1024*1024):
      if chunk: out.write(chunk)
   if temp.stat().st_size!=size: raise RuntimeError('Incomplete file')
   temp.replace(target); print('COMPLETE',name,size,flush=True); return name
  except Exception as e:
   print('RETRY',name,attempt,type(e).__name__,flush=True); time.sleep(min(3*(attempt+1),15))
 raise RuntimeError('Download failed: '+name)
try:
 r=requests.get(API+'/files',params={'Revision':'master','Recursive':'true'},timeout=30); r.raise_for_status()
 files=[f for f in r.json()['Data']['Files'] if '/' not in f['Path'] and (f['Path'].endswith(('.json','.safetensors')) or f['Path'] in {'README.md','LICENSE','USE_POLICY.md'})]
 files.sort(key=lambda f:f['Size'])
 plan={'status':'DOWNLOADING','source':'ModelScope LLM-Research/Meta-Llama-3.1-8B-Instruct','destination':str(DEST),'files':[{k:f[k] for k in ('Path','Size','Revision')} for f in files],'total_bytes':sum(f['Size'] for f in files)}
 state(plan); print('START',plan['total_bytes'],flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch,files))
 config=json.loads((DEST/'config.json').read_text())
 assert config['max_position_embeddings']==131072 and config['hidden_size']==4096 and config['num_hidden_layers']==32
 index=json.loads((DEST/'model.safetensors.index.json').read_text())
 assert all((DEST/p).is_file() for p in set(index['weight_map'].values()))
 plan.update(status='COMPLETE',config_native_length=config['max_position_embeddings']); state(plan); print('ALL_COMPLETE',flush=True)
except BaseException as e:
 state({'status':'FAILED','error':type(e).__name__+': '+str(e)}); traceback.print_exc(); raise
