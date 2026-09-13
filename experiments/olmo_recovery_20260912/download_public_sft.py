#!/usr/bin/env python3
"""Download public training sources on the server, with resume and no SHA scans."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

SPECS = [
 ('Yukang/LongAlpaca-12k','46dce924ed8786979556018e191c0f557d8f4aa2','LongAlpaca-12k.json','longalpaca.json'),
 ('HuggingFaceH4/ultrachat_200k','8049631c405ae6576f93f445c6b8166f76f5505a','data/train_sft-00000-of-00003-a3ecf92756993583.parquet','ultrachat_train_sft_0.parquet'),
 ('HuggingFaceH4/ultrachat_200k','8049631c405ae6576f93f445c6b8166f76f5505a','data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet','ultrachat_train_sft_1.parquet'),
 ('HuggingFaceH4/ultrachat_200k','8049631c405ae6576f93f445c6b8166f76f5505a','data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet','ultrachat_train_sft_2.parquet'),
 ('zai-org/LongCite-45k','main','long.jsonl','longcite.jsonl'),
]


def save(path, value):
 temporary=path.with_name(path.name+'.incomplete')
 temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def fetch(root,spec,endpoint):
 repo,revision,filename,local=spec;path=root/local;partial=root/(local+'.part')
 if path.is_file() and path.stat().st_size:
  return dict(repo=repo,revision=revision,path=str(path),bytes=path.stat().st_size,status='REUSED_USER_ATTESTED')
 if shutil.which('aria2c'):
  url=f'{endpoint}/datasets/{repo}/resolve/{revision}/{filename}?download=true'
  incoming=local+'.aria-download';log=root/(local+'.transfer.log')
  command=['aria2c','--continue=true','--split=8','--max-connection-per-server=8',
           '--min-split-size=4M','--file-allocation=none','--check-integrity=false',
           '--auto-file-renaming=false','--allow-overwrite=true','--max-tries=8',
           '--retry-wait=3','--connect-timeout=15','--timeout=30','--summary-interval=30',
           '--console-log-level=warn','--user-agent=Mozilla/5.0','--dir='+str(root),
           '--out='+incoming,url]
  print(json.dumps({'file':local,'transport':'aria2c','connections':8}),flush=True)
  with log.open('ab') as stream:
   subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
  target=root/incoming
  if not target.is_file() or not target.stat().st_size:raise RuntimeError('empty aria2 download')
  target.replace(path)
  return dict(repo=repo,revision=revision,path=str(path),bytes=path.stat().st_size,status='DOWNLOADED',url=url,transport='aria2c')
 # Query parameters keep public redirect caches from serving expired download URLs.
 url=f'{endpoint}/datasets/{repo}/resolve/{revision}/{filename}?download=true'
 errors=[]
 for attempt in range(6):
  offset=partial.stat().st_size if partial.exists() else 0
  headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity'}
  if offset:headers['Range']=f'bytes={offset}-'
  try:
   req=urllib.request.Request(url+f'&attempt={int(time.time())}',headers=headers)
   with urllib.request.urlopen(req,timeout=60) as response:
    content_range=response.headers.get('Content-Range')
    if offset and response.status==206 and content_range and content_range.startswith(f'bytes {offset}-'):
     mode='ab';total=int(content_range.rsplit('/',1)[1])
    elif response.status==200:
     mode='wb';offset=0;total=int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
    else:raise RuntimeError('unexpected partial download response')
    if total and shutil.disk_usage(root).free < total-offset+2_000_000_000:
     raise RuntimeError('insufficient free disk for remaining download')
    received=offset;next_log=received+128*1024*1024
    with partial.open(mode) as stream:
     while True:
      block=response.read(1024*1024)
      if not block:break
      stream.write(block);received+=len(block)
      if received>=next_log:
       print(json.dumps({'file':local,'bytes':received,'total':total}),flush=True);next_log=received+128*1024*1024
   if total and partial.stat().st_size!=total:raise RuntimeError('download ended before Content-Length')
   if not partial.stat().st_size:raise RuntimeError('empty download')
   partial.replace(path)
   return dict(repo=repo,revision=revision,path=str(path),bytes=path.stat().st_size,status='DOWNLOADED',url=url)
  except Exception as error:
   errors.append(str(error));print(json.dumps({'file':local,'attempt':attempt+1,'error':str(error)}),flush=True)
   if attempt<5:time.sleep(min(10,2**attempt))
 raise RuntimeError('; '.join(errors))


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
 parser.add_argument('--endpoint',default='https://hf-mirror.com');parser.add_argument('--execute',action='store_true')
 args=parser.parse_args()
 if not args.execute:
  print(json.dumps({'status':'PLAN_ONLY','files':[s[3] for s in SPECS],'training_splits_only':True,'sha_validation':False}));return
 args.root.mkdir(parents=True,exist_ok=True)
 state={'status':'DOWNLOADING','asset_identity_policy':'user_attested_clone/no_sha_validation','files':[],'failures':[]}
 save(args.root/'public_sources.json',state)
 with ThreadPoolExecutor(max_workers=3) as pool:
  futures={pool.submit(fetch,args.root,spec,args.endpoint):spec[3] for spec in SPECS}
  for future in as_completed(futures):
   try:result=future.result();state['files'].append(result);print(json.dumps(result),flush=True)
   except Exception as error:state['failures'].append({'file':futures[future],'error':str(error)})
   save(args.root/'public_sources.json',state)
 state['status']='RAW_PUBLIC_SOURCES_READY' if not state['failures'] else 'PARTIAL'
 state['bytes']=sum(r['bytes'] for r in state['files']);save(args.root/'public_sources.json',state)
 print(json.dumps(state),flush=True)
 if state['failures']:raise SystemExit(1)


if __name__=='__main__':main()
