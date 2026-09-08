"""Wait on the known preparation process, then run a discarded real-data probe."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/rotary_budget_20260908')
pid=int((root/'prepare.pid').read_text())
while not (root/'data/manifest.json').exists():
    proc=Path(f'/proc/{pid}')
    if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':
        raise RuntimeError('Preparation ended without READY manifest; inspect prepare.log')
    time.sleep(5)
cmd=['/root/miniconda3/bin/python',str(root/'train_budget.py'),'--original-root','/root/autodl-tmp/hybrid-rope',
     '--data',str(root/'data/manifest.json'),'--output',str(root/'probe_mb8_01'),'--arm','G32','--probe','--micro-batch','8']
env=os.environ.copy();env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
env['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor_cache');env['OMP_NUM_THREADS']='8'
with open(root/'probe_mb8_01.log','a') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    (root/'probe.pid').write_text(str(child.pid))
    (root/'probe_queue.json').write_text(json.dumps({'status':'RUNNING','pid':child.pid,'command':cmd}))
    rc=child.wait()
(root/'probe_queue.json').write_text(json.dumps({'status':'COMPLETE' if rc==0 else 'FAILED','returncode':rc,'pid':child.pid}))
