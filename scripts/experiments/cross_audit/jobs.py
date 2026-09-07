"""Serial exact-command supervisor for a later SSH operator. No automatic retries.

Generate jobs separately; running one requires the frozen plan digest and an
explicit user-authorized job. The flag checks plan identity, not permission.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from .contracts import sha_file


def atomic(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def run(plan_path,job_id,approved_sha):
    path=Path(plan_path)
    if sha_file(path)!=approved_sha:raise ValueError('plan changed after review')
    plan=json.loads(path.read_text());job=next(j for j in plan['jobs'] if j['id']==job_id)
    if job.get('blocked_reason'):raise ValueError(job['blocked_reason'])
    root=Path(plan['state_dir']);root.mkdir(parents=True,exist_ok=True)
    state_path=root/f'{job_id}.json'
    if state_path.exists():raise FileExistsError('job already attempted; no implicit resume/retry')
    for dep in job.get('dependencies',[]):
        dep_path=root/f'{dep}.json'
        if not dep_path.exists() or json.loads(dep_path.read_text())['status']!='COMPLETE':
            raise ValueError(f'incomplete dependency: {dep}')
    for rel,expected in plan['code_files'].items():
        if sha_file(Path(plan['cwd'])/rel)!=expected:raise ValueError(f'code drift: {rel}')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(gpu)!=1:raise ValueError('expected one dedicated GPU')
    name,mib=gpu[0].rsplit(',',1)
    if name.strip()!=plan['gpu_name'] or float(mib)/1024<plan['min_gpu_memory_gib']:
        raise ValueError('GPU differs from reviewed machine/memory contract')
    if not isinstance(job['timeout_seconds'],int) or job['timeout_seconds']<=0:
        raise ValueError('positive hard timeout required')
    with (root/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1',
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
        start=time.monotonic();log=root/f'{job_id}.log'
        with log.open('x') as f:
            child=subprocess.Popen(job['argv'],cwd=plan['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            state=dict(job=job_id,status='RUNNING',pid=child.pid,plan_sha256=approved_sha,
                       argv=job['argv'],start_unix=time.time(),timeout_seconds=job['timeout_seconds'])
            atomic(state_path,state)
            try:
                while child.poll() is None:
                    elapsed=time.monotonic()-start
                    if elapsed>=job['timeout_seconds']:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        state['status']='TIMEOUT';break
                    state.update(elapsed_seconds=elapsed,log_bytes=log.stat().st_size,last_update_unix=time.time())
                    atomic(state_path,state);time.sleep(5)
            except BaseException:
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                state['status']='SUPERVISOR_INTERRUPTED';atomic(state_path,state);raise
            state.update(exit_code=child.returncode,elapsed_seconds=time.monotonic()-start)
            if state['status']=='RUNNING':state['status']='COMPLETE' if child.returncode==0 else 'FAILED'
            atomic(state_path,state)
            if state['status']!='COMPLETE':raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','batch','status'))
    p.add_argument('--plan',required=True);p.add_argument('--job');p.add_argument('--approved-plan-sha')
    p.add_argument('--stage',choices=('E0','E1','E2','E3'))
    p.add_argument('--poweroff-after',action='store_true',help='use ONLY with explicit user shutdown authorization')
    a=p.parse_args()
    if a.mode=='run':
        if not a.job or not a.approved_plan_sha:p.error('run needs --job and --approved-plan-sha')
        run(a.plan,a.job,a.approved_plan_sha)
    elif a.mode=='batch':
        if not a.stage or not a.approved_plan_sha:p.error('batch needs stage and reviewed plan digest')
        plan=json.loads(Path(a.plan).read_text())
        selected=[j for j in plan['jobs'] if j['id'].startswith(a.stage+'_')]
        if not selected or any(j.get('blocked_reason') for j in selected):raise ValueError('stage is empty or contains blocked decisions')
        if sha_file(a.plan)!=a.approved_plan_sha:raise ValueError('plan changed after review')
        try:
            for job in selected:run(a.plan,job['id'],a.approved_plan_sha)
        finally:
            report=Path(plan['state_dir']).parent/f'{a.stage}_report_{int(time.time())}'
            subprocess.run([__import__('sys').executable,'-m','scripts.experiments.cross_audit.report',
                            '--plan',a.plan,'--out',str(report)],cwd=plan['cwd'],check=False)
            if a.poweroff_after:
                shutdown=shutil.which('shutdown')
                if not shutdown:raise RuntimeError('shutdown command unavailable; alert operator')
                subprocess.run([shutdown,'-h','now'],check=True)
    else:
        plan=json.loads(Path(a.plan).read_text());root=Path(plan['state_dir'])
        for job in plan['jobs']:
            path=root/f'{job["id"]}.json'
            state=json.loads(path.read_text()) if path.exists() else dict(job=job['id'],
                status='BLOCKED' if job.get('blocked_reason') else 'NOT_STARTED',reason=job.get('blocked_reason'))
            if state['status']=='RUNNING':
                try:os.kill(state['pid'],0)
                except ProcessLookupError:state['status']='STALE_STATE_PROCESS_ABSENT'
                if time.time()-state.get('last_update_unix',state['start_unix'])>60:
                    state['monitor_warning']='supervisor heartbeat stale; inspect, never auto-restart'
            print(json.dumps(state))


if __name__=='__main__':main()
