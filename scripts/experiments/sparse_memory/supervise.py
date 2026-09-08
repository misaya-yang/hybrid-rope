"""Detached serial jobs under one absolute phase deadline and exclusive lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .run import atomic

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    a = p.parse_args()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    path = Path(a.plan); plan = json.loads(path.read_text())
    root = Path(plan['root']); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for rel, expected in plan['code'].items():
        if hashlib.sha256((Path(plan['cwd'])/rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('frozen code drift: '+rel)
    for job in plan['jobs']:
        if STOP: raise SystemExit(2)
        receipt = root/(job['id']+'.json')
        if receipt.exists(): raise FileExistsError('attempt already has a receipt')
        remaining = plan['deadline']-time.time()
        if remaining < 60: raise RuntimeError('phase deadline, no new job')
        with (root/(job['id']+'.log')).open('x') as log:
            proc = subprocess.Popen(job['argv'], cwd=plan['cwd'], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'1',
                     'PYTHONUNBUFFERED':'1'})
            state = dict(id=job['id'], pid=proc.pid, start_unix=time.time(),
                phase_deadline=plan['deadline'], status='RUNNING', argv=job['argv'],
                plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            atomic(receipt, state)
            stop_at = time.monotonic()+min(remaining, job['timeout'])
            while proc.poll() is None and not STOP and time.monotonic() < stop_at:
                try: proc.wait(timeout=min(1.0, max(.01, stop_at-time.monotonic())))
                except subprocess.TimeoutExpired: pass
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                state['status'] = 'INTERRUPTED' if STOP else 'TIMEOUT'
            state.update(end_unix=time.time(), exit_code=proc.returncode)
            if state['status'] == 'RUNNING':
                state['status'] = 'COMPLETE' if proc.returncode == 0 else 'FAILED'
            atomic(receipt, state)
            if state['status'] != 'COMPLETE': raise SystemExit(1)


if __name__ == '__main__':
    main()
