"""Detached worker with explicit stop, progress and an optional authorized deadline."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .run import atomic, eligible_queue

STOP = False


def stopped(signum, frame):
    global STOP
    STOP = True


def phase_state(path, seconds, resume, now):
    path = Path(path)
    if seconds is not None and seconds <= 0:
        raise ValueError('positive total phase budget required')
    if path.exists():
        if not resume:
            raise FileExistsError('phase exists; explicit resume preserves its original deadline')
        state = json.loads(path.read_text())
        if seconds != state['budget_seconds']:
            raise ValueError('resume cannot reset or expand the phase budget')
        if state['deadline_unix'] is not None and now >= state['deadline_unix']:
            raise TimeoutError('original phase budget exhausted')
        return state
    if resume:
        raise FileNotFoundError('no phase to resume')
    state = dict(started_unix=now,deadline_unix=now+seconds if seconds is not None else None,
                 budget_seconds=seconds,attempts=[],budget_origin='explicit argument' if seconds is not None else 'no wall-time budget supplied; fixed work count and operator stop')
    atomic(path,state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--phase-seconds',type=int,help='Only if the user actually supplies a total wall-time budget')
    p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--detach',action='store_true')
    args = p.parse_args()
    args.prepared, args.out = args.prepared.resolve(), args.out.resolve()
    selected = eligible_queue(json.loads((args.prepared/'queue.json').read_text()))
    if not selected and not args.baseline_only:
        raise SystemExit('No reviewed candidate: launch refused before GPU loading.')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.detach:
        argv = [sys.executable,'-m','scripts.experiments.olmo_fast_screen.supervise',
                '--prepared',str(args.prepared),'--out',str(args.out)]
        if args.phase_seconds is not None:argv.extend(['--phase-seconds',str(args.phase_seconds)])
        if args.resume:argv.append('--resume')
        if args.baseline_only:argv.append('--baseline-only')
        with (args.out/'supervisor.log').open('a') as log:
            child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                     start_new_session=True)
        print(json.dumps({'supervisor_pid':child.pid,'status':'DISPATCHED_VERIFY_PHASE_STATE',
                          'state':str(args.out/'phase.json'),'stop_file':str(args.out/'STOP')}))
        return
    signal.signal(signal.SIGTERM,stopped)
    signal.signal(signal.SIGINT,stopped)
    with (args.out/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = phase_state(args.out/'phase.json',args.phase_seconds,args.resume,time.time())
        if (args.out/'STOP').exists():
            raise SystemExit('STOP remains set; no launch')
        attempt = len(state['attempts'])+1
        argv = [sys.executable,'-m','scripts.experiments.olmo_fast_screen.run',
                '--prepared',str(args.prepared),'--out',str(args.out)]
        if state['deadline_unix'] is not None:argv.extend(['--phase-deadline',str(state['deadline_unix'])])
        if args.baseline_only:argv.append('--baseline-only')
        env = {**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4',
               'OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        with (args.out/f'worker_attempt_{attempt:02d}.log').open('x') as log:
            child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                     start_new_session=True,env=env)
            record = dict(attempt=attempt,pid=child.pid,started_unix=time.time(),status='RUNNING')
            state['attempts'].append(record);atomic(args.out/'phase.json',state)
            reason = None
            while child.poll() is None:
                now = time.time()
                live_path = args.out/'live.json'
                live = json.loads(live_path.read_text()) if live_path.exists() else {}
                if (live.get('expected_seconds') and live.get('start_unix',0)>=record['started_unix']
                    and now-live['start_unix']>live['expected_seconds']
                    and record.get('last_estimate_warning_stage')!=live.get('stage')):
                    record['last_estimate_warning_stage']=live.get('stage')
                    print(json.dumps({'status':'ESTIMATE_EXCEEDED_CONTINUING',
                                      'stage':live.get('stage'),'elapsed_seconds':now-live['start_unix']}),flush=True)
                    atomic(args.out/'phase.json',state)
                if STOP or (args.out/'STOP').exists():reason='STOPPED'
                elif state['deadline_unix'] is not None and now >= state['deadline_unix']:reason='PHASE_TIMEOUT'
                if reason:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        child.wait()
                    break
                try:child.wait(timeout=1)
                except subprocess.TimeoutExpired:pass
            record.update(ended_unix=time.time(),exit_code=child.returncode,
                          status=reason or ('COMPLETE' if child.returncode==0 else 'FAILED'))
            state['elapsed_wall_seconds']=time.time()-state['started_unix']
            atomic(args.out/'phase.json',state)
            print(json.dumps(record))
            if record['status'] != 'COMPLETE':raise SystemExit(1)


if __name__ == '__main__':
    main()
