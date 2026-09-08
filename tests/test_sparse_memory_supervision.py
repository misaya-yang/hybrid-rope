"""A real operator interruption must reap the child and not launch the next job."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def test_supervisor_operator_stop_reaps_child_and_stops_queue(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    started = tmp_path/'started'
    forbidden = tmp_path/'second_started'
    script = 'from pathlib import Path; import time; Path('+repr(str(started))+').write_text("yes"); time.sleep(30)'
    second = 'from pathlib import Path; Path('+repr(str(forbidden))+').write_text("bad")'
    plan = dict(root=str(tmp_path/'state'), cwd=str(repo), deadline=time.time()+120, code={}, jobs=[
        dict(id='first', argv=[sys.executable,'-c',script], timeout=40),
        dict(id='second', argv=[sys.executable,'-c',second], timeout=5)])
    path=tmp_path/'plan.json'; path.write_text(json.dumps(plan))
    proc=subprocess.Popen([sys.executable,'-m','scripts.experiments.sparse_memory.supervise','--plan',str(path)], cwd=repo)
    try:
        end=time.monotonic()+15
        while not started.exists() and time.monotonic()<end:
            if proc.poll() is not None: raise AssertionError('supervisor exited before child')
            time.sleep(.05)
        assert started.exists()
        proc.send_signal(signal.SIGTERM); proc.wait(timeout=5)
        state=json.loads((tmp_path/'state/first.json').read_text())
        assert state['status']=='INTERRUPTED' and state['exit_code'] != 0
        assert not forbidden.exists()
        try: os.kill(state['pid'],0)
        except ProcessLookupError: pass
        else: raise AssertionError('child survived supervisor interruption')
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()
