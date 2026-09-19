#!/usr/bin/env python3
"""Run the four explicitly authorized arms in order, stopping on any failure."""
import json
import fcntl
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/mrrope_official_20260919')
OUTPUT = ROOT / 'results/four_arm_fa2'
ARMS = [('llama31', 'mrpro'), ('llama31', 'tailspline'),
        ('llama3', 'mrpro'), ('llama3', 'tailspline')]


def save(value):
    path = OUTPUT / 'queue_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / '.queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = []
    for family, method in ARMS:
        name = family + '_' + method
        checkpoint = 'Llama-3.1-8B-Instruct' if family == 'llama31' else 'Meta-Llama-3-8B-Instruct'
        if family == 'llama3':
            while True:
                status = json.loads((ROOT / 'llama3_download_status.json').read_text())
                if status['status'] == 'COMPLETE':
                    break
                if status['status'] == 'FAILED':
                    raise RuntimeError('Llama3 download failed')
                save({'status': 'WAITING_FOR_LLAMA3_DOWNLOAD', 'completed': completed})
                time.sleep(5)
        report_path = OUTPUT / name / 'report.json'
        if report_path.exists() and json.loads(report_path.read_text())['status'] == 'COMPLETE':
            completed.append(name)
            continue
        command = [sys.executable, '-u', str(Path(__file__).with_name('official_single_arm.py')),
            '--execute', '--family', family, '--method', method,
            '--attention', 'flash_attention_2', '--samples', '10',
            '--model', '/root/autodl-tmp/models/' + checkpoint,
            '--output', str(OUTPUT / name), '--resume']
        environment = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        with (OUTPUT / (name + '.log')).open('ab') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
            save({'status': 'RUNNING', 'active': name, 'pid': process.pid, 'completed': completed,
                  'order': [f + '_' + m for f, m in ARMS], 'command': command})
            print('START', name, process.pid, flush=True)
            code = process.wait()
        if code or not report_path.exists() or json.loads(report_path.read_text())['status'] != 'COMPLETE':
            save({'status': 'FAILED', 'active': name, 'returncode': code, 'completed': completed})
            raise RuntimeError('Arm did not complete: ' + name)
        completed.append(name)
        print('COMPLETE', name, flush=True)
    reports = {name: json.loads((OUTPUT / name / 'report.json').read_text()) for name in completed}
    raw = {name: [json.loads(line) for line in (OUTPUT / name / 'generations.jsonl').read_text().splitlines()]
           for name in completed}
    for rows in raw.values():
        assert len(rows) == 130
    reference = raw[completed[0]]
    for name, rows in raw.items():
        for a, b in zip(reference, rows):
            for key in ['task', 'source_index', 'row_id', 'input_ids_sha256', 'references', 'generate_kwargs', 'seed']:
                assert a[key] == b[key], (name, key)
    summary = {'status': 'COMPLETE', 'paired_inputs_and_decoder_verified': True,
               'rows_per_arm': 130, 'reports': reports}
    (OUTPUT / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    save({'status': 'COMPLETE', 'completed': completed})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__, str(exc), flush=True)
        raise
