#!/usr/bin/env python3
import subprocess
from pathlib import Path

PLINK = r'C:\Users\Admin\.ssh\plink.exe'
HOST = 'root@connect.bjb1.seetacloud.com'
PORT = '42581'
PW = 'htG0sD63/yG0'

local_dir = Path('e:/rope/hybrid-rope/results/anchored_sigmoid_v3_followup')
local_dir.mkdir(parents=True, exist_ok=True)

files = [
    ('exp1_robustness/results.json', 'exp1_results.json'),
    ('exp2_theta_substitution/results.json', 'exp2_results.json'),
    ('exp3_anchor_ablation/results.json', 'exp3_results.json'),
    ('summary.md', 'summary.md'),
]

for remote_name, local_name in files:
    remote_path = f'/root/autodl-tmp/dfrope/hybrid-rope/results/anchored_sigmoid_v3_followup/{remote_name}'
    local_path = local_dir / local_name
    
    cmd = [PLINK, '-batch', '-ssh', '-P', PORT, HOST, '-pw', PW, f'cat {remote_path}']
    result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
    
    if result.stdout and not 'No such' in result.stderr:
        with open(local_path, 'w', encoding='utf-8') as fp:
            fp.write(result.stdout)
        print(f'Downloaded: {local_name}')
    else:
        print(f'Failed: {remote_name}')
        if result.stderr:
            print(f'  Error: {result.stderr[:100]}')

print('\nDone!')