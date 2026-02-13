#!/usr/bin/env python3
import subprocess
import base64
import sys

# Read script
with open(r'e:\rope\hybrid-rope\scripts\run_llama_theta_matched.py', 'rb') as f:
    data = base64.b64encode(f.read()).decode()

# Create command without problematic characters
ssh_cmd = [
    r'C:\Users\Admin\.ssh\plink.exe',
    '-batch', '-ssh', '-P', '42581',
    'root@connect.bjb1.seetacloud.com',
    '-pw', 'htG0sD63/yG0',
    f'echo {data} | base64 -d > /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_llama_theta_matched.py'
]

result = subprocess.run(ssh_cmd, capture_output=True, text=True)
print(f'stdout: {result.stdout}')
print(f'stderr: {result.stderr}')
print(f'returncode: {result.returncode}')
print('Upload complete')