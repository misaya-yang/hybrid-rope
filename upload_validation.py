#!/usr/bin/env python3
import subprocess
import base64

# Read validation script
with open(r'e:\rope\hybrid-rope\scripts\run_anchored_sigmoid_validation.py', 'rb') as f:
    data = base64.b64encode(f.read()).decode()

# Upload
ssh_cmd = [
    r'C:\Users\Admin\.ssh\plink.exe',
    '-batch', '-ssh', '-P', '42581',
    'root@connect.bjb1.seetacloud.com',
    '-pw', 'htG0sD63/yG0',
    f'echo {data} | base64 -d > /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_anchored_sigmoid_validation.py'
]

result = subprocess.run(ssh_cmd, capture_output=True, text=True)
print(f'returncode: {result.returncode}')
print('Upload complete')