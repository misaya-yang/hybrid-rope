#!/usr/bin/env python3
"""分块上传脚本"""
import subprocess
import base64
import time

# 读取脚本
with open('scripts/run_llama13b_triangle.py', 'rb') as f:
    data = f.read()

# 分块
chunks = [data[i:i+5000] for i in range(0, len(data), 5000)]
print(f'Total {len(chunks)} chunks')

SSH = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# 清空目标文件
subprocess.run(SSH + ' "rm -f /tmp/script_part.txt"', shell=True)

# 上传每个块
for i, chunk in enumerate(chunks):
    b64 = base64.b64encode(chunk).decode()
    cmd = f'echo {b64} | base64 -d >> /tmp/script_part.txt'
    result = subprocess.run(SSH + ' "' + cmd + '"', shell=True, capture_output=True, text=True)
    print(f'Chunk {i+1}/{len(chunks)}: {len(b64)} bytes')
    time.sleep(0.3)

# 移动到目标位置
subprocess.run(SSH + ' "mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/scripts"', shell=True)
subprocess.run(SSH + ' "mv /tmp/script_part.txt /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_llama13b_triangle.py"', shell=True)

# 验证
verify = subprocess.run(SSH + ' "wc -l /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_llama13b_triangle.py"', shell=True, capture_output=True, text=True)
print('Lines:', verify.stdout)

# 启动实验
print('\nStarting experiment...')
run_cmd = 'cd /root/autodl-tmp/dfrope/hybrid-rope && nohup python scripts/run_llama13b_triangle.py > results/llama13b_triangle_boundary/run.log 2>&1 &'
subprocess.run(SSH + ' "' + run_cmd + '"', shell=True)

# 检查进程
time.sleep(2)
check = subprocess.run(SSH + ' "ps aux | grep run_llama13b_triangle | grep -v grep"', shell=True, capture_output=True, text=True)
print('Process:', check.stdout)

print('\nDone!')