#!/usr/bin/env python3
"""上传并运行LLaMA-13B三角对照实验"""
import subprocess
import sys

SSH_CMD = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# 读取脚本
with open('scripts/run_llama13b_triangle.py', 'r', encoding='utf-8') as f:
    script = f.read()

# 转义单引号
script_escaped = script.replace("'", "'\"'\"'")

# 上传命令
upload_cmd = f"mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/scripts && echo '{script_escaped}' > /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_llama13b_triangle.py"

print("Uploading script...")
result = subprocess.run(f'{SSH_CMD} "{upload_cmd}"', shell=True, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# 验证上传
verify_cmd = "ls -la /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_llama13b_triangle.py"
result = subprocess.run(f'{SSH_CMD} "{verify_cmd}"', shell=True, capture_output=True, text=True)
print(result.stdout)

# 启动实验
print("\nStarting experiment...")
run_cmd = "cd /root/autodl-tmp/dfrope/hybrid-rope && nohup python scripts/run_llama13b_triangle.py > results/llama13b_triangle_boundary/run.log 2>&1 &"
result = subprocess.run(f'{SSH_CMD} "{run_cmd}"', shell=True, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# 检查进程
check_cmd = "sleep 3 && ps aux | grep run_llama13b_triangle | grep -v grep"
result = subprocess.run(f'{SSH_CMD} "{check_cmd}"', shell=True, capture_output=True, text=True)
print("Process check:", result.stdout)

print("\nDone! Experiment started.")