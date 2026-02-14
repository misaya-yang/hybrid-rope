#!/usr/bin/env python3
"""上传comprehensive theta脚本到服务器"""

import subprocess
import sys

# SSH配置
SSH_HOST = "connect.bjb1.seetacloud.com"
SSH_PORT = "42581"
SSH_USER = "root"
SSH_PASS = "htG0sD63/yG0"

# 使用plink
PLINK = r"C:\Users\Admin\.ssh\plink.exe"
PSCP = r"C:\Users\Admin\.ssh\pscp.exe"

def run_ssh(cmd):
    """执行SSH命令"""
    full_cmd = f'{PLINK} -batch -ssh -P {SSH_PORT} {SSH_USER}@{SSH_HOST} -pw {SSH_PASS} "{cmd}"'
    print(f"执行: {cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=120)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def upload_file(local, remote):
    """上传文件"""
    cmd = f'{PSCP} -P {SSH_PORT} -pw {SSH_PASS} "{local}" {SSH_USER}@{SSH_HOST}:{remote}'
    print(f"上传: {local} -> {remote}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def main():
    print("=" * 50)
    print("上传全面Theta搜索脚本")
    print("=" * 50)
    
    # 上传脚本
    local_script = r"E:\rope\hybrid-rope\scripts\run_comprehensive_theta.py"
    remote_script = "/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_comprehensive_theta.py"
    
    if not upload_file(local_script, remote_script):
        print("上传失败!")
        return
    
    print("上传成功!")
    
    # 创建结果目录
    run_ssh("mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/results/comprehensive_theta")
    
    # 启动实验 (后台运行)
    print("\n启动长时间实验...")
    print("实验包含: 17种配置 x 24个长度点 = ~408次评估")
    print("预计时间: 30-60分钟")
    
    # 使用nohup后台运行
    run_ssh(f"cd /root/autodl-tmp/dfrope/hybrid-rope && nohup /root/miniconda3/bin/python scripts/run_comprehensive_theta.py > results/comprehensive_theta/run.log 2>&1 &")
    
    print("\n实验已在后台启动!")
    print("检查进度命令:")
    print(f'  {PLINK} -batch -ssh -P {SSH_PORT} {SSH_USER}@{SSH_HOST} -pw {SSH_PASS} "tail -50 /root/autodl-tmp/dfrope/hybrid-rope/results/comprehensive_theta/run.log"')

if __name__ == "__main__":
    main()