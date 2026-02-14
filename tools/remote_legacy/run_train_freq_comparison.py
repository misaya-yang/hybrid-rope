#!/usr/bin/env python3
"""
远程执行训练频率对比实验
========================
在R6000 (96GB) 上同时运行多个配置的训练

用法:
    python run_train_freq_comparison.py --config all
    python run_train_freq_comparison.py --config baseline
    python run_train_freq_comparison.py --config hybrid
    python run_train_freq_comparison.py --config geometric
"""

import subprocess
import argparse
import time
from datetime import datetime

# SSH配置
SSH_CMD = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# 实验配置 (500步快速对比实验)
EXPERIMENTS = {
    'baseline': {
        'model_size': '350m',
        'freq_type': 'orig',
        'max_steps': 500,
        'seq_length': 8192,
    },
    'hybrid': {
        'model_size': '350m',
        'freq_type': 'hybrid',
        'max_steps': 500,
        'seq_length': 8192,
        'alpha': 0.2,
        'p': 3.9,
        'omf': 0.3,
        'theta_base': 100000,
    },
    'geometric': {
        'model_size': '350m',
        'freq_type': 'geometric',
        'max_steps': 500,
        'seq_length': 8192,
        'theta_base': 100000,
    },
    '500m_baseline': {
        'model_size': '500m',
        'freq_type': 'orig',
        'max_steps': 500,
        'seq_length': 8192,
    },
    '500m_hybrid': {
        'model_size': '500m',
        'freq_type': 'hybrid',
        'max_steps': 500,
        'seq_length': 8192,
        'alpha': 0.2,
        'p': 3.9,
        'omf': 0.3,
        'theta_base': 100000,
    },
}


def upload_script():
    """上传训练脚本到服务器"""
    print("Uploading training script...")
    
    local_path = "scripts/run_train_freq_comparison.py"
    remote_path = "/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_train_freq_comparison.py"
    
    # 使用pscp上传
    pscp_cmd = f'pscp -P 42581 -pw htG0sD63/yG0 {local_path} root@connect.bjb1.seetacloud.com:{remote_path}'
    
    result = subprocess.run(pscp_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Upload failed: {result.stderr}")
        return False
    
    print("Upload successful!")
    return True


def build_command(exp_name, config):
    """构建训练命令"""
    cmd = f"cd /root/autodl-tmp/dfrope/hybrid-rope && "
    cmd += f"nohup python scripts/run_train_freq_comparison.py "
    
    for key, value in config.items():
        if isinstance(value, float):
            cmd += f"--{key} {value} "
        else:
            cmd += f"--{key} {value} "
    
    cmd += f"> logs/train_{exp_name}.log 2>&1 &"
    
    return cmd


def run_experiment(exp_name):
    """运行单个实验"""
    if exp_name not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_name}")
        return False
    
    config = EXPERIMENTS[exp_name]
    print(f"\n{'='*60}")
    print(f"Starting experiment: {exp_name}")
    print(f"Config: {config}")
    print(f"{'='*60}")
    
    cmd = build_command(exp_name, config)
    full_cmd = f'{SSH_CMD} "{cmd}"'
    
    print(f"Command: {cmd}")
    
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to start experiment: {result.stderr}")
        return False
    
    print(f"Experiment {exp_name} started!")
    return True


def run_all_experiments():
    """运行所有实验"""
    print("\n" + "="*60)
    print("TRAINING FREQUENCY COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    # 上传脚本
    if not upload_script():
        print("Failed to upload script, aborting.")
        return
    
    # 确保日志目录存在
    subprocess.run(f'{SSH_CMD} "mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/logs"', shell=True)
    
    # 运行实验
    results = {}
    for exp_name in ['baseline', 'hybrid', 'geometric']:
        results[exp_name] = run_experiment(exp_name)
        time.sleep(2)  # 等待启动
    
    # 打印状态
    print("\n" + "="*60)
    print("EXPERIMENT STATUS")
    print("="*60)
    for exp_name, success in results.items():
        status = "✓ Started" if success else "✗ Failed"
        print(f"  {exp_name}: {status}")
    
    print("\nTo check logs:")
    print(f'  {SSH_CMD} "tail -f /root/autodl-tmp/dfrope/hybrid-rope/logs/train_baseline.log"')
    
    print("\nTo check GPU usage:")
    print(f'  {SSH_CMD} "nvidia-smi"')


def check_status():
    """检查实验状态"""
    cmd = f'{SSH_CMD} "ps aux | grep run_train_freq"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("Running processes:")
    print(result.stdout)
    
    cmd = f'{SSH_CMD} "tail -20 /root/autodl-tmp/dfrope/hybrid-rope/logs/train_*.log"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("\nRecent logs:")
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="远程执行训练频率对比实验")
    parser.add_argument("--config", type=str, default="all",
                        choices=["all", "baseline", "hybrid", "geometric", "500m_baseline", "500m_hybrid", "status"],
                        help="要运行的实验配置")
    
    args = parser.parse_args()
    
    if args.config == "status":
        check_status()
    elif args.config == "all":
        run_all_experiments()
    else:
        upload_script()
        run_experiment(args.config)


if __name__ == "__main__":
    main()