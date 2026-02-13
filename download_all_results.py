import subprocess
import os

# 下载所有今天的结果目录
results_dirs = [
    'night_run_anchored_x20_9h',
    'night_run_9h_extended', 
    'anchored_sigmoid_v3_followup',
    'advisor_followup_2026-02-14'
]

for dir_name in results_dirs:
    local_path = f'e:/rope/hybrid-rope/results/{dir_name}'
    os.makedirs(local_path, exist_ok=True)
    
    # 使用pscp下载
    cmd = f'C:\\Users\\Admin\\.ssh\\pscp.exe -r -P 42581 -pw htG0sD63/yG0 root@connect.bjb1.seetacloud.com:/root/autodl-tmp/dfrope/hybrid-rope/results/{dir_name}/* e:/rope/hybrid-rope/results/{dir_name}/'
    print(f"Downloading {dir_name}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"  Return code: {result.returncode}")
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:200]}")

print("\nDone! Check e:/rope/hybrid-rope/results/")