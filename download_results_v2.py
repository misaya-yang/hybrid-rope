import subprocess
import os
import base64

# 确保目录存在
os.makedirs('e:/rope/hybrid-rope/results/night_run_anchored_x20_9h', exist_ok=True)

# 下载 results.json
cmd = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "cat /root/autodl-tmp/dfrope/hybrid-rope/results/night_run_anchored_x20_9h/results.json | base64 -w0"'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if result.returncode == 0:
    b64 = result.stdout.strip()
    content = base64.b64decode(b64).decode('utf-8')
    with open('e:/rope/hybrid-rope/results/night_run_anchored_x20_9h/results.json', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Downloaded results.json")
else:
    print(f"Failed: {result.stderr}")

# 下载 summary.md
cmd = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "cat /root/autodl-tmp/dfrope/hybrid-rope/results/night_run_anchored_x20_9h/summary.md | base64 -w0"'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if result.returncode == 0:
    b64 = result.stdout.strip()
    content = base64.b64decode(b64).decode('utf-8')
    with open('e:/rope/hybrid-rope/results/night_run_anchored_x20_9h/summary.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Downloaded summary.md")
else:
    print(f"Failed: {result.stderr}")

# 下载 run.log
cmd = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "cat /root/autodl-tmp/dfrope/hybrid-rope/results/night_run_anchored_x20_9h/run.log | base64 -w0"'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if result.returncode == 0:
    b64 = result.stdout.strip()
    content = base64.b64decode(b64).decode('utf-8')
    with open('e:/rope/hybrid-rope/results/night_run_anchored_x20_9h/run.log', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Downloaded run.log")
else:
    print(f"Failed: {result.stderr}")

print("Done!")