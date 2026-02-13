import subprocess
import base64

script_path = r'e:\rope\hybrid-rope\scripts\run_night_run_9h.py'
with open(script_path, 'r', encoding='utf-8') as f:
    script = f.read()

b64 = base64.b64encode(script.encode('utf-8')).decode('ascii')

# Split into chunks to avoid command line length limits
chunk_size = 4000
chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]

# First, clear the file
cmd = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "rm -f /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_night_run_9h.py"'
subprocess.run(cmd, shell=True)

# Upload each chunk
for i, chunk in enumerate(chunks):
    if i == 0:
        cmd = rf'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "echo {chunk} | base64 -d > /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_night_run_9h.py"'
    else:
        cmd = rf'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "echo {chunk} | base64 -d >> /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_night_run_9h.py"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Chunk {i+1}/{len(chunks)}: {result.returncode}")
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        break

# Verify
cmd = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "head -5 /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_night_run_9h.py"'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("Verification:")
print(result.stdout)