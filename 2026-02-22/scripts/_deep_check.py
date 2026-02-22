import subprocess, time, os, glob

pid_r = subprocess.run(["bash", "-c", "pgrep -f run_llama"], capture_output=True, text=True)
pid = pid_r.stdout.strip()
if not pid:
    print("NO TRAINING PROCESS FOUND!")
    exit(1)

def get_io(p):
    try:
        with open(f"/proc/{p}/io") as f:
            d = {}
            for line in f:
                k, v = line.strip().split(": ")
                d[k] = int(v)
            return d
    except:
        return None

io1 = get_io(pid)
gpu1 = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

print(f"PID={pid}  GPU={gpu1}")
print(f"IO snapshot 1: read={io1.get('read_bytes',0):,}  write={io1.get('write_bytes',0):,}")
print("Waiting 10 seconds...")
time.sleep(10)

io2 = get_io(pid)
gpu2 = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

read_delta = io2.get('read_bytes',0) - io1.get('read_bytes',0)
write_delta = io2.get('write_bytes',0) - io1.get('write_bytes',0)

print(f"IO snapshot 2: read={io2.get('read_bytes',0):,}  write={io2.get('write_bytes',0):,}")
print(f"IO delta (10s): read={read_delta:,} bytes  write={write_delta:,} bytes")
print(f"GPU after: {gpu2}")

d = "/root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h/train_baseline"
ckpts = sorted(glob.glob(d + "/checkpoint-*"))
print(f"Checkpoints: {len(ckpts)}")

if read_delta > 0 or write_delta > 0:
    print("\nVERDICT: ACTIVE - I/O detected, training is computing")
else:
    print("\nVERDICT: POSSIBLY STUCK - No I/O in 10 seconds")
