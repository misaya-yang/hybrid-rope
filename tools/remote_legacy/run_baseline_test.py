#!/usr/bin/env python3
"""上传并运行Baseline Passkey测试"""

import subprocess
import base64

PLINK = r"C:\Users\Admin\.ssh\plink.exe"
SSH_CMD = f'{PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# 简单的Passkey测试脚本
SCRIPT = r'''#!/usr/bin/env python3
"""
Baseline Passkey测试 - 确定模型的长度边界
"""

import os
import sys
import json
import time
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/baseline_passkey")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULT_DIR / "run.log"

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def generate_passkey_prompt(n_garbage, passkey, seed=42):
    """生成passkey检索prompt"""
    random.seed(seed)
    
    templates = [
        "The {} is located in the {} region.",
        "Many {} are found near the {}.",
        "The {} system includes various {}.",
        "Historical {} indicates {} presence.",
        "Scientific {} reveals {} patterns.",
    ]
    
    words = [
        ("forest", "northern"), ("river", "southern"), ("mountain", "eastern"),
        ("lake", "western"), ("valley", "central"), ("desert", "coastal"),
        ("island", "tropical"), ("peninsula", "arctic"), ("plateau", "temperate"),
    ]
    
    lines = []
    for i in range(n_garbage):
        template = random.choice(templates)
        word1, word2 = random.choice(words)
        lines.append(template.format(word1, word2))
    
    passkey_str = f"The passkey is {passkey}. Remember this."
    insert_pos = random.randint(len(lines) // 2, len(lines) - 1)
    lines.insert(insert_pos, passkey_str)
    
    context = " ".join(lines)
    prompt = f"{context}\n\nWhat is the passkey? The passkey is"
    
    return prompt, passkey

def evaluate_passkey(model, tokenizer, device, seq_len, n_tests=3):
    """评估passkey检索"""
    model.eval()
    correct = 0
    total_loss = 0.0
    
    for i in range(n_tests):
        n_garbage = max(10, seq_len // 15)
        passkey = random.randint(10000, 99999)
        prompt, _ = generate_passkey_prompt(n_garbage, passkey, seed=i*1000+seq_len)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
        actual_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            try:
                outputs = model(**inputs, labels=inputs["input_ids"])
                if outputs.loss is not None and not torch.isnan(outputs.loss):
                    total_loss += outputs.loss.item()
                
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated = tokenizer.decode(gen_outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                if str(passkey) in generated:
                    correct += 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return None, None, None, "OOM"
                log(f"[ERROR] {e}")
                return None, None, None, "ERROR"
    
    accuracy = correct / n_tests
    avg_loss = total_loss / n_tests if n_tests > 0 else 0
    ppl = math.exp(min(avg_loss, 10)) if avg_loss > 0 else 0
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    return accuracy, ppl, mem, "ok"

def main():
    log("=" * 60)
    log("Baseline Passkey测试")
    log("=" * 60)
    
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    # 测试长度 - 从短到长
    test_lengths = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
    
    log("[MODEL] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    log("[MODEL] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    log("[MODEL] Model loaded successfully!")
    
    results = {}
    start_time = time.time()
    
    for i, length in enumerate(test_lengths):
        log(f"\n[TEST] L={length} ({i+1}/{len(test_lengths)})")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        acc, ppl, mem, status = evaluate_passkey(model, tokenizer, device, length)
        
        if acc is not None:
            results[str(length)] = {
                "accuracy": round(acc, 3),
                "ppl": round(ppl, 3),
                "mem_gb": round(mem, 2),
                "status": status
            }
            log(f"[RESULT] Acc={acc:.1%}, PPL={ppl:.2f}, Mem={mem:.2f}GB")
        else:
            results[str(length)] = {"status": status}
            log(f"[RESULT] {status}")
        
        # 保存中间结果
        with open(RESULT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    total_time = (time.time() - start_time) / 60
    log(f"\n[TOTAL] {total_time:.1f} minutes")
    log("[DONE] 测试完成!")

if __name__ == "__main__":
    main()
'''

def run_ssh(cmd, timeout=60):
    full_cmd = f'{SSH_CMD} "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode

def main():
    print("=" * 50)
    print("启动Baseline Passkey测试")
    print("=" * 50)
    
    # 1. 创建目录
    print("\n[1] 创建目录...")
    run_ssh("mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/results/baseline_passkey")
    
    # 2. 上传脚本
    print("\n[2] 上传脚本...")
    script_b64 = base64.b64encode(SCRIPT.encode()).decode()
    chunk_size = 4000
    chunks = [script_b64[i:i+chunk_size] for i in range(0, len(script_b64), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            cmd = f"echo '{chunk}' > /tmp/script.b64"
        else:
            cmd = f"echo '{chunk}' >> /tmp/script.b64"
        run_ssh(cmd)
    print(f"  Uploaded {len(chunks)} chunks")
    
    # 3. 解码并启动
    print("\n[3] 解码并启动测试...")
    run_ssh("base64 -d /tmp/script.b64 > /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_baseline_passkey.py")
    
    run_ssh("cd /root/autodl-tmp/dfrope/hybrid-rope && nohup /root/miniconda3/bin/python scripts/run_baseline_passkey.py > results/baseline_passkey/run.log 2>&1 &", timeout=10)
    
    print("\n[4] 测试已启动!")
    print("\n检查进度:")
    print(f'  {PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -50 /root/autodl-tmp/dfrope/hybrid-rope/results/baseline_passkey/run.log"')

if __name__ == "__main__":
    main()