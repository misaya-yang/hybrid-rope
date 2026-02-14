#!/usr/bin/env python3
"""通过SSH启动长时间运行的Passkey检索实验"""

import subprocess
import base64

PLINK = r"C:\Users\Admin\.ssh\plink.exe"
SSH_CMD = f'{PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# Passkey检索实验 - 确保长时间运行
SCRIPT = r'''#!/usr/bin/env python3
"""
长时间Passkey检索实验
测试模型在不同长度下的检索能力
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

RESULT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/passkey_long")
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
    
    # 垃圾文本模板
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
    
    # 插入passkey
    passkey_str = f"The passkey is {passkey}. Remember this."
    insert_pos = random.randint(len(lines) // 2, len(lines) - 1)
    lines.insert(insert_pos, passkey_str)
    
    # 构建prompt
    context = " ".join(lines)
    prompt = f"{context}\n\nWhat is the passkey? The passkey is"
    
    return prompt, passkey

def test_passkey_retrieval(model, tokenizer, device, seq_len, n_tests=5):
    """测试passkey检索能力"""
    model.eval()
    correct = 0
    total_loss = 0.0
    
    for i in range(n_tests):
        # 估算需要的垃圾行数
        n_garbage = seq_len // 10
        
        # 生成随机passkey
        passkey = random.randint(10000, 99999)
        
        prompt, _ = generate_passkey_prompt(n_garbage, passkey, seed=i*1000+seq_len)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            try:
                # 计算loss
                outputs = model(**inputs, labels=inputs["input_ids"])
                if outputs.loss is not None and not torch.isnan(outputs.loss):
                    total_loss += outputs.loss.item()
                
                # 生成回答
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated = tokenizer.decode(gen_outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                # 检查是否包含正确passkey
                if str(passkey) in generated:
                    correct += 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return None, None, None, "OOM"
                raise
    
    accuracy = correct / n_tests
    avg_loss = total_loss / n_tests if n_tests > 0 else 0
    ppl = math.exp(avg_loss) if avg_loss > 0 else 0
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    return accuracy, ppl, mem, "ok"

def main():
    log("=" * 60)
    log("长时间Passkey检索实验")
    log("=" * 60)
    
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    # 测试长度 - 从短到长
    test_lengths = [
        1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
        10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576,
        26624, 28672, 30720, 32768, 36864, 40960, 45056, 49152,
        53248, 57344, 61440, 65536, 69632, 73728, 77824, 81920
    ]
    
    log("[MODEL] Loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    log(f"[MODEL] Loaded. Test lengths: {len(test_lengths)}")
    log(f"[PLAN] Est. time: 2-3 hours")
    
    all_results = {}
    start_time = time.time()
    
    for i, length in enumerate(test_lengths):
        log(f"\n{'#'*60}")
        log(f"[PROGRESS] {i+1}/{len(test_lengths)}: L={length}")
        if i > 0:
            elapsed = (time.time() - start_time) / 60
            remaining = (len(test_lengths) - i) * (elapsed / i)
            log(f"[TIME] Elapsed: {elapsed:.1f}min, ETA: {remaining:.1f}min")
        log(f"{'#'*60}")
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            accuracy, ppl, mem, status = test_passkey_retrieval(model, tokenizer, device, length)
            
            if status == "OOM":
                log(f"[RESULT] L={length}: OOM")
                all_results[str(length)] = {"status": "OOM"}
                # 继续尝试下一个长度（可能短的可以）
                continue
            
            all_results[str(length)] = {
                "accuracy": round(accuracy, 3) if accuracy else None,
                "ppl": round(ppl, 3) if ppl else None,
                "mem_gb": round(mem, 2) if mem else None,
                "status": status
            }
            
            log(f"[RESULT] L={length}: Acc={accuracy:.2%}, PPL={ppl:.2f}, Mem={mem:.2f}GB")
            
            # 保存中间结果
            with open(RESULT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
                
        except Exception as e:
            log(f"[ERROR] L={length}: {e}")
            all_results[str(length)] = {"status": "error", "message": str(e)}
            import traceback
            log(traceback.format_exc())
    
    log("\n" + "=" * 60)
    log("完成!")
    
    total_time = (time.time() - start_time) / 60
    log(f"[TOTAL] {total_time:.1f} minutes")
    
    # 报告
    lines = ["# Passkey检索实验报告", "", f"时间: {datetime.now()}", "", "## 结果", ""]
    lines.append("| 长度 | 准确率 | PPL | 显存(GB) |")
    lines.append("|------|--------|-----|----------|")
    
    for length_str, data in sorted(all_results.items(), key=lambda x: int(x[0])):
        acc = data.get("accuracy")
        ppl = data.get("ppl")
        mem = data.get("mem_gb")
        status = data.get("status", "unknown")
        
        if acc is not None:
            lines.append(f"| {length_str} | {acc:.1%} | {ppl:.2f} | {mem:.2f} |")
        else:
            lines.append(f"| {length_str} | - | - | {status} |")
    
    with open(RESULT_DIR / "report.md", "w") as f:
        f.write("\n".join(lines))
    
    log(f"[DONE] 结果: {RESULT_DIR}")

if __name__ == "__main__":
    main()
'''

def run_ssh(cmd, timeout=60):
    full_cmd = f'{SSH_CMD} "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode

def main():
    print("=" * 50)
    print("启动长时间Passkey检索实验")
    print("=" * 50)
    
    # 1. 创建目录
    print("\n[1] 创建目录...")
    run_ssh("mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/results/passkey_long")
    
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
    
    # 3. 解码
    print("\n[3] 解码脚本...")
    run_ssh("base64 -d /tmp/script.b64 > /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_passkey_long.py && chmod +x /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_passkey_long.py")
    
    # 4. 启动
    print("\n[4] 启动长时间实验...")
    print("  - 32个长度点 (1k到80k)")
    print("  - 每个长度5次passkey检索测试")
    print("  - 预计时间: 2-3小时")
    
    run_ssh("cd /root/autodl-tmp/dfrope/hybrid-rope && nohup /root/miniconda3/bin/python scripts/run_passkey_long.py > results/passkey_long/run.log 2>&1 &", timeout=10)
    
    print("\n[5] 实验已启动!")
    print("\n检查进度:")
    print(f'  {PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -50 /root/autodl-tmp/dfrope/hybrid-rope/results/passkey_long/run.log"')

if __name__ == "__main__":
    main()