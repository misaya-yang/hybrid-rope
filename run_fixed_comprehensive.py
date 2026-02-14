#!/usr/bin/env python3
"""通过SSH直接创建并运行修复版脚本"""

import subprocess
import base64

PLINK = r"C:\Users\Admin\.ssh\plink.exe"
SSH_CMD = f'{PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# 修复版脚本 - 使用正确的rope_scaling参数
SCRIPT = r'''#!/usr/bin/env python3
"""全面RoPE Scaling实验 - 修复版"""

import os
import sys
import json
import time
import math
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

RESULT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/rope_scaling_v2")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULT_DIR / "run.log"

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def evaluate_model(model, tokenizer, device, seq_len, num_windows=8):
    """评估模型perplexity"""
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    
    if len(tokens) < seq_len * 4:
        tokens = tokens.repeat(math.ceil(seq_len * 4 / len(tokens)) + 1)
    tokens = tokens.to(device)
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(num_windows):
            start = torch.randint(0, len(tokens) - seq_len - 1, (1,)).item()
            input_chunk = tokens[start:start+seq_len].unsqueeze(0)
            
            try:
                outputs = model(input_chunk, labels=input_chunk)
                loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item() * seq_len
                    total_tokens += seq_len
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return None, None, "OOM"
                raise
    
    if total_tokens == 0:
        return None, None, "ERROR"
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    return ppl, avg_loss, mem

def run_experiment(model_path, tokenizer, rope_config, device, test_lengths):
    """运行单个rope_scaling配置实验"""
    config_name = rope_config.get("name", "unknown")
    rope_scaling = rope_config.get("rope_scaling")
    
    results = {"config": rope_config, "data": {}, "boundary": None}
    
    log(f"[LOAD] Loading model with rope_scaling={rope_scaling}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            rope_scaling=rope_scaling,
            trust_remote_code=True
        )
    except Exception as e:
        log(f"[ERROR] Failed to load model: {e}")
        results["error"] = str(e)
        return results
    
    model.eval()
    baseline_ppl = None
    
    for length in test_lengths:
        log(f"[TEST] L={length}")
        start_time = time.time()
        
        try:
            torch.cuda.empty_cache()
            ppl, loss, mem = evaluate_model(model, tokenizer, device, length)
            elapsed = time.time() - start_time
            
            if ppl is None:
                log(f"[RESULT] L={length}: {mem}")
                results["data"][str(length)] = {"status": mem}
                if mem == "OOM":
                    results["oom_at"] = length
                    break
                continue
            
            results["data"][str(length)] = {
                "ppl": round(ppl, 3),
                "loss": round(loss, 4),
                "elapsed_sec": round(elapsed, 2),
                "mem_gb": round(mem, 2),
                "status": "ok"
            }
            
            log(f"[RESULT] L={length}: PPL={ppl:.3f}, time={elapsed:.1f}s, mem={mem:.2f}GB")
            
            if baseline_ppl is None:
                baseline_ppl = ppl
            
            if results["boundary"] is None and (ppl > 5 * baseline_ppl or ppl > 100):
                results["boundary"] = length
                log(f"[BOUNDARY] Found at L={length}")
            
            if ppl > 1000:
                log(f"[SKIP] PPL too high")
                break
                
        except Exception as e:
            log(f"[ERROR] L={length}: {e}")
            results["data"][str(length)] = {"status": "error", "message": str(e)}
            break
    
    del model
    torch.cuda.empty_cache()
    
    return results

def main():
    log("=" * 60)
    log("全面RoPE Scaling实验 - 修复版")
    log("=" * 60)
    
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    # 测试长度 - 更多点
    test_lengths = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                    18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768,
                    36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536]
    
    log("[TOKENIZER] Loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 实验配置 - 使用正确的rope_scaling格式
    experiments = []
    
    # 1. Baseline (无scaling)
    experiments.append({"name": "baseline", "rope_scaling": None})
    
    # 2. Linear scaling
    for factor in [2, 4, 8, 16, 32]:
        experiments.append({
            "name": f"linear_{factor}x",
            "rope_scaling": {"rope_type": "linear", "factor": factor}
        })
    
    # 3. Dynamic NTK
    for factor in [2, 4, 8, 16, 32]:
        experiments.append({
            "name": f"dynamic_{factor}x",
            "rope_scaling": {"rope_type": "dynamic", "factor": factor}
        })
    
    # 4. YaRN
    for factor in [2, 4, 8, 16, 32]:
        experiments.append({
            "name": f"yarn_{factor}x",
            "rope_scaling": {"rope_type": "yarn", "factor": factor}
        })
    
    # 5. Llama3
    for factor in [2, 4, 8, 16, 32]:
        experiments.append({
            "name": f"llama3_{factor}x",
            "rope_scaling": {"rope_type": "llama3", "factor": factor}
        })
    
    total = len(experiments)
    log(f"[PLAN] Total experiments: {total}")
    log(f"[PLAN] Test lengths: {len(test_lengths)} points")
    log(f"[PLAN] Estimated time: 60+ minutes")
    
    all_results = {}
    start_time = time.time()
    
    for i, config in enumerate(experiments):
        log(f"\n{'#'*60}")
        log(f"[PROGRESS] {i+1}/{total}: {config['name']}")
        if i > 0:
            elapsed = (time.time() - start_time) / 60
            remaining = (total - i) * (elapsed / i)
            log(f"[TIME] Elapsed: {elapsed:.1f}min, ETA: {remaining:.1f}min")
        log(f"{'#'*60}")
        
        try:
            result = run_experiment(model_path, tokenizer, config, device, test_lengths)
            all_results[config["name"]] = result
            
            with open(RESULT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
                
        except Exception as e:
            log(f"[ERROR] {config['name']} failed: {e}")
            import traceback
            log(traceback.format_exc())
            all_results[config["name"]] = {"error": str(e)}
    
    log("\n" + "=" * 60)
    log("生成报告...")
    
    total_time = (time.time() - start_time) / 60
    log(f"[TOTAL] {total_time:.1f} minutes")
    
    lines = ["# RoPE Scaling实验报告", "", f"时间: {datetime.now()}", "", "## 结果汇总", ""]
    lines.append("| 配置 | 边界 | 8k PPL | 16k PPL | 32k PPL | 48k PPL | 64k PPL |")
    lines.append("|------|------|--------|---------|---------|---------|---------|")
    
    for name, data in all_results.items():
        if "error" in data and "data" not in data:
            continue
        d = data.get("data", {})
        boundary = data.get("boundary", "N/A")
        
        def get_ppl(l):
            v = d.get(str(l), {}).get("ppl")
            return f"{v:.2f}" if isinstance(v, float) else "N/A"
        
        lines.append(f"| {name} | {boundary} | {get_ppl(8192)} | {get_ppl(16384)} | {get_ppl(32768)} | {get_ppl(49152)} | {get_ppl(65536)} |")
    
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
    print("上传并启动修复版RoPE Scaling实验")
    print("=" * 50)
    
    # 1. 创建目录
    print("\n[1] 创建目录...")
    run_ssh("mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/results/rope_scaling_v2")
    
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
    run_ssh("base64 -d /tmp/script.b64 > /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_rope_scaling_v2.py && chmod +x /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_rope_scaling_v2.py")
    
    # 4. 启动
    print("\n[4] 启动长时间实验...")
    print("  - 26种RoPE scaling配置")
    print("  - 24个长度点 (2k到64k)")
    print("  - 预计时间: 60+分钟")
    
    run_ssh("cd /root/autodl-tmp/dfrope/hybrid-rope && nohup /root/miniconda3/bin/python scripts/run_rope_scaling_v2.py > results/rope_scaling_v2/run.log 2>&1 &", timeout=10)
    
    print("\n[5] 实验已启动!")
    print("\n检查进度:")
    print(f'  {PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -50 /root/autodl-tmp/dfrope/hybrid-rope/results/rope_scaling_v2/run.log"')

if __name__ == "__main__":
    main()