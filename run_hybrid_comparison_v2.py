#!/usr/bin/env python3
"""上传并运行Hybrid-RoPE vs Baseline对比实验 - 修复版"""

import subprocess
import base64

PLINK = r"C:\Users\Admin\.ssh\plink.exe"
SSH_CMD = f'{PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# Hybrid-RoPE对比实验 - 修复版
SCRIPT = r'''#!/usr/bin/env python3
"""
Hybrid-RoPE vs Baseline 对比实验 - 修复版
使用rope_scaling参数实现长度外推，而不是monkey patching
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
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/hybrid_comparison_v2")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULT_DIR / "run.log"

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ============== 评估函数 ==============

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
        n_garbage = max(10, seq_len // 15)  # 减少garbage数量以避免超长
        passkey = random.randint(10000, 99999)
        prompt, _ = generate_passkey_prompt(n_garbage, passkey, seed=i*1000+seq_len)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
        actual_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            try:
                # 计算PPL
                outputs = model(**inputs, labels=inputs["input_ids"])
                if outputs.loss is not None and not torch.isnan(outputs.loss):
                    total_loss += outputs.loss.item()
                
                # 生成测试
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
    ppl = math.exp(min(avg_loss, 10)) if avg_loss > 0 else 0  # cap to avoid overflow
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    return accuracy, ppl, mem, "ok"

# ============== 主实验 ==============

def run_comparison():
    log("=" * 60)
    log("RoPE Scaling 对比实验 - 修复版")
    log("=" * 60)
    
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    # 测试长度 - 专注于8k边界附近
    test_lengths = [4096, 6144, 8192, 10240, 12288, 16384, 20480]
    
    # RoPE配置
    rope_configs = {
        "baseline": None,  # 标准RoPE
        "linear_2x": {"rope_type": "linear", "factor": 2.0},
        "linear_4x": {"rope_type": "linear", "factor": 4.0},
        "dynamic_2x": {"rope_type": "dynamic", "factor": 2.0},
        "dynamic_4x": {"rope_type": "dynamic", "factor": 4.0},
        "yarn_2x": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 8192},
        "yarn_4x": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192},
    }
    
    log("[MODEL] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    all_results = {}
    start_time = time.time()
    
    for config_name, rope_scaling in rope_configs.items():
        log("\n" + "=" * 60)
        log(f"[CONFIG] Testing {config_name}...")
        if rope_scaling:
            log(f"  rope_scaling: {rope_scaling}")
        log("=" * 60)
        
        # 加载模型
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                device_map="auto", 
                trust_remote_code=True,
                rope_scaling=rope_scaling
            )
            model.eval()
        except Exception as e:
            log(f"[ERROR] Failed to load model with {config_name}: {e}")
            all_results[config_name] = {"error": str(e)}
            continue
        
        config_results = {}
        for i, length in enumerate(test_lengths):
            log(f"[{config_name}] L={length} ({i+1}/{len(test_lengths)})")
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            acc, ppl, mem, status = evaluate_passkey(model, tokenizer, device, length)
            
            if acc is not None:
                config_results[str(length)] = {
                    "accuracy": round(acc, 3),
                    "ppl": round(ppl, 3),
                    "mem_gb": round(mem, 2),
                    "actual_accuracy": acc,
                    "status": status
                }
                log(f"[RESULT] Acc={acc:.1%}, PPL={ppl:.2f}, Mem={mem:.2f}GB")
            else:
                config_results[str(length)] = {"status": status}
                log(f"[RESULT] {status}")
            
            # 保存中间结果
            all_results[config_name] = config_results
            with open(RESULT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        
        del model
        torch.cuda.empty_cache()
    
    # 生成报告
    log("\n" + "=" * 60)
    log("生成对比报告...")
    
    total_time = (time.time() - start_time) / 60
    log(f"[TOTAL] {total_time:.1f} minutes")
    
    # Markdown报告
    lines = ["# RoPE Scaling 对比报告", "", f"时间: {datetime.now()}", ""]
    lines.append("## Passkey检索准确率对比")
    lines.append("")
    
    # 表头
    header = "| 长度 |"
    separator = "|------|"
    for config in rope_configs.keys():
        header += f" {config} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)
    
    # 数据行
    for length in test_lengths:
        row = f"| {length} |"
        for config in rope_configs.keys():
            data = all_results.get(config, {}).get(str(length), {})
            acc = data.get("accuracy")
            if acc is not None:
                row += f" {acc:.0%} |"
            else:
                status = data.get("status", "-")
                row += f" {status} |"
        lines.append(row)
    
    lines.append("")
    lines.append("## PPL对比")
    lines.append("")
    lines.append(header)
    lines.append(separator)
    
    for length in test_lengths:
        row = f"| {length} |"
        for config in rope_configs.keys():
            data = all_results.get(config, {}).get(str(length), {})
            ppl = data.get("ppl")
            if ppl is not None:
                row += f" {ppl:.2f} |"
            else:
                row += f" - |"
        lines.append(row)
    
    lines.append("")
    lines.append("## 最佳配置")
    lines.append("")
    
    # 找出最佳配置
    best_acc = {}
    for length in test_lengths:
        best_config = None
        best_val = -1
        for config in rope_configs.keys():
            data = all_results.get(config, {}).get(str(length), {})
            acc = data.get("actual_accuracy", 0)
            if acc > best_val:
                best_val = acc
                best_config = config
        if best_config:
            best_acc[length] = (best_config, best_val)
            lines.append(f"- **{length}**: {best_config} ({best_val:.0%})")
    
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated at {datetime.now().isoformat()}*")
    
    with open(RESULT_DIR / "report.md", "w") as f:
        f.write("\n".join(lines))
    
    log(f"[DONE] 结果: {RESULT_DIR}")
    log("=" * 60)

if __name__ == "__main__":
    run_comparison()
'''

def run_ssh(cmd, timeout=60):
    full_cmd = f'{SSH_CMD} "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode

def main():
    print("=" * 50)
    print("启动RoPE Scaling对比实验 - 修复版")
    print("=" * 50)
    
    # 1. 创建目录
    print("\n[1] 创建目录...")
    run_ssh("mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/results/hybrid_comparison_v2")
    
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
    print("\n[4] 启动对比实验...")
    print("  - Baseline (标准RoPE)")
    print("  - Linear scaling (2x, 4x)")
    print("  - Dynamic scaling (2x, 4x)")
    print("  - YaRN (2x, 4x)")
    print("  - 7个长度点 (4k-20k)")
    print("  - 预计时间: 5-10分钟")
    
    run_ssh("cd /root/autodl-tmp/dfrope/hybrid-rope && nohup /root/miniconda3/bin/python scripts/run_rope_scaling_v2.py > results/hybrid_comparison_v2/run.log 2>&1 &", timeout=10)
    
    print("\n[5] 实验已启动!")
    print("\n检查进度:")
    print(f'  {PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -50 /root/autodl-tmp/dfrope/hybrid-rope/results/hybrid_comparison_v2/run.log"')

if __name__ == "__main__":
    main()