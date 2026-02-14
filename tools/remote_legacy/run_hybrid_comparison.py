#!/usr/bin/env python3
"""上传并运行Hybrid-RoPE vs Baseline对比实验"""

import subprocess
import base64

PLINK = r"C:\Users\Admin\.ssh\plink.exe"
SSH_CMD = f'{PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# Hybrid-RoPE对比实验
SCRIPT = r'''#!/usr/bin/env python3
"""
Hybrid-RoPE vs Baseline 对比实验
测试我们的改进方法是否能提升长文本能力
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

RESULT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/hybrid_comparison")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULT_DIR / "run.log"

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ============== Hybrid-RoPE实现 ==============

class HybridRoPE:
    """
    Hybrid-RoPE: 混合几何衰减的RoPE
    核心思想: 在长距离上应用theta衰减，保持短距离的精度
    """
    def __init__(self, dim, base=10000, theta=2000000, device="cuda"):
        self.dim = dim
        self.base = base
        self.theta = theta  # 衰减参数
        self.device = device
        
        # 原始inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.inv_freq = inv_freq
    
    def get_cos_sin(self, seq_len):
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        
        # 几何衰减: decay = theta^(-i/L)
        # 这会平滑地减少长距离位置的频率
        decay = self.theta ** (-t / seq_len)
        
        # 计算频率
        freqs = torch.outer(t, self.inv_freq)
        
        # 应用衰减
        freqs = freqs * decay.unsqueeze(1)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin

def apply_hybrid_rope_to_model(model, theta=2000000):
    """
    将Hybrid-RoPE应用到模型
    通过monkey patching替换rotary embedding
    """
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    device = next(model.parameters()).device
    
    # 创建Hybrid-RoPE
    hybrid_rope = HybridRoPE(head_dim, theta=theta, device=device)
    
    # 保存原始forward
    for name, module in model.named_modules():
        if hasattr(module, 'rotary_emb') and module.rotary_emb is not None:
            # 保存原始
            if not hasattr(module.rotary_emb, '_original_forward'):
                module.rotary_emb._original_forward = module.rotary_emb.forward
                module.rotary_emb._hybrid_rope = hybrid_rope
                module.rotary_emb._theta = theta
                
                # 替换forward
                def make_hybrid_forward(orig_forward, rope_impl):
                    def hybrid_forward(x, position_ids):
                        seq_len = x.shape[2]
                        cos, sin = rope_impl.get_cos_sin(seq_len)
                        
                        # 调整维度以匹配
                        # cos, sin: [seq_len, head_dim/2]
                        # 需要: [batch, 1, seq_len, head_dim]
                        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
                        sin = sin.unsqueeze(0).unsqueeze(0)
                        
                        # 重复以匹配完整head_dim
                        cos = torch.cat([cos, cos], dim=-1)  # [1, 1, seq_len, head_dim]
                        sin = torch.cat([sin, sin], dim=-1)
                        
                        return cos.to(x.dtype), sin.to(x.dtype)
                    
                    return hybrid_forward
                
                module.rotary_emb.forward = make_hybrid_forward(
                    module.rotary_emb._original_forward,
                    hybrid_rope
                )
    
    return model

def restore_original_rope(model):
    """恢复原始RoPE"""
    for name, module in model.named_modules():
        if hasattr(module, 'rotary_emb') and hasattr(module.rotary_emb, '_original_forward'):
            module.rotary_emb.forward = module.rotary_emb._original_forward
            del module.rotary_emb._original_forward
            if hasattr(module.rotary_emb, '_hybrid_rope'):
                del module.rotary_emb._hybrid_rope

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

def evaluate_passkey(model, tokenizer, device, seq_len, n_tests=5):
    """评估passkey检索"""
    model.eval()
    correct = 0
    total_loss = 0.0
    
    for i in range(n_tests):
        n_garbage = seq_len // 10
        passkey = random.randint(10000, 99999)
        prompt, _ = generate_passkey_prompt(n_garbage, passkey, seed=i*1000+seq_len)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
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
                raise
    
    accuracy = correct / n_tests
    avg_loss = total_loss / n_tests
    ppl = math.exp(avg_loss) if avg_loss > 0 else 0
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    return accuracy, ppl, mem, "ok"

def evaluate_ppl(model, tokenizer, device, seq_len):
    """评估perplexity (简单版本)"""
    model.eval()
    
    # 简单测试文本
    test_text = "The quick brown fox jumps over the lazy dog. " * 500
    inputs = tokenizer(test_text[:seq_len*4], return_tensors="pt").to(device)
    inputs["input_ids"] = inputs["input_ids"][:, :seq_len]
    inputs["attention_mask"] = inputs["attention_mask"][:, :seq_len]
    
    with torch.no_grad():
        try:
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            ppl = math.exp(loss)
            mem = torch.cuda.max_memory_allocated() / 1e9
            return ppl, loss, mem, "ok"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return None, None, None, "OOM"
            raise

# ============== 主实验 ==============

def run_comparison():
    log("=" * 60)
    log("Hybrid-RoPE vs Baseline 对比实验")
    log("=" * 60)
    
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    # 测试长度 - 重点关注8k边界附近
    test_lengths = [8192, 10240, 12288, 14336, 16384, 20480, 24576, 32768, 40960, 49152]
    
    # Theta配置
    theta_values = [500000, 1000000, 2000000, 3000000, 5000000]
    
    log("[MODEL] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    all_results = {}
    start_time = time.time()
    
    # 1. Baseline
    log("\n" + "=" * 60)
    log("[BASELINE] Testing standard RoPE...")
    log("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    
    baseline_results = {}
    for i, length in enumerate(test_lengths):
        log(f"[BASELINE] L={length} ({i+1}/{len(test_lengths)})")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        acc, ppl, mem, status = evaluate_passkey(model, tokenizer, device, length)
        
        if acc is not None:
            baseline_results[str(length)] = {
                "accuracy": round(acc, 3),
                "ppl": round(ppl, 3),
                "mem_gb": round(mem, 2),
                "status": status
            }
            log(f"[RESULT] Acc={acc:.1%}, PPL={ppl:.2f}, Mem={mem:.2f}GB")
        else:
            baseline_results[str(length)] = {"status": status}
            log(f"[RESULT] {status}")
    
    all_results["baseline"] = baseline_results
    
    del model
    torch.cuda.empty_cache()
    
    # 2. Hybrid-RoPE with different theta
    for theta in theta_values:
        config_name = f"hybrid_theta_{theta//1000}k"
        log("\n" + "=" * 60)
        log(f"[HYBRID] Testing with theta={theta}...")
        log("=" * 60)
        
        # 重新加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        
        # 应用Hybrid-RoPE
        model = apply_hybrid_rope_to_model(model, theta=theta)
        model.eval()
        
        hybrid_results = {}
        for i, length in enumerate(test_lengths):
            log(f"[HYBRID-{theta//1000}k] L={length} ({i+1}/{len(test_lengths)})")
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            acc, ppl, mem, status = evaluate_passkey(model, tokenizer, device, length)
            
            if acc is not None:
                hybrid_results[str(length)] = {
                    "accuracy": round(acc, 3),
                    "ppl": round(ppl, 3),
                    "mem_gb": round(mem, 2),
                    "status": status
                }
                log(f"[RESULT] Acc={acc:.1%}, PPL={ppl:.2f}, Mem={mem:.2f}GB")
            else:
                hybrid_results[str(length)] = {"status": status}
                log(f"[RESULT] {status}")
        
        all_results[config_name] = hybrid_results
        
        # 保存中间结果
        with open(RESULT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        del model
        torch.cuda.empty_cache()
    
    # 3. 生成报告
    log("\n" + "=" * 60)
    log("生成对比报告...")
    
    total_time = (time.time() - start_time) / 60
    log(f"[TOTAL] {total_time:.1f} minutes")
    
    # Markdown报告
    lines = ["# Hybrid-RoPE vs Baseline 对比报告", "", f"时间: {datetime.now()}", ""]
    lines.append("## Passkey检索准确率对比")
    lines.append("")
    
    # 表头
    header = "| 长度 |"
    separator = "|------|"
    for config in all_results.keys():
        header += f" {config} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)
    
    # 数据行
    for length in test_lengths:
        row = f"| {length} |"
        for config in all_results.keys():
            data = all_results[config].get(str(length), {})
            acc = data.get("accuracy")
            if acc is not None:
                row += f" {acc:.0%} |"
            else:
                row += f" - |"
        lines.append(row)
    
    lines.append("")
    lines.append("## PPL对比")
    lines.append("")
    lines.append(header)
    lines.append(separator)
    
    for length in test_lengths:
        row = f"| {length} |"
        for config in all_results.keys():
            data = all_results[config].get(str(length), {})
            ppl = data.get("ppl")
            if ppl is not None:
                row += f" {ppl:.2f} |"
            else:
                row += f" - |"
        lines.append(row)
    
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
    print("启动Hybrid-RoPE vs Baseline对比实验")
    print("=" * 50)
    
    # 1. 创建目录
    print("\n[1] 创建目录...")
    run_ssh("mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/results/hybrid_comparison")
    
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
    run_ssh("base64 -d /tmp/script.b64 > /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_hybrid_comparison.py && chmod +x /root/autodl-tmp/dfrope/hybrid-rope/scripts/run_hybrid_comparison.py")
    
    # 4. 启动
    print("\n[4] 启动对比实验...")
    print("  - Baseline (标准RoPE)")
    print("  - Hybrid-RoPE (theta=500k, 1M, 2M, 3M, 5M)")
    print("  - 10个长度点 (8k-50k)")
    print("  - 预计时间: 10-15分钟")
    
    run_ssh("cd /root/autodl-tmp/dfrope/hybrid-rope && nohup /root/miniconda3/bin/python scripts/run_hybrid_comparison.py > results/hybrid_comparison/run.log 2>&1 &", timeout=10)
    
    print("\n[5] 实验已启动!")
    print("\n检查进度:")
    print(f'  {PLINK} -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -80 /root/autodl-tmp/dfrope/hybrid-rope/results/hybrid_comparison/run.log"')

if __name__ == "__main__":
    main()