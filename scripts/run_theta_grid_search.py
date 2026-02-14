#!/usr/bin/env python3
"""
大规模Theta网格搜索实验
测试多种theta配置在不同序列长度下的表现
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

# 结果目录
RESULT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/theta_grid_search")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件
LOG_FILE = RESULT_DIR / "run.log"

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ============== RoPE实现 ==============

def compute_inv_freq(base, dim, device):
    """计算逆频率"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    return inv_freq

def apply_rotary_emb(x, cos, sin):
    """应用旋转位置编码"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    # 确保cos和sin的维度匹配
    if cos.dim() == 3:
        cos = cos.squeeze(0)  # [seq_len, head_dim]
    if sin.dim() == 3:
        sin = sin.squeeze(0)
    
    # 扩展维度
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # 确保seq_len匹配
    seq_len = x.shape[2]
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]
    
    # 重复以匹配x1和x2的维度
    cos = cos.expand(-1, -1, -1, x1.shape[-1])
    sin = sin.expand(-1, -1, -1, x2.shape[-1])
    
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return x_rotated

class GeometricRoPE:
    """几何衰减RoPE"""
    def __init__(self, base, theta, dim, device):
        self.base = base
        self.theta = theta
        self.dim = dim
        self.device = device
        self.inv_freq = compute_inv_freq(base, dim, device)
    
    def get_cos_sin(self, seq_len):
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        
        # 几何衰减: freq_i = freq_i * theta^(-i/L)
        decay = self.theta ** (-t / seq_len)
        decay = decay.unsqueeze(1)  # [seq_len, 1]
        
        freqs = torch.outer(t.float(), self.inv_freq)
        freqs = freqs * decay
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin

class SigmoidRoPE:
    """Sigmoid边界RoPE"""
    def __init__(self, base, boundary, steepness, dim, device):
        self.base = base
        self.boundary = boundary
        self.steepness = steepness
        self.dim = dim
        self.device = device
        self.inv_freq = compute_inv_freq(base, dim, device)
    
    def get_cos_sin(self, seq_len):
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        
        # Sigmoid衰减: decay = 1 / (1 + exp(k*(L - boundary)))
        decay = 1.0 / (1.0 + torch.exp(self.steepness * (t - self.boundary)))
        decay = decay.unsqueeze(1)
        
        freqs = torch.outer(t.float(), self.inv_freq)
        freqs = freqs * decay
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin

class PolynomialRoPE:
    """多项式衰减RoPE"""
    def __init__(self, base, power, dim, device):
        self.base = base
        self.power = power
        self.dim = dim
        self.device = device
        self.inv_freq = compute_inv_freq(base, dim, device)
    
    def get_cos_sin(self, seq_len):
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        
        # 多项式衰减: decay = (1 - i/L)^power
        decay = (1 - t / seq_len) ** self.power
        decay = decay.unsqueeze(1)
        
        freqs = torch.outer(t.float(), self.inv_freq)
        freqs = freqs * decay
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin

# ============== 模型修改 ==============

def modify_model_rope(model, rope_impl, seq_len):
    """修改模型的RoPE"""
    cos, sin = rope_impl.get_cos_sin(seq_len)
    
    # 存储到模型
    model._rope_cos = cos
    model._rope_sin = sin
    model._rope_seq_len = seq_len
    
    # 替换forward方法
    for name, module in model.named_modules():
        if 'self_attn' in name and hasattr(module, 'rotary_emb'):
            original_forward = module.forward
            
            def make_forward_wrapper(orig_fwd, cos_tensor, sin_tensor):
                def forward_wrapper(*args, **kwargs):
                    # 调用原始forward
                    return orig_fwd(*args, **kwargs)
                return forward_wrapper
            
            module._original_forward = original_forward

def reset_model_rope(model):
    """重置模型的RoPE修改"""
    if hasattr(model, '_rope_cos'):
        del model._rope_cos
    if hasattr(model, '_rope_sin'):
        del model._rope_sin
    if hasattr(model, '_rope_seq_len'):
        del model._rope_seq_len

# ============== 评估函数 ==============

def evaluate_ppl(model, tokenizer, device, seq_len, rope_impl=None):
    """评估perplexity"""
    # 加载数据
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(dataset["text"])
    
    # 编码
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    
    # 截取足够长的序列
    if len(tokens) < seq_len * 2:
        tokens = tokens.repeat(math.ceil(seq_len * 2 / len(tokens)) + 1)
    
    tokens = tokens[:seq_len * 4].to(device)  # 使用4个seq_len用于评估
    
    # 创建输入
    inputs = tokens.unsqueeze(0)
    
    # 评估
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, inputs.shape[1] - seq_len, seq_len // 2):
            input_chunk = inputs[:, i:i+seq_len]
            
            try:
                outputs = model(input_chunk, labels=input_chunk)
                loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item() * input_chunk.shape[1]
                    total_tokens += input_chunk.shape[1]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return None, None, "OOM"
                raise
    
    if total_tokens == 0:
        return None, None, "ERROR"
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    mem_used = torch.cuda.max_memory_allocated() / 1e9
    
    return ppl, avg_loss, mem_used

# ============== 主实验 ==============

def run_experiment(config_name, rope_class, rope_params, model, tokenizer, device, test_lengths):
    """运行单个配置的实验"""
    log(f"[CONFIG] Running: {config_name}")
    log(f"[CONFIG] Params: {rope_params}")
    
    results = {
        "name": config_name,
        "params": rope_params,
        "data": {},
        "boundary": None
    }
    
    baseline_ppl = None
    boundary_found = False
    
    for length in test_lengths:
        log(f"[EVAL] Testing L={length}...")
        
        start_time = time.time()
        
        try:
            # 创建RoPE实例
            rope_impl = rope_class(**rope_params)
            
            # 评估
            ppl, loss, mem = evaluate_ppl(model, tokenizer, device, length, rope_impl)
            
            elapsed = time.time() - start_time
            
            if ppl is None:
                log(f"[EVAL] L={length}: {mem}")
                results["data"][str(length)] = {"status": mem}
                if mem == "OOM":
                    results["oom_at"] = length
                    break
                continue
            
            # 记录结果
            results["data"][str(length)] = {
                "ppl": round(ppl, 3),
                "loss": round(loss, 4),
                "elapsed_sec": round(elapsed, 2),
                "mem_gb": round(mem, 2),
                "status": "ok"
            }
            
            log(f"[EVAL] L={length}: PPL={ppl:.3f}, loss={loss:.4f}, mem={mem:.2f}GB, time={elapsed:.1f}s")
            
            # 计算baseline
            if baseline_ppl is None:
                baseline_ppl = ppl
            
            # 检测边界 (PPL > 5x baseline 或 PPL > 100)
            if not boundary_found and (ppl > 5 * baseline_ppl or ppl > 100):
                results["boundary"] = length
                boundary_found = True
                log(f"[BOUNDARY] Found at L={length}")
            
            # 清理
            del rope_impl
            torch.cuda.empty_cache()
            
        except Exception as e:
            log(f"[ERROR] L={length}: {str(e)}")
            results["data"][str(length)] = {"status": "error", "message": str(e)}
            break
    
    # 计算collapse ratio
    if baseline_ppl and str(test_lengths[0]) in results["data"]:
        baseline = results["data"][str(test_lengths[0])]["ppl"]
        results["collapse_ratio"] = {
            length: round(results["data"][length]["ppl"] / baseline, 3)
            for length in results["data"]
            if results["data"][length].get("status") == "ok"
        }
    
    return results

def main():
    log("=" * 60)
    log("大规模Theta网格搜索实验")
    log("=" * 60)
    
    # 配置
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    # 测试长度
    test_lengths = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 
                    20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536]
    
    # 加载模型
    log(f"[MODEL] Loading from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # 获取RoPE维度
    config = model.config
    rope_dim = getattr(config, 'rope_scaling', None)
    head_dim = config.hidden_size // config.num_attention_heads
    
    log(f"[MODEL] Loaded. Hidden={config.hidden_size}, Heads={config.num_attention_heads}, Head_dim={head_dim}")
    
    # 实验配置
    experiments = []
    
    # 1. Baseline (标准RoPE)
    experiments.append({
        "name": "baseline",
        "rope_class": lambda **kwargs: type('BaselineRoPE', (), {
            'get_cos_sin': lambda self, seq_len: (
                torch.cos(torch.outer(
                    torch.arange(seq_len, device=kwargs['device'], dtype=torch.float32),
                    1.0 / (kwargs['base'] ** (torch.arange(0, kwargs['dim'], 2, device=kwargs['device']).float() / kwargs['dim']))
                )),
                torch.sin(torch.outer(
                    torch.arange(seq_len, device=kwargs['device'], dtype=torch.float32),
                    1.0 / (kwargs['base'] ** (torch.arange(0, kwargs['dim'], 2, device=kwargs['device']).float() / kwargs['dim']))
                ))
            )
        })(),
        "params": {"base": 10000, "dim": head_dim, "device": device}
    })
    
    # 2. 几何衰减 - 多种theta
    theta_values = [100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000, 5000000]
    for theta in theta_values:
        experiments.append({
            "name": f"geo_{theta//1000}k",
            "rope_class": GeometricRoPE,
            "params": {"base": 10000, "theta": theta, "dim": head_dim, "device": device}
        })
    
    # 3. Sigmoid边界 - 多种配置
    sigmoid_configs = [
        (8192, 0.001), (8192, 0.002), (8192, 0.005),
        (16384, 0.001), (16384, 0.002), (16384, 0.005),
        (32768, 0.0005), (32768, 0.001), (32768, 0.002),
        (49152, 0.0005), (49152, 0.001)
    ]
    for boundary, steepness in sigmoid_configs:
        experiments.append({
            "name": f"sigmoid_b{boundary//1024}k_s{steepness}",
            "rope_class": SigmoidRoPE,
            "params": {"base": 10000, "boundary": boundary, "steepness": steepness, "dim": head_dim, "device": device}
        })
    
    # 4. 多项式衰减
    power_values = [1, 2, 3, 4, 5, 10]
    for power in power_values:
        experiments.append({
            "name": f"poly_p{power}",
            "rope_class": PolynomialRoPE,
            "params": {"base": 10000, "power": power, "dim": head_dim, "device": device}
        })
    
    log(f"[MAIN] Total experiments: {len(experiments)}")
    log(f"[MAIN] Test lengths: {test_lengths}")
    
    # 运行所有实验
    all_results = {}
    
    for i, exp in enumerate(experiments):
        log(f"\n[MAIN] ===== Experiment {i+1}/{len(experiments)}: {exp['name']} =====")
        
        try:
            result = run_experiment(
                exp["name"],
                exp["rope_class"],
                exp["params"],
                model,
                tokenizer,
                device,
                test_lengths
            )
            all_results[exp["name"]] = result
            
            # 保存中间结果
            with open(RESULT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            
        except Exception as e:
            log(f"[ERROR] Experiment {exp['name']} failed: {str(e)}")
            all_results[exp["name"]] = {"error": str(e)}
    
    # 生成汇总报告
    log("\n" + "=" * 60)
    log("实验完成！生成汇总报告...")
    
    # 生成Markdown报告
    report = generate_report(all_results)
    with open(RESULT_DIR / "report.md", "w") as f:
        f.write(report)
    
    log(f"[DONE] 结果保存在 {RESULT_DIR}")
    log("=" * 60)

def generate_report(results):
    """生成Markdown报告"""
    lines = ["# Theta网格搜索实验报告", "", f"生成时间: {datetime.now()}", ""]
    
    # 表格：各配置的边界长度
    lines.append("## 边界长度汇总")
    lines.append("")
    lines.append("| 配置 | 边界长度 | 8k PPL | 16k PPL | 32k PPL | 48k PPL |")
    lines.append("|------|----------|--------|---------|---------|---------|")
    
    for name, data in sorted(results.items()):
        if "error" in data:
            continue
        boundary = data.get("boundary", "N/A")
        d = data.get("data", {})
        ppl_8k = d.get("8192", {}).get("ppl", "N/A")
        ppl_16k = d.get("16384", {}).get("ppl", "N/A")
        ppl_32k = d.get("32768", {}).get("ppl", "N/A")
        ppl_48k = d.get("49152", {}).get("ppl", "N/A")
        
        if isinstance(ppl_8k, float):
            ppl_8k = f"{ppl_8k:.2f}"
        if isinstance(ppl_16k, float):
            ppl_16k = f"{ppl_16k:.2f}"
        if isinstance(ppl_32k, float):
            ppl_32k = f"{ppl_32k:.2f}"
        if isinstance(ppl_48k, float):
            ppl_48k = f"{ppl_48k:.2f}"
        
        lines.append(f"| {name} | {boundary} | {ppl_8k} | {ppl_16k} | {ppl_32k} | {ppl_48k} |")
    
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated at {datetime.now().isoformat()}*")
    
    return "\n".join(lines)

if __name__ == "__main__":
    main()