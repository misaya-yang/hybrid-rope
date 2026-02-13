## 任务：Llama-3 Zero-shot Hybrid Frequency 验证

### 环境
- GPU: A800 80GB (支持 BF16)
- 网络: 畅通 (需下载模型)
- 依赖: `pip install transformers torch accelerate datasets`

### 目标
加载开源 Llama-3-8B 模型，在**不进行任何训练**的情况下，强行替换其 RoPE 频率为 Hybrid 分布，验证是否能改善长上下文性能。

### 实验步骤

1.  **加载模型**: `Meta-Llama-3-8B-Instruct` (BF16, device_map="auto")
2.  **准备数据**: `pg19` (书籍长文本) 或 `wikitext`，取长度 > 8192 的样本。
3.  **定义频率生成器**:
    - `get_geo_freq(theta)`: 标准等比。
    - `get_hybrid_freq(theta_base, alpha, p, omf)`: 你的混合分布。
4.  **替换逻辑 (Patching)**:
    - 编写函数 `patch_llama_freq(model, new_freq_tensor)`。
    - 遍历 `model.model.layers`，修改 `layer.self_attn.rotary_emb.inv_freq`。
    - **关键**: 必须清空 `cos_cached`/`sin_cached` (如有)，强制模型重新计算相位。
5.  **评测流程**:
    - **Run 1 (Baseline)**: 不修改，直接测 PPL @ 8K, 16K, 32K。
    - **Run 2 (Hybrid)**: 替换为 `hybrid_a0.2_t100k` (或其他参数)，测 PPL。

### 核心代码片段 (run_llama3_zeroshot.py)

```python
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda"

def get_hybrid_freq(dim, base_theta=100000, alpha=0.2, p=3.9, omf=0.3):
    # 1. Geo Base
    k = torch.arange(dim // 2, dtype=torch.float32, device=DEVICE)
    geo = 1.0 / (base_theta ** (2 * k / dim))
    
    # 2. Poly Component
    omega_max = 1.0
    omega_min = geo[-1] * omf
    t = k / ((dim // 2) - 1)
    log_w = torch.log(torch.tensor(omega_max)) + (t**p) * (torch.log(omega_min) - torch.log(torch.tensor(omega_max)))
    poly = torch.exp(log_w)
    
    # 3. Hybrid
    return (1 - alpha) * geo + alpha * poly

def patch_model(model, new_freq):
    print(f"Patching model with new frequencies...")
    for layer in model.model.layers:
        rope = layer.self_attn.rotary_emb
        # Llama3 uses inv_freq buffer
        rope.inv_freq = new_freq
        # Reset cache to force re-computation
        if hasattr(rope, 'cos_cached'): rope.cos_cached = None
        if hasattr(rope, 'sin_cached'): rope.sin_cached = None
    print("Patch complete.")

def eval_ppl(model, tokenizer, text, seq_len=8192, stride=4096):
    inputs = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    nls = []
    # Sliding window PPL eval
    for i in range(0, inputs.size(1), stride):
        begin_loc = max(i + stride - seq_len, 0)
        end_loc = min(i + stride, inputs.size(1))
        trg_len = end_loc - i
        if trg_len <= 0: break
        
        input_ids = inputs[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Mask context
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nls.append(outputs.loss * trg_len)
            
    return torch.exp(torch.stack(nls).sum() / end_loc)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    
    # Load Long Text
    text = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")["text"]
    text = "\n\n".join(text)[:100000] # Take first 100k chars for speed
    
    print("=== Baseline (Standard Geo 500k) ===")
    ppl = eval_ppl(model, tokenizer, text)
    print(f"Baseline PPL: {ppl:.2f}")
    
    print("=== Hybrid (Theta 100k, Alpha 0.2) ===")
    # Note: Llama3 head_dim is 128
    new_freq = get_hybrid_freq(dim=128, base_theta=100000, alpha=0.2)
    patch_model(model, new_freq)
    ppl = eval_ppl(model, tokenizer, text)
    print(f"Hybrid PPL: {ppl:.2f}")

if __name__ == "__main__":
    main()import paramiko

# SSH连接配置
host = "117.50.220.66"
port = 23
username = "root"
password = "jHB67294F08oAaN5"

def run_command(client, cmd):
    """执行命令并返回输出"""
    stdin, stdout, stderr = client.exec_command(cmd)
    return stdout.read().decode()

try:
    # 创建SSH客户端
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print(f"正在连接到 {host}:{port}...")
    client.connect(host, port, username, password, 
                   timeout=30,
                   allow_agent=False,
                   look_for_keys=False,
                   banner_timeout=30)
    print("✓ 连接成功！\n")
    
    # 获取系统信息
    print("=" * 50)
    print("系统信息")
    print("=" * 50)
    print(run_command(client, "uname -a"))
    
    print("\n" + "=" * 50)
    print("CPU信息")
    print("=" * 50)
    print(run_command(client, "lscpu | head -20"))
    
    print("\n" + "=" * 50)
    print("内存信息")
    print("=" * 50)
    print(run_command(client, "free -h"))
    
    print("\n" + "=" * 50)
    print("GPU信息")
    print("=" * 50)
    print(run_command(client, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No NVIDIA GPU detected'"))
    
    print("\n" + "=" * 50)
    print("磁盘信息")
    print("=" * 50)
    print(run_command(client, "df -h /home"))
    
    print("\n" + "=" * 50)
    print("当前目录内容")
    print("=" * 50)
    print(run_command(client, "ls -la"))
    
    print("\n" + "=" * 50)
    print("Python版本")
    print("=" * 50)
    print(run_command(client, "python3 --version 2>/dev/null || python --version"))
    
    client.close()
    print("\n✓ 连接已关闭。")
    
except Exception as e:
    print(f"✗ 连接失败: {e}")