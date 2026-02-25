# SSH 连接与零训练验证笔记

此文档是一个工作笔记，用于记录如何在服务器上做 **LLaMA-3 zero-shot Hybrid frequency** 的快速验证。

安全约定：
- 不要把 SSH 密码、token、私钥写进仓库。
- 如需脚本化连接，用环境变量注入（例如 `SEETACLOUD_SSH_PW`），或改用 SSH key。

## 目标

加载开源 `Meta-Llama-3-8B-Instruct`，在不进行任何训练的情况下，强行替换其 RoPE 频率为 Hybrid 分布，验证是否能改善长上下文性能。

## 实验步骤（建议）

1. 加载模型：`Meta-Llama-3-8B-Instruct`（BF16/FP16，`device_map="auto"`）
2. 准备数据：`pg19`（长文本）或 `wikitext`
3. 定义频率生成器：
   - `get_geo_freq(theta)`：标准等比
   - `get_hybrid_freq(theta_base, alpha, p, omf)`：混合分布
4. 替换逻辑（patching）：
   - 遍历 RoPE 模块，修改 `rotary_emb.inv_freq`
   - 清空/重置 RoPE cache（如果模型实现里有）
5. 评测流程：
   - Baseline：不修改，测 PPL @ 8K/16K/32K
   - Hybrid：替换为 `hybrid_a0.2_t100k`（或其他参数），测 PPL

## 代码片段（仅示意）

```python
import math
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda"

def get_hybrid_freq(dim, base_theta=100000, alpha=0.2, p=3.9, omf=0.3):
    k = torch.arange(dim // 2, dtype=torch.float32, device=DEVICE)
    geo = 1.0 / (base_theta ** (2 * k / dim))
    omega_max = 1.0
    omega_min = geo[-1] * omf
    t = k / ((dim // 2) - 1)
    log_w = math.log(omega_max) + (t**p) * (torch.log(omega_min) - math.log(omega_max))
    poly = torch.exp(log_w)
    return (1 - alpha) * geo + alpha * poly

def patch_model(model, new_inv_freq):
    for name, module in model.named_modules():
        if "rotary" in name.lower() and hasattr(module, "inv_freq"):
            module.inv_freq = new_inv_freq.to(module.inv_freq.device, dtype=module.inv_freq.dtype)
            if hasattr(module, "max_seq_len_cached"):
                module.max_seq_len_cached = 0

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(ds["text"])[:100000]
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)

    # TODO: implement a stable sliding-window PPL eval here.
    # Then compare baseline vs patched.
```

