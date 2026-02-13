#!/usr/bin/env python3
"""
LLaMA Shape vs Theta Min Experiment
Compare geo_500k vs sigmoid_t100k on LLaMA-3-8B
Eval-only, WikiText-103-raw-v1, random_start slicing
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_PATH = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
DATA_NAME = "wikitext-103-raw-v1"
DATA_SPLIT = "validation"
LENGTHS = [2048, 16384]
SEED = 42
WINDOWS = 10
MAX_TOKENS = 250000

OUTPUT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/llama_shape_theta_min")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(tokenizer, max_tokens=MAX_TOKENS):
    from datasets import load_dataset
    ds = load_dataset(DATA_NAME, split=DATA_SPLIT)
    text = "\n\n".join(ds["text"])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:max_tokens]
    return torch.tensor(tokens, dtype=torch.long)

def get_random_windows(tokens, length, num_windows, seed):
    set_seed(seed)
    max_start = len(tokens) - length
    if max_start <= 0:
        return [tokens]
    starts = np.random.randint(0, max_start, size=num_windows)
    return [tokens[s:s+length] for s in starts]

def apply_rope_patch(model, rope_type, **kwargs):
    """Patch RoPE frequencies in the model"""
    import math
    
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    if rope_type == "geo_500k":
        # Geometric with theta=500k
        theta = kwargs.get('theta', 500000)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    elif rope_type == "sigmoid_t100k":
        # Sigmoid with T=100k
        T = kwargs.get('T', 100000)
        positions = torch.arange(0, 8192*2)  # Extended positions
        # Sigmoid frequency schedule
        inv_freq_min = 1e-6
        inv_freq_max = 1.0
        mid = T / 2
        steepness = 10.0 / T
        inv_freq = inv_freq_min + (inv_freq_max - inv_freq_min) * torch.sigmoid(
            steepness * (positions.float() - mid)
        )
        # Take only the needed dimension
        inv_freq = inv_freq[:head_dim//2]
    else:
        raise ValueError(f"Unknown rope_type: {rope_type}")
    
    # Patch the rotary embedding
    for name, module in model.named_modules():
        if 'rotary_emb' in name.lower():
            module.inv_freq = inv_freq.to(model.device)
            print(f"Patched {name} with {rope_type}, inv_freq shape: {inv_freq.shape}")
            break
    
    return model

@torch.no_grad()
def eval_ppl(model, tokenizer, tokens, length, device="cuda"):
    """Evaluate perplexity on a window of tokens"""
    model.eval()
    window = tokens[:length].unsqueeze(0).to(device)
    
    # Forward pass
    outputs = model(window, labels=window)
    loss = outputs.loss.item()
    ppl = np.exp(loss)
    
    return ppl

def run_experiment(rope_type, tokens, tokenizer, lengths, seed, windows):
    """Run experiment for one RoPE config"""
    print(f"\n{'='*60}")
    print(f"  {rope_type}")
    print(f"{'='*60}")
    
    # Load fresh model for each config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply RoPE patch
    if rope_type == "geo_500k":
        model = apply_rope_patch(model, "geo_500k", theta=500000)
    elif rope_type == "sigmoid_t100k":
        model = apply_rope_patch(model, "sigmoid_t100k", T=100000)
    
    results = {}
    
    for length in lengths:
        print(f"\n  Length {length}...")
        ppl_values = []
        
        for i in range(windows):
            set_seed(seed + i)
            max_start = len(tokens) - length
            if max_start > 0:
                start = np.random.randint(0, max_start)
                window = tokens[start:start+length]
            else:
                window = tokens[:length]
            
            ppl = eval_ppl(model, tokenizer, window, length)
            ppl_values.append(ppl)
            print(f"    Window {i+1}/{windows}: PPL={ppl:.3f}")
        
        results[length] = {
            "ppl_mean": np.mean(ppl_values),
            "ppl_std": np.std(ppl_values),
            "n": windows
        }
        print(f"  PPL@{length}: {results[length]['ppl_mean']:.3f} ± {results[length]['ppl_std']:.3f}")
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    return results

def main():
    print("="*60)
    print("LLaMA Shape vs Theta Min Experiment")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_NAME}/{DATA_SPLIT}")
    print(f"Lengths: {LENGTHS}")
    print(f"Seed: {SEED}")
    print(f"Windows: {WINDOWS}")
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading tokenizer and data...")
    tokenizer = load_tokenizer()
    tokens = load_data(tokenizer)
    print(f"Loaded {len(tokens)} tokens")
    
    # Run experiments
    all_results = {}
    
    for rope_type in ["geo_500k", "sigmoid_t100k"]:
        results = run_experiment(rope_type, tokens, tokenizer, LENGTHS, SEED, WINDOWS)
        all_results[rope_type] = results
    
    # Compute collapse ratios and format output
    output = {}
    for rope_type, results in all_results.items():
        ppl_2k = results[2048]["ppl_mean"]
        ppl_16k = results[16384]["ppl_mean"]
        collapse_ratio = ppl_16k / ppl_2k
        
        output[rope_type] = {
            "ppl_2k": round(ppl_2k, 3),
            "ppl_16k": round(ppl_16k, 3),
            "collapse_ratio": round(collapse_ratio, 3)
        }
    
    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Generate summary
    summary = f"""# LLaMA Shape vs Theta Min Experiment

## Setup
- **Model**: LLaMA-3-8B
- **Data**: WikiText-103-raw-v1
- **Slicing**: random_start
- **Seed**: {SEED}
- **Lengths**: 2048, 16384
- **Windows per config**: {WINDOWS}

## Results

| Config | PPL@2048 | PPL@16384 | Collapse Ratio |
|--------|----------|-----------|----------------|
| geo_500k | {output['geo_500k']['ppl_2k']:.3f} | {output['geo_500k']['ppl_16k']:.3f} | **{output['geo_500k']['collapse_ratio']:.2f}x** |
| sigmoid_t100k | {output['sigmoid_t100k']['ppl_2k']:.3f} | {output['sigmoid_t100k']['ppl_16k']:.3f} | **{output['sigmoid_t100k']['collapse_ratio']:.2f}x** |

## Collapse Ratio Comparison
- geo_500k: {output['geo_500k']['collapse_ratio']:.2f}x
- sigmoid_t100k: {output['sigmoid_t100k']['collapse_ratio']:.2f}x
- **Improvement**: {output['geo_500k']['collapse_ratio'] / output['sigmoid_t100k']['collapse_ratio']:.1f}x more stable

## Conclusion
"""
    
    if output['sigmoid_t100k']['collapse_ratio'] < output['geo_500k']['collapse_ratio']:
        summary += f"sigmoid_t100k 在 16K 的 collapse_ratio ({output['sigmoid_t100k']['collapse_ratio']:.2f}x) 明显低于 geo_500k ({output['geo_500k']['collapse_ratio']:.2f}x)，支持 **频谱形状 > θ 大小** 的稳定性假设。"
    else:
        summary += f"geo_500k 表现优于预期，需要进一步分析。"
    
    summary_path = OUTPUT_DIR / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved summary to {summary_path}")
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(output, indent=2))
    
    return output

if __name__ == "__main__":
    main()