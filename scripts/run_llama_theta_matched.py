#!/usr/bin/env python3
"""
LLaMA Theta-Matched Shape Control Experiment
Rigorous test of "shape > theta" hypothesis

Configs:
1. geo_100k - θ-aligned geometric
2. sigmoid_100k - θ-aligned sigmoid  
3. geo_500k - larger theta control

Expected results for "shape > theta":
- sigmoid_100k < geo_100k (shape effect)
- sigmoid_100k ≲ geo_500k (shape replaces large theta)
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_PATH = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
DATA_NAME = "wikitext-103-raw-v1"
DATA_SPLIT = "validation"
LENGTHS = [2048, 16384]
SEED = 42
WINDOWS = 10
MAX_TOKENS = 250000
OUTPUT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/llama_theta_matched_shape_control")

ROPE_CONFIGS = {
    "geo_100k": {"type": "geometric", "theta": 100000},
    "sigmoid_100k": {"type": "sigmoid", "T": 100000},
    "geo_500k": {"type": "geometric", "theta": 500000},
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(tokenizer, max_tokens=MAX_TOKENS):
    import pandas as pd
    # Direct load from parquet cache
    cache_path = "/root/.cache/huggingface/hub/datasets--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1"
    df = pd.read_parquet(f"{cache_path}/validation-00000-of-00001.parquet")
    text = "\n\n".join(df["text"].tolist())
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:max_tokens]
    return torch.tensor(tokens, dtype=torch.long)

def compute_inv_freq(rope_type, head_dim, **kwargs):
    """Compute inverse frequency for RoPE"""
    if rope_type == "geometric":
        theta = kwargs.get('theta', 10000)
        return 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    elif rope_type == "sigmoid":
        T = kwargs.get('T', 100000)
        inv_freq_min = 1e-6
        inv_freq_max = 1.0
        mid = T / 2
        steepness = 10.0 / T
        positions = torch.arange(0, head_dim // 2).float()
        return inv_freq_min + (inv_freq_max - inv_freq_min) * torch.sigmoid(
            steepness * (positions * 100 - mid)
        )
    else:
        raise ValueError(f"Unknown rope_type: {rope_type}")

def apply_rope_patch(model, rope_type, **kwargs):
    """Patch RoPE frequencies in the model"""
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    inv_freq = compute_inv_freq(rope_type, head_dim, **kwargs)
    
    for name, module in model.named_modules():
        if 'rotary_emb' in name.lower():
            module.inv_freq = inv_freq.to(model.device)
            break
    
    return model

@torch.no_grad()
def eval_ppl(model, tokens, length):
    """Evaluate perplexity on a window of tokens"""
    model.eval()
    window = tokens[:length].unsqueeze(0).cuda()
    outputs = model(window, labels=window)
    loss = outputs.loss.item()
    return np.exp(loss)

def run_config(config_name, config, tokens, tokenizer, lengths, seed, windows):
    """Run experiment for one config"""
    print(f"\n{'='*60}")
    print(f"  {config_name}")
    print(f"  type={config['type']}, theta/T={config.get('theta', config.get('T'))}")
    print(f"{'='*60}")
    
    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply RoPE patch
    model = apply_rope_patch(model, config['type'], 
                             theta=config.get('theta'), 
                             T=config.get('T'))
    
    results = {}
    
    for length in lengths:
        print(f"  Length {length}...")
        ppl_values = []
        
        for i in range(windows):
            set_seed(seed + i)
            max_start = len(tokens) - length
            if max_start > 0:
                start = np.random.randint(0, max_start)
                window = tokens[start:start+length]
            else:
                window = tokens[:length]
            
            ppl = eval_ppl(model, window, length)
            ppl_values.append(ppl)
            print(f"    Window {i+1}/{windows}: PPL={ppl:.3f}")
        
        results[length] = {
            "ppl_mean": float(np.mean(ppl_values)),
            "ppl_std": float(np.std(ppl_values)),
            "n": windows
        }
        print(f"  PPL@{length}: {results[length]['ppl_mean']:.3f} ± {results[length]['ppl_std']:.3f}")
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    return results

def main():
    print("="*60)
    print("LLaMA Theta-Matched Shape Control Experiment")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_NAME}/{DATA_SPLIT}")
    print(f"Lengths: {LENGTHS}")
    print(f"Seed: {SEED}")
    print(f"Windows: {WINDOWS}")
    print(f"Configs: {list(ROPE_CONFIGS.keys())}")
    
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
    
    for config_name, config in ROPE_CONFIGS.items():
        results = run_config(config_name, config, tokens, tokenizer, LENGTHS, SEED, WINDOWS)
        all_results[config_name] = results
    
    # Compute collapse ratios and format output
    output = {}
    for config_name, results in all_results.items():
        ppl_2k = results[2048]["ppl_mean"]
        ppl_16k = results[16384]["ppl_mean"]
        collapse_ratio = ppl_16k / ppl_2k
        
        output[config_name] = {
            "ppl_2k": round(ppl_2k, 3),
            "ppl_16k": round(ppl_16k, 3),
            "collapse_ratio": round(collapse_ratio, 3)
        }
    
    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Analyze hypothesis
    geo_100k_collapse = output["geo_100k"]["collapse_ratio"]
    sigmoid_100k_collapse = output["sigmoid_100k"]["collapse_ratio"]
    geo_500k_collapse = output["geo_500k"]["collapse_ratio"]
    
    shape_effect = sigmoid_100k_collapse < geo_100k_collapse
    shape_replace_theta = sigmoid_100k_collapse <= geo_500k_collapse * 1.1  # within 10%
    
    # Generate summary
    summary = f"""# LLaMA Theta-Matched Shape Control Experiment

## Purpose
Rigorous test of "shape > theta" hypothesis with controlled θ alignment.

## Setup
- **Model**: LLaMA-3-8B
- **Data**: WikiText-103-raw-v1
- **Slicing**: random_start
- **Seed**: {SEED}
- **Lengths**: 2048, 16384
- **Windows per config**: {WINDOWS}

## Configurations

| Config | Type | θ/T | Description |
|--------|------|-----|-------------|
| geo_100k | geometric | 100k | θ-aligned baseline |
| sigmoid_100k | sigmoid | 100k | θ-aligned shape test |
| geo_500k | geometric | 500k | Large θ control |

## Results

| Config | PPL@2048 | PPL@16384 | Collapse Ratio |
|--------|----------|-----------|----------------|
| geo_100k | {output['geo_100k']['ppl_2k']} | {output['geo_100k']['ppl_16k']} | **{output['geo_100k']['collapse_ratio']}x** |
| sigmoid_100k | {output['sigmoid_100k']['ppl_2k']} | {output['sigmoid_100k']['ppl_16k']} | **{output['sigmoid_100k']['collapse_ratio']}x** |
| geo_500k | {output['geo_500k']['ppl_2k']} | {output['geo_500k']['ppl_16k']} | **{output['geo_500k']['collapse_ratio']}x** |

## Hypothesis Testing

### Test 1: Shape Effect (θ-matched)
- geo_100k collapse: {geo_100k_collapse}x
- sigmoid_100k collapse: {sigmoid_100k_collapse}x
- **Result**: {'✅ PASSED' if shape_effect else '❌ FAILED'} - Sigmoid is {geo_100k_collapse/sigmoid_100k_collapse:.1f}x more stable than geometric at same θ

### Test 2: Shape Replaces Large θ
- geo_500k collapse: {geo_500k_collapse}x
- sigmoid_100k collapse: {sigmoid_100k_collapse}x
- **Result**: {'✅ PASSED' if shape_replace_theta else '❌ FAILED'} - Sigmoid at 100k {'is comparable to' if shape_replace_theta else 'is worse than'} geometric at 500k

## Conclusion

"""
    
    if shape_effect and shape_replace_theta:
        summary += f"**Strong support for 'shape > θ' hypothesis**: Sigmoid shape at θ=100k achieves {sigmoid_100k_collapse}x collapse ratio, which is:\n"
        summary += f"- {geo_100k_collapse/sigmoid_100k_collapse:.1f}x better than geometric at same θ (shape effect)\n"
        summary += f"- Comparable to geometric at 5x larger θ (shape replaces large θ)\n"
    elif shape_effect:
        summary += f"**Partial support for 'shape > θ' hypothesis**: Shape effect confirmed ({geo_100k_collapse/sigmoid_100k_collapse:.1f}x improvement), but sigmoid at 100k cannot fully replace large θ."
    else:
        summary += f"**No support for 'shape > θ' hypothesis**: Shape does not provide benefit over geometric at same θ."
    
    summary_path = OUTPUT_DIR / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved summary to {summary_path}")
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(output, indent=2))
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    print(f"Shape effect (sigmoid_100k < geo_100k): {'✅ PASSED' if shape_effect else '❌ FAILED'}")
    print(f"Shape replaces θ (sigmoid_100k ≲ geo_500k): {'✅ PASSED' if shape_replace_theta else '❌ FAILED'}")
    
    return output

if __name__ == "__main__":
    main()