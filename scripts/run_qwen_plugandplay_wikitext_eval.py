#!/usr/bin/env python3
"""
Qwen即插即用RoPE PPL评测实验

严格协议：
- 数据集: wikitext-103-raw-v1 validation, streaming
- 长度: [2048, 4096, 8192, 16384, 32768]
- Seeds: [42, 123, 777]
- 每个seed每个长度采样 WINDOWS_PER_SEED 个随机窗口
- 只允许base-only评测（无LoRA/finetune）
- 干预方式: inv_freq patch 或 YaRN rope_scaling baseline

配置比较：
- qwen_orig: 原模型
- qwen_yarn8: YaRN factor=8 baseline
- qwen_geo_100k: geometric(theta=100000)
- qwen_sigmoid_best_t100k: sigmoid(theta_base=100000, steepness=8, midpoint=0.5, omf=0.3)
- qwen_hybrid_a0.2_t100k: hybrid(geo, anchored_poly, alpha=0.2)
- qwen_random_control: 随机打乱频率顺序的反证
"""
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 环境变量默认值
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 协议参数
OUT_DIR = Path(os.environ.get("OUT_DIR", "/opt/dfrope/results/qwen_plugandplay_wikitext_v1"))
OUT_JSON = OUT_DIR / "results.json"

LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "2048,4096,8192,16384,32768").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,123,777").split(",")]
WINDOWS_PER_SEED = int(os.environ.get("WINDOWS_PER_SEED", "10"))
MAX_EVAL_TOKENS = int(os.environ.get("MAX_EVAL_TOKENS", "400000"))

# 模型路径
QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
)
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/root/autodl-tmp/dfrope/ms_models")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"


# ==================== 频率定义 ====================

def geometric_freq(K: int, theta: float) -> torch.Tensor:
    """标准几何频率: 1 / theta^(2k/2K)"""
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))


def sigmoid_freq(
    K: int,
    theta_base: float,
    steepness: float = 8.0,
    midpoint: float = 0.5,
    omf: float = 0.3,
) -> torch.Tensor:
    """Sigmoid频率插值"""
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = torch.arange(K, dtype=torch.float32) / (K - 1)
    s = 1.0 / (1.0 + torch.exp(-steepness * (t - midpoint)))
    log_omega = math.log(omega_max) + s * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)


def anchored_poly_freq(
    K: int,
    theta_base: float,
    p: float = 3.9,
    omf: float = 0.3,
) -> torch.Tensor:
    """锚定多项式频率
    
    geo = geometric(K, theta_base)
    omega_max = geo[0]
    omega_min = geo[-1] * omf
    t = k/(K-1)
    log_omega = log(omega_max) + (t^p) * (log(omega_min) - log(omega_max))
    omega = exp(log_omega)
    """
    k = torch.arange(K, dtype=torch.float32)
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = k / (K - 1)
    log_omega = math.log(omega_max) + (t ** p) * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)


def hybrid_freq(
    freq_a: torch.Tensor,
    freq_b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """混合频率: (1-alpha)*freq_a + alpha*freq_b"""
    return (1 - alpha) * freq_a + alpha * freq_b


def random_control_freq(
    K: int,
    theta: float,
    seed: int,
    omf: float = 0.3,
) -> torch.Tensor:
    """随机控制频率 - 保持omega_max/omega_min同geo，但随机打乱顺序"""
    geo = geometric_freq(K, theta)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    
    # 在log空间均匀采样
    t = torch.arange(K, dtype=torch.float32) / (K - 1)
    log_omega = math.log(omega_max) + t * (math.log(omega_min) - math.log(omega_max))
    freqs = torch.exp(log_omega)
    
    # 随机打乱
    rng = random.Random(seed)
    perm = list(range(K))
    rng.shuffle(perm)
    return freqs[perm]


# ==================== RoPE Patch ====================

def find_rope_modules(model) -> List[Tuple[str, object]]:
    """查找所有可能的RoPE模块"""
    modules: List[Tuple[str, object]] = []
    
    # 全局rotary_emb
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        modules.append(("model.model.rotary_emb", model.model.rotary_emb))
    
    # 每层的self_attn.rotary_emb
    for i, layer in enumerate(getattr(model.model, "layers", [])):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "rotary_emb"):
            modules.append((f"layers[{i}].self_attn.rotary_emb", attn.rotary_emb))
        if hasattr(attn, "rotary_fn"):
            rf = attn.rotary_fn
            if hasattr(rf, "inv_freq"):
                modules.append((f"layers[{i}].self_attn.rotary_fn", rf))
            if hasattr(rf, "rotary_emb") and hasattr(rf.rotary_emb, "inv_freq"):
                modules.append((f"layers[{i}].self_attn.rotary_fn.rotary_emb", rf.rotary_emb))
    
    return modules


def patch_rope_freq(model, inv_freq: torch.Tensor) -> Dict[str, object]:
    """Patch RoPE的inv_freq"""
    modules = find_rope_modules(model)
    seen = set()
    patched_names = []
    
    for name, rope in modules:
        if id(rope) in seen or not hasattr(rope, "inv_freq"):
            continue
        seen.add(id(rope))
        old_inv = rope.inv_freq
        rope.inv_freq = inv_freq.to(device=old_inv.device, dtype=old_inv.dtype)
        if hasattr(rope, "max_seq_len_cached"):
            rope.max_seq_len_cached = 0  # 避免缓存混用
        patched_names.append(name)
    
    if not patched_names:
        raise RuntimeError("No rotary module with inv_freq found")
    
    return {
        "patched_count": len(patched_names),
        "patched_examples": patched_names[:4],
        "inv_min": float(inv_freq.min().item()),
        "inv_max": float(inv_freq.max().item()),
    }


def build_inv_freq_from_spec(model, spec: Dict) -> Optional[torch.Tensor]:
    """根据spec构建inv_freq"""
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    K = head_dim // 2
    freq_type = spec.get("freq_type", "orig")
    
    if freq_type == "orig":
        return None
    elif freq_type == "geometric":
        return geometric_freq(K, float(spec.get("theta", 100000)))
    elif freq_type == "sigmoid":
        return sigmoid_freq(
            K,
            theta_base=float(spec.get("theta_base", 100000)),
            steepness=float(spec.get("steepness", 8.0)),
            midpoint=float(spec.get("midpoint", 0.5)),
            omf=float(spec.get("omf", 0.3)),
        )
    elif freq_type == "anchored_poly":
        return anchored_poly_freq(
            K,
            theta_base=float(spec.get("theta_base", 100000)),
            p=float(spec.get("p", 3.9)),
            omf=float(spec.get("omf", 0.3)),
        )
    elif freq_type == "hybrid":
        geo = geometric_freq(K, float(spec.get("theta_base", 100000)))
        poly = anchored_poly_freq(
            K,
            theta_base=float(spec.get("theta_base", 100000)),
            p=float(spec.get("p", 3.9)),
            omf=float(spec.get("omf", 0.3)),
        )
        alpha = float(spec.get("alpha", 0.2))
        return hybrid_freq(geo, poly, alpha)
    elif freq_type == "random_control":
        return random_control_freq(
            K,
            theta=float(spec.get("theta", 100000)),
            seed=int(spec.get("seed", 42)),
            omf=float(spec.get("omf", 0.3)),
        )
    else:
        raise ValueError(f"Unknown freq_type: {freq_type}")


# ==================== 模型加载 ====================

def get_bnb_config():
    """获取BitsAndBytes配置"""
    if not LOAD_IN_4BIT:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def ensure_model_path(model_path: str, ms_id: str) -> str:
    """确保模型路径存在，必要时从ModelScope下载"""
    p = Path(model_path)
    if p.exists():
        print(f"[Model] Found at: {model_path}")
        return str(p)
    
    # 尝试从ModelScope下载
    print(f"[Model] Path not found: {model_path}")
    print(f"[Model] Downloading from ModelScope: {ms_id}")
    try:
        from modelscope import snapshot_download
        dl_path = snapshot_download(
            ms_id,
            cache_dir=MODEL_CACHE_DIR,
            ignore_patterns=["original/*", "*.pth"],
        )
        print(f"[Model] Downloaded to: {dl_path}")
        return dl_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model {ms_id}: {e}")


def load_model_with_spec(spec: Dict):
    """根据spec加载模型"""
    model_path = ensure_model_path(
        str(spec.get("model_path", QWEN_MODEL_PATH)),
        str(spec.get("ms_id", "Qwen/Qwen2.5-7B-Instruct"))
    )
    
    freq_type = spec.get("freq_type", "orig")
    use_yarn = spec.get("use_yarn", False)
    
    bnb_config = get_bnb_config()
    
    if use_yarn:
        # YaRN baseline - 尝试两种key style
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        rope_variants = [
            {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 32768},
            {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 32768},
        ]
        
        model = None
        last_err = None
        for rv in rope_variants:
            try:
                cfg.rope_scaling = rv
                cfg.max_position_embeddings = max(
                    getattr(cfg, "max_position_embeddings", 32768),
                    32768
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=cfg,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                print(f"[Model] Loaded with YaRN rope_scaling: {rv}")
                break
            except Exception as e:
                last_err = e
        
        if model is None:
            raise RuntimeError(f"Failed to load with YaRN: {last_err}")
    else:
        # 正常加载
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    
    # Patch inv_freq（除了orig和yarn）
    patch_info = None
    if freq_type != "orig" and not use_yarn:
        inv_freq = build_inv_freq_from_spec(model, spec)
        if inv_freq is not None:
            patch_info = patch_rope_freq(model, inv_freq)
            print(f"[Patch] {spec['name']}: patched={patch_info['patched_count']}, "
                  f"inv_range=[{patch_info['inv_min']:.3e}, {patch_info['inv_max']:.3e}]")
    
    return model, model_path, patch_info


def load_tokenizer(model_path: str):
    """加载tokenizer"""
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ==================== 评测 ====================

def load_eval_tokens(tokenizer, max_tokens: int) -> List[int]:
    """加载wikitext-103 validation tokens"""
    print(f"[Data] Loading wikitext-103-raw-v1 validation, max_tokens={max_tokens}")
    ds = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    ids: List[int] = []
    for row in ds:
        text = row.get("text")
        if not text or not text.strip():
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    print(f"[Data] Loaded {len(ids)} tokens")
    return ids[:max_tokens]


def sample_starts(total_tokens: int, length: int, seed: int, count: int) -> List[int]:
    """采样随机窗口起点"""
    rng = random.Random(seed * 100000 + length)
    max_start = total_tokens - length - 1
    if max_start < 1:
        raise RuntimeError(f"Not enough tokens: total={total_tokens}, need>{length + 1}")
    return [rng.randint(0, max_start) for _ in range(count)]


@torch.no_grad()
def eval_one_window(model, ids: List[int], start: int, length: int) -> float:
    """评测单个窗口的loss"""
    x = torch.tensor(
        ids[start: start + length + 1],
        dtype=torch.long,
        device=model.device
    ).unsqueeze(0)
    logits = model(x[:, :-1]).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
    lv = float(loss.item())
    if not math.isfinite(lv):
        raise RuntimeError("Non-finite loss detected")
    return lv


def eval_model_ppl(spec: Dict, tokenizer, eval_ids: List[int]) -> Dict:
    """评测单个配置的PPL"""
    print(f"\n{'='*60}")
    print(f"[Eval] {spec['name']}")
    print(f"{'='*60}")
    
    t0 = time.time()
    model, resolved_path, patch_info = load_model_with_spec(spec)
    load_time = time.time() - t0
    print(f"[Eval] Model loaded in {load_time/60:.1f} min")
    
    by_length: Dict[str, object] = {}
    
    for length in LENGTHS:
        print(f"\n[Eval] Length={length}")
        seed_to_ppl: Dict[int, float] = {}
        seed_to_loss_mean: Dict[int, float] = {}
        
        for seed in SEEDS:
            starts = sample_starts(len(eval_ids), length, seed, WINDOWS_PER_SEED)
            losses: List[float] = []
            
            for i, st in enumerate(starts):
                try:
                    loss = eval_one_window(model, eval_ids, st, length)
                    losses.append(loss)
                    if (i + 1) % 5 == 0:
                        print(f"  [Seed={seed}] Window {i+1}/{WINDOWS_PER_SEED}: loss={loss:.4f}")
                except torch.cuda.OutOfMemoryError:
                    print(f"  [Seed={seed}] Window {i+1}: OOM, skipping")
                    torch.cuda.empty_cache()
                    continue
            
            if losses:
                mean_loss = sum(losses) / len(losses)
                seed_to_loss_mean[seed] = float(mean_loss)
                seed_to_ppl[seed] = float(math.exp(mean_loss))
                print(f"  [Seed={seed}] Mean loss={mean_loss:.4f}, PPL={seed_to_ppl[seed]:.4f}")
        
        # 计算mean±std
        ppls = [float(v) for _, v in sorted(seed_to_ppl.items())]
        if len(ppls) == 0:
            mean_ppl, std_ppl = float('nan'), float('nan')
        elif len(ppls) == 1:
            mean_ppl, std_ppl = ppls[0], 0.0
        else:
            mean_ppl = float(statistics.mean(ppls))
            std_ppl = float(statistics.pstdev(ppls))
        
        by_length[str(length)] = {
            "ppl_by_seed": {str(k): round(v, 6) for k, v in seed_to_ppl.items()},
            "loss_mean_by_seed": {str(k): round(v, 6) for k, v in seed_to_loss_mean.items()},
            "mean_ppl": round(mean_ppl, 6),
            "std_ppl": round(std_ppl, 6),
            "windows_per_seed": WINDOWS_PER_SEED,
        }
    
    # 计算collapse ratio
    min_len = min(LENGTHS)
    max_len = max(LENGTHS)
    min_key = str(min_len)
    max_key = str(max_len)
    
    collapse_ratio = None
    if min_key in by_length and max_key in by_length:
        min_ppl = by_length[min_key]["mean_ppl"]
        max_ppl = by_length[max_key]["mean_ppl"]
        if math.isfinite(min_ppl) and min_ppl > 0:
            collapse_ratio = round(max_ppl / min_ppl, 6)
    
    result = {
        "name": spec["name"],
        "freq_type": spec.get("freq_type", "orig"),
        "use_yarn": spec.get("use_yarn", False),
        "resolved_model_path": resolved_path,
        "patch_info": patch_info,
        "load_time_sec": round(load_time, 2),
        "lengths": by_length,
        f"collapse_ratio_{max_len}_over_{min_len}": collapse_ratio,
    }
    
    # 清理
    del model
    torch.cuda.empty_cache()
    
    return result


def compute_ppl_ratio_vs_orig(results: Dict, orig_name: str = "qwen_orig") -> Dict:
    """计算各配置相对于orig的PPL比率"""
    models = results.get("models", [])
    orig = next((m for m in models if m["name"] == orig_name), None)
    if not orig:
        return {}
    
    orig_lengths = orig.get("lengths", {})
    ratios = {}
    
    for model in models:
        if model["name"] == orig_name:
            continue
        model_ratios = {}
        for len_key, len_data in model.get("lengths", {}).items():
            if len_key in orig_lengths:
                orig_ppl = orig_lengths[len_key].get("mean_ppl")
                model_ppl = len_data.get("mean_ppl")
                if orig_ppl and model_ppl and math.isfinite(orig_ppl) and orig_ppl > 0:
                    model_ratios[len_key] = round(model_ppl / orig_ppl, 4)
        ratios[model["name"]] = model_ratios
    
    return ratios


def print_summary_table(results: Dict):
    """打印汇总表格"""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Qwen Plug-and-Play RoPE PPL Evaluation")
    print("=" * 100)
    
    # 表头
    header = f"{'Config':<30}"
    for L in LENGTHS:
        header += f" | PPL@{L:<8}"
    header += f" | Collapse"
    print(header)
    print("-" * 100)
    
    # 数据行
    for model in results.get("models", []):
        name = model["name"]
        row = f"{name:<30}"
        for L in LENGTHS:
            len_data = model.get("lengths", {}).get(str(L), {})
            mean = len_data.get("mean_ppl", float('nan'))
            std = len_data.get("std_ppl", 0.0)
            if math.isfinite(mean):
                row += f" | {mean:.2f}±{std:.2f}"
            else:
                row += f" | {'N/A':<10}"
        
        # collapse ratio
        max_len = max(LENGTHS)
        min_len = min(LENGTHS)
        collapse_key = f"collapse_ratio_{max_len}_over_{min_len}"
        collapse = model.get(collapse_key)
        if collapse and math.isfinite(collapse):
            row += f" | {collapse:.2f}"
        else:
            row += f" | {'N/A':<8}"
        print(row)
    
    # PPL ratio vs orig
    print("\n" + "-" * 100)
    print("PPL Ratio vs qwen_orig (lower is better)")
    print("-" * 100)
    
    ratios = results.get("ppl_ratio_vs_orig", {})
    for model_name, model_ratios in ratios.items():
        row = f"{model_name:<30}"
        for L in LENGTHS:
            ratio = model_ratios.get(str(L))
            if ratio:
                row += f" | {ratio:.4f}    "
            else:
                row += f" | {'N/A':<10}"
        print(row)
    
    print("=" * 100)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Qwen Plug-and-Play RoPE PPL Evaluation")
    print("=" * 60)
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"LENGTHS: {LENGTHS}")
    print(f"SEEDS: {SEEDS}")
    print(f"WINDOWS_PER_SEED: {WINDOWS_PER_SEED}")
    print(f"MAX_EVAL_TOKENS: {MAX_EVAL_TOKENS}")
    print(f"LOAD_IN_4BIT: {LOAD_IN_4BIT}")
    print(f"QWEN_MODEL_PATH: {QWEN_MODEL_PATH}")
    print("=" * 60)
    
    # 配置列表
    MODEL_SPECS = [
        {
            "name": "qwen_orig",
            "model_path": QWEN_MODEL_PATH,
            "ms_id": "Qwen/Qwen2.5-7B-Instruct",
            "freq_type": "orig",
            "use_yarn": False,
        },
        {
            "name": "qwen_yarn8",
            "model_path": QWEN_MODEL_PATH,
            "ms_id": "Qwen/Qwen2.5-7B-Instruct",
            "freq_type": "orig",  # YaRN不需要patch
            "use_yarn": True,
        },
        {
            "name": "qwen_geo_100k",
            "model_path": QWEN_MODEL_PATH,
            "ms_id": "Qwen/Qwen2.5-7B-Instruct",
            "freq_type": "geometric",
            "theta": 100000,
        },
        {
            "name": "qwen_sigmoid_best_t100k",
            "model_path": QWEN_MODEL_PATH,
            "ms_id": "Qwen/Qwen2.5-7B-Instruct",
            "freq_type": "sigmoid",
            "theta_base": 100000,
            "steepness": 8.0,
            "midpoint": 0.5,
            "omf": 0.3,
        },
        {
            "name": "qwen_hybrid_a0.2_t100k",
            "model_path": QWEN_MODEL_PATH,
            "ms_id": "Qwen/Qwen2.5-7B-Instruct",
            "freq_type": "hybrid",
            "theta_base": 100000,
            "p": 3.9,
            "omf": 0.3,
            "alpha": 0.2,
        },
        {
            "name": "qwen_random_control",
            "model_path": QWEN_MODEL_PATH,
            "ms_id": "Qwen/Qwen2.5-7B-Instruct",
            "freq_type": "random_control",
            "theta": 100000,
            "seed": 42,
            "omf": 0.3,
        },
    ]
    
    # 加载tokenizer和数据
    resolved_path = ensure_model_path(QWEN_MODEL_PATH, "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = load_tokenizer(resolved_path)
    eval_ids = load_eval_tokens(tokenizer, MAX_EVAL_TOKENS)
    
    if len(eval_ids) < max(LENGTHS) + 2:
        raise RuntimeError(f"Eval tokens too short: {len(eval_ids)}")
    
    # 初始化结果
    results: Dict = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "protocol": {
            "dataset": "wikitext-103-raw-v1:validation",
            "lengths": LENGTHS,
            "seeds": SEEDS,
            "windows_per_seed": WINDOWS_PER_SEED,
            "max_eval_tokens": MAX_EVAL_TOKENS,
            "random_start": True,
            "train_or_lora": "none (base-only)",
            "load_in_4bit": LOAD_IN_4BIT,
        },
        "model_specs": MODEL_SPECS,
        "models": [],
    }
    
    start_time = time.time()
    
    # 评测每个配置
    for spec in MODEL_SPECS:
        try:
            model_result = eval_model_ppl(spec, tokenizer, eval_ids)
            results["models"].append(model_result)
        except Exception as e:
            print(f"[Error] Failed to eval {spec['name']}: {e}")
            import traceback
            traceback.print_exc()
            results["models"].append({
                "name": spec["name"],
                "error": str(e),
            })
        
        # 每完成一个配置就保存
        OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    
    # 计算PPL ratio vs orig
    results["ppl_ratio_vs_orig"] = compute_ppl_ratio_vs_orig(results)
    
    # 完成
    elapsed_min = (time.time() - start_time) / 60
    results["elapsed_minutes"] = round(elapsed_min, 3)
    
    # 保存最终结果
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    
    # 打印汇总表
    print_summary_table(results)
    
    print(f"\n[Done] Results saved to: {OUT_JSON}")
    print(f"[Done] Total time: {elapsed_min:.1f} minutes")


if __name__ == "__main__":
    main()