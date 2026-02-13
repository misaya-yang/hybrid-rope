import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT_DIR = Path(os.environ.get("OUT_DIR", "/opt/dfrope/results/cross_model_wikitext_v1"))
OUT_JSON = OUT_DIR / "results.json"

LENGTHS = [2048, 16384]
SEEDS = [42, 123, 777]
WINDOWS_PER_SEED = int(os.environ.get("WINDOWS_PER_SEED", "10"))
MAX_EVAL_TOKENS = int(os.environ.get("MAX_EVAL_TOKENS", "280000"))

AUTO_DOWNLOAD = os.environ.get("AUTO_DOWNLOAD_MODELSCOPE", "1") == "1"
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/root/autodl-tmp/dfrope/ms_models")

QWEN_DEFAULT_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct",
)
LLAMA_DEFAULT_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
)

MODEL_SPECS = [
    {
        "name": "llama_geo_10k",
        "family": "llama",
        "model_path": LLAMA_DEFAULT_PATH,
        "ms_id": "LLM-Research/Meta-Llama-3-8B-Instruct",
        "freq_type": "geometric",
        "theta": 10000.0,
    },
    {
        "name": "llama_sigmoid_best_t100k",
        "family": "llama",
        "model_path": LLAMA_DEFAULT_PATH,
        "ms_id": "LLM-Research/Meta-Llama-3-8B-Instruct",
        "freq_type": "sigmoid",
        "theta_base": 100000.0,
        "steepness": 8.0,
        "midpoint": 0.5,
        "omf": 0.3,
    },
    {
        "name": "qwen_orig_theta",
        "family": "qwen",
        "model_path": QWEN_DEFAULT_PATH,
        "ms_id": "Qwen/Qwen2.5-7B-Instruct",
        "freq_type": "orig",
    },
    {
        "name": "qwen_geo_100k",
        "family": "qwen",
        "model_path": QWEN_DEFAULT_PATH,
        "ms_id": "Qwen/Qwen2.5-7B-Instruct",
        "freq_type": "geometric",
        "theta": 100000.0,
    },
]


def geometric_freq(k_count: int, theta: float) -> torch.Tensor:
    idx = torch.arange(k_count, dtype=torch.float32)
    return 1.0 / (theta ** (2 * idx / (2 * k_count)))


def sigmoid_freq(
    k_count: int,
    theta_base: float,
    steepness: float,
    midpoint: float,
    omf: float,
) -> torch.Tensor:
    geo = geometric_freq(k_count, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = torch.arange(k_count, dtype=torch.float32) / (k_count - 1)
    s = 1.0 / (1.0 + torch.exp(-steepness * (t - midpoint)))
    log_omega = math.log(omega_max) + s * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)


def ensure_model_path(model_path: str, ms_id: str) -> str:
    if AUTO_DOWNLOAD:
        try:
            from modelscope import snapshot_download
        except Exception as exc:
            raise RuntimeError(f"modelscope import failed while downloading {ms_id}: {exc}") from exc
        print(f"[Download] ensure snapshot_download: {ms_id}")
        # Skip original checkpoint files (e.g. original/consolidated.00.pth) to avoid
        # downloading unnecessary 10GB+ artifacts for inference/eval.
        dl_path = snapshot_download(
            ms_id,
            cache_dir=MODEL_CACHE_DIR,
            ignore_patterns=["original/*", "*.pth"],
        )
        print(f"[Download] ready: {dl_path}")
        return dl_path

    p = Path(model_path)
    if p.exists():
        return str(p)
    raise FileNotFoundError(f"Missing model path: {model_path}")


def find_rope_modules(model) -> List[Tuple[str, object]]:
    modules: List[Tuple[str, object]] = []
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        modules.append(("model.model.rotary_emb", model.model.rotary_emb))
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
            rope.max_seq_len_cached = 0
        patched_names.append(name)
    if not patched_names:
        raise RuntimeError("No rotary module with inv_freq found")
    return {"patched_count": len(patched_names), "patched_examples": patched_names[:4]}


def build_inv_freq_from_spec(model, spec: Dict[str, object]) -> torch.Tensor:
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    k_count = head_dim // 2
    ftype = spec["freq_type"]
    if ftype == "geometric":
        return geometric_freq(k_count, float(spec["theta"]))
    if ftype == "sigmoid":
        return sigmoid_freq(
            k_count,
            theta_base=float(spec["theta_base"]),
            steepness=float(spec["steepness"]),
            midpoint=float(spec["midpoint"]),
            omf=float(spec["omf"]),
        )
    raise ValueError(f"Unsupported freq_type for patch: {ftype}")


def load_model_and_tokenizer(spec: Dict[str, object]):
    resolved_path = ensure_model_path(str(spec["model_path"]), str(spec["ms_id"]))
    print(f"[Load] {spec['name']} <- {resolved_path}")
    tok = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    patch_info = None
    if spec["freq_type"] != "orig":
        inv = build_inv_freq_from_spec(model, spec)
        patch_info = patch_rope_freq(model, inv)
        patch_info["inv_min"] = float(inv.min().item())
        patch_info["inv_max"] = float(inv.max().item())
    return model, tok, resolved_path, patch_info


def load_eval_tokens(tokenizer, max_tokens: int) -> List[int]:
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
    return ids[:max_tokens]


def sample_starts(total_tokens: int, length: int, seed: int, count: int) -> List[int]:
    rng = random.Random(seed * 100000 + length)
    max_start = total_tokens - length - 1
    if max_start < 1:
        raise RuntimeError(f"not enough tokens: total={total_tokens}, need>{length + 1}")
    return [rng.randint(0, max_start) for _ in range(count)]


@torch.no_grad()
def eval_one_window(model, ids: List[int], start: int, length: int) -> float:
    x = torch.tensor(ids[start : start + length + 1], dtype=torch.long, device=model.device).unsqueeze(0)
    logits = model(x[:, :-1]).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
    lv = float(loss.item())
    if not math.isfinite(lv):
        raise RuntimeError("non-finite loss detected")
    return lv


def summarize_seed_ppls(seed_to_ppl: Dict[int, float]) -> Dict[str, float]:
    vals = [float(v) for _, v in sorted(seed_to_ppl.items())]
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    return {"mean": float(statistics.mean(vals)), "std": float(statistics.pstdev(vals))}


def eval_model(spec: Dict[str, object]) -> Dict[str, object]:
    model, tok, resolved_path, patch_info = load_model_and_tokenizer(spec)
    ids = load_eval_tokens(tok, MAX_EVAL_TOKENS)
    if len(ids) < max(LENGTHS) + 2:
        raise RuntimeError(f"eval tokens too short: {len(ids)}")

    by_length: Dict[str, object] = {}
    min_len_key = str(min(LENGTHS))
    max_len_key = str(max(LENGTHS))
    for length in LENGTHS:
        seed_to_ppl: Dict[int, float] = {}
        seed_to_loss_mean: Dict[int, float] = {}
        for seed in SEEDS:
            starts = sample_starts(len(ids), length, seed, WINDOWS_PER_SEED)
            losses: List[float] = []
            for st in starts:
                losses.append(eval_one_window(model, ids, st, length))
            mean_loss = sum(losses) / len(losses)
            seed_to_loss_mean[seed] = float(mean_loss)
            seed_to_ppl[seed] = float(math.exp(mean_loss))
        summary = summarize_seed_ppls(seed_to_ppl)
        by_length[str(length)] = {
            "ppl_by_seed": {str(k): round(v, 6) for k, v in seed_to_ppl.items()},
            "loss_mean_by_seed": {str(k): round(v, 6) for k, v in seed_to_loss_mean.items()},
            "mean_ppl": round(summary["mean"], 6),
            "std_ppl": round(summary["std"], 6),
            "windows_per_seed": WINDOWS_PER_SEED,
        }

    c_ratio = by_length[max_len_key]["mean_ppl"] / by_length[min_len_key]["mean_ppl"]
    out = {
        "name": spec["name"],
        "family": spec["family"],
        "resolved_model_path": resolved_path,
        "freq_type": spec["freq_type"],
        "patch_info": patch_info,
        "eval_dataset": "wikitext-103-raw-v1:validation",
        "token_count_used": len(ids),
        "lengths": by_length,
        "collapse_ratio_16k_over_2k": round(float(c_ratio), 6),
    }

    del model
    torch.cuda.empty_cache()
    return out


def print_table(results: Dict[str, object]) -> None:
    print("\n=== Main Table (random_start, WikiText-103 val) ===")
    print("Model | PPL@2K(mean±std) | PPL@16K(mean±std) | Collapse(16K/2K)")
    for m in results["models"]:
        r2k = m["lengths"]["2048"]
        r16 = m["lengths"]["16384"]
        print(
            f"{m['name']} | "
            f"{r2k['mean_ppl']:.4f}±{r2k['std_ppl']:.4f} | "
            f"{r16['mean_ppl']:.4f}±{r16['std_ppl']:.4f} | "
            f"{m['collapse_ratio_16k_over_2k']:.4f}"
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "protocol": {
            "dataset": "wikitext-103-raw-v1:validation",
            "lengths": LENGTHS,
            "seeds": SEEDS,
            "windows_per_seed": WINDOWS_PER_SEED,
            "max_eval_tokens": MAX_EVAL_TOKENS,
            "random_start": True,
            "train_or_lora": "none (base-only)",
        },
        "models": [],
    }

    for spec in MODEL_SPECS:
        print(f"\n[Run] {spec['name']}")
        model_result = eval_model(spec)
        results["models"].append(model_result)
        OUT_JSON.write_text(json.dumps(results, indent=2))

    elapsed_min = (time.time() - start) / 60
    results["elapsed_minutes"] = round(elapsed_min, 3)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
