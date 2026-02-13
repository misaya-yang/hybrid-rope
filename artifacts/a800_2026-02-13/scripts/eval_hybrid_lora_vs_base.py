import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/opt/dfrope/models_alt/LLM-Research/Meta-Llama-3-8B-Instruct"
HYBRID_LORA_PATH = "/opt/dfrope/results/llama3_hybrid_lora/final_lora"
OUT_DIR = "/opt/dfrope/results/llama3_hybrid_lora_eval"

DTYPE = torch.bfloat16
EVAL_LENGTHS = [2048, 8192, 16384]
EVAL_CHUNKS = 5
VAL_TOKENS = 1_200_000
SEED = 42


def geometric_freq(K, theta):
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))


def anchored_poly_freq(K, theta_base, p=3.9, omf=0.3):
    k = torch.arange(K, dtype=torch.float32)
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = k / (K - 1)
    log_omega = math.log(omega_max) + (t ** p) * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)


def hybrid_freq(freq_a, freq_b, alpha):
    return (1 - alpha) * freq_a + alpha * freq_b


def build_hybrid_inv_freq(head_dim: int):
    K = head_dim // 2
    geo_100k = geometric_freq(K, 100000)
    poly_100k = anchored_poly_freq(K, 100000, p=3.9, omf=0.3)
    return hybrid_freq(geo_100k, poly_100k, alpha=0.2)


def patch_model_rope(model):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    inv = build_hybrid_inv_freq(head_dim)

    cands = []
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        cands.append(model.model.rotary_emb)

    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "rotary_emb"):
            cands.append(attn.rotary_emb)
        if hasattr(attn, "rotary_fn"):
            rf = attn.rotary_fn
            if hasattr(rf, "inv_freq"):
                cands.append(rf)
            if hasattr(rf, "rotary_emb") and hasattr(rf.rotary_emb, "inv_freq"):
                cands.append(rf.rotary_emb)

    seen = set()
    patched = 0
    for rope in cands:
        if id(rope) in seen or not hasattr(rope, "inv_freq"):
            continue
        seen.add(id(rope))
        rope.inv_freq = inv.to(device=rope.inv_freq.device, dtype=rope.inv_freq.dtype)
        if hasattr(rope, "max_seq_len_cached"):
            rope.max_seq_len_cached = 0
        patched += 1

    if patched == 0:
        raise RuntimeError("No rotary module found to patch")


def load_val_tokens(tokenizer, max_tokens=VAL_TOKENS):
    ids = []
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    for x in ds:
        txt = x.get("text")
        if not txt:
            continue
        ids.extend(tokenizer.encode(txt, add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    return torch.tensor(ids[:max_tokens], dtype=torch.long)


@torch.no_grad()
def eval_ppl(model, tokens, lengths, n_chunks):
    model.eval()
    out = {}
    device = model.device
    for L in lengths:
        losses = []
        for i in range(n_chunks):
            s = i * L
            e = s + L
            if e > len(tokens):
                break
            chunk = tokens[s:e].unsqueeze(0).to(device)
            logits = model(chunk[:, :-1]).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1))
            losses.append(float(loss.item()))
        if losses:
            mean_loss = sum(losses) / len(losses)
            ppl = math.exp(mean_loss)
            std = (sum((math.exp(x) - ppl) ** 2 for x in losses) / len(losses)) ** 0.5
            out[str(L)] = {"ppl": round(ppl, 3), "std": round(std, 3), "n": len(losses)}
            print(f"L={L}: PPL={ppl:.3f} +- {std:.3f} (n={len(losses)})", flush=True)
    return out


def load_base_model(use_yarn=False):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if use_yarn:
        orig = getattr(config, "max_position_embeddings", 8192)
        config.rope_scaling = {
            "type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": int(orig),
        }
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = True
    return model


def load_hybrid_lora_model():
    model = load_base_model(use_yarn=False)
    patch_model_rope(model)
    model = PeftModel.from_pretrained(model, HYBRID_LORA_PATH)
    return model


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading validation tokens...", flush=True)
    val_tokens = load_val_tokens(tok)
    print(f"Validation tokens loaded: {len(val_tokens)}", flush=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "lengths": EVAL_LENGTHS,
        "chunks": EVAL_CHUNKS,
        "models": {},
    }

    print("\n=== Eval: base_unfinetuned ===", flush=True)
    m_base = load_base_model(use_yarn=False)
    results["models"]["base_unfinetuned"] = eval_ppl(m_base, val_tokens, EVAL_LENGTHS, EVAL_CHUNKS)
    del m_base
    torch.cuda.empty_cache()

    print("\n=== Eval: hybrid_lora ===", flush=True)
    m_hybrid = load_hybrid_lora_model()
    results["models"]["hybrid_lora"] = eval_ppl(m_hybrid, val_tokens, EVAL_LENGTHS, EVAL_CHUNKS)
    del m_hybrid
    torch.cuda.empty_cache()

    print("\n=== Eval: base_yarn_x2 (optional) ===", flush=True)
    try:
        m_yarn = load_base_model(use_yarn=True)
        results["models"]["base_yarn_x2"] = eval_ppl(m_yarn, val_tokens, EVAL_LENGTHS, EVAL_CHUNKS)
        del m_yarn
        torch.cuda.empty_cache()
    except Exception as e:
        results["models"]["base_yarn_x2"] = {"error": str(e)}
        print(f"base_yarn_x2 skipped: {e}", flush=True)

    out_json = f"{OUT_DIR}/results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary Table ===", flush=True)
    print("Length | base_unfinetuned | hybrid_lora | base_yarn_x2", flush=True)
    for L in EVAL_LENGTHS:
        b = results["models"].get("base_unfinetuned", {}).get(str(L), {}).get("ppl", "N/A")
        h = results["models"].get("hybrid_lora", {}).get(str(L), {}).get("ppl", "N/A")
        y = results["models"].get("base_yarn_x2", {}).get(str(L), {}).get("ppl", "N/A")
        print(f"{L:<6} | {b!s:<16} | {h!s:<11} | {y!s:<12}", flush=True)

    print(f"Saved: {out_json}", flush=True)


if __name__ == "__main__":
    main()
