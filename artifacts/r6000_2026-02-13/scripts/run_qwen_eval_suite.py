import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

MODEL_PATH = "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
ADAPTER_PATH = "/opt/dfrope/results/qwen_hybrid_lora/final_lora"
OUT_PATH = "/opt/dfrope/results/qwen_hybrid_lora/eval_suite.json"

SEED = 42
DEVICE = "cuda"
LENGTHS = [8192, 16384, 24576, 32768]
PPL_CHUNKS = 3
TASK_SAMPLES = 6


def bnb_cfg():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def try_load_base_with_rope_scaling(rope_scaling):
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cfg.rope_scaling = rope_scaling
    cfg.max_position_embeddings = max(getattr(cfg, "max_position_embeddings", 32768), 32768)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=cfg,
        quantization_config=bnb_cfg(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_model(mode):
    if mode == "base":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_cfg(),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return model

    if mode == "base_yarn8":
        # Qwen config variants differ across transformers versions; try both key styles.
        rope_variants = [
            {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 32768},
            {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 32768},
        ]
        last_err = None
        for rv in rope_variants:
            try:
                return try_load_base_with_rope_scaling(rv)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"failed to load base_yarn8: {last_err}")

    if mode == "hybrid_lora":
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_cfg(),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        return model

    raise ValueError(f"unknown mode: {mode}")


def score_continuation_logprob(model, prefix_ids, cont_ids):
    # Returns total logprob of cont_ids conditioned on prefix_ids.
    ids = torch.tensor(prefix_ids + cont_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids[:, :-1]).logits
        logp = F.log_softmax(logits, dim=-1)

    # cont token i is predicted by position (prefix_len-1 + i)
    start = len(prefix_ids) - 1
    total = 0.0
    for i, tid in enumerate(cont_ids):
        total += float(logp[0, start + i, tid].item())
    return total


def load_eval_tokens(tok, target_tokens=2_500_000):
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True, trust_remote_code=True)
    ids = []
    for x in ds:
        txt = x.get("text")
        if not txt:
            continue
        ids.extend(tok.encode(txt, add_special_tokens=False))
        if len(ids) >= target_tokens:
            break
    return ids


def load_filler_tokens(tok, target_tokens=3_000_000):
    # Use wikitext as open-domain filler for synthetic downstream tasks.
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True, trust_remote_code=True)
    ids = []
    for x in ds:
        txt = x.get("text")
        if not txt:
            continue
        ids.extend(tok.encode(txt, add_special_tokens=False))
        if len(ids) >= target_tokens:
            break
    if len(ids) < 500_000:
        # fallback to TinyStories if wikitext stream is slow/unavailable
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)
        for x in ds:
            txt = x.get("text")
            if not txt:
                continue
            ids.extend(tok.encode(txt, add_special_tokens=False))
            if len(ids) >= target_tokens:
                break
    return ids


def eval_ppl(model, val_ids, lengths):
    out = {}
    for L in lengths:
        losses = []
        for i in range(PPL_CHUNKS):
            s = i * L
            e = s + L
            if e > len(val_ids):
                break
            x = torch.tensor(val_ids[s:e], dtype=torch.long, device=DEVICE).unsqueeze(0)
            try:
                with torch.no_grad():
                    logits = model(x[:, :-1]).logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
                lv = float(loss.item())
                if not math.isfinite(lv):
                    continue
                losses.append(lv)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        if losses:
            out[str(L)] = {
                "ppl": round(float(math.exp(sum(losses) / len(losses))), 4),
                "loss_mean": round(float(sum(losses) / len(losses)), 6),
                "n": len(losses),
            }
        else:
            out[str(L)] = {"ppl": None, "loss_mean": None, "n": 0}
    return out


def build_base_context_ids(filler_ids, total_len, rng):
    if total_len >= len(filler_ids):
        return filler_ids[:total_len]
    start = rng.randint(0, len(filler_ids) - total_len - 1)
    return filler_ids[start:start + total_len]


def eval_passkey_mc(model, tok, filler_ids, lengths, rng):
    result = {}
    option_letters = ["A", "B", "C", "D"]

    for L in lengths:
        accs = []
        for _ in range(TASK_SAMPLES):
            passkey = "".join(str(rng.randint(0, 9)) for _ in range(6))
            wrong = set()
            while len(wrong) < 3:
                w = "".join(str(rng.randint(0, 9)) for _ in range(6))
                if w != passkey:
                    wrong.add(w)
            options = [passkey] + list(wrong)
            rng.shuffle(options)
            correct_idx = options.index(passkey)

            snippet = f" Important record: the passkey is {passkey}. "
            depth = rng.choice([0.1, 0.5, 0.9])

            q = (
                "\nQuestion: Which option is the passkey? "
                f"A){options[0]} B){options[1]} C){options[2]} D){options[3]}\n"
                "Answer:"
            )

            snippet_ids = tok.encode(snippet, add_special_tokens=False)
            q_ids = tok.encode(q, add_special_tokens=False)
            room = L - len(snippet_ids) - len(q_ids) - 16
            if room < 256:
                continue
            base_ids = build_base_context_ids(filler_ids, room, rng)
            pos = int(len(base_ids) * depth)
            prefix_ids = base_ids[:pos] + snippet_ids + base_ids[pos:] + q_ids

            try:
                scores = []
                for letter in option_letters:
                    cont = tok.encode(" " + letter, add_special_tokens=False)
                    scores.append(score_continuation_logprob(model, prefix_ids, cont))
                pred = max(range(4), key=lambda i: scores[i])
                accs.append(1.0 if pred == correct_idx else 0.0)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        result[str(L)] = {
            "acc": round(float(sum(accs) / len(accs)), 4) if accs else None,
            "n": len(accs),
        }
    return result


def eval_kv_mc(model, tok, filler_ids, lengths, rng):
    result = {}
    letters = ["A", "B", "C", "D"]

    for L in lengths:
        accs = []
        for _ in range(TASK_SAMPLES):
            kvs = []
            for i in range(8):
                k = f"k{i}_{rng.randint(100, 999)}"
                v = "".join(str(rng.randint(0, 9)) for _ in range(5))
                kvs.append((k, v))
            target_k, target_v = rng.choice(kvs)

            # compose a small key-value memory block
            mem = " ".join([f"{k}:{v};" for k, v in kvs])
            snippet = f" Memory table => {mem} "

            wrong = set()
            while len(wrong) < 3:
                w = "".join(str(rng.randint(0, 9)) for _ in range(5))
                if w != target_v:
                    wrong.add(w)
            options = [target_v] + list(wrong)
            rng.shuffle(options)
            correct_idx = options.index(target_v)

            q = (
                f"\nQuestion: For key {target_k}, which value is correct? "
                f"A){options[0]} B){options[1]} C){options[2]} D){options[3]}\nAnswer:"
            )

            snippet_ids = tok.encode(snippet, add_special_tokens=False)
            q_ids = tok.encode(q, add_special_tokens=False)
            room = L - len(snippet_ids) - len(q_ids) - 16
            if room < 256:
                continue
            base_ids = build_base_context_ids(filler_ids, room, rng)
            pos = int(len(base_ids) * rng.choice([0.1, 0.5, 0.9]))
            prefix_ids = base_ids[:pos] + snippet_ids + base_ids[pos:] + q_ids

            try:
                scores = []
                for letter in letters:
                    cont = tok.encode(" " + letter, add_special_tokens=False)
                    scores.append(score_continuation_logprob(model, prefix_ids, cont))
                pred = max(range(4), key=lambda i: scores[i])
                accs.append(1.0 if pred == correct_idx else 0.0)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        result[str(L)] = {
            "acc": round(float(sum(accs) / len(accs)), 4) if accs else None,
            "n": len(accs),
        }
    return result


def run_one(mode, tok, val_ids, filler_ids, rng):
    print(f"\n[Eval] loading model: {mode}")
    t0 = time.time()
    model = load_model(mode)
    print(f"[Eval] model ready in {(time.time()-t0)/60:.1f} min")

    out = {
        "ppl": eval_ppl(model, val_ids, LENGTHS),
        "passkey_mc": eval_passkey_mc(model, tok, filler_ids, LENGTHS, rng),
        "kv_mc": eval_kv_mc(model, tok, filler_ids, LENGTHS, rng),
    }

    del model
    torch.cuda.empty_cache()
    return out


def main():
    if not Path(ADAPTER_PATH).exists():
        raise FileNotFoundError(f"Adapter not found: {ADAPTER_PATH}")

    random.seed(SEED)
    torch.manual_seed(SEED)

    tok = load_tokenizer()
    print("[Data] loading eval tokens")
    val_ids = load_eval_tokens(tok)
    print(f"[Data] val tokens={len(val_ids)}")

    print("[Data] loading filler tokens")
    filler_ids = load_filler_tokens(tok)
    print(f"[Data] filler tokens={len(filler_ids)}")

    results = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "model_path": MODEL_PATH,
        "adapter_path": ADAPTER_PATH,
        "lengths": LENGTHS,
        "seed": SEED,
        "metrics": {},
    }

    modes = ["base", "base_yarn8", "hybrid_lora"]
    for m in modes:
        rng = random.Random(SEED)
        try:
            results["metrics"][m] = run_one(m, tok, val_ids, filler_ids, rng)
        except Exception as e:
            results["metrics"][m] = {"error": str(e)}

        Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Save] {OUT_PATH}")

    print("[Done]")


if __name__ == "__main__":
    main()
