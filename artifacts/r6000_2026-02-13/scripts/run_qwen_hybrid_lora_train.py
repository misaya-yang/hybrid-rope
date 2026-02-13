import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# Mirror-friendly defaults (works in CN env)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

MODEL_PATH = "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
OUT_DIR = "/opt/dfrope/results/qwen_hybrid_lora"
SEED = 42
SEQ_LEN = int(os.environ.get('SEQ_LEN', '8192'))
TARGET_TOKENS = int(os.environ.get('TARGET_TOKENS', '40000000'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '500'))
DATASET_MODE = os.environ.get('DATASET_MODE', 'tinystories').lower()


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
        cands.append(("model.model.rotary_emb", model.model.rotary_emb))

    for i, layer in enumerate(getattr(model.model, "layers", [])):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "rotary_emb"):
            cands.append((f"layers[{i}].self_attn.rotary_emb", attn.rotary_emb))
        if hasattr(attn, "rotary_fn"):
            rf = attn.rotary_fn
            if hasattr(rf, "inv_freq"):
                cands.append((f"layers[{i}].self_attn.rotary_fn", rf))
            if hasattr(rf, "rotary_emb") and hasattr(rf.rotary_emb, "inv_freq"):
                cands.append((f"layers[{i}].self_attn.rotary_fn.rotary_emb", rf.rotary_emb))

    patched = 0
    seen = set()
    for name, rope in cands:
        if id(rope) in seen:
            continue
        seen.add(id(rope))
        if not hasattr(rope, "inv_freq"):
            continue
        old = rope.inv_freq
        new_inv = inv.to(device=old.device, dtype=old.dtype)
        rope.inv_freq = new_inv
        if hasattr(rope, "max_seq_len_cached"):
            rope.max_seq_len_cached = 0
        patched += 1
        print(f"[RoPE] patched {name}, shape={tuple(new_inv.shape)}")

    if patched == 0:
        raise RuntimeError("No rotary module with inv_freq found")

    return patched, head_dim, inv


def stream_tokens(tokenizer, ds_name, split, text_key, target_tokens, config=None):
    kwargs = {
        "split": split,
        "streaming": True,
        "trust_remote_code": True,
    }
    ds = load_dataset(ds_name, **kwargs) if config is None else load_dataset(ds_name, config, **kwargs)
    ids = []
    next_mark = 2_000_000
    for x in ds:
        txt = x.get(text_key)
        if not txt:
            continue
        ids.extend(tokenizer.encode(txt, add_special_tokens=False))
        if len(ids) >= next_mark:
            print(f"[Data] {ds_name} tokens={len(ids)/1e6:.1f}M")
            next_mark += 2_000_000
        if len(ids) >= target_tokens:
            break
    return ids


def candidate_sources(mode):
    # Keep TinyStories as robust fallback; PG19/SlimPajama are optional for long-text adaptation.
    if mode == 'pg19':
        return [
            ("deepmind/pg19", None, "train", "text"),
            ("roneneldan/TinyStories", None, "train", "text"),
        ]
    if mode == 'slimpajama':
        return [
            ("cerebras/SlimPajama-627B", None, "train", "text"),
            ("roneneldan/TinyStories", None, "train", "text"),
        ]
    # default: fastest/most stable in constrained network env
    return [
        ("roneneldan/TinyStories", None, "train", "text"),
        ("wikitext", "wikitext-103-raw-v1", "train", "text"),
        ("deepmind/pg19", None, "train", "text"),
    ]


def load_long_tokens(tokenizer, target_tokens, seq_len, mode):
    print(f"[Data] mode={mode}, target={target_tokens/1e6:.0f}M tokens")
    last_err = None
    for ds_name, cfg, split, text_key in candidate_sources(mode):
        try:
            t0 = time.time()
            ids = stream_tokens(tokenizer, ds_name, split, text_key, target_tokens, config=cfg)
            n = len(ids) // seq_len
            usable = n * seq_len
            if usable < seq_len:
                raise RuntimeError(f"not enough tokens from {ds_name}")
            toks = torch.tensor(ids[:usable], dtype=torch.long).view(n, seq_len)
            print(
                f"[Data] source={ds_name} cfg={cfg} chunks={n} usable_tokens={usable} "
                f"time={(time.time()-t0)/60:.1f}min"
            )
            return toks, f"{ds_name}:{cfg}"
        except Exception as e:
            last_err = e
            print(f"[Data] source failed {ds_name} cfg={cfg}: {e}")
    raise RuntimeError(f"All data sources failed: {last_err}")


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        x = self.arr[idx]
        return {
            "input_ids": x,
            "labels": x.clone(),
            "attention_mask": torch.ones_like(x),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
    }


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    print("[Env] HF_ENDPOINT=", os.environ.get('HF_ENDPOINT'))
    print(f"[Cfg] seq_len={SEQ_LEN}, target_tokens={TARGET_TOKENS}, max_steps={MAX_STEPS}, mode={DATASET_MODE}")

    print("[Load] tokenizer")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[Load] model 4bit+bf16")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.max_position_embeddings = max(getattr(model.config, "max_position_embeddings", 0), SEQ_LEN)

    patched, head_dim, inv = patch_model_rope(model)
    print(f"[RoPE] patched={patched}, head_dim={head_dim}, inv_range=({inv.min().item():.3e},{inv.max().item():.3e})")

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tokens, source = load_long_tokens(tok, TARGET_TOKENS, SEQ_LEN, DATASET_MODE)
    ds = ChunkDataset(tokens)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=MAX_STEPS,
        warmup_steps=30,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)

    t0 = time.time()
    trainer.train()
    hours = (time.time() - t0) / 3600

    trainer.save_model(f"{OUT_DIR}/final_lora")
    tok.save_pretrained(f"{OUT_DIR}/final_lora")

    out = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "model_path": MODEL_PATH,
        "seq_len": SEQ_LEN,
        "target_tokens": TARGET_TOKENS,
        "max_steps": MAX_STEPS,
        "dataset_mode": DATASET_MODE,
        "data_source": source,
        "train_hours": round(hours, 3),
        "rope": {
            "type": "hybrid_a0.2_t100k",
            "inv_min": float(inv.min().item()),
            "inv_max": float(inv.max().item()),
        },
    }
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[Done]", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
