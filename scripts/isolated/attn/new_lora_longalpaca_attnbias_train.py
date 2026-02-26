#!/usr/bin/env python3
"""
Isolated LoRA training pipeline with:
1) Anchored-Sigmoid inv_freq.copy_() injection
2) Optional attention-logit bias monkey patch
3) Optional macro/micro KL regularization

Safety constraints:
- Never modifies existing training scripts.
- Exits early when sacred Qwen jobs are running.
- Locks base model to Meta-Llama-3-8B-Instruct (8K native).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from attn_patch_llama_attention_bias import AttentionBiasConfig, apply_llama_attention_bias_patch


MODEL_LOCK_NAME = "Meta-Llama-3-8B-Instruct"
MODEL_LOCK_MAX_POS = 8192
MODEL_LOCK_DEFAULT = "meta-llama/Meta-Llama-3-8B-Instruct"
ROPE_THETA_LOCK = 500000.0


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def find_sacred_qwen_jobs() -> List[str]:
    cmd = "ps -eo pid,args | grep -E 'python .*\\.py|queue_qwen|qwen' | grep -v grep || true"
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, check=False)
    rows: List[str] = []
    for raw in p.stdout.splitlines():
        low = raw.lower()
        if "qwen" in low and (
            "eval_longbench.py" in low
            or "train_cross_model_lora" in low
            or "queue_qwen" in low
        ):
            rows.append(raw.strip())
    return rows


def assert_model_lock(base_model_path: str) -> Dict[str, object]:
    low = base_model_path.lower()
    if MODEL_LOCK_NAME.lower() not in low:
        raise RuntimeError(
            f"Model lock violation: require {MODEL_LOCK_NAME}, got {base_model_path}"
        )
    if "3.1" in low or "128k" in low:
        raise RuntimeError("Model lock violation: Llama-3.1 or 128K-native model is forbidden.")

    cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    max_pos = int(getattr(cfg, "max_position_embeddings", -1))
    rope_theta = float(getattr(cfg, "rope_theta", ROPE_THETA_LOCK))
    if max_pos != MODEL_LOCK_MAX_POS:
        raise RuntimeError(
            f"Model lock violation: expected max_position_embeddings={MODEL_LOCK_MAX_POS}, got {max_pos}"
        )
    if abs(rope_theta - ROPE_THETA_LOCK) > 1e-6:
        raise RuntimeError(
            f"Theta lock violation: expected rope_theta={ROPE_THETA_LOCK}, got {rope_theta}"
        )
    return {
        "base_model_path": base_model_path,
        "model_type": str(getattr(cfg, "model_type", "")),
        "max_position_embeddings": max_pos,
        "rope_theta": rope_theta,
    }


def iter_json_records(path: Path) -> Iterable[Dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        return

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item


def normalize_messages(obj: Dict) -> List[Dict[str, str]]:
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        out: List[Dict[str, str]] = []
        for m in msgs:
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role and content:
                out.append({"role": role, "content": content})
        if out:
            return out

    inst = str(obj.get("instruction", "")).strip()
    inp = str(obj.get("input", "")).strip()
    out = str(obj.get("output", obj.get("response", obj.get("answer", "")))).strip()
    if inst and out:
        user = inst if not inp else f"{inst}\n\n{inp}"
        return [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
    return []


def build_wikitext_samples(path: Path, max_samples: int, seed: int) -> List[Dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) >= 500]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    blocks = blocks[:max_samples]
    rows: List[Dict] = []
    for b in blocks:
        cut = max(180, int(len(b) * 0.58))
        cut = min(cut, len(b) - 120)
        if cut <= 0:
            continue
        rows.append(
            {
                "instruction": "Continue the text while preserving topic and style.",
                "input": b[:cut],
                "output": b[cut:],
            }
        )
    return rows


def _random_filler(rng: random.Random, n_tokens: int) -> str:
    vocab = [
        "science",
        "method",
        "context",
        "document",
        "retrieval",
        "analysis",
        "token",
        "sequence",
        "reasoning",
        "anchor",
        "schedule",
        "evidence",
    ]
    return " ".join(vocab[rng.randrange(len(vocab))] for _ in range(max(32, n_tokens)))


def build_synthetic_long_samples(n_samples: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    out: List[Dict] = []
    for idx in range(n_samples):
        key = f"K{rng.randint(100000, 999999)}"
        filler_len = rng.randint(1800, 5200)
        filler = _random_filler(rng, filler_len)
        if idx % 2 == 0:
            prompt = (
                f"{filler}\n\nRemember this key: {key}\n\n"
                "Question: What is the exact key?"
            )
            answer = key
        else:
            needle = f"The hidden passcode is {key}."
            prefix = _random_filler(rng, filler_len // 2)
            suffix = _random_filler(rng, filler_len // 2)
            prompt = (
                f"{prefix}\n\n{needle}\n\n{suffix}\n\n"
                "Question: extract the hidden passcode exactly."
            )
            answer = key
        out.append({"instruction": "Read the long context and answer exactly.", "input": prompt, "output": answer})
    return out


def render_text_samples(
    tokenizer: AutoTokenizer,
    rows: List[Dict],
    max_records: int,
) -> List[str]:
    rendered: List[str] = []
    for row in rows[:max_records]:
        msgs = normalize_messages(row)
        if not msgs:
            continue
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        txt = str(txt).strip()
        if txt:
            rendered.append(txt)
    return rendered


def build_mixed_corpus(args: argparse.Namespace, tokenizer: AutoTokenizer) -> Tuple[List[str], Dict[str, object]]:
    rng = random.Random(args.seed)
    long_rows: List[Dict] = []
    for p in [Path(args.longalpaca_path), Path(args.longqa_path)]:
        if p.as_posix().strip() and p.exists():
            for obj in iter_json_records(p):
                long_rows.append(obj)
                if len(long_rows) >= args.max_long_records:
                    break
        if len(long_rows) >= args.max_long_records:
            break
    if not long_rows:
        raise RuntimeError("No LongAlpaca/LongQA data found.")

    long_texts = render_text_samples(tokenizer, long_rows, max_records=args.max_long_records)
    if len(long_texts) < 500:
        raise RuntimeError(f"Too few valid long-instruction samples: {len(long_texts)}")

    synthetic_count = int(round(len(long_texts) * float(args.synthetic_ratio)))
    synthetic_rows = build_synthetic_long_samples(synthetic_count, seed=args.seed + 17)
    synthetic_texts = render_text_samples(tokenizer, synthetic_rows, max_records=synthetic_count)

    wiki_texts: List[str] = []
    if float(args.wikitext_ratio) > 0:
        wiki_rows = build_wikitext_samples(Path(args.wikitext_train_path), args.max_wiki_samples, args.seed + 23)
        wiki_keep = int(round(len(long_texts) * float(args.wikitext_ratio)))
        wiki_texts = render_text_samples(tokenizer, wiki_rows, max_records=wiki_keep)

    merged = long_texts + synthetic_texts + wiki_texts
    rng.shuffle(merged)

    info = {
        "long_texts": len(long_texts),
        "synthetic_texts": len(synthetic_texts),
        "wikitext_texts": len(wiki_texts),
        "total_texts": len(merged),
        "synthetic_ratio_target": float(args.synthetic_ratio),
        "wikitext_ratio_target": float(args.wikitext_ratio),
    }
    return merged, info


def tokenize_dataset(texts: List[str], tokenizer: AutoTokenizer, max_seq_len: int) -> Dataset:
    ds = Dataset.from_dict({"text": texts})

    def _tok(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        tok["labels"] = [x[:] for x in tok["input_ids"]]
        return tok

    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    return ds


def split_dataset(ds: Dataset, seed: int) -> Tuple[Dataset, Dataset]:
    ds = ds.shuffle(seed=seed)
    n = len(ds)
    n_val = max(32, int(n * 0.01))
    n_train = max(1, n - n_val)
    train_ds = ds.select(range(0, n_train))
    val_ds = ds.select(range(n_train, n))
    return train_ds, val_ds


def find_rotary_modules_with_inv_freq(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    mods: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and isinstance(getattr(module, "inv_freq"), torch.Tensor):
            mods.append((name, module))
    return mods


def build_anchored_inv_freq(head_dim: int, base: float, anchor: float, slope_raw: float, center_ratio: float) -> torch.Tensor:
    n = head_dim // 2
    idx = torch.arange(n, dtype=torch.float32)
    base_inv = 1.0 / (float(base) ** (idx / float(n)))
    slope = float(slope_raw) / float(head_dim)
    center = float(center_ratio) * float(n)
    sig = torch.sigmoid(slope * (idx - center))
    return base_inv / (1.0 + (float(anchor) - 1.0) * sig)


def inject_inv_freq_copy(
    model: torch.nn.Module,
    inv_freq_path: str,
    anchor_factor: float,
    slope_raw: float,
    center_ratio: float,
    rope_base: float,
) -> Dict[str, object]:
    modules = find_rotary_modules_with_inv_freq(model)
    if not modules:
        raise RuntimeError("No rotary modules with inv_freq found.")
    ref = modules[0][1].inv_freq
    target_dim = int(ref.numel() * 2)

    if inv_freq_path.strip():
        inv = torch.load(inv_freq_path, map_location="cpu")
        if not isinstance(inv, torch.Tensor):
            raise RuntimeError(f"Invalid inv_freq tensor at {inv_freq_path}")
        inv = inv.to(dtype=ref.dtype)
    else:
        inv = build_anchored_inv_freq(
            head_dim=target_dim,
            base=float(rope_base),
            anchor=float(anchor_factor),
            slope_raw=float(slope_raw),
            center_ratio=float(center_ratio),
        ).to(dtype=ref.dtype)

    with torch.no_grad():
        for _, module in modules:
            module.inv_freq.copy_(inv.to(device=module.inv_freq.device, dtype=module.inv_freq.dtype))

    return {
        "rotary_module_count": len(modules),
        "inv_freq_numel": int(inv.numel()),
        "inv_sha256": hashlib.sha256(inv.detach().cpu().numpy().tobytes()).hexdigest(),
        "from_file": bool(inv_freq_path.strip()),
    }


def distance_histogram_from_attentions(attentions: Tuple[torch.Tensor, ...], max_delta: int) -> torch.Tensor:
    hist = torch.zeros(max_delta, dtype=torch.float32, device=attentions[0].device)
    for att in attentions:
        # [bs, heads, q, k]
        probs = att.float().mean(dim=(0, 1))  # [q, k]
        q, k = probs.shape
        q_idx = torch.arange(q, device=probs.device)
        k_idx = torch.arange(k, device=probs.device)
        delta = (q_idx[:, None] - k_idx[None, :]).abs().reshape(-1)
        weights = probs.reshape(-1)
        hist = hist + torch.bincount(delta, weights=weights, minlength=max_delta).to(hist.dtype)
    hist = hist / hist.sum().clamp_min(1e-8)
    return hist


class MacroMicroTrainer(Trainer):
    def __init__(self, *args, p_ref: Optional[torch.Tensor], lambda_micro: float, lambda_macro: float, s2_power: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_ref = p_ref
        self.lambda_micro = float(lambda_micro)
        self.lambda_macro = float(lambda_macro)
        self.s2_power = float(s2_power)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        outputs = model(**inputs, output_attentions=True)
        loss = outputs.loss
        if outputs.attentions:
            att = outputs.attentions[0].float()  # first layer for efficiency
            q = int(att.shape[-2])
            k = int(att.shape[-1])
            q_idx = torch.arange(q, device=att.device)
            k_idx = torch.arange(k, device=att.device)
            delta = (q_idx[:, None] - k_idx[None, :]).abs().to(torch.float32)
            s2 = torch.pow(delta + 1.0, -self.s2_power)
            micro = (att.mean(dim=1) * s2.view(1, q, k)).mean()

            macro = torch.tensor(0.0, device=att.device)
            if self.p_ref is not None:
                hist = distance_histogram_from_attentions((att,), max_delta=max(q, k))
                pref = self.p_ref.to(device=hist.device, dtype=hist.dtype)
                if pref.numel() != hist.numel():
                    pref = torch.nn.functional.interpolate(
                        pref.view(1, 1, -1),
                        size=hist.numel(),
                        mode="linear",
                        align_corners=False,
                    ).view(-1)
                    pref = pref / pref.sum().clamp_min(1e-8)
                macro = torch.sum(hist * (hist.clamp_min(1e-8).log() - pref.clamp_min(1e-8).log()))
            loss = loss + self.lambda_micro * micro + self.lambda_macro * macro
        return (loss, outputs) if return_outputs else loss


def estimate_reference_prior(model: torch.nn.Module, train_ds: Dataset, n_batches: int, batch_size: int) -> torch.Tensor:
    model.eval()
    rows = train_ds.select(range(0, min(len(train_ds), n_batches * batch_size)))
    hist_acc: Optional[torch.Tensor] = None
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            input_ids = torch.tensor(batch["input_ids"], device=model.device)
            attn_mask = torch.tensor(batch["attention_mask"], device=model.device)
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True, use_cache=False)
            if out.attentions:
                h = distance_histogram_from_attentions(out.attentions[:1], max_delta=input_ids.shape[1])
                hist_acc = h if hist_acc is None else (hist_acc + h)
    model.train()
    if hist_acc is None:
        return torch.ones(64, dtype=torch.float32) / 64.0
    hist_acc = hist_acc / hist_acc.sum().clamp_min(1e-8)
    return hist_acc.detach().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated attention-integrated LoRA trainer.")
    ap.add_argument("--base_model_path", type=str, default=MODEL_LOCK_DEFAULT)
    ap.add_argument("--output_dir", type=Path, default=Path("artifacts/new_attnbias_v1/train"))
    ap.add_argument("--run_name", type=str, default="attn_lora_r32_s800")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--longalpaca_path", type=str, default="/root/autodl-tmp/dfrope/datasets/LongAlpaca-12k.json")
    ap.add_argument("--longqa_path", type=str, default="/root/autodl-tmp/dfrope/datasets/LongQA.jsonl")
    ap.add_argument("--wikitext_train_path", type=str, default="/root/autodl-tmp/wikitext_data/train.txt")
    ap.add_argument("--max_long_records", type=int, default=12000)
    ap.add_argument("--max_wiki_samples", type=int, default=4000)
    ap.add_argument("--synthetic_ratio", type=float, default=0.30)
    ap.add_argument("--wikitext_ratio", type=float, default=0.20)
    ap.add_argument("--max_seq_len", type=int, default=16384)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--attn_implementation", type=str, default="auto")
    ap.add_argument("--attn_bias_mode", type=str, choices=["off", "bias", "bias+gate"], default="off")
    ap.add_argument("--gamma_mode", type=str, choices=["constant", "per-layer", "head-group"], default="constant")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--gamma_by_layer", type=str, default="")
    ap.add_argument("--gamma_head_low", type=float, default=0.0)
    ap.add_argument("--gamma_head_high", type=float, default=0.0)
    ap.add_argument("--gate_tau", type=float, default=0.0)
    ap.add_argument("--gate_tg", type=float, default=1.0)
    ap.add_argument("--use_macro_micro_kl", action="store_true")
    ap.add_argument("--lambda_micro", type=float, default=0.01)
    ap.add_argument("--lambda_macro", type=float, default=0.01)
    ap.add_argument("--pref_warmup_batches", type=int, default=4)
    ap.add_argument("--inv_freq_path", type=str, default="")
    ap.add_argument("--anchor_factor", type=float, default=4.0)
    ap.add_argument("--slope_raw", type=float, default=20.0)
    ap.add_argument("--center_ratio", type=float, default=0.70)
    ap.add_argument("--rope_base", type=float, default=ROPE_THETA_LOCK)
    ap.add_argument("--local_files_only", action="store_true", default=True)
    args = ap.parse_args()

    sacred = find_sacred_qwen_jobs()
    if sacred:
        print("[SAFETY] Sacred Qwen process detected. Exit without training.")
        for row in sacred:
            print(row)
        raise SystemExit(2)

    model_info = assert_model_lock(args.base_model_path)
    if abs(float(args.rope_base) - ROPE_THETA_LOCK) > 1e-6:
        raise RuntimeError(f"rope_base must remain {ROPE_THETA_LOCK}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = args.output_dir / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    texts, data_info = build_mixed_corpus(args, tokenizer)
    ds = tokenize_dataset(texts, tokenizer=tokenizer, max_seq_len=int(args.max_seq_len))
    train_ds, val_ds = split_dataset(ds, seed=args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation if args.attn_implementation != "auto" else None,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )

    rope_info = inject_inv_freq_copy(
        model=model,
        inv_freq_path=args.inv_freq_path,
        anchor_factor=float(args.anchor_factor),
        slope_raw=float(args.slope_raw),
        center_ratio=float(args.center_ratio),
        rope_base=float(args.rope_base),
    )
    custom_inv_path = artifacts / "custom_inv_freq.pt"
    first_rotary = find_rotary_modules_with_inv_freq(model)[0][1].inv_freq.detach().cpu()
    torch.save(first_rotary, custom_inv_path)

    patch_cfg = AttentionBiasConfig(
        mode=args.attn_bias_mode,
        gamma_mode=args.gamma_mode,
        gamma=float(args.gamma),
        gamma_by_layer=args.gamma_by_layer,
        gamma_head_low=float(args.gamma_head_low),
        gamma_head_high=float(args.gamma_head_high),
        tau=float(args.gate_tau),
        tg=float(args.gate_tg),
        enabled=(args.attn_bias_mode != "off"),
    )
    patch_handle = apply_llama_attention_bias_patch(model, patch_cfg)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=0.0,
        bias="none",
        target_modules=parse_csv(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    p_ref: Optional[torch.Tensor] = None
    trainer_cls = Trainer
    trainer_kwargs: Dict[str, object] = {}
    if args.use_macro_micro_kl:
        p_ref = estimate_reference_prior(
            model=model,
            train_ds=train_ds,
            n_batches=int(args.pref_warmup_batches),
            batch_size=int(args.per_device_train_batch_size),
        )
        trainer_cls = MacroMicroTrainer
        trainer_kwargs = {
            "p_ref": p_ref,
            "lambda_micro": float(args.lambda_micro),
            "lambda_macro": float(args.lambda_macro),
            "s2_power": 2.0,
        }

    targs = TrainingArguments(
        output_dir=str(out),
        run_name=args.run_name,
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=max(1, int(args.per_device_train_batch_size)),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to=[],
        evaluation_strategy="steps",
        eval_steps=int(args.save_steps),
        save_total_limit=2,
        dataloader_num_workers=2,
    )

    trainer = trainer_cls(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
        **trainer_kwargs,
    )
    train_result = trainer.train()
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))

    run_cfg = {
        "timestamp": now(),
        "script": Path(__file__).name,
        "model_lock": model_info,
        "data_info": data_info,
        "rope_info": rope_info,
        "attn_bias": vars(patch_cfg),
        "macro_micro_kl": {
            "enabled": bool(args.use_macro_micro_kl),
            "lambda_micro": float(args.lambda_micro),
            "lambda_macro": float(args.lambda_macro),
            "p_ref_len": int(p_ref.numel()) if p_ref is not None else 0,
        },
        "train_result": dict(train_result.metrics),
        "custom_inv_freq_path": custom_inv_path.as_posix(),
        "custom_inv_freq_sha256": sha256_file(custom_inv_path),
        "code_hashes": {
            "new_lora_longalpaca_attnbias_train.py": sha256_file(Path(__file__)),
            "attn_patch_llama_attention_bias.py": sha256_file(Path(__file__).resolve().parent / "attn_patch_llama_attention_bias.py"),
        },
    }
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    patch_handle.restore()
    print(f"[DONE] saved run_config.json -> {(out / 'run_config.json').as_posix()}")


if __name__ == "__main__":
    main()
