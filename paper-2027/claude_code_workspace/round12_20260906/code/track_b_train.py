#!/usr/bin/env python3
"""Track B: two-arm long-context adaptation under a FROZEN static position system.

Recipe name (new, per plan Section 7.7): R12_DUAL_ARM_CPT_SFT_V1.

One run trains ONE arm (Y or Z). The static table + rotary gain of the arm are
installed once and never touched by the optimizer; only:
  - LoRA r=16 alpha=16 dropout=0 on every attention/MLP linear,
  - every RMSNorm affine weight,
  - the input embedding,
are trainable. Base linears, LM head and the rotary (inv_freq + gain) stay frozen.

Phase A (CPT): 512 updates; each update = 4 x 16K dense causal-LM microsteps
(one 16385-token real document segment each; 33,554,432 tokens total cap) plus
ONE native replay row (<=2K tokens) matched to the frozen native teacher with a
full-vocab KL at sampled positions, weight 1.0.
Phase B (SFT): 64 updates; effective batch 8 = 2 qa_8192 + 2 qa_16384 + 4
binding views (micro-batch 4 x accum 2), loss on answer tokens only (incl. EOS),
plus 2 native replay rows/update. Optimizer is RESET at the phase boundary.

Optimizer both phases: AdamW lr 2e-5, betas (0.9,0.95), wd 0, grad clip 1.0,
5% linear warmup then cosine to 2e-6.

Checkpoints: CPT128 / CPT256 / CPT512 / SFT64, each = PEFT adapter + extra
safetensors holding norm/embedding deltas + manifest (hashes + recipe record).

Fused linear CE via liger_kernel (installed on the server). Teacher = same base
weights with NATIVE inv_freq and amplitude 1.0, frozen, bf16, no grad.

Usage (Phase A then B, arm Y):
  python track_b_train.py --arm Y --model <olmo1b> --tables <round12_tables> \
      --cpt-data <data/cpt> --sft-views <data/sft/views.jsonl> \
      --replay-manifest <native_pool_v3/manifest.json> \
      --out-root <runs/Y_CPT_SFT> --phase all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RECIPE = "R12_DUAL_ARM_CPT_SFT_V1"
SEQ_LEN = 16384
TOKEN_CAP = 33_554_432
CPT_STEPS = 512
CPT_SAVES = (128, 256, 512)
SFT_STEPS = 64
LORA_R = LORA_ALPHA = 16
LR = 2e-5
MIN_LR = 2e-6
WARMUP_FRAC = 0.05
KL_WEIGHT = 1.0
KL_SAMPLE_POSITIONS = 512
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Model + frozen table install (byte-compatible with frozen engine semantics)
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_student(model_path: str, tables_dir: Path, arm: str):
    from transformers import AutoModelForCausalLM
    manifest = json.loads((tables_dir / "manifest_round12.json").read_text())
    assert manifest["status"] == "ROUND12_STATIC_TABLES_FROZEN_V1"
    entry = manifest["arms"][arm]
    table = np.load(tables_dir / entry["path"], allow_pickle=False)
    assert table.dtype == np.float32
    assert sha256_bytes(np.ascontiguousarray(table).tobytes()) == entry["float32_sha256"]
    gain = float(entry["rotary_amplitude"])

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    rotary = model.model.rotary_emb
    assert rotary.inv_freq.shape == table.shape
    rotary.inv_freq.copy_(torch.from_numpy(table).to(rotary.inv_freq))
    rotary.original_inv_freq = rotary.inv_freq.detach().clone()
    rotary.attention_scaling = gain
    # rotary params must never enter the optimizer
    rotary.inv_freq.requires_grad_(False)
    return model, gain, entry["float32_sha256"]


def load_teacher(model_path: str):
    """Frozen native teacher: default inv_freq, amplitude exactly 1.0."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.model.rotary_emb.attention_scaling = 1.0
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def mark_extra_trainable(peft_model):
    """Enable all RMSNorm affine weights + input embedding (never lm_head/rotary)."""
    n_norm = n_emb = 0
    for name, p in peft_model.named_parameters():
        if "lm_head" in name or "rotary" in name:
            continue
        if "norm" in name and name.endswith(".weight"):
            p.requires_grad_(True)
            n_norm += 1
        elif "embed_tokens" in name:
            p.requires_grad_(True)
            n_emb += 1
    assert n_norm > 0 and n_emb > 0, f"norm/embedding not found: {n_norm}/{n_emb}"
    return n_norm, n_emb


def apply_trainable_config(model):
    """LoRA on all linears; all RMSNorm affine + input embedding trainable."""
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                     target_modules=list(TARGET_MODULES), bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    mark_extra_trainable(model)
    n_lora = sum(1 for n, p in model.named_parameters() if p.requires_grad_ and "lora_" in n)
    assert n_lora == 2 * len(TARGET_MODULES) * _n_layers(model), \
        f"lora param count unexpected: {n_lora}"
    return model


def _n_layers(model) -> int:
    return model.config.num_hidden_layers if hasattr(model, "config") \
        else model.base_model.config.num_hidden_layers


def load_resume(model, adapter_dir: Path, extra_path: Path):
    """Phase-B resume: attach saved adapter (trainable), load norm/embed deltas,
    re-enable the norm+embedding requires_grad flags."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    if extra_path.exists():
        from safetensors.torch import load_file
        extra = load_file(str(extra_path))
        names = dict(model.named_parameters())
        for k, v in extra.items():
            assert k in names, f"resume param {k} not found"
            names[k].data.copy_(v.to(names[k].dtype))
    mark_extra_trainable(model)
    return model


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def hidden_and_lmhead(student):
    """Return (backbone OLMo2Model, lm_head weight) through the PEFT wrappers."""
    inner = student.base_model.model          # OLMo2ForCausalLM
    return inner.model, inner.lm_head.weight


def dense_lm_loss(student, input_ids, labels, flce):
    """Liger fused linear CE wants flat (B*L, H) hidden states and flat labels
    already shifted by +1 (input_ids[:-1] vs ids[1:]); no internal shift."""
    backbone, head_w = hidden_and_lmhead(student)
    h = backbone(input_ids=input_ids, attention_mask=None).last_hidden_state
    return flce(h.reshape(-1, h.shape[-1]), head_w, labels.reshape(-1))


def kl_replay_loss(student, teacher, replay_rows, device, rng, max_len: int = 2048):
    """Full-vocab KL(teacher || student) at sampled positions of each replay row.

    Positions: the pool's prediction_positions when available (the answer
    positions), else a deterministic random sample of KL_SAMPLE_POSITIONS.
    Rows are truncated to max_len tokens (plan: <=2K replay tokens per update).
    """
    backbone_s, head_s = hidden_and_lmhead(student)
    total, n_rows = 0.0, 0
    for row in replay_rows:
        ids = row["ids"][:max_len]
        L = len(ids)
        if L < 8:
            continue
        cand = [p for p in (row["positions"] or []) if 1 <= p < L]
        if not cand:
            cand = sorted(rng.sample(range(1, L), min(KL_SAMPLE_POSITIONS, L - 1)))
        pos = torch.tensor(cand, dtype=torch.long, device=device)
        ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            h_t = teacher.model(input_ids=ids).last_hidden_state
            logits_t = F.linear(h_t[0, pos], teacher.lm_head.weight.to(h_t.dtype))
        h_s = backbone_s(input_ids=ids).last_hidden_state
        logits_s = F.linear(h_s[0, pos], head_s)
        logp_s = F.log_softmax(logits_s.float(), dim=-1)
        p_t = F.softmax(logits_t.float(), dim=-1)
        total = total + F.kl_div(logp_s, p_t, reduction="batchmean")
        n_rows += 1
    return (total / n_rows) if n_rows else None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cpt(data_dir: Path):
    m = json.loads((data_dir / "manifest.json").read_text())
    assert m["status"] == "CPT_DATA_FROZEN_V1"
    train = np.load(data_dir / "train_2048x16385.npy")
    val = np.load(data_dir / "validation.npy")
    assert train.shape == (2048, SEQ_LEN + 1), train.shape
    return train, val, m


def load_replay_pool(manifest_path: Path):
    """native_pool_v3 (NATIVE_REPLAY_POOL_V1): train-split rows with full
    input_ids and optional prediction_positions (the answer positions)."""
    m = json.loads(Path(manifest_path).read_text())
    assert m.get("status") == "NATIVE_REPLAY_POOL_V1", "unexpected replay pool"
    rows_file = Path(manifest_path).parent / m["rows_path"]
    assert hashlib.sha256(rows_file.read_bytes()).hexdigest() == m["rows_sha256"], \
        "replay pool rows hash mismatch"
    seqs = []
    for line in rows_file.read_text().splitlines():
        r = json.loads(line)
        if r.get("split", "train") != "train":
            continue
        ids = r.get("input_ids") or (list(r["prompt_ids"]) + list(r["target_ids"]))
        seqs.append({"ids": list(ids),
                     "positions": list(r["prediction_positions"])
                     if r.get("prediction_positions") else None})
    assert len(secs) == 512, f"expected 512 train replay rows, got {len(secs)}"
    return seqs, m


def load_sft_views(path: Path):
    views = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert len(views) == 512, len(views)
    return views


def make_sft_batches(views, seed=7):
    import random as _r
    rng = _r.Random(seed)
    strata = {}
    for v in views:
        strata.setdefault(v["stratum"], []).append(v)
    for s in strata:
        rng.shuffle(strata[s])
    qa8, qa16, bind = strata["qa_8192"], strata["qa_16384"], strata["binding"]
    assert (len(qa8), len(qa16), len(bind)) == (128, 128, 256)
    batches = []
    for i in range(SFT_STEPS):
        b = qa8[2 * i:2 * i + 2] + qa16[2 * i:2 * i + 2] + bind[4 * i:4 * i + 4]
        assert len(b) == 8
        b.sort(key=lambda v: len(v["prompt_ids"]))  # pad-friendly order
        batches.append(b)
    return batches


def collate_views(batch, pad_id):
    """Pad prompts+targets; labels = -100 outside the target span."""
    full = [v["prompt_ids"] + v["target_ids"] for v in batch]
    L = max(len(f) for f in full)
    input_ids, labels, mask = [], [], []
    for v, f in zip(batch, full):
        pad = L - len(f)
        input_ids.append(f[:-1] + [pad_id] * pad)
        lab = [-100] * (len(v["prompt_ids"]) - 1) + f[len(v["prompt_ids"]):] + [-100] * pad
        labels.append(lab)
        mask.append([1] * (len(f) - 1) + [0] * pad)
    dev = lambda x: torch.tensor(x, dtype=torch.long)
    return dev(input_ids), dev(labels), dev(mask)


# ---------------------------------------------------------------------------
# Optimizer / schedule
# ---------------------------------------------------------------------------
def build_optim_sched(model, total_steps):
    params = [p for p in model.parameters() if p.requires_grad_]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    warmup = max(1, int(WARMUP_FRAC * total_steps))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return max(MIN_LR / LR, cos)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    return opt, sched


# ---------------------------------------------------------------------------
# Checkpointing (norm + embedding deltas travel with the product)
# ---------------------------------------------------------------------------
def save_checkpoint(student, step_dir: Path, meta: dict):
    from safetensors.torch import save_file
    step_dir.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(step_dir / "adapter")
    extra = {n: p.detach().float().cpu() for n, p in student.named_parameters()
             if p.requires_grad_ and "lora_" not in n}
    save_file(extra, str(step_dir / "extra_norm_embedding.safetensors"))
    manifest = {"status": "R12_TRACKB_CHECKPOINT_V1", "recipe": RECIPE, **meta,
                "extra_params": sorted(extra),
                "adapter_sha256": None}
    (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {step_dir.name}: {len(extra)} extra tensors", flush=True)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def phase_a(args, student, teacher, flce, device, rng):
    train, val, _ = load_cpt(Path(args.cpt_data))
    replay, _rep_m = load_replay_pool(Path(args.replay_manifest))
    opt, sched = build_optim_sched(student, CPT_STEPS)
    order = np.random.RandomState(args.seed).permutation(len(train))
    tokens_seen = 0
    t0 = time.time()
    if args.grad_ckpt:
        try:
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.gradient_checkpointing_enable()
    for step in range(1, CPT_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(args.cpt_accum):
            row = train[order[(step - 1) * args.cpt_accum + micro]]
            ids = torch.tensor(row, dtype=torch.long, device=device)
            inp, lab = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
            loss = dense_lm_loss(student, inp, lab, flce) / args.cpt_accum
            loss.backward()
            loss_acc += loss.item()
            tokens_seen += SEQ_LEN
        # one <=2K native replay row per update, KL weight 1
        rrow = replay[(step - 1) % len(replay)]
        kl = kl_replay_loss(student, teacher, [rrow], device, rng)
        if kl is not None:
            (KL_WEIGHT * kl).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad_], 1.0)
        opt.step(); sched.step()
        if step % 8 == 0 or step in CPT_SAVES:
            print(f"A step {step}/{CPT_STEPS} lm={loss_acc:.4f} "
                  f"kl={kl.item() if kl is not None else 0:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} tok={tokens_seen} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if step % 64 == 0:
            val_loss = eval_dense(student, val[: args.val_rows], flce, device)
            print(f"A step {step} val_loss={val_loss:.4f}", flush=True)
        if step in CPT_SAVES:
            save_checkpoint(student, Path(args.out_root) / f"CPT{step}", {
                "phase": "A", "step": step, "tokens_seen": tokens_seen,
                "token_cap": TOKEN_CAP, "table_sha256": args.table_sha,
                "arm": args.arm, "wall_seconds": round(time.time() - t0, 1)})
        if tokens_seen >= TOKEN_CAP:
            break
    assert tokens_seen <= TOKEN_CAP, "token cap violated"
    return student


def eval_dense(student, val, flce, device):
    student.eval()
    tot, n = 0.0, 0
    with torch.inference_mode():
        for row in val:
            ids = torch.tensor(row, dtype=torch.long, device=device)
            tot += dense_lm_loss(student, ids[:-1].unsqueeze(0),
                                 ids[1:].unsqueeze(0), flce).item()
            n += 1
    student.train()
    return tot / max(1, n)


def phase_b(args, student, teacher, flce, device, rng):
    views = load_sft_views(Path(args.sft_views))
    batches = make_sft_batches(views, seed=args.seed + 1)
    replay, _ = load_replay_pool(Path(args.replay_manifest))
    opt, sched = build_optim_sched(student, SFT_STEPS)  # optimizer RESET per plan
    pad_id = student.config.pad_token_id if student.config.pad_token_id is not None \
        else student.config.eos_token_id
    t0 = time.time()
    for step in range(1, SFT_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for mb in range(0, 8, args.sft_micro):
            ids, lab, mask = collate_views(batches[step - 1][mb:mb + args.sft_micro], pad_id)
            ids, lab, mask = ids.to(device), lab.to(device), mask.to(device)
            backbone, head_w = hidden_and_lmhead(student)
            h = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            loss = flce(h.reshape(-1, h.shape[-1]), head_w, lab.reshape(-1)) \
                / (8 // args.sft_micro)
            loss.backward()
            loss_acc += loss.item()
        rrows = [replay[(step * 2 + j) % len(replay)] for j in range(2)]
        kl = kl_replay_loss(student, teacher, rrows, device, rng)
        if kl is not None:
            (KL_WEIGHT * kl).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad_], 1.0)
        opt.step(); sched.step()
        if step % 4 == 0 or step == SFT_STEPS:
            print(f"B step {step}/{SFT_STEPS} sft={loss_acc:.4f} "
                  f"kl={kl.item() if kl is not None else 0:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.0f}s", flush=True)
    save_checkpoint(student, Path(args.out_root) / "SFT64", {
        "phase": "B", "step": SFT_STEPS, "arm": args.arm,
        "table_sha256": args.table_sha, "wall_seconds": round(time.time() - t0, 1)})
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["Y", "Z"])
    ap.add_argument("--model", required=True, help="base OLMo-2-0425-1B-Instruct dir")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--cpt-data", required=True)
    ap.add_argument("--sft-views", required=True)
    ap.add_argument("--replay-manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--phase", default="all", choices=["A", "B", "all"])
    ap.add_argument("--resume-adapter", default=None, help="CPT512/adapter for phase B-only")
    ap.add_argument("--resume-extra", default=None)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--cpt-accum", type=int, default=4)
    ap.add_argument("--sft-micro", type=int, default=4)
    ap.add_argument("--val-rows", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random as _r
    rng = _r.Random(args.seed)
    device = "cuda"

    out_root = Path(args.out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"{out_root} not empty; preserved, not overwritten")
    out_root.mkdir(parents=True, exist_ok=True)

    student, gain, table_sha = load_student(args.model, Path(args.tables), args.arm)
    args.table_sha = table_sha
    if args.phase == "B":
        assert args.resume_adapter, "phase B requires --resume-adapter (e.g. CPT512/adapter)"
        student = load_resume(student, Path(args.resume_adapter),
                              Path(args.resume_extra or "nonexistent.safetensors"))
    else:
        student = apply_trainable_config(student)
    # Required for gradient flow to the trainable input embedding when
    # gradient checkpointing is on.
    student.enable_input_require_grads()
    student.to(device)
    student.train()

    teacher = load_teacher(args.model).to(device)

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

    run_meta = {"recipe": RECIPE, "arm": args.arm, "table_sha256": table_sha,
                "gain": gain, "base_model": args.model,
                "phase_plan": "A:512 CPT updates + B:64 SFT updates (optimizer reset)",
                "trainable": "lora r16 all linears + all RMSNorm affine + embed_tokens",
                "frozen": "base linears, lm_head, rotary inv_freq + gain"}
    (out_root / "run_config.json").write_text(json.dumps(run_meta, indent=2))

    if args.phase in ("A", "all"):
        phase_a(args, student, teacher, flce, device, rng)
    if args.phase == "all":
        # continue in-process: optimizer reset happens inside phase_b
        phase_b(args, student, teacher, flce, device, rng)
    elif args.phase == "B":
        phase_b(args, student, teacher, flce, device, rng)
    print("DONE", args.arm, args.phase, flush=True)


if __name__ == "__main__":
    main()
