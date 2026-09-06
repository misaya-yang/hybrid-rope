#!/usr/bin/env python3
"""Track B v2: 500M-token CPT under a FROZEN static position system (arm Z).

Recipe name: R12_7B_CPT_500M_V1. Derived from R12_DUAL_ARM_CPT_SFT_V1
(track_b_train.py, kept untouched) with exactly THREE engineering changes,
all required by the 32GB card + >=500M-token scale; supervision, schedule,
and constraints are otherwise unchanged:

  1. Data: CPT_DATA_FROZEN_V2_500M (30,592 x 16K PG19 segments, 501.2M
     tokens, one epoch; mmap-loaded). Validation reuses frozen V1 npy.
  2. KL teacher: native_pool_v3's 512 train rows all carry fixed
     prediction_positions (512/512 verified), so V1's kl_replay_loss never
     resamples; the teacher distribution is precomputed once
     (build_kl_cache.py) and the trainer never loads the teacher
     (student-only VRAM; math identical up to bf16 storage).
  3. Scale: CPT_STEPS/TOKEN_CAP derived from the data manifest (7,648
     updates x 65,536 tokens); token-milestone checkpoints 125M/250M/375M +
     final; rolling resume_latest every 16 steps (adapter + norm/emb deltas +
     optimizer + scheduler) because a multi-day run must survive restarts.

Everything else identical to V1: LoRA r16 a16 all linears + all RMSNorm
affine + embed_tokens trainable; base linears, lm_head, rotary (inv_freq +
gain) frozen; AdamW 2e-5 betas(.9,.95) wd0 clip 1.0; 5% warmup cosine to
2e-6; KL weight 1.0 on one <=2K replay row per update; Phase B = 64 SFT
updates with optimizer reset (views/replay unchanged).

Usage:
  python track_b_train_v2.py --arm Z --model <olmo7b dir> --tables <tables> \
      --cpt-data <data/cpt_500m> --kl-cache <kl_cache.npz> \
      --sft-views <data/sft/views.jsonl> \
      --replay-manifest <native_pool_v3/manifest.json> \
      --out-root <runs/Z_CPT_500M> --phase all
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

RECIPE = "R12_7B_CPT_500M_V1"
SEQ_LEN = 16384
LORA_R = LORA_ALPHA = 16
LR = 2e-5
MIN_LR = 2e-6
WARMUP_FRAC = 0.05
KL_WEIGHT = 1.0
SFT_STEPS = 64
RESUME_EVERY = 16
TOKEN_MILESTONES = (125_000_000, 250_000_000, 375_000_000)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Model + frozen table install (identical to V1)
# ---------------------------------------------------------------------------
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
    rotary.inv_freq.requires_grad_(False)
    return model, gain, entry["float32_sha256"]


def mark_extra_trainable(peft_model):
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


def _n_layers(model) -> int:
    return model.config.num_hidden_layers if hasattr(model, "config") \
        else model.base_model.config.num_hidden_layers


def apply_trainable_config(model):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                     target_modules=list(TARGET_MODULES), bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    mark_extra_trainable(model)
    n_lora = sum(1 for n, p in model.named_parameters()
                 if p.requires_grad_ and "lora_" in n)
    assert n_lora == 2 * len(TARGET_MODULES) * _n_layers(model)
    return model


def load_resume(student, adapter_dir: Path, extra_path: Path):
    from peft import PeftModel
    model = PeftModel.from_pretrained(student, str(adapter_dir), is_trainable=True)
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
    inner = student.base_model.model
    return inner.model, inner.lm_head.weight


def dense_lm_loss(student, input_ids, labels, flce):
    backbone, head_w = hidden_and_lmhead(student)
    h = backbone(input_ids=input_ids, attention_mask=None).last_hidden_state
    return flce(h.reshape(-1, h.shape[-1]), head_w, labels.reshape(-1))


def kl_from_cache(student, cache, replay_row, device):
    """KL(teacher || student) for one replay row, teacher from frozen cache.

    replay_row: {"ids", "positions", "file_row"}; cache entry p_<file_row>
    already holds teacher probs at exactly the surviving positions
    (1 <= p < min(len(ids), max_len)), so this mirrors V1's kl_replay_loss.
    """
    if replay_row["cache_n_pos"] == 0:
        return None
    ids = replay_row["ids"][:2048]
    pos = torch.tensor(replay_row["positions"][:replay_row["cache_n_pos"]],
                       dtype=torch.long, device=device)
    backbone_s, head_s = hidden_and_lmhead(student)
    h_s = backbone_s(input_ids=torch.tensor([ids], dtype=torch.long,
                                            device=device)).last_hidden_state
    logits_s = F.linear(h_s[0, pos], head_s)
    logp_s = F.log_softmax(logits_s.float(), dim=-1)
    p_t = replay_row["cache_probs"].to(device).float()
    return F.kl_div(logp_s, p_t, reduction="batchmean")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_cpt_v2(data_dir: Path):
    m = json.loads((data_dir / "manifest.json").read_text())
    assert m["status"] == "CPT_DATA_FROZEN_V2_500M", m["status"]
    train = np.load(data_dir / m["train"]["path"], mmap_mode="r")
    assert tuple(train.shape) == tuple(m["train"]["shape"]), train.shape
    val = np.load(m["validation"]["path"])
    return train, val, m


def load_replay_pool_cached(manifest_path: Path, cache_path: Path):
    """Pool rows WITH cache attached; file_row = key into the cache npz."""
    m = json.loads(Path(manifest_path).read_text())
    assert m.get("status") == "NATIVE_REPLAY_POOL_V1"
    rows_file = Path(manifest_path).parent / m["rows_path"]
    assert hashlib.sha256(rows_file.read_bytes()).hexdigest() == m["rows_sha256"]
    cache = np.load(cache_path, allow_pickle=False)
    cache_meta = json.loads((Path(str(cache_path) + ".meta.json")).read_text())
    assert cache_meta["status"] == "KL_TEACHER_CACHE_V1"
    assert cache_meta["replay_manifest_sha256"] == m["rows_sha256"]
    index = json.loads(bytes(cache["index_json"]).decode())
    by_row = {ix["row"]: ix for ix in index}
    seqs = []
    for i, line in enumerate(rows_file.read_text().splitlines()):
        r = json.loads(line)
        if r.get("split", "train") != "train":
            continue
        ids = r.get("input_ids") or (list(r["prompt_ids"]) + list(r["target_ids"]))
        ix = by_row[i]
        seqs.append({"ids": list(ids),
                     "positions": ix.get("positions") or [],
                     "cache_n_pos": ix["n_pos"],
                     "cache_probs": torch.from_numpy(cache[f"p_{i}"])
                     if ix["n_pos"] > 0 else None,
                     "file_row": i})
    assert len(seqs) == 512
    assert sum(1 for s in seqs if s["cache_n_pos"] > 0) == \
        cache_meta["n_rows_with_positions"]
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
        b.sort(key=lambda v: len(v["prompt_ids"]))
        batches.append(b)
    return batches


def collate_views(batch, pad_id):
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
# Optimizer / schedule (identical to V1)
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

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(student, step_dir: Path, meta: dict):
    from safetensors.torch import save_file
    step_dir.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(step_dir / "adapter")
    extra = {n: p.detach().float().cpu() for n, p in student.named_parameters()
             if p.requires_grad_ and "lora_" not in n}
    save_file(extra, str(step_dir / "extra_norm_embedding.safetensors"))
    manifest = {"status": "R12_TRACKB_CHECKPOINT_V2", "recipe": RECIPE, **meta,
                "extra_params": sorted(extra)}
    (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {step_dir.name}: {len(extra)} extra tensors", flush=True)


def save_resume_latest(student, opt, sched, step: int, out_root: Path, meta: dict):
    """Rolling crash-resume bundle (single copy, overwritten)."""
    d = out_root / "resume_latest"
    d.mkdir(parents=True, exist_ok=True)
    tmp = out_root / "resume_latest_tmp"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    student.save_pretrained(tmp / "adapter")
    from safetensors.torch import save_file
    extra = {n: p.detach().float().cpu() for n, p in student.named_parameters()
             if p.requires_grad_ and "lora_" not in n}
    save_file(extra, str(tmp / "extra_norm_embedding.safetensors"))
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    torch.save(sched.state_dict(), tmp / "scheduler.pt")
    (tmp / "meta.json").write_text(json.dumps({"step": step, **meta}, indent=2))
    import shutil
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)


def restore_resume_latest(student, out_root: Path, opt, sched):
    d = out_root / "resume_latest"
    meta = json.loads((d / "meta.json").read_text())
    student = load_resume(student, d / "adapter",
                          d / "extra_norm_embedding.safetensors")
    opt.load_state_dict(torch.load(d / "optimizer.pt", map_location="cpu",
                                   weights_only=True))
    sched.load_state_dict(torch.load(d / "scheduler.pt", map_location="cpu",
                                     weights_only=True))
    return student, meta


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def eval_dense(student, val, flce, device):
    student.eval()
    tot, n = 0.0, 0
    with torch.inference_mode():
        for row in val:
            ids = torch.tensor(np.asarray(row), dtype=torch.long, device=device)
            tot += dense_lm_loss(student, ids[:-1].unsqueeze(0),
                                 ids[1:].unsqueeze(0), flce).item()
            n += 1
    student.train()
    return tot / max(1, n)


def phase_a(args, student, replay, flce, device, cpt_steps: int, train, val):
    opt, sched = build_optim_sched(student, cpt_steps)
    order = np.random.RandomState(args.seed).permutation(len(train))
    start_step, tokens_seen, milestones_done = 0, 0, set()
    if args.resume_latest:
        student, rmeta = restore_resume_latest(student, Path(args.out_root), opt, sched)
        start_step, tokens_seen = rmeta["step"], rmeta["tokens_seen"]
        milestones_done = set(rmeta.get("milestones_done", []))
        print(f"RESUMED at step {start_step} ({tokens_seen} tokens)", flush=True)
        # re-install on device after CPU-side state-dict surgery
        student.to(device)
        student.train()
        student.enable_input_require_grads()
    t0 = time.time()
    if args.grad_ckpt:
        try:
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.gradient_checkpointing_enable()
    for step in range(start_step + 1, cpt_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(args.cpt_accum):
            row = train[order[(step - 1) * args.cpt_accum + micro]]
            ids = torch.tensor(np.asarray(row), dtype=torch.long, device=device)
            inp, lab = ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)
            loss = dense_lm_loss(student, inp, lab, flce) / args.cpt_accum
            loss.backward()
            loss_acc += loss.item()
            tokens_seen += SEQ_LEN
        rrow = replay[(step - 1) % len(replay)]
        kl = kl_from_cache(student, None, rrow, device)
        if kl is not None:
            (KL_WEIGHT * kl).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad_], 1.0)
        opt.step(); sched.step()
        if step % 8 == 0:
            print(f"A step {step}/{cpt_steps} lm={loss_acc:.4f} "
                  f"kl={kl.item() if kl is not None else 0:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} tok={tokens_seen} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if step % 64 == 0:
            val_loss = eval_dense(student, val[: args.val_rows], flce, device)
            print(f"A step {step} val_loss={val_loss:.4f}", flush=True)
        for ms in TOKEN_MILESTONES:
            if tokens_seen >= ms and ms not in milestones_done:
                milestones_done.add(ms)
                save_checkpoint(student, Path(args.out_root) / f"CPT{ms//10**6}M", {
                    "phase": "A", "step": step, "tokens_seen": tokens_seen,
                    "arm": args.arm, "table_sha256": args.table_sha,
                    "wall_seconds": round(time.time() - t0, 1)})
        if step % RESUME_EVERY == 0 or step == cpt_steps:
            save_resume_latest(student, opt, sched, step, Path(args.out_root),
                               {"tokens_seen": tokens_seen, "arm": args.arm,
                                "phase": "A",
                                "milestones_done": sorted(milestones_done)})
        if tokens_seen >= args.token_cap:
            break
    assert tokens_seen <= args.token_cap, "token cap violated"
    save_checkpoint(student, Path(args.out_root) / "CPT_FINAL", {
        "phase": "A", "step": cpt_steps, "tokens_seen": tokens_seen,
        "arm": args.arm, "table_sha256": args.table_sha,
        "wall_seconds": round(time.time() - t0, 1)})
    return student


def phase_b(args, student, replay, flce, device):
    views = load_sft_views(Path(args.sft_views))
    batches = make_sft_batches(views, seed=args.seed + 1)
    opt, sched = build_optim_sched(student, SFT_STEPS)  # optimizer RESET per plan
    pad_id = student.config.pad_token_id if student.config.pad_token_id is not None \
        else student.config.eos_token_id
    t0 = time.time()
    for step in range(1, SFT_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for mb in range(0, 8, args.sft_micro):
            ids, lab, mask = collate_views(batches[step - 1][mb:mb + args.sft_micro],
                                           pad_id)
            ids, lab, mask = ids.to(device), lab.to(device), mask.to(device)
            backbone, head_w = hidden_and_lmhead(student)
            h = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            loss = flce(h.reshape(-1, h.shape[-1]), head_w, lab.reshape(-1)) \
                / math.ceil(8 / args.sft_micro)
            loss.backward()
            loss_acc += loss.item()
        rrows = [replay[(step * 2 + j) % len(replay)] for j in range(2)]
        kls = [kl_from_cache(student, None, r, device) for r in rrows]
        kls = [k for k in kls if k is not None]
        if kls:
            kl = sum(kls) / len(kls)
            (KL_WEIGHT * kl).backward()
        else:
            kl = None
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad_], 1.0)
        opt.step(); sched.step()
        if step % 4 == 0 or step == SFT_STEPS:
            print(f"B step {step}/{SFT_STEPS} sft={loss_acc:.4f} "
                  f"kl={kl.item() if kl is not None else 0:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    save_checkpoint(student, Path(args.out_root) / "SFT64", {
        "phase": "B", "step": SFT_STEPS, "arm": args.arm,
        "table_sha256": args.table_sha, "wall_seconds": round(time.time() - t0, 1)})
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["Z"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--cpt-data", required=True)
    ap.add_argument("--kl-cache", required=True)
    ap.add_argument("--sft-views", required=True)
    ap.add_argument("--replay-manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--phase", default="all", choices=["A", "B", "all"])
    ap.add_argument("--resume-adapter", default=None)
    ap.add_argument("--resume-extra", default=None)
    ap.add_argument("--resume-latest", action="store_true",
                    help="continue phase A from resume_latest bundle")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--cpt-accum", type=int, default=4)
    ap.add_argument("--sft-micro", type=int, default=1)
    ap.add_argument("--val-rows", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    out_root = Path(args.out_root)
    fresh = not (out_root / "resume_latest").exists()
    if out_root.exists() and any(out_root.iterdir()) and fresh \
            and not args.resume_latest:
        raise FileExistsError(f"{out_root} not empty and no resume requested")
    out_root.mkdir(parents=True, exist_ok=True)

    train, val, data_m = load_cpt_v2(Path(args.cpt_data))
    assert tuple(train.shape)[1] == SEQ_LEN + 1
    args.token_cap = data_m["total_train_tokens"]
    cpt_steps = len(train) // args.cpt_accum
    assert cpt_steps * args.cpt_accum == len(train)

    student, gain, table_sha = load_student(args.model, Path(args.tables), args.arm)
    args.table_sha = table_sha
    if args.phase == "B":
        assert args.resume_adapter
        student = load_resume(student, Path(args.resume_adapter),
                              Path(args.resume_extra or "nonexistent.safetensors"))
    elif not args.resume_latest:
        student = apply_trainable_config(student)
    student.enable_input_require_grads()
    student.to(device)
    student.train()

    replay, _ = load_replay_pool_cached(Path(args.replay_manifest),
                                        Path(args.kl_cache))

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)

    run_meta = {"recipe": RECIPE, "arm": args.arm, "table_sha256": table_sha,
                "gain": gain, "base_model": args.model,
                "cpt_data_manifest": data_m["status"],
                "token_cap": args.token_cap, "cpt_steps": cpt_steps,
                "cpt_accum": args.cpt_accum, "seq_len": SEQ_LEN,
                "kl_mode": "precomputed teacher cache (build_kl_cache.py)",
                "phase_plan": f"A:{cpt_steps} CPT updates + B:{SFT_STEPS} SFT "
                              "updates (optimizer reset)",
                "trainable": "lora r16 all linears + all RMSNorm affine + embed_tokens",
                "frozen": "base linears, lm_head, rotary inv_freq + gain",
                "differences_vs_R12_DUAL_ARM_CPT_SFT_V1":
                    "data V2 500M; KL teacher precomputed cache (identical math, "
                    "pool rows all have fixed prediction_positions); milestone + "
                    "rolling-resume checkpointing"}
    if fresh or args.phase == "B":
        (out_root / "run_config.json").write_text(json.dumps(run_meta, indent=2))

    if args.phase in ("A", "all"):
        student = phase_a(args, student, replay, flce, device, cpt_steps, train, val)
    if args.phase == "all" or args.phase == "B":
        student = phase_b(args, student, replay, flce, device)
    print("DONE", args.arm, args.phase, flush=True)


if __name__ == "__main__":
    main()
