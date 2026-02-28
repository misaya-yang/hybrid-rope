#!/usr/bin/env python3
"""
grad_diag.py — Minimal diagnostic patch for HuggingFace Trainer + ResponseOnlyTrainer.

Usage (2 lines added to training script):
    from grad_diag import install_grad_diag
    install_grad_diag()  # call BEFORE trainer.train()

Prints per-step: pre/post clip grad norm, clip coef, supervised tokens,
NaN/Inf detection, lr, scaler status. Zero changes to training logic.
"""
import torch
import sys
from typing import Dict

# ═══════════════════════════════════════════════════════
# COMPONENT 1: Per-step grad norm via clip_grad_norm_ hook
# ═══════════════════════════════════════════════════════

_orig_clip = torch.nn.utils.clip_grad_norm_
_step = [0]
_last_pre = [0.0]
_last_clip = [1.0]
_last_nan = [False]
_last_inf = [False]


def _clip_with_diag(parameters, max_norm, norm_type=2.0, **kwargs):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params = list(parameters)
    graded = [p for p in params if p.grad is not None]

    # NaN / Inf check (sample first 8 param groups for speed)
    has_nan = any(p.grad.isnan().any().item() for p in graded[:8]) if graded else False
    has_inf = any(p.grad.isinf().any().item() for p in graded[:8]) if graded else False

    # Call original: returns pre-clip total norm, clips in-place
    result = _orig_clip(params, max_norm, norm_type=norm_type, **kwargs)
    pre = result.item() if isinstance(result, torch.Tensor) else float(result)

    mn = float(max_norm)
    clip_coef = min(1.0, mn / max(pre, 1e-9))

    _step[0] += 1
    _last_pre[0] = pre
    _last_clip[0] = clip_coef
    _last_nan[0] = has_nan
    _last_inf[0] = has_inf

    # Print on anomaly or every 50 steps
    anomaly = pre < 0.5 or clip_coef < 0.3 or has_nan or has_inf
    if anomaly or _step[0] % 50 == 0:
        flags = []
        if pre < 0.5:       flags.append("LOW_GRAD")
        if clip_coef < 0.3:  flags.append("HEAVY_CLIP")
        if has_nan:          flags.append("NaN!")
        if has_inf:          flags.append("Inf!")
        tag = " ".join(flags) or "ok"
        post = pre * clip_coef
        print(
            f"[GRAD] step={_step[0]:>4d} pre={pre:.6f} post={post:.6f} "
            f"clip_coef={clip_coef:.4f} nan={has_nan} inf={has_inf} [{tag}]",
            flush=True,
        )

    return result


# ═══════════════════════════════════════════════════════
# COMPONENT 2: Enhanced compute_loss with per-step token diagnostics
# ═══════════════════════════════════════════════════════

def _patch_response_only_trainer():
    """Monkey-patch ResponseOnlyTrainer.compute_loss to print per-step label stats."""
    try:
        # Import from the training script's namespace
        from new_lora_longinst_train_v1 import ResponseOnlyTrainer
    except ImportError:
        print("[DIAG] WARNING: Could not import ResponseOnlyTrainer, skipping compute_loss patch", flush=True)
        return

    _orig_compute_loss = ResponseOnlyTrainer.compute_loss

    def _diag_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is not None:
            total = labels.numel()
            supervised = int((labels != -100).sum().item())
            masked = total - supervised
            batch_size = labels.shape[0]
            seq_len = labels.shape[1] if labels.dim() > 1 else total

            # Per-sample supervised token counts
            if labels.dim() > 1:
                per_sample = [(labels[i] != -100).sum().item() for i in range(batch_size)]
            else:
                per_sample = [supervised]

            # Print on anomaly or every 50 steps
            step = getattr(self.state, "global_step", 0)
            is_anomaly = supervised < 200 or min(per_sample) < 10
            if is_anomaly or step % 50 == 0:
                flags = []
                if supervised < 200:        flags.append("FEW_SUP_TOK")
                if min(per_sample) < 10:    flags.append("EMPTY_SAMPLE")
                tag = " ".join(flags) or "ok"
                print(
                    f"[MASK] step={step:>4d} batch={batch_size} seq={seq_len} "
                    f"sup={supervised} masked={masked} per_sample={per_sample} [{tag}]",
                    flush=True,
                )

        return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs,
                                   num_items_in_batch=num_items_in_batch)

    ResponseOnlyTrainer.compute_loss = _diag_compute_loss
    print("[DIAG] Patched ResponseOnlyTrainer.compute_loss", flush=True)


# ═══════════════════════════════════════════════════════
# COMPONENT 3: Enhanced on_log callback with full summary
# ═══════════════════════════════════════════════════════

from transformers import TrainerCallback


class DiagLogCallback(TrainerCallback):
    """Enriched per-logging-step summary using captured grad state."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        step = state.global_step
        loss = logs.get("loss", float("nan"))
        grad = logs.get("grad_norm", float("nan"))
        lr = logs.get("learning_rate", float("nan"))
        eff = int(logs.get("effective_label_tokens", 0))

        mn = float(getattr(args, "max_grad_norm", 1.0) or 1e9)
        clip = min(1.0, mn / max(grad, 1e-9)) if grad > 0 else 1.0
        post = grad * clip

        # AMP / scaler status
        using_bf16 = getattr(args, "bf16", False)
        using_fp16 = getattr(args, "fp16", False)
        if using_bf16:
            amp = "bf16(no_scaler)"
        elif using_fp16:
            amp = "fp16(scaler_active)"
        else:
            amp = "fp32"

        flags = []
        if grad < 0.5:                     flags.append("LOW_GRAD")
        if clip < 0.3:                     flags.append("HEAVY_CLIP")
        if eff < 200:                      flags.append("FEW_SUP_TOK")
        if loss < 0.3:                     flags.append("VERY_LOW_LOSS")
        if loss < 0.3 and grad < 0.5:     flags.append("MEMORIZATION?")
        if _last_nan[0]:                   flags.append("NaN!")
        if _last_inf[0]:                   flags.append("Inf!")
        tag = " ".join(flags) or "OK"

        print(
            f"[DIAG] s={step:>4d} L={loss:.4f} g_pre={grad:.4f} g_post={post:.4f} "
            f"clip={clip:.4f} sup={eff:>4d} lr={lr:.2e} amp={amp} | {tag}",
            flush=True,
        )


# ═══════════════════════════════════════════════════════
# INSTALL: one-call setup
# ═══════════════════════════════════════════════════════

_installed = False

def install_grad_diag():
    """Call once before trainer.train(). Installs all diagnostics."""
    global _installed
    if _installed:
        return
    _installed = True

    # 1. Hook clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = _clip_with_diag
    print("[DIAG] Installed clip_grad_norm_ hook", flush=True)

    # 2. Patch compute_loss (optional — fails gracefully if import doesn't work)
    try:
        _patch_response_only_trainer()
    except Exception as e:
        print(f"[DIAG] compute_loss patch skipped: {e}", flush=True)

    print("[DIAG] Ready. Anomalous steps will be printed automatically.", flush=True)


# Also export the callback for manual addition
diag_callback = DiagLogCallback()
