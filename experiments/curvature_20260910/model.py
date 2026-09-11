"""Frozen-model plumbing shared by every probe in this package.

Two transformers generations are in play: the local CPU box runs 4.57, the
experiment server runs 5.15 (where `rope_scaling` became `rope_parameters` and
`inv_freq` became an `nn.Buffer`).  Everything version-specific is isolated
here so a probe never has to care.

Design notes that matter for correctness:

* `Qwen2RotaryEmbedding.forward` is decorated `@torch.no_grad()` in both
  generations.  `grad_forward()` swaps in a grad-capable twin with identical
  arithmetic, so a backward pass through the frequencies exists at all.  Every
  probe here is forward-only by default; the grad path exists so the two can be
  cross-checked rather than trusted.
* The vocabulary projection dominates memory at long length (131072 x 151936
  x 2 bytes = 38 GiB if materialised).  Every long forward passes
  `logits_to_keep`, which slices hidden states BEFORE the head, so a 128K
  forward costs activations and nothing else.
"""
from __future__ import annotations

import math
import types

import numpy as np
import torch

K = 64


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_frozen(path, dtype, device, attn=None):
    """Load a frozen model across transformers generations and device setups.

    Three independent variations have to be absorbed: `dtype` (5.x) vs
    `torch_dtype` (4.x); `device_map` needing accelerate, which may or may not
    be installed; and CPU, where `device_map` buys nothing anyway.  Each is a
    fallback rather than a version switch, so an unknown future rename still
    loads instead of crashing in the plumbing before any science happens.
    """
    from transformers import AutoModelForCausalLM

    kw = dict(attn_implementation=attn) if attn else {}
    base = dict(local_files_only=True, **kw)
    attempts = []
    if device != "cpu":
        attempts.append(dict(base, device_map={"": device}))
    attempts.append(dict(base))
    last = None
    for dev_map in attempts:
        for key in ("dtype", "torch_dtype"):
            try:
                model = AutoModelForCausalLM.from_pretrained(path, **{key: dtype}, **dev_map)
                return model.to(device).eval()
            except (TypeError, ValueError) as exc:
                last = exc
    raise RuntimeError(f"could not load {path}: {last!r}")


class FrozenRoPE:
    """A frozen causal LM whose 64 RoPE frequencies are a settable knob."""

    def __init__(self, path, dtype="bf16", device=None, attn=None, log_softmax_dtype=torch.float32):
        self.device = device or pick_device()
        self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[dtype] \
            if isinstance(dtype, str) else dtype
        self.path = path
        if self.device == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        self.model = load_frozen(path, self.dtype, self.device, attn)
        self.model.requires_grad_(False)
        # Lazy, and allowed to fail: every probe that reads a prepared .npy
        # token stream never needs a tokenizer, and a tokenizer that will not
        # load (missing protobuf, a gated repo, an incomplete download) should
        # not stop the science.  Only the gold-answer path asks for one.
        self._tokenizer = None
        self._tokenizer_tried = False

        self.rotary = self.model.model.rotary_emb
        if getattr(self.rotary, "rope_type", "default") != "default":
            raise ValueError("cannot stack dynamic scaling on a frozen reference")
        n = int(self.rotary.inv_freq.numel())
        if n != K:
            raise ValueError(f"expected {K} frequencies, got {n}")
        self.native_inv_freq = self.rotary.inv_freq.detach().float().cpu().numpy().copy()
        self.head_dim = n * 2
        self._grad_patched = False

    @property
    def tokenizer(self):
        if not self._tokenizer_tried:
            self._tokenizer_tried = True
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            except Exception as exc:                                   # noqa: BLE001
                self._tokenizer = None
                self._tokenizer_error = repr(exc)
        if self._tokenizer is None:
            raise RuntimeError(
                f"no tokenizer at {self.path} ({getattr(self, '_tokenizer_error', 'unknown')}). "
                "Everything that reads prepared token ids is unaffected; only the "
                "gold-answer long objective needs one.")
        return self._tokenizer

    # -- table installation -------------------------------------------------
    def install(self, values, gain, track_grad=False):
        v = torch.tensor(np.asarray(values, dtype=np.float32), device=self.device)
        if v.shape != (K,) or not torch.isfinite(v).all() or (v < 0).any():
            raise ValueError("expected 64 finite nonnegative frequencies in slot order")
        if not math.isfinite(float(gain)) or float(gain) <= 0:
            raise ValueError("invalid gain")
        v = v.clone().requires_grad_(track_grad)
        self.rotary.inv_freq = v
        self.rotary.original_inv_freq = v.detach().clone()
        self.rotary.attention_scaling = float(gain)
        return v

    def install_table(self, table):
        return self.install(table["values_float32"], table["gain"],
                            track_grad=table.get("track_grad", False))

    def _patch_grad_rotary(self):
        if self._grad_patched:
            return

        # Remove only the outer no_grad decorator from the installed model's
        # implementation. Reimplementing Qwen's arithmetic here used to cast
        # OLMo2's FP32 cos/sin to BF16 and changed the model being differentiated.
        # The stock body retains its model-specific dtype, disabled-autocast
        # region and inner dynamic-rope wrapper (default RoPE is required above).
        if type(self.rotary).__name__ not in ("Qwen2RotaryEmbedding", "Olmo2RotaryEmbedding"):
            raise ValueError("gradient rotary is qualified only for Qwen2 and OLMo2")
        stock = self.rotary.forward.__func__
        forward = getattr(stock, "__wrapped__", None)
        if forward is None:
            raise RuntimeError("stock rotary lacks the expected outer no_grad wrapper")
        self.rotary.forward = types.MethodType(forward, self.rotary)
        self._grad_patched = True

    def enable_grad_path(self):
        self._patch_grad_rotary()

    # -- forward ------------------------------------------------------------
    def logits(self, ids, keep, want_grad=False):
        """Last `keep` positions' logits, in float32.  ids: (1, L) long tensor."""
        if keep > ids.shape[1]:
            raise ValueError(f"keep={keep} exceeds sequence {ids.shape[1]}")
        self.enable_grad_path()
        with torch.set_grad_enabled(want_grad):
            out = self.model(ids, use_cache=False, logits_to_keep=keep)
        return out.logits[0].float()

    def nll(self, ids, keep=512):
        """Mean next-token NLL over the last `keep` targets.  Forward only."""
        with torch.no_grad():
            lg = self.logits(ids, keep + 1)
        return torch.nn.functional.cross_entropy(lg[:-1], ids[0, -keep:], reduction="mean")

    def nll_per_token(self, ids, keep=512):
        with torch.no_grad():
            lg = self.logits(ids, keep + 1)
        return torch.nn.functional.cross_entropy(lg[:-1], ids[0, -keep:], reduction="none")

    def log_probs(self, ids, keep, want_grad=False):
        lg = self.logits(ids, keep, want_grad=want_grad)
        return torch.log_softmax(lg, dim=-1)

    def grad_wrt_freq(self, ids, keep):
        """Exact d(mean NLL)/d(inv_freq) for all 64 slots in one backward."""
        self.install(self.native_inv_freq, 1.0, track_grad=False)
        v = self.install(self.rotary.inv_freq.detach().float().cpu().numpy(),
                         float(self.rotary.attention_scaling), track_grad=True)
        lg = self.logits(ids, keep + 1, want_grad=True)
        loss = torch.nn.functional.cross_entropy(lg[:-1], ids[0, -keep:])
        g, = torch.autograd.grad(loss, v)
        return loss.detach().float().item(), g.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Forward-only measurements.  No backward pass, no activation storage: these
# are the primitives the probes are built from, and they are what makes a 128K
# sweep affordable on a 32 GiB card.
# ---------------------------------------------------------------------------
@torch.no_grad()
def output_kl(model, ids, keep, base_logp, table):
    """E_u[ KL( p_base(.|u) || p_table(.|u) ) ] over the last `keep` positions.

    This is the native-preservation metric: it has no first-order term, its
    quadratic form is the Fisher information F_N = E[J^T(diag p - pp^T)J] >= 0,
    it measures behaviour drift rather than weight drift, and it needs no
    assumption about which frequencies matter.  A plain cross-entropy
    difference would have a non-zero first-order term and an indefinite
    second-order form; that is why it is not used.

    Evaluated as sum_v p (log p_base - log p_table) with the two log-probability
    tensors taken directly, in float64.  The obvious `p_base.exp().log()` round
    trip costs ~1e-7 absolute on each log-probability, which is the same size as
    the whole KL for a small delta, and on a near-uniform output it is what
    decides the sign of a Fisher diagonal.  That was a real failure, not a
    hypothetical one: an earlier version of this function returned negative
    Fisher entries.
    """
    model.install_table(table)
    lp = model.log_probs(ids, keep).double()
    b = base_logp.double()
    p = b.exp()
    return float((p * (b - lp)).sum(-1).mean())


@torch.no_grad()
def pair_nll(model, ids, keep, table):
    model.install_table(table)
    return float(model.nll(ids, keep))


def fisher_diagonal(model, ids, keep, base_logp, base_table, delta, slots):
    """F_N[j,j] from one forward per slot: D_N(delta*e_j) = 0.5 F_jj delta^2 + O(d^3).

    D_N(0) = 0 exactly, so the first-order term is absent by construction and a
    single one-sided forward per slot is enough.  Cheaper than a Fisher-vector
    product and it needs no backward at any length.
    """
    nu0 = np.asarray(base_table["values_float32"], dtype=np.float64)
    out = {}
    for j in slots:
        v = nu0.copy()
        v[j] = v[j] * math.exp(delta)
        t = dict(base_table, values_float32=v)
        out[j] = 2.0 * output_kl(model, ids, keep, base_logp, t) / (delta * delta)
    return out


def fisher_cross(model, ids, keep, base_logp, base_table, delta, pairs):
    """F_N[j,k] for named pairs, from one forward per pair.

    D_N(delta(e_j+e_k)) = 0.5 delta^2 (F_jj + 2 F_jk + F_kk), so a pair forward
    combined with the two diagonals isolates the cross term.  This is what
    makes the s28/s29 coupling prediction testable without a 64x64 Hessian.
    """
    nu0 = np.asarray(base_table["values_float32"], dtype=np.float64)
    diag = fisher_diagonal(model, ids, keep, base_logp, base_table, delta,
                           sorted({j for p in pairs for j in p}))
    out = {}
    for j, k in pairs:
        v = nu0.copy()
        v[j] *= math.exp(delta)
        v[k] *= math.exp(delta)
        d = 2.0 * output_kl(model, ids, keep, base_logp, dict(base_table, values_float32=v)) / (delta * delta)
        out[(j, k)] = 0.5 * (d - diag[j] - diag[k])
    return out, diag


def fisher_scaling(model, ids, keep, base_logp, base_table, slots, delta,
                   factors=(1.0, 2.0, 4.0)):
    """Is the output KL actually quadratic in the step, at this delta?

    F_jj is read as D_N(delta e_j) / (delta^2 / 2), which equals the Fisher only
    while the cubic term is negligible.  Probing the same slots at delta,
    2 delta and 4 delta and fitting the log-log slope gives the exponent
    directly.  2 is the model; well below 2 means the probes are measuring
    outside the region where F_N describes the model, and both the budget eps
    and the predicted gain stop meaning what they say.

    Cheap -- 3 forwards per slot on a handful of slots -- and it is the one
    check that decides whether the Fisher numbers are usable at all.
    """
    nu0 = np.asarray(base_table["values_float32"], dtype=np.float64)
    per_slot, exponent = {}, {}
    for j in slots:
        vals = []
        for f in factors:
            v = nu0.copy()
            v[j] = v[j] * math.exp(delta * f)
            vals.append(output_kl(model, ids, keep, base_logp,
                                  dict(base_table, values_float32=v)))
        a = [delta * f for f in factors]
        y = np.asarray(vals, dtype=np.float64)
        per_slot[j] = {f"{f:g}": val for f, val in zip(factors, vals)}
        # local slopes between consecutive pairs: the pair between the two
        # smallest deltas is the one that says whether the delta in use is
        # inside the quadratic region; a falling last slope means it is not
        local = [float(np.log(y[i + 1] / y[i]) / np.log(a[i + 1] / a[i]))
                 if y[i] > 0 and y[i + 1] > 0 else None
                 for i in range(len(a) - 1)]
        ok = y > 0
        exponent[j] = dict(
            local=local,
            first=local[0] if local else None,
            overall=(float(np.sum(np.log(y[ok]) * np.log(np.asarray(a)[ok]))
                           / np.sum(np.log(np.asarray(a)[ok]) ** 2)) if ok.sum() >= 2 else None))
    firsts = [e["first"] for e in exponent.values() if e["first"] is not None]
    med = float(np.median(firsts)) if firsts else None
    return dict(per_slot=per_slot, exponent=exponent, exponent_median=med,
                quadratic=bool(med is not None and 1.7 <= med <= 2.3))


def fisher_all_at_once(model, ids, keep, base_logp, base_table, delta, diag):
    """D_N(delta * 1) against 0.5 delta^2 sum_j F_jj: joint vs diagonal.

    One forward.  Read this as a bounded warning, not as a diagonality test.
    Moving all 64 slots at once creates 64*63 off-diagonal terms against 64
    diagonal ones, so a large ratio here is expected even when every individual
    F_jk is small -- it says the joint move is not explained by the diagonal,
    which is true of any correlated metric.  The precise version of the question
    is asked in solve_kkt.py, where the diagonal and full quadratic forms are
    compared on the step that is actually taken.
    """
    nu0 = np.asarray(base_table["values_float32"], dtype=np.float64)
    v = nu0 * math.exp(delta)
    d = output_kl(model, ids, keep, base_logp, dict(base_table, values_float32=v))
    pred = 0.5 * delta * delta * float(sum(diag.values()))
    return dict(measured=d, predicted_diag=pred,
                joint_over_diag=(d / pred) if pred else None)


__all__ = [
    "K", "FrozenRoPE", "pick_device", "output_kl", "pair_nll",
    "fisher_diagonal", "fisher_cross", "fisher_scaling", "fisher_all_at_once",
]
