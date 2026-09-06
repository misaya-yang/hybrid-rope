#!/usr/bin/env python3
"""P0 diagnostic B — where does the table/long-shift error happen?
(Pro synthesis §11.1, computing the §2.3 and §5.2 exact identities.)

Read-only. For pre-registered diagnostic instances (transport_views
single_evidence, far 16384, worlds 0/1) and declared observation locations
(layers x ALL heads x the answer-decision query row):

  (A) §5.2 attention-write identity, compact' vs far within ONE system:
        o~ - o0 =  sum_S a0_j (u~_j - u0_j)          (value-write change)
                 + sum_S (abar_j - a0_j) u~_j        (routing change)
                 + beta (ubar_D - ubar_S)            (added-token competition)
      compact' = prefix + evidence block + suffix: exactly the original
      tokens of the far input (token-preserving). The far layout is
      prefix + background body[block @ source_block] + suffix; prefix/suffix/
      block are re-derived with the tokenizer and verified token-exact against
      the frozen record before any decomposition.

  (B) §2.3 direct/inherited logit identity between TWO systems (ref=T0 vs
      test in {Z0, ZF, ON}) on the SAME far input, per aligned token pair:
        sqrt(dh)(s'-s) = q^T(M'-M)k + dq^T M' k + q^T M' dk + dq^T M' dk
      aggregated over original tokens (S) vs inserted background tokens (D).

Captured activations are the model's own post-QK-norm / pre-RoPE q,k, v, the
RoPE cos/sin (attention_scaling baked in), and the attention module output.
ALL heads are kept (no head selection). Only declared query rows are
materialized; no N x N attention matrix is kept resident.

Modes:
  --mode system : run ONE system (arm [+adapter]); capture + §5.2; plus §2.3
                  against --ref-dump (T0 activations) when provided.
                  T0's own run additionally dumps its activations for the
                  test systems.
  --mode report : combine all system outputs into theory_observation_map.json.

Usage (sequential single-GPU runs; T0 FIRST, then each test system):
  python diag_write_decomp.py --mode system --system-id T0 --arm N \
      --model <1B> --tables <tables> --views <transport_views.jsonl> \
      --proofs <source_proofs.jsonl> --instances <sid1> <sid2> \
      --out $B12/diag/wd
  python diag_write_decomp.py --mode system --system-id ZF --arm Z \
      --adapter <ZF step_128 dir> --ref-dump $B12/diag/wd/T0/ref_dump.npz \
      ... (same flags) ...
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

from track_a_eval import load_and_install

# Pre-registered observation grid (frozen in PREGLUCTION_ROUND12_V2_ADDENDUM).
DEFAULT_LAYERS = (0, 4, 8, 12, 15)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """transformers OLMo2 convention: x*cos + rotate_half(x)*sin (fp32)."""
    return x * cos + rotate_half(x) * sin


def rot_by_delta_matrix(k: torch.Tensor, inv_freq: torch.Tensor,
                        qpos: int, scale_sq: float) -> torch.Tensor:
    """For all keys j<=qpos: M k_j = scale^2 * R_Omega(j-qpos) k_j, vectorized.
    k: [J, D] fp32; inv_freq: [K]. Returns [J, D]."""
    J, D = k.shape
    deltas = torch.arange(J, dtype=torch.float32) - float(qpos)       # [J]
    ang = torch.outer(deltas, inv_freq)                                # [J, K]
    c = ang.cos().repeat(1, 2)                                         # [J, D]
    s = ang.sin().repeat(1, 2)
    return scale_sq * (k * c + rotate_half(k) * s)


class Capture:
    """Hooks on declared layers: q/k after q_norm/k_norm (pre-RoPE), v,
    the attention module's (cos,sin) inputs, and its output (post o_proj).
    Each forward overwrites the stored tensors; snapshot after each run."""

    def __init__(self, layers_model, declared):
        self.declared = list(declared)
        self.data = {L: {} for L in declared}
        self._handles = []
        for L in declared:
            attn = layers_model[L].self_attn
            d = self.data[L]

            def mk_out_hook(store, name):
                def hook(module, inp, out):
                    if isinstance(out, tuple):  # attn module returns a tuple
                        out = out[0]
                    store[name] = out.detach()
                return hook

            def mk_pre_hook(store):
                def hook(module, args, kwargs):
                    pe = kwargs.get("position_embeddings")
                    if pe is None and len(args) >= 2:
                        pe = args[1]
                    store["cos"], store["sin"] = (pe[0].detach(), pe[1].detach())
                return hook

            self._handles.append(attn.q_norm.register_forward_hook(
                mk_out_hook(d, "q_pre")))
            self._handles.append(attn.k_norm.register_forward_hook(
                mk_out_hook(d, "k_pre")))
            self._handles.append(attn.v_proj.register_forward_hook(
                mk_out_hook(d, "v")))
            self._handles.append(attn.register_forward_pre_hook(
                mk_pre_hook(d), with_kwargs=True))
            self._handles.append(attn.register_forward_hook(
                mk_out_hook(d, "attn_out")))

    def close(self):
        for h in self._handles:
            h.remove()

    def snapshot_cpu(self):
        return {L: {k: v.detach().cpu() for k, v in self.data[L].items()}
                for L in self.declared}


def get_base(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def build_compact_prime(tok, far_row, proof):
    """Reconstruct prefix/suffix/block exactly as the round-11 builder did and
    verify token-exact against the frozen far record. Returns ids, the
    compact'->far alignment map, and the S/D membership of far positions."""
    instruction = ("Use only the supplied passages, even if they describe a "
                   "counterfactual world. Give only the answer, without "
                   "explanation.\n\n")
    q = proof["question"]
    marked = tok.apply_chat_template(
        [{"role": "user",
          "content": instruction + "SOURCE_BANK_MARKER" + "\n\nQuestion: " + q}],
        tokenize=False, add_generation_prompt=True)
    left, right = marked.split("SOURCE_BANK_MARKER")
    prefix = tok.encode(left, add_special_tokens=False)
    suffix = tok.encode(right, add_special_tokens=False)
    contexts = proof["worlds"][int(far_row["world"])]["contexts"]
    block = tok.encode("\n\n".join(contexts) + "\n", add_special_tokens=False)

    far_ids = list(far_row["prompt_ids"])
    L = len(far_ids)
    p0, blen = far_row["source_block"]
    assert blen == len(block), f"block len mismatch {blen} vs {len(block)}"
    assert far_ids[p0:p0 + blen] == block, "evidence block tokens mismatch vs record"
    assert far_ids[:len(prefix)] == prefix, "prefix reconstruction mismatch"
    assert far_ids[L - len(suffix):] == suffix, "suffix reconstruction mismatch"

    in_s = np.zeros(L, dtype=bool)
    for a, b in ((0, len(prefix)), (p0, p0 + blen), (L - len(suffix), L)):
        in_s[a:b] = True
    npre, nblk, nsuf = len(prefix), len(block), len(suffix)
    compact_prime = prefix + block + suffix
    map_c2f = (list(range(0, npre)) + list(range(p0, p0 + nblk))
               + list(range(L - nsuf, L)))
    assert len(map_c2f) == len(compact_prime)
    return {"compact_prime": compact_prime, "map_c2f": map_c2f, "in_s": in_s,
            "block_range": (p0, p0 + blen),
            "prefix_len": npre, "suffix_len": nsuf}


def softmax_row(scores: torch.Tensor) -> torch.Tensor:
    m = scores.max(dim=-1, keepdim=True).values
    e = torch.exp(scores - m)
    return e / e.sum(dim=-1, keepdim=True)


@torch.inference_mode()
def head_value_writes(o_proj, v_rows: torch.Tensor, head: int, head_dim: int,
                      n_heads: int, device, chunk=4096) -> torch.Tensor:
    """u_j = W_O^{(head)} v_j using the actual (possibly LoRA-wrapped) o_proj.
    v_rows: [n, head_dim] bf16 CPU. Returns [n, hidden] fp32 CPU."""
    assert getattr(o_proj, "bias", None) is None, \
        "o_proj bias would break per-head isolation"
    outs = []
    for s in range(0, v_rows.shape[0], chunk):
        cv = v_rows[s:s + chunk].to(device)
        e = torch.zeros(cv.shape[0], n_heads * head_dim, dtype=cv.dtype, device=device)
        e[:, head * head_dim:(head + 1) * head_dim] = cv
        outs.append(o_proj(e).float().cpu())
    return torch.cat(outs, dim=0)


def decomp_5_2(cap_c, cap_f, cp, layers, qf, attn_modules, device,
               head_dim, n_heads) -> list[dict]:
    """§5.2 three-term identity per (layer, head) at declared query row qf."""
    in_s_t = torch.tensor(cp["in_s"], dtype=torch.bool)
    map_idx = torch.tensor(cp["map_c2f"], dtype=torch.long)
    blk_a, blk_b = cp["block_range"]
    qc = len(cp["compact_prime"]) - 1
    rows = []
    for Lnum in layers:
        dc, df = cap_c[Lnum], cap_f[Lnum]
        o_proj = attn_modules[Lnum].o_proj
        hidden = df["attn_out"].shape[-1]
        cos_f = df["cos"][0].float()
        sin_f = df["sin"][0].float()
        cos_c = dc["cos"][0].float()
        sin_c = dc["sin"][0].float()
        qf_pre = df["q_pre"][0].float().view(-1, n_heads, head_dim)
        kf_pre = df["k_pre"][0].float().view(-1, n_heads, head_dim)
        qc_pre = dc["q_pre"][0].float().view(-1, n_heads, head_dim)
        kc_pre = dc["k_pre"][0].float().view(-1, n_heads, head_dim)
        qf_rot = rope_apply(qf_pre, cos_f[:, None, :], sin_f[:, None, :])
        kf_rot = rope_apply(kf_pre, cos_f[:, None, :], sin_f[:, None, :])
        qc_rot = rope_apply(qc_pre, cos_c[:, None, :], sin_c[:, None, :])
        kc_rot = rope_apply(kc_pre, cos_c[:, None, :], sin_c[:, None, :])
        sq = math.sqrt(head_dim)
        # kf_rot[:qf+1] is [J,H,D] -> [H,D,J]; explicit [H,1,D] batch so the
        # matmul cannot gain a spurious broadcast batch dim -> [H,1,J] -> [H,J]
        af = softmax_row((qf_rot[qf].unsqueeze(1)
                          @ kf_rot[:qf + 1].permute(1, 2, 0)).squeeze(1) / sq)
        ac = softmax_row((qc_rot[qc].unsqueeze(1)
                          @ kc_rot[:qc + 1].permute(1, 2, 0)).squeeze(1) / sq)
        v_f = df["v"][0].view(-1, n_heads, head_dim)
        v_c = dc["v"][0].view(-1, n_heads, head_dim)
        capt_f = df["attn_out"][0][qf].float()
        capt_c = dc["attn_out"][0][qc].float()
        o_rec_f = torch.zeros(hidden)
        o_rec_c = torch.zeros(hidden)
        heads = {}
        for h in range(n_heads):
            u_f = head_value_writes(o_proj, v_f[:, h, :], h, head_dim, n_heads, device)
            u_c = head_value_writes(o_proj, v_c[:, h, :], h, head_dim, n_heads, device)
            o_f_h = af[h] @ u_f[:qf + 1]
            o_c_h = ac[h] @ u_c[:qc + 1]
            o_rec_f += o_f_h
            o_rec_c += o_c_h
            # map_idx: compact position -> far position. Compact attention is
            # already aligned position-for-position with map_idx; far attention
            # must be gathered THROUGH map_idx.
            a0 = ac[h][:len(map_idx)]            # compact attention over S
            af_s = af[h][map_idx]                # far attention on the same S
            s_causal = in_s_t[:qf + 1]
            beta = float(af[h][:qf + 1][~s_causal].sum())
            abar = af_s / max(1.0 - beta, 1e-9)
            u_f_s = u_f[map_idx]
            T1 = (a0[:, None] * (u_f_s - u_c)).sum(0)
            T2 = ((abar - a0)[:, None] * u_f_s).sum(0)
            if beta > 0:
                dpos = torch.where(~s_causal)[0]
                ubar_d = (af[h][dpos][:, None] * u_f[dpos]).sum(0) / beta
                ubar_s = (abar[:, None] * u_f_s).sum(0)
                T3 = beta * (ubar_d - ubar_s)
            else:
                T3 = torch.zeros_like(T1)
            do = o_f_h - o_c_h
            resid = (T1 + T2 + T3) - do
            heads[h] = {
                "T1_norm": float(T1.norm()), "T2_norm": float(T2.norm()),
                "T3_norm": float(T3.norm()), "delta_o_norm": float(do.norm()),
                "proj_T1": float(T1 @ do) / max(float(do.norm()), 1e-9),
                "proj_T2": float(T2 @ do) / max(float(do.norm()), 1e-9),
                "proj_T3": float(T3 @ do) / max(float(do.norm()), 1e-9),
                "identity_rel_residual": float(
                    resid.norm() / max(float(do.norm()), 1e-9)),
                "beta": beta,
                "mass_block": float(af[h][blk_a:blk_b].sum()),
                "mass_S": float(af[h][:qf + 1][s_causal].sum()),
                "mass_D": beta,
                "argmax_key": int(af[h].argmax()),
                "argmax_in_S": bool(s_causal[int(af[h].argmax())]),
                # signed full vectors for post-hoc checks (fp32)
                "vec_T1": T1.numpy(), "vec_T2": T2.numpy(),
                "vec_T3": T3.numpy(), "vec_delta_o": do.numpy(),
            }
        rows.append({
            "layer": Lnum,
            "outer_recon_rel_far": float(
                (o_rec_f - capt_f).norm() / max(float(capt_f.norm()), 1e-9)),
            "outer_recon_rel_compact": float(
                (o_rec_c - capt_c).norm() / max(float(capt_c.norm()), 1e-9)),
            "heads": heads,
        })
    return rows


def decomp_2_3(ref, cap_t, layers, qpos, inv_test, scale_t, in_s,
               head_dim, n_heads, top_k=16) -> list[dict]:
    """§2.3 identity, ref (T0 dump) vs test capture, same far input.

    s_r/s_t are built from the model's own (captured) ABSOLUTE rotations:
      s[j] = (g·R(qpos θ) q)^T (g·R(j θ) k) / sqrt(d).
    The §2.3 expansion is in the RELATIVE operator M(j) = g^2 R((j-qpos) θ):
      ds = q0^T(M'-M)k0 + dq^T M'k0 + q0^T M'dk0 + dq^T M'dk0   (exact).
    M-application comes from rot_by_delta_matrix (inv_freq rebuild, fp32).
    Cross-check: the same M·k built from captured quantities,
      M k = R(-qpos θ)[g R(j θ) k] · g  =  rk·cos[qpos] - rotate_half(rk)·sin[qpos]
    (the captured cos/sin already carry one factor of g)."""
    inv_t = torch.tensor(np.asarray(inv_test), dtype=torch.float32)
    scale_t_sq = scale_t * scale_t
    scale_r_sq = float(ref["scale"]) ** 2
    inv_r = torch.tensor(np.asarray(ref["inv_freq"]), dtype=torch.float32)
    sq = math.sqrt(head_dim)
    s_mask = torch.tensor(in_s[:qpos + 1], dtype=torch.bool)
    rows = []
    for Lnum in layers:
        dt = cap_t[Lnum]
        cos_r = torch.tensor(np.asarray(ref[f"cos_{Lnum}"])).float()   # [J, D]
        sin_r = torch.tensor(np.asarray(ref[f"sin_{Lnum}"])).float()
        cos_t = dt["cos"][0].float()
        sin_t = dt["sin"][0].float()
        qr_r = torch.tensor(np.asarray(ref[f"q_pre_{Lnum}"])).float().view(-1, n_heads, head_dim)
        kr_r = torch.tensor(np.asarray(ref[f"k_pre_{Lnum}"])).float().view(-1, n_heads, head_dim)
        qr_t = dt["q_pre"][0].float().view(-1, n_heads, head_dim)
        kr_t = dt["k_pre"][0].float().view(-1, n_heads, head_dim)
        heads = {}
        for h in range(n_heads):
            q0 = qr_r[qpos, h]                        # [D]
            k0 = kr_r[:qpos + 1, h]                   # [J, D]
            dq = qr_t[qpos, h] - q0
            dk = kr_t[:qpos + 1, h] - k0
            # captured ABSOLUTE rotations (what the model actually applied)
            rq0 = q0 * cos_r[qpos] + rotate_half(q0) * sin_r[qpos]
            rk0 = k0 * cos_r[:qpos + 1] + rotate_half(k0) * sin_r[:qpos + 1]
            rqt = qr_t[qpos, h] * cos_t[qpos] + rotate_half(qr_t[qpos, h]) * sin_t[qpos]
            rkt = kr_t[:qpos + 1, h] * cos_t[:qpos + 1] \
                + rotate_half(kr_t[:qpos + 1, h]) * sin_t[:qpos + 1]
            s_r = (rq0 @ rk0.transpose(0, 1)) / sq    # [J]
            s_t = (rqt @ rkt.transpose(0, 1)) / sq    # [J]
            ds = s_t - s_r
            # RELATIVE operator applied to ref keys (M), test keys (M')
            m0k = rot_by_delta_matrix(k0, inv_r, qpos, scale_r_sq)
            m1k = rot_by_delta_matrix(k0, inv_t, qpos, scale_t_sq)
            m1dk = rot_by_delta_matrix(dk, inv_t, qpos, scale_t_sq)
            direct = (q0 @ (m1k - m0k).transpose(0, 1)) / sq              # [J]
            inherited = (dq @ m1k.transpose(0, 1)
                         + q0 @ m1dk.transpose(0, 1)
                         + dq @ m1dk.transpose(0, 1)) / sq                # [J]
            recon = direct + inherited
            resid = (recon - ds).abs()
            norm = max(float(ds.abs().max()), 1e-9)
            # cross-checks: relative operators rebuilt from captured cos/sin.
            # ref side: W k0 (rk0 is the captured rotation of the REF key k0).
            m0k_cross = rk0 * cos_r[qpos] - rotate_half(rk0) * sin_r[qpos]
            cross_r = float((m0k_cross - m0k).abs().max()
                            / max(float(m0k.abs().max()), 1e-9))
            # test side: rkt is the captured rotation of the TEST key, so the
            # un-rotated object is W' k_t — compare against the inv_freq
            # rebuild of W' k_t (NOT W' k0, which m1k is).
            m1kt_cross = rkt * cos_t[qpos] - rotate_half(rkt) * sin_t[qpos]
            m1kt = rot_by_delta_matrix(kr_t[:qpos + 1, h], inv_t, qpos,
                                       scale_t_sq)
            cross_t = float((m1kt_cross - m1kt).abs().max()
                            / max(float(m1kt.abs().max()), 1e-9))
            heads[h] = {
                "actual_dlogit_sum_S": float(ds[s_mask].sum()),
                "actual_dlogit_sum_D": float(ds[~s_mask].sum()),
                "direct_sum_S": float(direct[s_mask].sum()),
                "direct_sum_D": float(direct[~s_mask].sum()),
                "inherited_sum_S": float(inherited[s_mask].sum()),
                "inherited_sum_D": float(inherited[~s_mask].sum()),
                "rel_residual_max": float(resid.max() / norm),
                "rel_residual_mean": float(resid.mean() / norm),
                "rotation_crosscheck_rel_ref": cross_r,
                "rotation_crosscheck_rel_test": cross_t,
                "top_keys": [
                    {"j": int(j), "in_S": bool(s_mask[j]),
                     "direct": float(direct[j]), "inherited": float(inherited[j]),
                     "actual": float(ds[j])}
                    for j in resid.topk(min(top_k, qpos + 1)).indices.tolist()],
            }
        rows.append({"layer": Lnum, "heads": heads})
    return rows


def write_observation_map(out: Path):
    """§12 write_theory_observation_map: combine system outputs, preserve all
    negative results, make no method/checkpoint selection."""
    systems = {}
    for d in sorted(out.iterdir()):
        m = d / "manifest.json"
        if d.is_dir() and m.exists():
            systems[d.name] = json.loads(m.read_text())
    rep = {
        "status": "THEORY_OBSERVATION_MAP_V1",
        "systems": systems,
        "preserve_all_registered_negative_results": True,
        "no_claim_that_large_component_norm_proves_causality": True,
        "no_method_or_checkpoint_selection": True,
        "no_new_training": True,
    }
    (out / "theory_observation_map.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["system", "report"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--system-id", default=None)
    ap.add_argument("--arm", choices=["N", "Z"], default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--views", default=None)
    ap.add_argument("--proofs", default=None)
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--split", default="validation",
                    help="views split filter (pre-registered: validation)")
    ap.add_argument("--layers", type=int, nargs="*", default=list(DEFAULT_LAYERS))
    ap.add_argument("--ref-dump", default=None)
    ap.add_argument("--dtype", default="float32",
                    help="model dtype for this diagnostic (fp32: identity "
                         "residuals at arithmetic precision; registered "
                         "generation receipts stay bf16 elsewhere)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "report":
        write_observation_map(out)
        return

    for flag in ("model", "tables", "system_id", "arm", "views", "proofs",
                 "instances"):
        assert getattr(args, flag), f"system mode needs --{flag.replace('_','-')}"
    sysdir = out / args.system_id
    sysdir.mkdir(exist_ok=True)
    if (sysdir / "manifest.json").exists():
        raise FileExistsError("system output exists; preserved, not overwritten")

    proofs = {json.loads(l)["semantic_id"]: json.loads(l)
              for l in Path(args.proofs).read_text().splitlines()}
    want = set(args.instances)
    views = {}
    for line in Path(args.views).read_text().splitlines():
        r = json.loads(line)
        if (r["semantic_id"] in want and r["layout"] == "far"
                and r["length_cap"] == 16384
                and r.get("split") == args.split):
            views[(r["semantic_id"], int(r["world"]))] = r
    missing = [k for k in [(s, w) for s in args.instances for w in (0, 1)]
               if k not in views]
    assert not missing, f"missing far-16384 views: {missing}"

    model, tok, identity = load_and_install(args.model, Path(args.tables), args.arm,
                                            adapter_dir=args.adapter,
                                            extra_path=args.extra,
                                            dtype=getattr(torch, args.dtype))
    model.to(args.device)
    base = get_base(model)
    layers_model = base.model.layers
    rotary = base.model.rotary_emb
    cfg = base.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    attn_modules = {L: layers_model[L].self_attn for L in args.layers}
    inv_freq = rotary.inv_freq.detach().float().cpu().numpy().copy()
    attn_scaling = float(rotary.attention_scaling)

    cap = Capture(layers_model, args.layers)
    ref = None
    if args.ref_dump:
        ref = dict(np.load(args.ref_dump, allow_pickle=False))

    results = {"system": args.system_id, "identity": identity,
               "attn_scaling": attn_scaling,
               "inv_freq_sha256": hashlib.sha256(
                   np.ascontiguousarray(inv_freq, dtype=np.float32).tobytes()).hexdigest(),
               "layers": sorted(args.layers),
               "query_rule": "last prompt position of the far view",
               "instances": {}}
    ref_tensors = {"scale": attn_scaling,
                   "inv_freq": np.ascontiguousarray(inv_freq, dtype=np.float32)}
    t0 = time.time()

    @torch.inference_mode()
    def fwd(ids):
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        model(x, use_cache=False, logits_to_keep=1)

    for sid in args.instances:
        proof = proofs[sid]
        per_world = {}
        for w in (0, 1):
            far = views[(sid, w)]
            cp = build_compact_prime(tok, far, proof)
            qf = len(far["prompt_ids"]) - 1

            fwd(far["prompt_ids"])
            cap_f = cap.snapshot_cpu()
            fwd(cp["compact_prime"])
            cap_c = cap.snapshot_cpu()

            wd = decomp_5_2(cap_c, cap_f, cp, args.layers, qf, attn_modules,
                            args.device, head_dim, n_heads)
            entry = {"query_pos_far": qf,
                     "compact_prime_len": len(cp["compact_prime"]),
                     "far_len": len(far["prompt_ids"]),
                     "write_decomp_5_2": wd}
            if ref is not None:
                entry["logit_decomp_2_3_vs_T0"] = decomp_2_3(
                    ref, cap_f, args.layers, qf, inv_freq, attn_scaling,
                    cp["in_s"], head_dim, n_heads)
            if args.system_id == "T0":
                for Lnum in args.layers:
                    d = cap_f[Lnum]
                    for name in ("q_pre", "k_pre", "cos", "sin"):
                        ref_tensors[f"{name}_{Lnum}"] = \
                            d[name][0].to(torch.float32).numpy()
            per_world[f"world_{w}"] = entry
            print(f"[{sid[:8]} w{w}] §5.2 outer-recon layer "
                  + " ".join(f"L{r['layer']}:{r['outer_recon_rel_far']:.2e}"
                             for r in wd), flush=True)
        results["instances"][sid] = per_world

    cap.close()
    del model
    torch.cuda.empty_cache()

    # ---- save ------------------------------------------------------------
    vecs = {}
    for sid, pw in results["instances"].items():
        for w in (0, 1):
            for row in pw[f"world_{w}"]["write_decomp_5_2"]:
                for h, hd in row["heads"].items():
                    for name in ("vec_T1", "vec_T2", "vec_T3", "vec_delta_o"):
                        vecs[f"{sid[:8]}_w{w}_L{row['layer']}_h{h}_{name}"] = \
                            hd[name]
    np.savez_compressed(sysdir / "details.npz", **vecs)
    # strip big vectors from the json copy
    for sid, pw in results["instances"].items():
        for w in (0, 1):
            for row in pw[f"world_{w}"]["write_decomp_5_2"]:
                for hd in row["heads"].values():
                    for name in ("vec_T1", "vec_T2", "vec_T3", "vec_delta_o"):
                        hd.pop(name)
    (sysdir / "results.json").write_text(json.dumps(results, indent=2))
    if args.system_id == "T0":
        np.savez_compressed(sysdir / "ref_dump.npz", **ref_tensors)
    manifest = {
        "status": "DIAG_WRITE_DECOMP_COMPLETE",
        "diagnostic": "pro_synthesis_11.1_write_and_logit_decomposition",
        "system": args.system_id,
        "wall_seconds": round(time.time() - t0, 1),
        "summary": {
            sid: {
                f"w{w}": {
                    "max_outer_recon_rel": max(
                        r["outer_recon_rel_far"]
                        for r in pw[f"world_{w}"]["write_decomp_5_2"]),
                    "max_5_2_identity_residual": max(
                        hd["identity_rel_residual"]
                        for r in pw[f"world_{w}"]["write_decomp_5_2"]
                        for hd in r["heads"].values()),
                } for w in (0, 1)
            } for sid, pw in results["instances"].items()
        },
        "notes": [
            "all heads kept; no head selection; query = last prompt position",
            "§5.2 identity is exact algebra; residuals measure hook/alignment fidelity",
            "large component norms are NOT causal claims (§11.1 boundary)",
        ],
    }
    (sysdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
