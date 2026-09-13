"""GPU runner for the Llama-3-8B stage: 5 controls + the 16 surviving candidates.

Plan section 9.1 is explicit that the review package does NOT ship a GPU model
loader.  This file is that loader, written against Meta-Llama-3-8B-Instruct's
own module layout rather than a rewritten attention, so that the only thing
that changes between arms is the operator.

How the operator is injected
----------------------------
`modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)` is replaced.  The standard
signature does not receive `position_ids`, so `LlamaRotaryEmbedding.forward` is
wrapped to stash the position ids it was just called with; the patched apply
reads that stash and asserts it is fresh.  Everything upstream and downstream of
the rotation -- attention, MLP, norms, the KV cache -- is untouched.

WHAT MUST NOT GO WRONG
----------------------
* **Pair layout.**  Llama's `rotate_half` pairs (i, i + K), not (2j, 2j + 1).
  `operators.apply_rotation` takes this as an explicit argument and the runner
  passes "half".  Getting it wrong yields a different operator that still runs
  and still emits fluent text.  See the plan's section 9.3 warning.
* **No cross-arm KV reuse.**  Plan section 3 forbids sharing KV or prefix cache
  across tables.  Each arm gets a fresh generate() call with its own cache.
* **Phase precision.**  Phases are built in float64 on CPU, cast to float32,
  and the trig is evaluated in float32; the model itself stays bfloat16 and is
  never quantised (plan section 3).
* **Fixed position rule.**  Non-linear phase directions depend only on p and the
  fixed D, never on the current sequence length, so already-written K keeps its
  phase when generation continues (plan section 8.3).

Status: **not executed on a GPU.**  The operator construction, the cache
shapes, the pair layout and the rotation algebra are covered by CPU tests
(`test_tools.py::test_rotation`); the model integration itself is untested until
a card is available.  `--dry-run` is written to exercise every step except the
forward pass, because the OLMo runner's dry-run returns before its arm
construction and that gap has already cost this campaign a cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import sys
import time
from pathlib import Path

import numpy as np

import operators as O
import planb_matched_controls as MC

# --------------------------------------------------------------------------
# the five controls (plan: Native / Official YaRN / MR / BM / Resonance-YaRN)
# --------------------------------------------------------------------------


def _band(g, theta=None):
    lo, hi = O.find_correction_range(32.0, 1.0, g.head_dim, theta or g.theta, g.window)
    return lo, hi


def _mrpro(g, low=None, n=None):
    low = g.low if low is None else low
    n = g.n if n is None else n
    q = np.clip(np.arange(g.K) - low, 0, n).astype(np.float64)
    m = q * (q + 1.0) / (n * (n + 1.0))
    m[np.arange(g.K) >= low + n + 1] = 1.0
    return m


def _incr_beta(g, b, low=None, n=None, shape_a=1.0):
    """eps_k ~ k^a (n+1-k)^b; b=0 is MrRoPE-Pro, b=1 is BM."""
    low = g.low if low is None else low
    n = g.n if n is None else n
    kk = np.arange(1, n + 1, dtype=np.float64)
    w = kk ** shape_a * (n + 1 - kk) ** b
    mq = np.concatenate([[0.0], np.cumsum(w / w.sum())])
    out = np.zeros(g.K)
    out[low:low + n + 1] = mq
    out[low + n + 1:] = 1.0
    return out


def _mr_area_smooth(g, low=None, n=None):
    """Two-boundary taper with exactly MrRoPE's discrete exponent area.

    The increment distribution is BetaBinomial(n-1, alpha=4, beta=2) on
    k-1. Its mean increment index is (2n+1)/3, so the cumulative exponent
    area is (n+2)/3, exactly matching MrRoPE. The continuum CDF is
    5*x**4 - 4*x**5, whose slope vanishes at both endpoints.
    """
    low = g.low if low is None else low
    n = g.n if n is None else n
    kk = np.arange(1, n + 1, dtype=np.float64)
    weights = kk * (kk + 1.0) * (kk + 2.0) * (n + 1.0 - kk)
    mq = np.concatenate([[0.0], np.cumsum(weights / weights.sum())])
    out = np.zeros(g.K)
    out[low:low + n + 1] = mq
    out[low + n + 1:] = 1.0
    return out


class TableOperator(O.Operator):
    """An operator defined by an explicit frequency table (the controls)."""

    def __init__(self, geom, nu, name, policy, pair_gain=None, scope="control"):
        super().__init__(geom)
        self._nu = np.asarray(nu, dtype=np.float64)
        self._pair_gain = geom.gain if pair_gain is None else float(pair_gain)
        self.config = name
        self.direction = name
        self.policy = policy
        self.scope = scope

    def q_amp(self, p):
        return np.full((p.size, self.geom.K), self._pair_gain, dtype=np.float64)

    def k_amp(self, p):
        return np.full((p.size, self.geom.K), self._pair_gain, dtype=np.float64)


def build_static_table_operator(payload, g, label):
    """Compile one explicit 64-slot fixed table without changing runner semantics."""
    table = payload.get("table", payload)
    nu = np.asarray(table.get("values_float32"), dtype=np.float64)
    gain = float(table.get("gain"))
    if (
        nu.shape != (g.K,)
        or not np.isfinite(nu).all()
        or np.any(nu <= 0.0)
        or np.any(nu[:-1] <= nu[1:])
        or not math.isfinite(gain)
        or gain <= 0.0
    ):
        raise ValueError("invalid explicit static table")
    if label in CONTROLS or label in MATCHED_CONTROLS or label in CONTROLLED_MODULES:
        raise ValueError(f"static table label collides with a registered arm: {label}")
    return TableOperator(
        g, nu, label,
        "explicit one-fixed-table JSON; no runtime length or layer switching",
        pair_gain=gain, scope="frequency")


def build_control(name, g):
    if name == "Native":
        return TableOperator(g, g.omega.copy(), "Native", "untouched trained frequencies",
                             pair_gain=1.0)
    if name == "MR":
        return TableOperator(g, g.omega * g.scale ** (-_mrpro(g)), "MR", "MrRoPE-Pro Eq.14")
    if name == "BM":
        return TableOperator(g, g.omega * g.scale ** (-_incr_beta(g, 1.0)), "BM",
                             "boundary-matched, eps ~ k(n+1-k)")
    if name == "UNI":
        q = np.clip(np.arange(g.K) - g.low, 0, g.n).astype(np.float64)
        m = q / float(g.n)
        return TableOperator(g, g.omega * g.scale ** (-m), "UNI",
                             "MrRoPE-Uni linear ramp")
    if name == "OfficialYaRN":
        j = np.arange(g.K)
        ramp = np.clip((j - g.low) / float(g.high - g.low), 0.0, 1.0)
        nu = g.omega * (1.0 - (1.0 - 1.0 / g.scale) * ramp)
        return TableOperator(g, nu, "OfficialYaRN",
                             "jquesnelle@995db5b index ramp, nu'/nu = 1-(1-1/s)t")
    if name == "ResonanceYaRN":
        # Resonance RoPE is defined as an addition to an existing scaling scheme,
        # so the control stacks it on YaRN, not on MR.
        base = build_control("OfficialYaRN", g)
        op = O.D01(g, 0)
        nu = base.nu().copy()
        j = np.arange(g.K)
        sel = j > g.low
        T = 2.0 * math.pi / nu[sel]
        nu[sel] = 2.0 * math.pi / np.maximum(np.floor(T + 0.5), 1.0)
        return TableOperator(g, nu, "ResonanceYaRN", "integer-period rounding on top of official YaRN")
    raise KeyError(f"unknown control {name}")


CONTROLS = ("Native", "OfficialYaRN", "MR", "BM", "UNI", "ResonanceYaRN")


def build_matched_control(name, g):
    """Construct Plan B controls that are not part of the original 60 IDs."""
    if name.startswith("AREA_D06") and name[-1] in "abc":
        return MC._AreaControl(g, name.removeprefix("AREA_"))
    if name == "DROP_D05c":
        return MC._DropControl(g, "D05c")
    if name == "SIGN_D04b":
        return MC.build_sign(g, "D04b")
    if name in {f"D13{x}_STATIC_ENDPOINT" for x in "abc"}:
        return MC.build_static_endpoint(g, name[:4])
    raise KeyError(name)


MATCHED_CONTROLS = (
    "AREA_D06a", "AREA_D06b", "AREA_D06c",
    "DROP_D05c", "SIGN_D04b",
    "D13a_STATIC_ENDPOINT", "D13b_STATIC_ENDPOINT", "D13c_STATIC_ENDPOINT",
)


L1_AMPLITUDES = {
    "A087": 0.87,
    "A100": 1.00,
    # Llama MR has sum_m=103/3 while BM has sum_m=37.  This exact amplitude
    # matches total log compression: (103/3) * (111/103) = 37.  It is only a
    # diagnostic, not a pure shape control, because it also changes MR's slow
    # endpoint exponent from 1 to 111/103.  UNI is the endpoint-and-area-matched
    # comparator for BM.
    "AEQBM": 111.0 / 103.0,
    "A113": 1.13,
    "A130": 1.30,
}
L2_BETAS = {"BF32_BS1": (32.0, 1.0), "BF32_BS05": (32.0, 0.5),
            "BF64_BS1": (64.0, 1.0), "BF64_BS05": (64.0, 0.5)}


def build_controlled_module(name, g):
    """Build Plan B L1/L2 cells; these are modules, not original D IDs."""
    parts = name.split("_")
    if len(parts) == 3 and parts[0] == "L1" and parts[1] in {"MR", "BM"}:
        amplitude = L1_AMPLITUDES[parts[2]]
        profile = _mrpro(g) if parts[1] == "MR" else _incr_beta(g, 1.0)
        nu = g.omega * g.scale ** (-(amplitude * profile))
        return TableOperator(
            g, nu, name,
            f"Plan B L1 {parts[1]} profile; compression amplitude a={amplitude}")
    if len(parts) == 4 and parts[0] == "L2" and parts[1] in {"MR", "BM"}:
        beta_key = f"{parts[2]}_{parts[3]}"
        beta_fast, beta_slow = L2_BETAS[beta_key]
        low, high = O.find_correction_range(
            beta_fast, beta_slow, g.head_dim, g.theta, g.window)
        varied = O.Geometry.from_native(
            g.native_inv_freq, window=g.window, theta=g.theta, scale=g.scale,
            K=g.K, head_dim=g.head_dim, low=low, high=high, n=high - low)
        profile = _mrpro(varied) if parts[1] == "MR" else _incr_beta(varied, 1.0)
        nu = g.omega * g.scale ** (-profile)
        return TableOperator(
            g, nu, name,
            f"Plan B L2 {parts[1]} profile; beta_fast={beta_fast}, "
            f"beta_slow={beta_slow}, band=({low},{high})")
    raise KeyError(name)


CONTROLLED_MODULES = tuple(
    f"L1_{profile}_{amp}" for amp in L1_AMPLITUDES for profile in ("MR", "BM")
) + tuple(
    f"L2_{profile}_{beta}" for beta in L2_BETAS for profile in ("MR", "BM")
)

THEORY_CANDIDATES = (
    "MR_AREA_SMOOTH", "RIBB_A2_B1P5", "RIBB_A2_B1P5_G1",
)

TABLE_GAIN_CONTROLS = ("MR_G1", "BM_G1")

SCALE_GAIN_CONTROLS = (
    "MR_FS16_GS1", "MR_FS1_GS16",
    "MR_FS16_GS4", "MR_FS4_GS16",
)


def build_scale_gain_control(name, g):
    """Separate MrRoPE's frequency-scale and attention-gain factors."""
    parts = name.split("_")
    if (len(parts) != 3 or parts[0] != "MR" or
            not parts[1].startswith("FS") or not parts[2].startswith("GS")):
        raise KeyError(name)
    frequency_scale = float(parts[1][2:])
    gain_scale = float(parts[2][2:])
    if frequency_scale < 1.0 or gain_scale < 1.0:
        raise ValueError("frequency and gain scales must be at least one")
    profile = _mrpro(g)
    nu = g.omega * frequency_scale ** (-profile)
    pair_gain = 1.0 + 0.1 * math.log(gain_scale)
    return TableOperator(
        g, nu, name,
        f"MrRoPE frequency scale={frequency_scale:g}; gain scale={gain_scale:g}",
        pair_gain=pair_gain, scope="frequency")


def build_arm(name, g, authorized_scopes=("frequency", "frequency_assignment")):
    """`name` is a control name or one of the 16 candidate config ids."""
    if name in SCALE_GAIN_CONTROLS:
        return build_scale_gain_control(name, g)
    if name in TABLE_GAIN_CONTROLS:
        base_name = name.removesuffix("_G1")
        base = build_control(base_name, g)
        return TableOperator(
            g, base.nu(), name,
            f"{base_name} frequency table with unit Q/K gain",
            pair_gain=1.0, scope="frequency")
    if name == "MR_AREA_SMOOTH":
        return TableOperator(
            g, g.omega * g.scale ** (-_mr_area_smooth(g)), name,
            "BetaBinomial(4,2) increments: MR-area, two-boundary-smooth",
            scope="frequency")
    if name == "RIBB_A2_B1P5":
        return TableOperator(
            g, g.omega * g.scale ** (-_incr_beta(g, 0.5)), name,
            "RIBB alpha=2,beta=1.5: exact one-dimensional MR-to-BM bridge",
            scope="frequency")
    if name == "RIBB_A2_B1P5_G1":
        return TableOperator(
            g, g.omega * g.scale ** (-_incr_beta(g, 0.5)), name,
            "RIBB alpha=2,beta=1.5 with unit Q/K gain",
            pair_gain=1.0, scope="frequency")
    if name in CONTROLS:
        return build_control(name, g)
    if name in MATCHED_CONTROLS:
        return build_matched_control(name, g)
    if name in CONTROLLED_MODULES:
        return build_controlled_module(name, g)
    op = O.build(name, g)
    if op.scope not in set(authorized_scopes):
        raise ValueError(
            f"SCOPE_BLOCKED: {name} needs {op.scope!r}; authorized scopes are "
            f"{sorted(set(authorized_scopes))}")
    return op


# --------------------------------------------------------------------------
# position-stashed rotary patch
# --------------------------------------------------------------------------


def make_rotary_patch(torch, modeling, cache):
    """Return (patched_apply, patched_rotary_forward).

    `cache` holds float32 tensors for the whole run: phase/amp for Q and K, and
    the pair metric.  Positions index directly into them, which is what makes
    the rule independent of the current sequence length.
    """
    state = {"position_ids": None}

    def rotary_forward(self, x, position_ids, *a, **kw):
        state["position_ids"] = position_ids
        # the base implementation still runs so that callers relying on cos/sin
        # keep working; the patched apply ignores its output.
        return orig_rotary_forward(self, x, position_ids, *a, **kw)

    orig_rotary_forward = modeling.LlamaRotaryEmbedding.forward

    def apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        pos = position_ids if position_ids is not None else state["position_ids"]
        if pos is None:
            raise RuntimeError(
                "no position_ids available: the rotary embedding wrapper was not "
                "installed, or apply_rotary_pos_emb was called before it")
        # q/k: (B, H, P, D).  Positions are (B, P); this runner uses batch 1 and
        # asserts it, because a ragged batch would need per-row phase gather.
        if pos.shape[0] != 1:
            raise RuntimeError(f"runner assumes batch 1, got {pos.shape[0]}")
        idx = pos[0].to(torch.long)

        def rot(x, ph, am, dg):
            K = ph.shape[-1]
            input_dtype = x.dtype
            # Match HF's stock numerical path: positions, phases and trig are
            # evaluated in FP32, then cos/sin are cast back to the model dtype
            # before multiplying BF16 Q/K.  Promoting the whole rotation to
            # FP32 changed Native logits by up to 0.39 on the real checkpoint.
            a = x[..., :K]
            b = x[..., K:]
            if dg is not None:
                a = a * dg[:, 0].to(input_dtype)[None, None, None, :]
                b = b * dg[:, 1].to(input_dtype)[None, None, None, :]
            # gather the phase at each absolute position: (P, K) -> (1,1,P,K)
            c = torch.cos(ph[idx]).to(input_dtype)[None, None, :, :]
            s = torch.sin(ph[idx]).to(input_dtype)[None, None, :, :]
            ar = a * c - b * s
            br = a * s + b * c
            if am is not None:
                m = am[idx].to(input_dtype)[None, None, :, :]
                ar = ar * m
                br = br * m
            return torch.cat([ar, br], dim=-1).to(input_dtype)

        q_out = rot(q, cache["q_phase"], cache["q_amp"], cache["q_diag"])
        k_out = rot(k, cache["k_phase"], cache["k_amp"], cache["k_diag"])
        return q_out, k_out

    return apply, rotary_forward


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def build_cache(op, g, torch, dtype, device, max_pos):
    """Precompute the four surfaces over [0, max_pos) in float32."""
    p = np.arange(max_pos, dtype=np.float64)
    qp = op.q_phase(p).astype(np.float32)
    kp = op.k_phase(p).astype(np.float32)
    qa = op.q_amp(p).astype(np.float32)
    ka = op.k_amp(p).astype(np.float32)
    qd = op.q_diag().astype(np.float32)
    kd = op.k_diag().astype(np.float32)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return {
        "q_phase": t(qp), "k_phase": t(kp),
        # Only an all-ones amplitude can be elided.  The common YaRN/MrRoPE
        # gain is a real Q/K multiplier and must not disappear as an
        # "optimisation"; attention receives its square.
        "q_amp": None if np.all(qa == 1.0) else t(qa),
        "k_amp": None if np.all(ka == 1.0) else t(ka),
        "q_diag": t(qd) if not np.allclose(qd, 1.0) else None,
        "k_diag": t(kd) if not np.allclose(kd, 1.0) else None,
        "gain": g.gain,
        "max_pos": max_pos,
    }


DEFAULT_SCORER = ("/root/autodl-tmp/olmo_fast_screen_20260908/code/"
                  "scripts/experiments/olmo_fast_screen/bench.py")
SCORING_CONTRACT_REVISION = "llama-planb-ruler-derived-v1"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scorer(path=None):
    """Import the FROZEN RULER scorer and refuse anything that is not it.

    Plan section 5.1 freezes the scoring logic, and section 9.2 has the reporting
    tool explicitly not implement it.  So this package does not contain a second
    scorer: it imports the campaign's own `bench.score`, whose definition is

        score(row, text) = int(normalize_answer(text) == row['answer'])

    and checks the module really exposes it before trusting a single number.
    """
    import importlib.util

    path = path or DEFAULT_SCORER
    if not Path(path).exists():
        raise SystemExit(f"REFUSING: frozen scorer not found at {path}; pass --scorer")
    spec = importlib.util.spec_from_file_location("frozen_ruler_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("score",):
        if not hasattr(mod, fn):
            raise SystemExit(f"REFUSING: {path} has no {fn}(); this is not the frozen scorer")
    mod.source_sha256 = sha_file(path)
    contract = Path(__file__).with_name("SCORING_CONTRACT.md")
    if not contract.exists():
        raise SystemExit(f"REFUSING: missing frozen scoring contract {contract}")
    mod.scoring_contract_revision = SCORING_CONTRACT_REVISION
    mod.scoring_contract_sha256 = sha_file(contract)
    return mod


def row_template(row, arm, out_text, output_ids, operator_id, operator_sha,
                 scorer=None, eos=None, cap_hit=None):
    """The per-row identity record required by plan section 5.4."""
    format_ok = bool(out_text.strip())
    validity = ("VALID_SCORED_CAP_HIT" if cap_hit else
                "VALID_SCORED_NO_EOS" if not eos else
                "VALID_SCORED_EMPTY" if not format_ok else "VALID")
    rec = {
        "row_id": row["row_id"], "task": row["task"], "length_cap": row["length_cap"],
        "source_id": row.get("source_id") or row.get("source_document_id"),
        "source_document_id": row.get("source_document_id") or row.get("source_id"),
        "group_id": row.get("group_id") or row.get("semantic_group_id"),
        "semantic_group_id": row.get("semantic_group_id") or row.get("group_id"),
        "split": row.get("split"),
        "prompt_sha256": row.get("prompt_sha256") or row.get("prompt_input_ids_sha256"),
        "prompt_input_ids_sha256": row.get("prompt_input_ids_sha256") or row.get("prompt_sha256"),
        "references": row.get("references"), "gold": row.get("gold") or row.get("references"),
        "input_tokens": len(row["prompt_ids"]),
        "actual_length": row.get("actual_length", len(row["prompt_ids"])),
        "max_new_tokens": int(row["max_new_tokens"]),
        "evidence_positions": row.get("evidence_positions"),
        "distractor_positions": row.get("distractor_positions"),
        "output": out_text, "raw_text": out_text, "output_ids": list(output_ids),
        "arm": arm, "operator_id": operator_id,
        "operator_sha256": operator_sha,
        "ended_eos": eos, "eos_seen": eos, "cap_hit": cap_hit,
        "format_ok": format_ok, "validity_status": validity,
        "scorer_sha256": getattr(scorer, "source_sha256", None),
        "scorer_revision": row.get("scorer_revision"),
        "scoring_contract_revision": getattr(
            scorer, "scoring_contract_revision", None),
        "scoring_contract_sha256": getattr(scorer, "scoring_contract_sha256", None),
        "checkpoint_id": getattr(scorer, "checkpoint_id", None),
        "model_config_sha256": getattr(scorer, "model_config_sha256", None),
        "tokenizer_sha256": getattr(scorer, "tokenizer_sha256", None),
        "weight_index_sha256": getattr(scorer, "weight_index_sha256", None),
        "checkpoint_manifest_sha256": getattr(
            scorer, "checkpoint_manifest_sha256", None),
        "runner_sha256": getattr(scorer, "runner_sha256", None),
        "operators_sha256": getattr(scorer, "operators_sha256", None),
    }
    if scorer is not None:
        partial = float(scorer.score(row, out_text))
        strict = strict_task_score(row, out_text)
        full_exact = full_string_exact(row, out_text)
        qa_em, qa_f1 = qa_scores(row, out_text)
        rec.update({
            "correct": partial, "partial_score": partial,
            "strict_score": strict,
            "full_string_exact": full_exact,
            "full_string_exact_and_eos": float(bool(full_exact and eos)),
            "qa_em": qa_em, "qa_f1": qa_f1,
        })
    return rec


def _official_text(text):
    return re.sub(r"[\x00-\x1f]", "\n", text.strip()).strip().lower()


def _answer_tokens(text):
    """Punctuation-insensitive tokens, preserving answer order and extras."""
    return re.findall(r"\w+", _official_text(text), flags=re.UNICODE)


def _squad_normalize(text):
    text = _official_text(text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_scores(row, text):
    """Standard SQuAD-style max-over-reference EM/F1 for QA rows."""
    if not str(row["task"]).startswith("qa_"):
        return None, None
    prediction = _squad_normalize(text)
    p_tokens = prediction.split()
    ems, f1s = [], []
    from collections import Counter
    for reference in dict.fromkeys(row["references"]):
        target = _squad_normalize(reference)
        t_tokens = target.split()
        ems.append(float(prediction == target))
        common = Counter(p_tokens) & Counter(t_tokens)
        overlap = sum(common.values())
        if not p_tokens or not t_tokens:
            f1s.append(float(p_tokens == t_tokens))
        elif overlap == 0:
            f1s.append(0.0)
        else:
            precision = overlap / len(p_tokens)
            recall = overlap / len(t_tokens)
            f1s.append(2.0 * precision * recall / (precision + recall))
    return max(ems, default=0.0), max(f1s, default=0.0)


def strict_task_score(row, text):
    """Frozen strict whole-question metric.

    Synthetic multi-answer tasks require every gold substring.  Extra text,
    order and key/value binding are not penalized by this metric and therefore
    are not claimed; ``full_string_exact`` separately rejects extras and order
    changes.  Natural QA uses standard normalized exact match.
    """
    if str(row["task"]).startswith("qa_"):
        return qa_scores(row, text)[0]
    cleaned = _official_text(text)
    return float(all(_official_text(ref) in cleaned for ref in row["references"]))


def full_string_exact(row, text):
    """Exact answer-token sequence after punctuation/whitespace normalization."""
    got = _answer_tokens(text)
    refs = list(dict.fromkeys(row["references"]))
    if str(row["task"]).startswith("qa_"):
        return float(any(got == _answer_tokens(ref) for ref in refs))
    want = [token for ref in refs for token in _answer_tokens(ref)]
    return float(got == want)


def ids_digest(ids):
    return hashlib.sha256(json.dumps(
        list(ids), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def row_identity(row):
    return {
        "row_id": row["row_id"], "task": row["task"],
        "length_cap": int(row["length_cap"]),
        "source_document_id": row.get("source_document_id") or row.get("source_id"),
        "semantic_group_id": row.get("semantic_group_id") or row.get("group_id"),
        "prompt_sha256": row.get("prompt_sha256") or row.get("prompt_input_ids_sha256"),
        "references": row.get("references"),
        "max_new_tokens": int(row["max_new_tokens"]),
        "scorer_revision": row.get("scorer_revision"),
    }


def validate_panel(rows, expected=None):
    if not rows:
        raise SystemExit("REFUSING: empty panel")
    index = {}
    for row in rows:
        rid = row.get("row_id") or row.get("example_id")
        if not rid or rid in index:
            raise SystemExit(f"REFUSING: missing or duplicate row_id {rid!r}")
        row["row_id"] = rid
        ids = row.get("prompt_ids")
        if not ids or not all(type(x) is int and x >= 0 for x in ids):
            raise SystemExit(f"REFUSING: {rid} has invalid prompt_ids")
        cap = int(row["length_cap"])
        budget = int(row["max_new_tokens"])
        if len(ids) + budget > cap:
            raise SystemExit(f"REFUSING: {rid} exceeds cap ({len(ids)}+{budget}>{cap})")
        declared = row.get("prompt_sha256") or row.get("prompt_input_ids_sha256")
        if not declared or declared != ids_digest(ids):
            raise SystemExit(f"REFUSING: {rid} prompt hash mismatch")
        if not row.get("references"):
            raise SystemExit(f"REFUSING: {rid} has no references")
        index[rid] = row
    if expected is not None:
        exp = {}
        for row in expected:
            rid = row.get("row_id") or row.get("example_id")
            if not rid or rid in exp:
                raise SystemExit(f"REFUSING: frozen index has duplicate row {rid!r}")
            row = dict(row); row["row_id"] = rid; exp[rid] = row
        if set(exp) != set(index):
            raise SystemExit(
                f"REFUSING: panel has {len(index)} rows, frozen index has {len(exp)}; "
                f"difference {len(set(exp) ^ set(index))}")
        for rid in index:
            if row_identity(index[rid]) != row_identity(exp[rid]):
                raise SystemExit(f"REFUSING: frozen row identity differs for {rid}")
    return index


def cache_sha256(cache):
    h = hashlib.sha256()
    for key in ("q_phase", "k_phase", "q_amp", "k_amp", "q_diag", "k_diag"):
        h.update(key.encode())
        value = cache[key]
        if value is None:
            h.update(b"NONE")
        else:
            h.update(np.ascontiguousarray(value.detach().float().cpu().numpy()).tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dry_run(g, arm_names, max_pos, authorized_scopes, resolver=build_arm):
    """Exercise everything except the forward pass.

    This is deliberately NOT the OLMo runner's dry-run, which returns before its
    arm-construction block and therefore cannot vouch for the arm at all.
    """
    report = {"arms": [], "pair_layout": "half", "max_pos": max_pos}
    for name in arm_names:
        op = resolver(name, g, authorized_scopes)
        p = np.array([0.0, 1.0, g.window, g.target - 1.0])
        qp = op.q_phase(p)
        qa = op.q_amp(p)
        qd = op.q_diag()
        # the identity the runner relies on: phase is indexed by absolute position
        assert qp.shape == (4, g.K), qp.shape
        assert np.all(np.isfinite(qp)), name
        assert np.all(np.isfinite(qa)), name
        assert qd.shape == (g.K, 2)
        # a rotation applied at p must not depend on anything but p
        again = op.q_phase(p)
        assert np.array_equal(qp, again), f"{name}: q_phase is not deterministic"
        nu = op.nu()
        # sum_m is log(omega/nu)/log s, so it exists only where nu > 0.  D04
        # (negated frequency) and D05 (zeroed slot) leave that coordinate
        # entirely: reporting a bare nan/inf would let an undefined quantity
        # travel downstream as if it were a measurement.
        defined = bool(np.all(nu > 0) and np.all(np.isfinite(nu)))
        report["arms"].append({
            "arm": name, "scope": op.scope, "policy": op.policy,
            "sum_m": float(np.sum(np.log(g.omega / nu) / math.log(g.scale))) if defined else None,
            "m_coordinate_defined": defined,
            "m_undefined_reason": None if defined else
            ("negative frequency: m = log(omega/nu)/log s is undefined"
             if np.any(nu < 0) else
             "zero frequency: the slot is an identity rotation and has no m"),
            "nu_sha256": O._sha(nu),
            "phase_finite": bool(np.all(np.isfinite(qp))),
            "diag_is_identity": bool(np.allclose(qd, 1.0)),
        })
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="local Meta-Llama-3-8B-Instruct dir")
    ap.add_argument("--panel", required=True, help="RULER panel jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="MR,OfficialYaRN,BM,UNI",
                    help="comma-separated; controls or candidate ids (D01a, D03a, ...)")
    ap.add_argument("--scorer", default=None,
                    help=f"path to the frozen RULER scorer (default {DEFAULT_SCORER})")
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--theta", type=float, default=500000.0)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--native-npy", default=None,
                    help="stock inv_freq exported from an UNPATCHED checkpoint")
    ap.add_argument("--checkpoint-manifest", default=None,
                    help="one-time full checkpoint SHA-256 manifest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--expected-data", default=None,
                    help="frozen row index; the panel must match it exactly")
    ap.add_argument("--lengths", default=None,
                    help="optional comma-separated caps to run from the frozen panel")
    ap.add_argument("--authorized-scopes", default="frequency,frequency_assignment",
                    help="explicit Plan B scopes allowed for non-control arms")
    ap.add_argument("--static-table-json", default=None,
                    help="explicit fixed table containing values_float32[64] and gain")
    ap.add_argument("--static-table-label", default="StaticTable",
                    help="arm label used with --static-table-json")
    a = ap.parse_args(argv)

    g = O.Geometry.from_native(
        np.load(a.native_npy) if a.native_npy else None,
        window=a.window, theta=a.theta, scale=a.scale)
    static_op = None
    if a.static_table_json:
        payload = json.loads(Path(a.static_table_json).read_text())
        static_op = build_static_table_operator(payload, g, a.static_table_label)
        arm_names = [a.static_table_label]
    else:
        arm_names = [s.strip() for s in a.arms.split(",") if s.strip()]
    if not arm_names:
        raise SystemExit("no arms requested")

    scopes = tuple(x.strip() for x in a.authorized_scopes.split(",") if x.strip())
    if not scopes:
        raise SystemExit("REFUSING: --authorized-scopes may not be empty")
    def resolve_arm(name, geometry, authorized_scopes):
        if static_op is not None and name == a.static_table_label:
            return static_op
        return build_arm(name, geometry, authorized_scopes)

    for name in arm_names:
        resolve_arm(name, g, scopes)

    raw_panel = [json.loads(line) for line in Path(a.panel).read_text().splitlines() if line]
    if a.lengths:
        lengths = {int(x) for x in a.lengths.split(",") if x.strip()}
        raw_panel = [row for row in raw_panel if int(row["length_cap"]) in lengths]
    expected = None
    if a.expected_data:
        expected = [json.loads(line) for line in Path(a.expected_data).read_text().splitlines() if line]
        if a.lengths:
            expected = [row for row in expected if int(row["length_cap"]) in lengths]
    validate_panel(raw_panel, expected)
    if a.limit and a.expected_data:
        raise SystemExit("REFUSING: --limit cannot be combined with --expected-data")
    panel = raw_panel[:a.limit] if a.limit else raw_panel
    max_pos = max(len(row["prompt_ids"]) + int(row["max_new_tokens"]) for row in panel)
    if max_pos > g.target:
        raise SystemExit(f"REFUSING: panel needs position {max_pos}, target is {g.target}")

    if a.dry_run:
        rep = dry_run(g, arm_names, max_pos, scopes, resolver=resolve_arm)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2))
        print(f"DRY_RUN_OK: {len(rep['arms'])} arms constructed, "
              f"geometry low={g.low} high={g.high} n={g.n} target={g.target}")
        return 0

    # ---- from here on a GPU is required ----
    import torch
    import transformers
    from transformers.models.llama import modeling_llama as M

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING: torch reports no CUDA device")
    scorer = load_scorer(a.scorer)

    model_path = Path(a.model)
    scorer.checkpoint_id = str(model_path.resolve())
    scorer.model_config_sha256 = sha_file(model_path / "config.json")
    scorer.tokenizer_sha256 = sha_file(model_path / "tokenizer.json")
    weight_index = model_path / "model.safetensors.index.json"
    scorer.weight_index_sha256 = sha_file(weight_index) if weight_index.exists() else None
    scorer.checkpoint_manifest_sha256 = None
    if a.checkpoint_manifest:
        checkpoint_manifest = json.loads(Path(a.checkpoint_manifest).read_text())
        if checkpoint_manifest.get("status") != "COMPLETE" or \
                Path(checkpoint_manifest.get("model", "")).resolve() != model_path.resolve():
            raise SystemExit("REFUSING: checkpoint manifest is incomplete or for another path")
        expected_small = {
            "config.json": scorer.model_config_sha256,
            "tokenizer.json": scorer.tokenizer_sha256,
            "model.safetensors.index.json": scorer.weight_index_sha256,
        }
        if any(checkpoint_manifest.get("files", {}).get(name, {}).get("sha256") != value
               for name, value in expected_small.items()):
            raise SystemExit("REFUSING: checkpoint manifest disagrees on identity files")
        scorer.checkpoint_manifest_sha256 = sha_file(a.checkpoint_manifest)
    scorer.runner_sha256 = sha_file(__file__)
    scorer.operators_sha256 = sha_file(Path(__file__).with_name("operators.py"))

    tok = transformers.AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    stop_ids = model.generation_config.eos_token_id
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    stop_ids = set(stop_ids or []) | {tok.eos_token_id}
    stop_ids.discard(None)
    if not stop_ids:
        raise SystemExit("REFUSING: model/tokenizer define no EOS/EOT token")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else min(stop_ids)

    for arm in arm_names:
        op = resolve_arm(arm, g, scopes)
        cache = build_cache(op, g, torch, torch.float32, model.device, max_pos)
        operator_sha = cache_sha256(cache)
        orig_apply = M.apply_rotary_pos_emb
        orig_rot = M.LlamaRotaryEmbedding.forward
        patched_apply, patched_rot = make_rotary_patch(torch, M, cache)
        M.apply_rotary_pos_emb = patched_apply
        M.LlamaRotaryEmbedding.forward = patched_rot
        try:
            final_path = out_dir / f"{arm}.jsonl"
            partial_path = out_dir / f"{arm}.partial.jsonl"
            prior = load_jsonl(final_path if final_path.exists() else partial_path)
            prior_by_id = {row["row_id"]: row for row in prior}
            if len(prior_by_id) != len(prior) or not set(prior_by_id) <= {
                    row["row_id"] for row in panel}:
                raise SystemExit(f"REFUSING: invalid resume rows for {arm}")
            if any(row.get("operator_sha256") != operator_sha for row in prior):
                raise SystemExit(f"REFUSING: operator drift in resumed {arm} rows")
            if any(row.get("scorer_sha256") != scorer.source_sha256 or
                   row.get("scoring_contract_sha256") != scorer.scoring_contract_sha256
                   for row in prior):
                raise SystemExit(f"REFUSING: scorer drift in resumed {arm} rows")
            if any(row.get("model_config_sha256") != scorer.model_config_sha256 or
                   row.get("tokenizer_sha256") != scorer.tokenizer_sha256 or
                   row.get("weight_index_sha256") != scorer.weight_index_sha256 or
                   row.get("checkpoint_manifest_sha256") !=
                   scorer.checkpoint_manifest_sha256 or
                   row.get("runner_sha256") != scorer.runner_sha256 or
                   row.get("operators_sha256") != scorer.operators_sha256
                   for row in prior):
                raise SystemExit(f"REFUSING: checkpoint/code drift in resumed {arm} rows")
            panel_by_id = {row["row_id"]: row for row in panel}
            for rid, previous in prior_by_id.items():
                if row_identity(previous) != row_identity(panel_by_id[rid]):
                    raise SystemExit(f"REFUSING: panel identity drift in resumed {arm}/{rid}")
            t0 = time.time()
            for r in panel:
                if r["row_id"] in prior_by_id:
                    continue
                ids = torch.tensor([r["prompt_ids"]], device=model.device)
                with torch.no_grad():
                    gen = model.generate(
                        ids, max_new_tokens=r["max_new_tokens"], do_sample=False,
                        num_beams=1, use_cache=True,
                        pad_token_id=pad_id, eos_token_id=sorted(stop_ids))
                new_ids = gen[0, ids.shape[1]:]
                text = tok.decode(new_ids, skip_special_tokens=True)
                output_ids = [int(x) for x in new_ids.tolist()]
                eos = bool(output_ids and output_ids[-1] in stop_ids)
                rec = row_template(
                    r, arm, text, output_ids, f"{op.direction}:{op.policy}",
                    operator_sha, scorer=scorer, eos=eos,
                    cap_hit=bool(len(output_ids) >= int(r["max_new_tokens"]) and not eos))
                with partial_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                prior_by_id[r["row_id"]] = rec
            if set(prior_by_id) != {row["row_id"] for row in panel}:
                raise SystemExit(f"REFUSING: {arm} result is partial")
            if not final_path.exists():
                partial_path.replace(final_path)
            nu = op.nu()
            m_defined = bool(np.all(np.isfinite(nu)) and np.all(nu > 0))
            summary.append({
                "arm": arm, "rows": len(prior_by_id), "seconds": time.time() - t0,
                "operator_sha256": operator_sha, "m_coordinate_defined": m_defined,
                "sum_m": (float(np.sum(np.log(g.omega / nu) / math.log(g.scale)))
                          if m_defined else None),
            })
            print(json.dumps(summary[-1]), flush=True)
            atomic_json(out_dir / "run_summary.json", {
                "status": "RUNNING", "arms": summary,
                "scorer_sha256": scorer.source_sha256,
                "scoring_contract_sha256": scorer.scoring_contract_sha256})
        finally:
            M.apply_rotary_pos_emb = orig_apply
            M.LlamaRotaryEmbedding.forward = orig_rot

    atomic_json(out_dir / "run_summary.json",
        {"status": "COMPLETE", "geometry": {"low": g.low, "high": g.high, "n": g.n, "theta": g.theta,
                      "window": g.window, "scale": g.scale, "target": g.target,
                      "gain": g.gain},
         "pair_layout": "half", "arms": summary,
         "scorer": a.scorer or DEFAULT_SCORER, "scorer_sha256": scorer.source_sha256,
         "scoring_contract_revision": scorer.scoring_contract_revision,
         "scoring_contract_sha256": scorer.scoring_contract_sha256,
         "checkpoint_id": scorer.checkpoint_id,
         "model_config_sha256": scorer.model_config_sha256,
         "tokenizer_sha256": scorer.tokenizer_sha256,
         "weight_index_sha256": scorer.weight_index_sha256,
         "checkpoint_manifest": a.checkpoint_manifest,
         "checkpoint_manifest_sha256": scorer.checkpoint_manifest_sha256,
         "runner_sha256": scorer.runner_sha256,
         "operators_sha256": scorer.operators_sha256,
         "static_table_json": a.static_table_json,
         "static_table_label": a.static_table_label if a.static_table_json else None,
         "panel_sha256": sha_file(a.panel), "scored": True,
         "note": "scored with the frozen external scorer; not reimplemented here"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
