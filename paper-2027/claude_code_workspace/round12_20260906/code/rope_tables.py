#!/usr/bin/env python3
"""Round 12 static position-system tables: N / Z / Y(YaRN-s4) / M(MrRoPE-Pro-s4).

Reads the frozen round-11 fixed_controls N/Z/Y .npy tables verbatim (byte-verified),
derives M per MrRoPE ICLR2026 Section 3.2, and freezes a round-12 manifest.

Table semantics (must match frozen engine code_release_008 exactly):
  - A table is the FULL float32 inv_freq vector (K = head_dim/2 = 64 values),
    installed by replacing `model.model.rotary_emb.inv_freq`.
  - Native profile: inv_freq_j = base ** (-(2j)/head_dim)  (HF default, fp32).
  - The gain (rotary amplitude) is installed on `rotary_emb.attention_scaling`
    and multiplies BOTH cos and sin; effective logit multiplier = gain**2.
    (Frozen-engine manifest convention: effective_logit_multiplier = amplitude**2.)

Geometry check: OLMo-2-0425-1B and OLMo-2-1124-7B both have head_dim=128,
rope_theta=500000, max_position_embeddings=4096, vocab=100352 -> identical
default frequency profile -> the same numeric Z/Y/N tables apply to both.

CPU-only. Usage:
  python rope_tables.py verify --config1b <1b config.json> --config7b <7b config.json> \
      --controls <fixed_controls dir> --out <round12 tables dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = 500_000.0
HEAD_DIM = 128          # both OLMo-2 models
L0 = 4096               # native window of both models
SCALE = 4               # s = Ldesign / L0
NATIVE_FLOAT32_SHA = "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34"

# Frozen round-11 fixed_controls manifest (FIXED_NZGY_CONTROLS_FROZEN_V1).
FROZEN = {
    "N": {"file_sha256": "71cccdfb052eacba55d4e530881c916084a0ba3ed706643d60f6499fe664a217",
          "float32_sha256": "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34",
          "rotary_amplitude": 1.0},
    "Z": {"file_sha256": "ee968fcefe9f91a6bec7bff6eae45585d2876f03625378a739031a3155b59316",
          "float32_sha256": "56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b",
          "rotary_amplitude": 1.102585782722872},
    "Y": {"file_sha256": "b3644ff4f0aa94441c535bc64c1add4bae6da17146d6541ae5062ed3110f01b8",
          "float32_sha256": "cc9da456982ffce5ca0558e9ea661abc4a880ec002179ce6b9149d45aa4a016c",
          "rotary_amplitude": 1.138629436111989},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_float32(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def native_inv_freq(base: float = BASE, head_dim: int = HEAD_DIM) -> np.ndarray:
    """HF default RoPE inverse frequencies, fp32, exactly as transformers computes them.

    MUST use the torch recipe byte-for-byte: numpy's pow differs in the last ulp
    and would break sha-identity with the frozen engine (verified 2026-09-06:
    only the torch path reproduces the frozen native sha).
    """
    import torch
    idx = torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float32)
    inv = 1.0 / (base ** (idx / head_dim))
    return inv.numpy().astype(np.float32)


def yarn_amplitude(scale: float) -> float:
    """Official YaRN attention factor t = 1 + 0.1 ln(s); installed as cos/sin multiplier."""
    return 1.0 + 0.1 * math.log(scale)


def check_config(config_path: Path) -> dict:
    cfg = json.loads(Path(config_path).read_text())
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    facts = {
        "path": str(config_path),
        "hidden_size": cfg["hidden_size"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": head_dim,
        "rope_theta": cfg["rope_theta"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "vocab_size": cfg["vocab_size"],
        "tie_word_embeddings": cfg["tie_word_embeddings"],
        "rope_scaling": cfg.get("rope_scaling"),
    }
    assert head_dim == HEAD_DIM, f"head_dim {head_dim} != {HEAD_DIM}: table not reusable"
    assert abs(facts["rope_theta"] - BASE) < 1e-9, "rope_theta mismatch"
    assert facts["max_position_embeddings"] == L0, "L0 mismatch"
    assert facts["vocab_size"] == 100352, "vocab mismatch"
    assert facts["rope_scaling"] is None, "unexpected rope_scaling"
    return facts


def mrrope_pro_table(base: float = BASE, head_dim: int = HEAD_DIM, l0: int = L0,
                     scale: float = SCALE, beta: float = 32.0, alpha: float = 1.0):
    """MrRoPE-Pro (ICLR2026) Section 3.2 mixed-radix allocation at scale s.

    Band boundaries on NATIVE frequencies (turns at the native window L0):
        d_l = max{j : L0*theta_j > beta*2pi}   (fast edge: 32-turn boundary)
        d_h = min{j : L0*theta_j < alpha*2pi}  (slow edge: 1-turn boundary)
    Inside the band [d_l, d_h), increasing radix increments
        eps_j = 2*(1 + j - d_l) / (n*(n+1)),  n = d_h - d_l,  sum(eps) = 1
        lam_j = s**eps_j
    Modified frequencies divide by the CUMULATIVE product:
        theta'_j = theta_j / prod_{d < j} lam_d     (lam_d = 1 outside band)
    so dims below d_l are untouched and dims at/after d_h carry the full 1/s.
    Amplitude: YaRN factor t = 1 + 0.1 ln(s) as cos/sin multiplier (paper Eq. on 1/t).
    Returns (table, amplitude, record).
    """
    native = native_inv_freq(base, head_dim)
    turns = native * l0 / (2.0 * math.pi)
    fast = np.where(turns > beta)[0]
    slow = np.where(turns < alpha)[0]
    assert fast.size and slow.size, "band boundaries not found"
    d_l, d_h = int(fast[-1]), int(slow[0])
    assert d_l < d_h, "empty band"
    n = d_h - d_l
    lam = np.ones(head_dim // 2, dtype=np.float64)
    eps = np.array([2.0 * (1 + j - d_l) / (n * (n + 1)) for j in range(d_l, d_h)])
    lam[d_l:d_h] = scale ** eps
    table = native.astype(np.float64) / np.cumprod(np.concatenate(([1.0], lam[:-1])))
    table = table.astype(np.float32)
    assert np.all(table > 0) and np.all(table[:-1] > table[1:]), "table must be positive ordered"
    amp = yarn_amplitude(scale)
    record = {
        "method": "MrRoPE-Pro",
        "paper": "MrRoPE: Mixed-radix Rotary Position Embedding, ICLR2026, Section 3.2",
        "base": base, "head_dim": head_dim, "L0": l0, "scale": scale,
        "band_beta_turns": beta, "band_alpha_turns": alpha,
        "d_l_32turn_boundary": d_l, "d_h_1turn_boundary": d_h, "n_band": n,
        "turns_at_d_l": float(turns[d_l]), "turns_at_d_h": float(turns[d_h]),
        "note_band_convention": "d_l=max{j: L0*theta_j>32*2pi} (32-turn fast edge); "
                                "d_h=min{j: L0*theta_j<1*2pi} (1-turn slow edge); "
                                "indices are 0-based into the 64 inv_freq pairs.",
        "sum_eps": float(eps.sum()),
        "rotary_amplitude": amp,
        "amplitude_rule": "YaRN factor t=1+0.1*ln(s) installed as cos/sin multiplier "
                          "(effective logit multiplier t**2, same convention as frozen Y)",
    }
    return table, amp, record


def verify(controls: Path, out: Path, config1b: Path, config7b: Path) -> None:
    controls, out = Path(controls), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    facts_1b = check_config(config1b)
    facts_7b = check_config(config7b)

    # 1. Native profile identity: recomputed fp32 native must equal frozen native sha.
    native = native_inv_freq()
    assert sha256_float32(native) == NATIVE_FLOAT32_SHA, \
        "recomputed native profile does not match frozen engine native sha"
    np.save(out / "N.npy", native)  # byte copy for round-12 deploy convenience

    arms = {"N": {"path": "N.npy", "rotary_amplitude": 1.0,
                  "float32_sha256": sha256_float32(native),
                  "source": "recomputed HF default; sha matches frozen native"}}

    # 2. Frozen Z and Y: copy bytes, verify both hashes and amplitude, byte-identical reuse.
    for arm in ("Z", "Y"):
        src = controls / f"{arm}.npy"
        assert sha256_file(src) == FROZEN[arm]["file_sha256"], f"frozen {arm} file hash mismatch"
        arr = np.load(src, allow_pickle=False)
        assert arr.dtype == np.float32 and arr.shape == (HEAD_DIM // 2,)
        assert sha256_float32(arr) == FROZEN[arm]["float32_sha256"], f"frozen {arm} f32 hash mismatch"
        assert np.all(arr > 0) and np.all(arr[:-1] > arr[1:])
        (out / f"{arm}.npy").write_bytes(src.read_bytes())
        arms[arm] = {"path": f"{arm}.npy", "rotary_amplitude": FROZEN[arm]["rotary_amplitude"],
                     "float32_sha256": FROZEN[arm]["float32_sha256"],
                     "file_sha256": FROZEN[arm]["file_sha256"],
                     "source": "round-11 fixed_controls frozen bytes, reused verbatim"}

    # 3. M: derived per published formula, no benchmark fitting.
    table, amp, record = mrrope_pro_table()
    np.save(out / "M.npy", table)
    arms["M"] = {"path": "M.npy", "rotary_amplitude": amp,
                 "float32_sha256": sha256_float32(table),
                 "file_sha256": sha256_file(out / "M.npy"),
                 "source": record}

    manifest = {
        "status": "ROUND12_STATIC_TABLES_FROZEN_V1",
        "geometry": {"base": BASE, "head_dim": HEAD_DIM, "L0": L0, "scale": SCALE,
                     "K_pairs": HEAD_DIM // 2,
                     "profile_identity": "1B and 7B configs verified identical on "
                                         "head_dim/rope_theta/L0/vocab; native fp32 sha "
                                         "matches frozen engine native sha"},
        "config_1b": facts_1b, "config_7b": facts_7b,
        "arms": arms,
        "install_recipe": "rotary_emb.inv_freq.copy_(table); "
                          "rotary_emb.original_inv_freq = inv_freq.clone(); "
                          "rotary_emb.attention_scaling = rotary_amplitude; "
                          "same table+gain for ALL request lengths; no native routing",
        "provenance_controls": str(controls),
    }
    (out / "manifest_round12.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": manifest["status"],
                      "shas": {k: v["float32_sha256"] for k, v in arms.items()},
                      "M_band": [arms["M"]["source"]["d_l_32turn_boundary"],
                                 arms["M"]["source"]["d_h_1turn_boundary"]],
                      "out": str(out)}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--controls", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--config1b", required=True)
    v.add_argument("--config7b", required=True)
    args = ap.parse_args()
    if args.cmd == "verify":
        verify(args.controls, args.out, args.config1b, args.config7b)
