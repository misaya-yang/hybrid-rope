#!/usr/bin/env python3
"""Passkey retrieval with NLL-gap scoring for from-scratch EVQ models.

Provides two evaluation methods for passkey retrieval:
  A. Teacher-Forcing NLL Gap: compute NLL for correct vs. wrong passkey digits
     at probe positions.  Positive gap = model retrieved the passkey.
  B. Autoregressive exact-match: truncate at <<PASS:, generate 9 tokens,
     check if they match the inserted passkey.

Also provides MixedDataset and make_passkey_training_sample for mixing passkey
retrieval supervision into the language-modelling training loop.

Passkey format uses dashes between digits so each digit tokenises independently:
    <<PASS:7-4-2-9-1>>

Usage (standalone test):
    python scripts/m4_evq_sweep/eval_passkey_scratch.py \
        --work_dir ~/evq_m4_sweep --tier 50m --tau 1.5

Importable components (used by run_evq_sweep.py):
    from eval_passkey_scratch import MixedDataset, make_passkey_training_sample
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

# ---------------------------------------------------------------------------
# Import model & utilities from sweep script
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_evq_sweep import GPT, TIER_CONFIGS, evq_cosh_inv_freq, get_device_and_dtype

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASSKEY_PREFIX = "<<PASS:"
PASSKEY_SUFFIX = ">>"
DASH = "-"


# ===================================================================
# 1. make_passkey_training_sample
# ===================================================================

def make_passkey_training_sample(
    filler_tokens: torch.Tensor,
    tokenizer,
    seq_len: int = 2048,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate a single passkey training sample.

    Format:
        [filler] <<PASS:X-X-X-X-X>> [filler] <<PASS:X-X-X-X-X>>

    The passkey is a random 5-digit string with dashes (e.g. ``7-4-2-9-1``)
    so that each digit is a separate token.  The passkey is inserted at a
    random depth (10 %--90 % of the available filler) and repeated at the very
    end as the target for next-token prediction.

    Args:
        filler_tokens: 1-D tensor of token IDs used as filler material.
        tokenizer: HuggingFace tokenizer instance.
        seq_len: Total length of the returned sequence.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Tensor of shape ``(seq_len,)`` with dtype ``torch.long``.
    """
    rng = random.Random(seed)

    # Generate random 5-digit passkey
    digits = [str(rng.randint(0, 9)) for _ in range(5)]
    passkey_str = DASH.join(digits)  # e.g. "7-4-2-9-1"
    passkey_text = f"{PASSKEY_PREFIX}{passkey_str}{PASSKEY_SUFFIX}"

    passkey_ids = tokenizer.encode(passkey_text, add_special_tokens=False)

    # We need two copies of the passkey marker in the sequence
    total_passkey_tokens = 2 * len(passkey_ids)
    filler_budget = seq_len - total_passkey_tokens

    if filler_budget < 2:
        raise ValueError(
            f"seq_len={seq_len} too short for passkey marker "
            f"(2 x {len(passkey_ids)} = {total_passkey_tokens} tokens)"
        )

    # Random depth for the first passkey: 10 %--90 % of filler budget
    depth_frac = rng.uniform(0.10, 0.90)
    before_len = max(1, int(filler_budget * depth_frac))
    after_len = filler_budget - before_len

    # Sample filler tokens
    total_filler = len(filler_tokens)
    start = rng.randint(0, max(0, total_filler - filler_budget - 1))
    filler_slice = filler_tokens[start : start + filler_budget].tolist()
    # Pad if filler_tokens is shorter than filler_budget
    while len(filler_slice) < filler_budget:
        filler_slice.extend(filler_tokens[: filler_budget - len(filler_slice)].tolist())

    before = filler_slice[:before_len]
    after = filler_slice[before_len : before_len + after_len]

    # Assemble: [filler_before] <<PASS:...>> [filler_after] <<PASS:...>>
    seq = before + passkey_ids + after + passkey_ids

    # Trim or pad to exact seq_len
    seq = seq[:seq_len]
    while len(seq) < seq_len:
        seq.append(filler_tokens[len(seq) % total_filler].item())

    return torch.tensor(seq, dtype=torch.long)


# ===================================================================
# 2. MixedDataset
# ===================================================================

class MixedDataset(torch.utils.data.Dataset):
    """Dataset that mixes language-modelling data with passkey retrieval samples.

    With probability ``passkey_ratio``, ``__getitem__`` returns a passkey
    training sample; otherwise it returns the corresponding row from
    ``lm_data``.  Passkey samples use a deterministic seed derived from the
    index so the same sample is returned across epochs / runs.

    Args:
        lm_data: Tensor of shape ``(N, seq_len)`` containing pre-tokenised
            language-modelling chunks.
        filler_tokens: 1-D tensor of token IDs used as filler for passkey
            samples.
        tokenizer: HuggingFace tokenizer instance.
        passkey_ratio: Probability of returning a passkey sample for any
            given index.  Default 0.005 (0.5 %).
        seq_len: Sequence length for passkey samples.  Should match the
            second dimension of ``lm_data``.
    """

    def __init__(
        self,
        lm_data: torch.Tensor,
        filler_tokens: torch.Tensor,
        tokenizer,
        passkey_ratio: float = 0.005,
        seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.lm_data = lm_data
        self.filler_tokens = filler_tokens
        self.tokenizer = tokenizer
        self.passkey_ratio = passkey_ratio
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.lm_data)

    def __getitem__(self, idx) -> torch.Tensor:
        # Support batch indexing (tensor/list of indices) for perm[slice] access
        if isinstance(idx, torch.Tensor) and idx.dim() > 0:
            return torch.stack([self._get_single(int(i)) for i in idx])
        return self._get_single(int(idx))

    def _get_single(self, idx: int) -> torch.Tensor:
        # Deterministic decision: use a hash of the index to decide
        # whether this sample is passkey or LM.
        det_rng = random.Random(idx * 6364136223846793005 + 1)
        if det_rng.random() < self.passkey_ratio:
            # Deterministic seed based on idx for cross-run reproducibility
            return make_passkey_training_sample(
                self.filler_tokens,
                self.tokenizer,
                seq_len=self.seq_len,
                seed=idx,
            )
        return self.lm_data[idx]


# ===================================================================
# 3. build_passkey_eval_sequence
# ===================================================================

def build_passkey_eval_sequence(
    filler_tokens: torch.Tensor,
    passkey: str,
    tokenizer,
    total_length: int,
    depth_percent: float,
) -> Tuple[torch.Tensor, int, int]:
    """Build an evaluation sequence for passkey NLL-gap scoring.

    Layout::

        [filler_before] <<PASS:{passkey}>> [filler_after] <<PASS:

    The probe starts at the final ``<<PASS:`` -- the model must predict the
    passkey digits that follow.

    Args:
        filler_tokens: 1-D tensor of filler token IDs.
        passkey: Dash-separated digit string, e.g. ``"7-4-2-9-1"``.
        tokenizer: HuggingFace tokenizer.
        total_length: Desired total length of ``input_ids`` in tokens.
        depth_percent: Fraction (0--1) controlling where the first passkey
            is placed within the filler.

    Returns:
        ``(input_ids, passkey_start, probe_start)`` where

        - ``input_ids``: 1-D tensor of shape ``(total_length,)``.
        - ``passkey_start``: token index where the first passkey marker
          (``<<PASS:...>>``) begins.
        - ``probe_start``: token index where the trailing ``<<PASS:`` begins.
    """
    full_marker_text = f"{PASSKEY_PREFIX}{passkey}{PASSKEY_SUFFIX}"
    probe_text = PASSKEY_PREFIX  # just "<<PASS:"

    full_marker_ids = tokenizer.encode(full_marker_text, add_special_tokens=False)
    probe_ids = tokenizer.encode(probe_text, add_special_tokens=False)

    # Filler budget = total_length - len(full_marker) - len(probe)
    filler_budget = total_length - len(full_marker_ids) - len(probe_ids)
    if filler_budget < 2:
        raise ValueError(
            f"total_length={total_length} too short for marker+probe "
            f"({len(full_marker_ids)} + {len(probe_ids)} tokens)"
        )

    # Split filler into before / after the full marker
    depth = max(0.0, min(1.0, depth_percent))
    before_len = max(1, int(filler_budget * depth))
    after_len = filler_budget - before_len

    # Grab filler tokens (wrap-around if needed)
    total_filler = len(filler_tokens)
    filler_slice = filler_tokens[:filler_budget].tolist()
    while len(filler_slice) < filler_budget:
        filler_slice.extend(filler_tokens[: filler_budget - len(filler_slice)].tolist())

    before = filler_slice[:before_len]
    after = filler_slice[before_len : before_len + after_len]

    # Assemble
    passkey_start = before_len
    seq = before + full_marker_ids + after + probe_ids
    probe_start = len(before) + len(full_marker_ids) + len(after)

    # Trim to exact length
    seq = seq[:total_length]

    input_ids = torch.tensor(seq, dtype=torch.long)
    return input_ids, passkey_start, probe_start


# ===================================================================
# 4. eval_passkey_nll_gap
# ===================================================================

def _make_wrong_passkey(correct: str, rng: random.Random) -> str:
    """Return a dash-separated digit string that differs in every digit."""
    digits = correct.split(DASH)
    wrong = []
    for d in digits:
        choices = [str(x) for x in range(10) if str(x) != d]
        wrong.append(rng.choice(choices))
    return DASH.join(wrong)


@torch.no_grad()
def eval_passkey_nll_gap(
    model: GPT,
    tokenizer,
    filler_tokens: torch.Tensor,
    lengths: List[int],
    depths: List[float],
    num_trials: int = 10,
    seed: int = 42,
) -> Dict:
    """Evaluate passkey retrieval via NLL gap and autoregressive generation.

    **Method A -- Teacher-Forcing NLL Gap**

    For each ``(length, depth, trial)`` triple the function:

    1. Builds an eval sequence containing a random 5-digit passkey at the
       given depth, followed by the probe prefix ``<<PASS:``.
    2. Teacher-forces the correct passkey digits after the probe and
       records their per-token NLL.
    3. Teacher-forces a wrong passkey (every digit different) and records
       its NLL.
    4. ``gap = NLL_wrong - NLL_correct``.  A positive gap means the model
       successfully retrieved the passkey.

    **Method B -- Autoregressive Generation** (auxiliary)

    Truncates the sequence at ``<<PASS:``, greedily generates 9 tokens
    (5 digits + 4 dashes), and checks for an exact match.

    Args:
        model: Trained GPT model.
        tokenizer: HuggingFace tokenizer.
        filler_tokens: 1-D tensor of filler token IDs.
        lengths: List of total context lengths to test.
        depths: List of depth fractions (0.0 -- 1.0).
        num_trials: Number of random passkeys per (length, depth).
        seed: Base random seed.

    Returns:
        Dictionary with keys:

        - ``"details"``: nested dict keyed by
          ``"L={length}_d={depth:.1f}_t={trial}"`` containing per-trial
          metrics.
        - ``"summary"``: per-``(length, depth)`` aggregates (mean gap,
          retrieval rate, AR exact match rate).
        - ``"global"``: overall retrieval rate and mean NLL gap.
    """
    model.eval()
    rng = random.Random(seed)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    details: Dict[str, dict] = {}
    summary: Dict[str, dict] = {}
    global_gaps: List[float] = []
    global_retrieved: List[bool] = []
    global_ar_match: List[bool] = []

    for L in lengths:
        # Extend RoPE cos/sin cache once per length
        try:
            model.extend_rope(L + 64)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] Cannot extend rope to L={L}, skipping")
                _clear_cache()
                continue
            raise

        for d in depths:
            key_prefix = f"L={L}_d={d:.1f}"
            print(f"  {key_prefix} ...", end=" ", flush=True)

            trial_gaps: List[float] = []
            trial_retrieved: List[bool] = []
            trial_ar_match: List[bool] = []

            for t in range(num_trials):
                trial_seed = seed + L * 10000 + int(d * 1000) + t
                trial_rng = random.Random(trial_seed)

                # Random 5-digit passkey
                digits = [str(trial_rng.randint(0, 9)) for _ in range(5)]
                correct_passkey = DASH.join(digits)
                wrong_passkey = _make_wrong_passkey(correct_passkey, trial_rng)

                try:
                    result = _eval_single_trial(
                        model, tokenizer, filler_tokens, ctx,
                        L, d, correct_passkey, wrong_passkey,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM at trial {t}", end=" ")
                        _clear_cache()
                        break
                    raise

                trial_key = f"{key_prefix}_t={t}"
                details[trial_key] = result

                trial_gaps.append(result["nll_gap"])
                trial_retrieved.append(result["retrieved"])
                trial_ar_match.append(result["ar_exact_match"])

            n = len(trial_gaps)
            if n > 0:
                mean_gap = sum(trial_gaps) / n
                retrieval_rate = sum(trial_retrieved) / n
                ar_rate = sum(trial_ar_match) / n
                summary[key_prefix] = {
                    "mean_nll_gap": round(mean_gap, 4),
                    "retrieval_rate": round(retrieval_rate, 4),
                    "ar_exact_match_rate": round(ar_rate, 4),
                    "trials": n,
                }
                print(
                    f"gap={mean_gap:+.3f}  "
                    f"ret={retrieval_rate:.0%}  "
                    f"AR={ar_rate:.0%}  "
                    f"({n} trials)"
                )
                global_gaps.extend(trial_gaps)
                global_retrieved.extend(trial_retrieved)
                global_ar_match.extend(trial_ar_match)
            else:
                summary[key_prefix] = {
                    "mean_nll_gap": float("nan"),
                    "retrieval_rate": float("nan"),
                    "ar_exact_match_rate": float("nan"),
                    "trials": 0,
                }
                print("no valid trials")

    # Global summary
    n_global = len(global_gaps)
    global_summary = {
        "total_trials": n_global,
        "mean_nll_gap": round(sum(global_gaps) / n_global, 4) if n_global else float("nan"),
        "retrieval_rate": round(sum(global_retrieved) / n_global, 4) if n_global else float("nan"),
        "ar_exact_match_rate": round(sum(global_ar_match) / n_global, 4) if n_global else float("nan"),
    }

    return {
        "details": details,
        "summary": summary,
        "global": global_summary,
    }


def _eval_single_trial(
    model: GPT,
    tokenizer,
    filler_tokens: torch.Tensor,
    ctx,
    total_length: int,
    depth: float,
    correct_passkey: str,
    wrong_passkey: str,
) -> Dict:
    """Run a single NLL-gap trial + autoregressive generation.

    Returns a dict with nll_correct, nll_wrong, nll_gap, retrieved,
    ar_generated, ar_exact_match.
    """
    # Build eval sequence: [filler] <<PASS:correct>> [filler] <<PASS:
    input_ids, passkey_start, probe_start = build_passkey_eval_sequence(
        filler_tokens, correct_passkey, tokenizer, total_length, depth,
    )

    # Tokenise the answer portions (digits after <<PASS:)
    correct_answer_text = f"{correct_passkey}{PASSKEY_SUFFIX}"
    wrong_answer_text = f"{wrong_passkey}{PASSKEY_SUFFIX}"
    correct_answer_ids = tokenizer.encode(correct_answer_text, add_special_tokens=False)
    wrong_answer_ids = tokenizer.encode(wrong_answer_text, add_special_tokens=False)

    # ------- Method A: Teacher-Forcing NLL Gap -------
    # Append correct answer to the input to teacher-force
    full_correct = torch.cat([input_ids, torch.tensor(correct_answer_ids, dtype=torch.long)])
    full_wrong = torch.cat([input_ids, torch.tensor(wrong_answer_ids, dtype=torch.long)])

    nll_correct = _compute_nll_at_positions(
        model, ctx, full_correct, probe_start, len(correct_answer_ids)
    )
    nll_wrong = _compute_nll_at_positions(
        model, ctx, full_wrong, probe_start, len(wrong_answer_ids)
    )

    gap = nll_wrong - nll_correct
    retrieved = nll_correct < nll_wrong

    # ------- Method B: Autoregressive Generation -------
    ar_generated, ar_match = _autoregressive_probe(
        model, ctx, tokenizer, input_ids, correct_passkey,
    )

    return {
        "nll_correct": round(nll_correct, 4),
        "nll_wrong": round(nll_wrong, 4),
        "nll_gap": round(gap, 4),
        "retrieved": retrieved,
        "correct_passkey": correct_passkey,
        "wrong_passkey": wrong_passkey,
        "ar_generated": ar_generated,
        "ar_exact_match": ar_match,
    }


def _compute_nll_at_positions(
    model: GPT,
    ctx,
    token_ids: torch.Tensor,
    start: int,
    n_answer_tokens: int,
) -> float:
    """Compute mean NLL for ``n_answer_tokens`` starting at ``start``.

    The model receives the full ``token_ids`` as input and we extract the
    cross-entropy loss for positions ``start`` through
    ``start + n_answer_tokens - 1`` where each position predicts the *next*
    token.
    """
    input_ids = token_ids.unsqueeze(0).to(DEVICE)
    with ctx:
        logits = model(input_ids[:, :-1])  # (1, T-1, vocab)

    # Positions in logits that predict the answer tokens
    # logits[0, pos, :] predicts token_ids[pos+1]
    # We want to score token_ids[start+1 ... start+n_answer_tokens]
    # which are predicted by logits at positions [start ... start+n_answer_tokens-1]
    target_positions = list(range(start, start + n_answer_tokens))
    target_ids = token_ids[start + 1 : start + 1 + n_answer_tokens]

    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    nll_sum = 0.0
    for i, pos in enumerate(target_positions):
        if pos >= log_probs.size(0):
            break
        nll_sum -= log_probs[pos, target_ids[i]].item()

    count = min(n_answer_tokens, log_probs.size(0) - start)
    return nll_sum / max(count, 1)


def _autoregressive_probe(
    model: GPT,
    ctx,
    tokenizer,
    input_ids: torch.Tensor,
    correct_passkey: str,
) -> Tuple[str, bool]:
    """Greedy autoregressive generation from the probe position.

    Generates 9 tokens (5 digits + 4 dashes) and checks exact match against
    the correct passkey + ``>>``.

    Returns ``(generated_text, exact_match)``.
    """
    expected_text = f"{correct_passkey}{PASSKEY_SUFFIX}"
    expected_ids = tokenizer.encode(expected_text, add_special_tokens=False)
    gen_len = len(expected_ids)

    cur = input_ids.unsqueeze(0).to(DEVICE)
    generated: List[int] = []

    for _ in range(gen_len):
        with ctx:
            logits = model(cur)
        next_token = logits[0, -1, :].argmax().item()
        generated.append(next_token)
        cur = torch.cat(
            [cur, torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)],
            dim=1,
        )

    gen_text = tokenizer.decode(generated, skip_special_tokens=False)
    exact_match = generated == expected_ids

    return gen_text, exact_match


def _clear_cache() -> None:
    """Free device memory after OOM."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


# ===================================================================
# 5. sanity_check_tokenizer
# ===================================================================

def sanity_check_tokenizer(tokenizer) -> bool:
    """Verify that each digit in ``<<PASS:7-4-2-9-1>>`` is an independent token.

    Prints the tokenisation and returns ``True`` if every single digit
    (0--9) is tokenised as its own token within the passkey marker.

    Args:
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        ``True`` if all digits are independently tokenised, ``False``
        otherwise.
    """
    test_str = f"{PASSKEY_PREFIX}7-4-2-9-1{PASSKEY_SUFFIX}"
    token_ids = tokenizer.encode(test_str, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    print(f"  Tokenisation of {test_str!r}:")
    print(f"    IDs:    {token_ids}")
    print(f"    Tokens: {tokens}")

    # Check that each digit 7, 4, 2, 9, 1 appears as a standalone token
    target_digits = ["7", "4", "2", "9", "1"]
    found_digits: List[str] = []
    ok = True

    for digit in target_digits:
        # Look for a token that is exactly the digit (possibly with a leading
        # space or BPE prefix stripped)
        digit_found = False
        for tok in tokens:
            stripped = tok.strip()
            if stripped == digit:
                digit_found = True
                break
        if digit_found:
            found_digits.append(digit)
        else:
            ok = False
            print(f"    WARNING: digit '{digit}' is NOT an independent token")

    if ok:
        print(f"    OK: all digits {target_digits} are independent tokens")
    else:
        print(f"    FAIL: only {found_digits} out of {target_digits} are independent")

    return ok


# ===================================================================
# Standalone main for testing
# ===================================================================

def main() -> None:
    """Run passkey NLL-gap evaluation on a trained checkpoint."""
    parser = argparse.ArgumentParser(
        description="Passkey NLL-gap eval for from-scratch EVQ models"
    )
    parser.add_argument(
        "--work_dir", type=str, required=True,
        help="Sweep work directory containing checkpoints",
    )
    parser.add_argument("--tier", type=str, default="50m")
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base", type=float, default=500000.0)
    parser.add_argument(
        "--lengths", type=str, default="2048,4096,8192",
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--depths", type=str, default="0.1,0.5,0.9",
        help="Comma-separated depth fractions",
    )
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument(
        "--val_tokens", type=int, default=5_000_000,
        help="Max tokens for filler text",
    )
    args = parser.parse_args()

    import json

    work_dir = Path(args.work_dir)
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Sanity check
    print("\n--- Tokenizer sanity check ---")
    tok_ok = sanity_check_tokenizer(tokenizer)
    if not tok_ok:
        print("  WARNING: tokenizer sanity check failed, results may be unreliable")

    # Load filler data
    print("\n--- Loading filler data ---")
    from run_evq_sweep import load_val

    filler_tokens = load_val(tokenizer, args.val_tokens)
    print(f"  Filler tokens: {len(filler_tokens)}")

    # Load model
    print("\n--- Loading model ---")
    cfg = TIER_CONFIGS[args.tier].copy()
    run_id = f"{args.tier}_tau{args.tau:.2f}_seed{args.seed}"
    ckpt_path = work_dir / run_id / "model.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    inv_freq = evq_cosh_inv_freq(cfg["head_dim"], args.tau, args.base)
    model = GPT(cfg, inv_freq)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])
    model = model.to(DEVICE)
    model.eval()
    print(f"  Loaded {run_id}")

    # Run evaluation
    print(f"\n--- Passkey NLL-gap evaluation ---")
    print(f"  lengths={lengths}  depths={depths}  trials={args.num_trials}")
    results = eval_passkey_nll_gap(
        model, tokenizer, filler_tokens,
        lengths, depths, args.num_trials, seed=args.seed,
    )

    # Print global summary
    g = results["global"]
    print(f"\n{'='*60}")
    print(f"  GLOBAL SUMMARY ({run_id})")
    print(f"    Total trials:       {g['total_trials']}")
    print(f"    Mean NLL gap:       {g['mean_nll_gap']:+.4f}")
    print(f"    Retrieval rate:     {g['retrieval_rate']:.1%}")
    print(f"    AR exact match:     {g['ar_exact_match_rate']:.1%}")
    print(f"{'='*60}")

    # Save results
    out_path = work_dir / f"passkey_nll_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
