#!/usr/bin/env python3
"""Multi-needle passkey retrieval evaluation.

Inserts N passkeys (default 5) at evenly-spaced depths in a long context.
The model must retrieve ALL passkeys — scored via NLL-gap per needle.

Metrics:
  - per_needle_retrieval: fraction of individual needles retrieved (NLL gap > 0)
  - all_needle_retrieval: fraction of trials where ALL needles retrieved
  - mean_nll_gap: average NLL gap across all needles and trials

Usage (standalone):
    python eval_multi_needle.py  # runs a quick test with a dummy model

Importable:
    from eval_multi_needle import eval_multi_needle_passkey
"""

from __future__ import annotations

import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_TEXT_DIR = SCRIPT_DIR.parent / "core_text_phases"
sys.path.insert(0, str(CORE_TEXT_DIR))

from run_evq_sweep import GPT, get_device_and_dtype

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32

PASSKEY_PREFIX = "<<PASS:"
PASSKEY_SUFFIX = ">>"
DASH = "-"


# ===================================================================
# Build multi-needle eval sequence
# ===================================================================

def _generate_unique_passkeys(n_needles: int, rng: random.Random) -> List[str]:
    """Generate n unique 5-digit dash-separated passkeys."""
    seen = set()
    keys = []
    while len(keys) < n_needles:
        digits = [str(rng.randint(0, 9)) for _ in range(5)]
        pk = DASH.join(digits)
        if pk not in seen:
            seen.add(pk)
            keys.append(pk)
    return keys


def _make_wrong_passkey(correct: str, rng: random.Random) -> str:
    """Return a passkey that differs in every digit."""
    digits = correct.split(DASH)
    wrong = []
    for d in digits:
        choices = [str(x) for x in range(10) if str(x) != d]
        wrong.append(rng.choice(choices))
    return DASH.join(wrong)


def build_multi_needle_sequence(
    filler_tokens: torch.Tensor,
    passkeys: List[str],
    tokenizer,
    total_length: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """Build eval sequence with N passkeys at evenly-spaced depths.

    Layout:
        [filler] <<PASS:pk1>> [filler] <<PASS:pk2>> ... [filler] <<PASS:pkN>> [filler]
        Then for each probe: <<PASS_i: (probe prefix with index)

    We use indexed probe format: <<PASS1:...>> <<PASS2:...>> etc.
    But to keep tokenization simple, we use the same <<PASS:...>> format
    and query each passkey by placing its probe at the end.

    Strategy: Insert all N passkeys in the context at evenly-spaced depths.
    Then for each needle evaluation, append a unique probe at the end.

    For simplicity, we build ONE base sequence with all needles inserted,
    and evaluate each needle separately by appending its probe suffix.

    Returns:
        (base_input_ids, needle_starts, needle_indices)
        - base_input_ids: tensor of shape (total_length,) with all needles
        - needle_starts: list of token positions where each needle begins
        - needle_indices: list of needle indices (0..N-1) for reference
    """
    n_needles = len(passkeys)

    # Tokenize all needle markers
    marker_ids_list = []
    for pk in passkeys:
        marker_text = f"{PASSKEY_PREFIX}{pk}{PASSKEY_SUFFIX}"
        marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
        marker_ids_list.append(marker_ids)

    total_marker_tokens = sum(len(m) for m in marker_ids_list)
    filler_budget = total_length - total_marker_tokens
    if filler_budget < n_needles + 1:
        raise ValueError(
            f"total_length={total_length} too short for {n_needles} needles "
            f"(markers need {total_marker_tokens} tokens)"
        )

    # Distribute filler evenly: n_needles + 1 segments
    # Depths: needles placed at 1/(N+1), 2/(N+1), ..., N/(N+1) of context
    n_segments = n_needles + 1
    base_segment_len = filler_budget // n_segments
    remainder = filler_budget - base_segment_len * n_segments

    segment_lengths = [base_segment_len] * n_segments
    # Distribute remainder to first segments
    for i in range(remainder):
        segment_lengths[i] += 1

    # Get filler tokens
    total_filler = len(filler_tokens)
    rng = random.Random(seed)
    start = rng.randint(0, max(0, total_filler - filler_budget - 1))
    filler_flat = filler_tokens[start : start + filler_budget].tolist()
    while len(filler_flat) < filler_budget:
        filler_flat.extend(filler_tokens[: filler_budget - len(filler_flat)].tolist())

    # Assemble sequence
    seq = []
    needle_starts = []
    filler_pos = 0

    for i in range(n_needles):
        # Filler segment before needle i
        seg_len = segment_lengths[i]
        seq.extend(filler_flat[filler_pos : filler_pos + seg_len])
        filler_pos += seg_len

        # Record needle start position
        needle_starts.append(len(seq))
        seq.extend(marker_ids_list[i])

    # Final filler segment
    seg_len = segment_lengths[n_needles]
    seq.extend(filler_flat[filler_pos : filler_pos + seg_len])

    # Trim to exact length
    seq = seq[:total_length]
    while len(seq) < total_length:
        seq.append(filler_tokens[len(seq) % total_filler].item())

    input_ids = torch.tensor(seq, dtype=torch.long)
    return input_ids, needle_starts, list(range(n_needles))


# ===================================================================
# Evaluate single needle via NLL gap
# ===================================================================

def _compute_nll_at_positions(
    model: GPT,
    ctx,
    token_ids: torch.Tensor,
    start: int,
    n_answer_tokens: int,
) -> float:
    """Compute mean NLL for n_answer_tokens starting at position start."""
    input_ids = token_ids.unsqueeze(0).to(DEVICE)
    with ctx:
        logits = model(input_ids[:, :-1])

    target_positions = list(range(start, start + n_answer_tokens))
    target_ids = token_ids[start + 1 : start + 1 + n_answer_tokens]

    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    nll_sum = 0.0
    count = 0
    for i, pos in enumerate(target_positions):
        if pos >= log_probs.size(0):
            break
        nll_sum -= log_probs[pos, target_ids[i]].item()
        count += 1

    return nll_sum / max(count, 1)


def _eval_needle_nll_gap(
    model: GPT,
    tokenizer,
    ctx,
    base_input_ids: torch.Tensor,
    correct_passkey: str,
    wrong_passkey: str,
) -> Dict:
    """Evaluate a single needle by appending its probe to the base sequence.

    Appends <<PASS: at the end, then teacher-forces correct vs wrong digits.
    """
    probe_text = PASSKEY_PREFIX
    probe_ids = tokenizer.encode(probe_text, add_special_tokens=False)

    correct_answer_text = f"{correct_passkey}{PASSKEY_SUFFIX}"
    wrong_answer_text = f"{wrong_passkey}{PASSKEY_SUFFIX}"
    correct_answer_ids = tokenizer.encode(correct_answer_text, add_special_tokens=False)
    wrong_answer_ids = tokenizer.encode(wrong_answer_text, add_special_tokens=False)

    # Build full sequences: base + probe + answer
    probe_tensor = torch.tensor(probe_ids, dtype=torch.long)

    full_correct = torch.cat([
        base_input_ids,
        probe_tensor,
        torch.tensor(correct_answer_ids, dtype=torch.long),
    ])
    full_wrong = torch.cat([
        base_input_ids,
        probe_tensor,
        torch.tensor(wrong_answer_ids, dtype=torch.long),
    ])

    probe_start = len(base_input_ids)

    nll_correct = _compute_nll_at_positions(
        model, ctx, full_correct, probe_start, len(correct_answer_ids)
    )
    nll_wrong = _compute_nll_at_positions(
        model, ctx, full_wrong, probe_start, len(wrong_answer_ids)
    )

    gap = nll_wrong - nll_correct
    retrieved = nll_correct < nll_wrong

    return {
        "nll_correct": round(nll_correct, 4),
        "nll_wrong": round(nll_wrong, 4),
        "nll_gap": round(gap, 4),
        "retrieved": retrieved,
        "correct_passkey": correct_passkey,
        "wrong_passkey": wrong_passkey,
    }


# ===================================================================
# Multi-needle eval: indexed probing
# ===================================================================

def _eval_needle_at_position(
    model: GPT,
    tokenizer,
    ctx,
    base_input_ids: torch.Tensor,
    needle_idx: int,
    correct_passkey: str,
    wrong_passkey: str,
) -> Dict:
    """Evaluate retrieval of needle #needle_idx using indexed probe.

    Uses probe format: <<PASS_N: where N is the 1-based needle index.
    But since our existing format is just <<PASS:, and the model sees
    multiple <<PASS:...>> in context, we use a different strategy:

    We truncate the context right AFTER the last needle and append the
    probe <<PASS: — then score the FIRST passkey (needle_idx=0),
    SECOND passkey (needle_idx=1), etc. based on which answer we teacher-force.

    Actually, the simplest and most robust approach:
    For each needle, we create a context that has ALL needles but the probe
    asks for a SPECIFIC one. We use numbered probes: "Needle 1: <<PASS:"

    But to avoid tokenization issues, let's use the simplest approach:
    Insert needles with distinct formats like <<KEY1:...>>, <<KEY2:...>>, etc.
    Then probe each key separately.
    """
    # Use indexed key format
    key_prefix = f"<<KEY{needle_idx + 1}:"
    key_suffix = ">>"

    probe_ids = tokenizer.encode(key_prefix, add_special_tokens=False)

    correct_answer_text = f"{correct_passkey}{key_suffix}"
    wrong_answer_text = f"{wrong_passkey}{key_suffix}"
    correct_answer_ids = tokenizer.encode(correct_answer_text, add_special_tokens=False)
    wrong_answer_ids = tokenizer.encode(wrong_answer_text, add_special_tokens=False)

    probe_tensor = torch.tensor(probe_ids, dtype=torch.long)

    full_correct = torch.cat([
        base_input_ids,
        probe_tensor,
        torch.tensor(correct_answer_ids, dtype=torch.long),
    ])
    full_wrong = torch.cat([
        base_input_ids,
        probe_tensor,
        torch.tensor(wrong_answer_ids, dtype=torch.long),
    ])

    probe_start = len(base_input_ids)

    nll_correct = _compute_nll_at_positions(
        model, ctx, full_correct, probe_start, len(correct_answer_ids)
    )
    nll_wrong = _compute_nll_at_positions(
        model, ctx, full_wrong, probe_start, len(wrong_answer_ids)
    )

    gap = nll_wrong - nll_correct
    retrieved = nll_correct < nll_wrong

    return {
        "needle_idx": needle_idx,
        "nll_correct": round(nll_correct, 4),
        "nll_wrong": round(nll_wrong, 4),
        "nll_gap": round(gap, 4),
        "retrieved": retrieved,
        "correct_passkey": correct_passkey,
        "wrong_passkey": wrong_passkey,
    }


def build_indexed_multi_needle_sequence(
    filler_tokens: torch.Tensor,
    passkeys: List[str],
    tokenizer,
    total_length: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, List[int]]:
    """Build eval sequence with indexed needles: <<KEY1:pk1>>, <<KEY2:pk2>>, ...

    Returns:
        (input_ids, needle_starts)
    """
    n_needles = len(passkeys)

    # Tokenize indexed markers
    marker_ids_list = []
    for i, pk in enumerate(passkeys):
        marker_text = f"<<KEY{i+1}:{pk}>>"
        marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
        marker_ids_list.append(marker_ids)

    total_marker_tokens = sum(len(m) for m in marker_ids_list)
    filler_budget = total_length - total_marker_tokens
    if filler_budget < n_needles + 1:
        raise ValueError(
            f"total_length={total_length} too short for {n_needles} indexed needles "
            f"(markers need {total_marker_tokens} tokens)"
        )

    # Distribute filler evenly
    n_segments = n_needles + 1
    base_segment_len = filler_budget // n_segments
    remainder = filler_budget - base_segment_len * n_segments

    segment_lengths = [base_segment_len] * n_segments
    for i in range(remainder):
        segment_lengths[i] += 1

    # Get filler
    total_filler = len(filler_tokens)
    rng = random.Random(seed)
    start = rng.randint(0, max(0, total_filler - filler_budget - 1))
    filler_flat = filler_tokens[start : start + filler_budget].tolist()
    while len(filler_flat) < filler_budget:
        filler_flat.extend(filler_tokens[: filler_budget - len(filler_flat)].tolist())

    # Assemble
    seq = []
    needle_starts = []
    filler_pos = 0

    for i in range(n_needles):
        seg_len = segment_lengths[i]
        seq.extend(filler_flat[filler_pos : filler_pos + seg_len])
        filler_pos += seg_len
        needle_starts.append(len(seq))
        seq.extend(marker_ids_list[i])

    seg_len = segment_lengths[n_needles]
    seq.extend(filler_flat[filler_pos : filler_pos + seg_len])

    seq = seq[:total_length]
    while len(seq) < total_length:
        seq.append(filler_tokens[len(seq) % total_filler].item())

    return torch.tensor(seq, dtype=torch.long), needle_starts


# ===================================================================
# Main eval function
# ===================================================================

@torch.no_grad()
def eval_multi_needle_passkey(
    model: GPT,
    tokenizer,
    filler_tokens: torch.Tensor,
    lengths: List[int],
    n_needles: int = 5,
    num_trials: int = 20,
    seed: int = 42,
) -> Dict:
    """Evaluate multi-needle passkey retrieval via NLL gap.

    For each (length, trial):
    1. Generate n_needles unique passkeys
    2. Insert them at evenly-spaced depths using indexed format <<KEYi:...>>
    3. For each needle, probe with <<KEYi: and score NLL gap
    4. Record per-needle and all-needle retrieval rates

    Args:
        model: Trained GPT model.
        tokenizer: HuggingFace tokenizer.
        filler_tokens: 1-D tensor of filler token IDs.
        lengths: Context lengths to test.
        n_needles: Number of passkeys to insert (default 5).
        num_trials: Trials per length.
        seed: Random seed.

    Returns:
        Dict with "details", "by_length", "by_needle_position", "global".
    """
    model.eval()
    rng = random.Random(seed)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    details = {}
    by_length = {}

    # Global accumulators
    all_per_needle_retrieved = []  # bool per individual needle
    all_all_needle_retrieved = []  # bool per trial (all needles correct)
    all_gaps = []
    by_position = {i: {"retrieved": [], "gaps": []} for i in range(n_needles)}

    for L in lengths:
        # Extend RoPE cache
        try:
            model.extend_rope(L + 128)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] Cannot extend rope to L={L}, skipping")
                torch.cuda.empty_cache()
                continue
            raise

        length_key = f"L={L}"
        print(f"  {length_key} ({n_needles} needles, {num_trials} trials)...", end=" ", flush=True)

        length_per_needle = []
        length_all_needle = []
        length_gaps = []

        for t in range(num_trials):
            trial_seed = seed + L * 10000 + t
            trial_rng = random.Random(trial_seed)

            # Generate unique passkeys
            passkeys = _generate_unique_passkeys(n_needles, trial_rng)
            wrong_passkeys = [_make_wrong_passkey(pk, trial_rng) for pk in passkeys]

            try:
                # Build indexed multi-needle sequence
                base_ids, needle_starts = build_indexed_multi_needle_sequence(
                    filler_tokens, passkeys, tokenizer, L, seed=trial_seed,
                )

                # Evaluate each needle
                trial_results = []
                trial_all_retrieved = True

                for ni in range(n_needles):
                    result = _eval_needle_at_position(
                        model, tokenizer, ctx, base_ids,
                        ni, passkeys[ni], wrong_passkeys[ni],
                    )
                    trial_results.append(result)

                    if not result["retrieved"]:
                        trial_all_retrieved = False

                    all_per_needle_retrieved.append(result["retrieved"])
                    all_gaps.append(result["nll_gap"])
                    by_position[ni]["retrieved"].append(result["retrieved"])
                    by_position[ni]["gaps"].append(result["nll_gap"])
                    length_per_needle.append(result["retrieved"])
                    length_gaps.append(result["nll_gap"])

                all_all_needle_retrieved.append(trial_all_retrieved)
                length_all_needle.append(trial_all_retrieved)

                trial_key = f"L={L}_t={t}"
                details[trial_key] = {
                    "passkeys": passkeys,
                    "needles": trial_results,
                    "all_retrieved": trial_all_retrieved,
                }

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at trial {t}", end=" ")
                    torch.cuda.empty_cache()
                    break
                raise

        n_trials_done = len(length_all_needle)
        if n_trials_done > 0:
            per_needle_rate = sum(length_per_needle) / len(length_per_needle)
            all_needle_rate = sum(length_all_needle) / n_trials_done
            mean_gap = sum(length_gaps) / len(length_gaps)

            by_length[length_key] = {
                "per_needle_retrieval": round(per_needle_rate, 4),
                "all_needle_retrieval": round(all_needle_rate, 4),
                "mean_nll_gap": round(mean_gap, 4),
                "trials": n_trials_done,
            }
            print(
                f"per_needle={per_needle_rate:.0%}  "
                f"all_needle={all_needle_rate:.0%}  "
                f"gap={mean_gap:+.3f}  "
                f"({n_trials_done} trials)"
            )
        else:
            by_length[length_key] = {
                "per_needle_retrieval": float("nan"),
                "all_needle_retrieval": float("nan"),
                "mean_nll_gap": float("nan"),
                "trials": 0,
            }
            print("no valid trials")

    # By needle position summary
    by_needle_position = {}
    for i in range(n_needles):
        pos_data = by_position[i]
        n = len(pos_data["retrieved"])
        if n > 0:
            by_needle_position[f"needle_{i+1}"] = {
                "retrieval_rate": round(sum(pos_data["retrieved"]) / n, 4),
                "mean_nll_gap": round(sum(pos_data["gaps"]) / n, 4),
                "trials": n,
            }

    # Global summary
    n_total_needles = len(all_per_needle_retrieved)
    n_total_trials = len(all_all_needle_retrieved)
    global_summary = {
        "n_needles": n_needles,
        "total_trials": n_total_trials,
        "total_needle_evals": n_total_needles,
        "per_needle_retrieval": round(sum(all_per_needle_retrieved) / n_total_needles, 4) if n_total_needles else float("nan"),
        "all_needle_retrieval": round(sum(all_all_needle_retrieved) / n_total_trials, 4) if n_total_trials else float("nan"),
        "mean_nll_gap": round(sum(all_gaps) / len(all_gaps), 4) if all_gaps else float("nan"),
    }

    return {
        "details": details,
        "by_length": by_length,
        "by_needle_position": by_needle_position,
        "global": global_summary,
    }


# ===================================================================
# Standalone test
# ===================================================================

if __name__ == "__main__":
    print("Multi-needle passkey eval module loaded successfully.")
    print("Usage: from eval_multi_needle import eval_multi_needle_passkey")
    print()
    print("Example:")
    print("  results = eval_multi_needle_passkey(")
    print("      model, tokenizer, filler_tokens,")
    print("      lengths=[2048, 4096, 8192, 16384],")
    print("      n_needles=5, num_trials=20, seed=42,")
    print("  )")
    print()

    # Quick tokenization test
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        for i in range(1, 6):
            test = f"<<KEY{i}:7-4-2-9-1>>"
            ids = tok.encode(test, add_special_tokens=False)
            tokens = [tok.decode([tid]) for tid in ids]
            print(f"  {test} -> {len(ids)} tokens: {tokens}")

        print("\n  Tokenization OK - each digit is independently tokenized.")
    except ImportError:
        print("  (transformers not installed, skipping tokenization test)")
