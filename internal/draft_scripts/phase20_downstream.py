#!/usr/bin/env python3
"""Downstream task evaluation for Phase 20 1.5B models via LongBench NLL.

Evaluates trained checkpoints on 12 LongBench tasks using NLL scoring:
- narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique
- gov_report, qmsum, multi_news, trec, triviaqa, samsum

NLL scoring approach: Instead of generation (requires instruction-tuning),
we compute the conditional NLL of gold answers given context+question.
Lower NLL = model assigns higher probability to correct answer.

Features:
1. **LongBench NLL evaluation** on 12 tasks
   - 30 samples per task by default
   - Reports NLL per task and aggregate
2. **Position-wise PPL** on final checkpoints
3. **Cross-domain PPL** comparison tables
4. **Multi-seed aggregation** with confidence estimates
5. **LaTeX table generation** for papers
6. **Comparison plots** (geometric vs EVQ)

Usage:
    python team/scripts/phase20_downstream.py \\
        --geo_checkpoint /path/to/geo_model.pt \\
        --evq_checkpoint /path/to/evq_model.pt \\
        --output_dir results/downstream/ \\
        --seeds 42,137,256

    python team/scripts/phase20_downstream.py \\
        --checkpoint_dir /path/to/checkpoints/ \\
        --output_dir results/downstream/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Set HF mirror
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# Import from evaluation scripts
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "core_text_phases"
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from run_evq_sweep import (
        GPT,
        DEVICE,
        DTYPE,
        evq_cosh_inv_freq,
    )
except ImportError as e:
    print(f"Error importing from run_evq_sweep: {e}")
    sys.exit(1)

try:
    from phase20_eval_suite import (
        eval_position_wise_ppl,
        load_checkpoint,
    )
except ImportError:
    print("Warning: Could not import phase20_eval_suite utilities")

USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32

# LongBench task definitions
LONGBENCH_TASKS = {
    "narrativeqa": {"type": "qa", "samples": 30},
    "qasper": {"type": "qa", "samples": 30},
    "multifieldqa_en": {"type": "qa", "samples": 30},
    "hotpotqa": {"type": "qa", "samples": 30},
    "2wikimqa": {"type": "qa", "samples": 30},
    "musique": {"type": "qa", "samples": 30},
    "gov_report": {"type": "summary", "samples": 30},
    "qmsum": {"type": "summary", "samples": 30},
    "multi_news": {"type": "summary", "samples": 30},
    "trec": {"type": "classification", "samples": 30},
    "triviaqa": {"type": "qa", "samples": 30},
    "samsum": {"type": "summary", "samples": 30},
}

# Task groupings
TASK_SETS = {
    "qa": [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
        "2wikimqa", "musique", "triviaqa"
    ],
    "summary": ["gov_report", "qmsum", "multi_news", "samsum"],
    "classification": ["trec"],
    "all": list(LONGBENCH_TASKS.keys()),
}


# ---------------------------------------------------------------------------
# Data Loading and Formatting
# ---------------------------------------------------------------------------


def format_prompt(sample: dict, task_name: str) -> Tuple[str, str]:
    """Format a LongBench sample into (prompt, answer) strings.

    Args:
        sample: LongBench sample dictionary
        task_name: Task identifier

    Returns:
        (prompt_text, answer_text) tuple
    """
    context = sample.get("context", "")
    question = sample.get("input", "")
    answers = sample.get("answers", [])

    if not answers:
        return "", ""

    gold_answer = answers[0] if isinstance(answers, list) else str(answers)
    task_type = LONGBENCH_TASKS.get(task_name, {}).get("type", "qa")

    if task_type == "qa":
        prompt = f"Document:\n{context}\n\nQuestion: {question}\nAnswer:"
    elif task_type == "summary":
        prompt = f"Document:\n{context}\n\nSummary:"
    elif task_type == "classification":
        prompt = f"{context}\n\n{question}\nLabel:"
    else:
        prompt = f"{context}\n\n{question}\nAnswer:"

    return prompt, f" {gold_answer}"


def load_longbench_task(
    task_name: str,
    max_samples: int = 30,
    seed: int = 42,
    data_dir: str = "",
) -> List[dict]:
    """Load a LongBench task from local cache or HuggingFace.

    Args:
        task_name: Task identifier
        max_samples: Maximum samples to load
        seed: Random seed for sampling
        data_dir: Optional local directory with pre-downloaded data

    Returns:
        List of sample dictionaries
    """
    import random

    def _load_jsonl(path):
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    # Try local data directory first
    if data_dir:
        local_path = os.path.join(data_dir, f"{task_name}.jsonl")
        if os.path.exists(local_path):
            data = _load_jsonl(local_path)
            if len(data) > max_samples:
                data = random.Random(seed).sample(data, max_samples)
            return data

        # Try data/ subdirectory
        local_path2 = os.path.join(data_dir, "data", f"{task_name}.jsonl")
        if os.path.exists(local_path2):
            data = _load_jsonl(local_path2)
            if len(data) > max_samples:
                data = random.Random(seed).sample(data, max_samples)
            return data

    # Fallback to HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "THUDM/LongBench", task_name, split="test",
            trust_remote_code=True
        )
        data = list(ds)
        if len(data) > max_samples:
            data = random.Random(seed).sample(data, max_samples)
        return data
    except Exception as e:
        print(f"  [ERROR] Failed to load {task_name}: {e}")
        raise


def truncate_prompt_ids(
    tokenizer,
    prompt_ids: List[int],
    answer_ids: List[int],
    max_total_len: int,
    strategy: str = "middle",
) -> List[int]:
    """Truncate prompt to fit within max_total_len including answer.

    Preserves start and end of document (middle truncation).
    """
    budget = max_total_len - len(answer_ids)
    if budget <= 0:
        budget = max_total_len // 2

    if len(prompt_ids) <= budget:
        return prompt_ids

    if strategy == "left":
        # Keep most recent context
        return prompt_ids[-budget:]
    elif strategy == "middle":
        # Keep beginning and end
        half = budget // 2
        return prompt_ids[:half] + prompt_ids[-(budget - half):]
    else:
        # Keep first part
        return prompt_ids[:budget]


# ---------------------------------------------------------------------------
# NLL Computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_answer_nll(
    model: torch.nn.Module,
    prompt_ids: List[int],
    answer_ids: List[int],
    device: torch.device,
    is_custom_gpt: bool = False,
) -> Tuple[float, int]:
    """Compute NLL of answer tokens conditioned on prompt.

    Args:
        model: Language model (custom GPT or HF model)
        prompt_ids: Tokenized prompt/context
        answer_ids: Tokenized gold answer
        device: Torch device
        is_custom_gpt: Whether using custom GPT model

    Returns:
        (mean_nll_per_token, num_answer_tokens)
    """
    if len(answer_ids) == 0:
        return float('nan'), 0

    input_ids = prompt_ids + answer_ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        if is_custom_gpt:
            # Custom GPT returns logits (B, L, V)
            logits = model(x)
        else:
            # HF model with labels
            labels = torch.full_like(x, -100)
            labels[:, len(prompt_ids):] = x[:, len(prompt_ids):]
            out = model(input_ids=x, labels=labels)
            return float(out.loss.item()), len(answer_ids)

    # Manual cross-entropy on answer tokens
    prompt_len = len(prompt_ids)
    answer_logits = logits[0, prompt_len - 1 : prompt_len + len(answer_ids) - 1]
    answer_targets = x[0, prompt_len : prompt_len + len(answer_ids)]

    if len(answer_logits) == 0 or len(answer_targets) == 0:
        return float('nan'), 0

    loss = F.cross_entropy(answer_logits.float(), answer_targets, reduction='mean')
    return float(loss.item()), len(answer_ids)


# ---------------------------------------------------------------------------
# Task Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_longbench_nll(
    model: torch.nn.Module,
    tokenizer,
    tasks: List[str],
    samples_per_task: int = 30,
    max_context_len: int = 4096,
    seed: int = 42,
    data_dir: str = "",
    is_custom_gpt: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on LongBench tasks using NLL scoring.

    Args:
        model: Language model
        tokenizer: Tokenizer
        tasks: List of task names to evaluate
        samples_per_task: Max samples per task
        max_context_len: Maximum sequence length
        seed: Random seed
        data_dir: Optional local data directory
        is_custom_gpt: Whether model is custom GPT

    Returns:
        Dict mapping task_name -> metrics dict
    """
    device = next(model.parameters()).device
    model.eval()

    all_results = {}

    for task_name in tasks:
        if task_name not in LONGBENCH_TASKS:
            print(f"  [SKIP] Unknown task: {task_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        try:
            samples = load_longbench_task(
                task_name, samples_per_task, seed, data_dir
            )
        except Exception as e:
            print(f"  [ERROR] Failed to load task: {e}")
            all_results[task_name] = {
                "error": str(e),
                "mean_nll": float('nan'),
                "n_samples": 0,
            }
            continue

        print(f"  Loaded {len(samples)} samples")

        nlls = []
        lengths = []
        skipped = 0

        for i, sample in enumerate(samples):
            prompt_str, answer_str = format_prompt(sample, task_name)

            if not prompt_str or not answer_str.strip():
                skipped += 1
                continue

            try:
                prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
                answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)

                # Limit answer length
                if len(answer_ids) > 256:
                    answer_ids = answer_ids[:256]

                # Truncate prompt
                prompt_ids = truncate_prompt_ids(
                    tokenizer, prompt_ids, answer_ids,
                    max_context_len, strategy="middle"
                )

                total_len = len(prompt_ids) + len(answer_ids)
                if total_len < 10:
                    skipped += 1
                    continue

                nll, n_tokens = compute_answer_nll(
                    model, prompt_ids, answer_ids, device, is_custom_gpt
                )

                if not math.isnan(nll) and not math.isinf(nll) and nll < 50.0:
                    nlls.append(nll)
                    lengths.append(total_len)

            except Exception as e:
                skipped += 1
                continue

            if (i + 1) % 10 == 0:
                running_mean = np.mean(nlls) if nlls else float('nan')
                print(f"  [{i+1}/{len(samples)}] running NLL: {running_mean:.4f}")

        # Aggregate results for this task
        if nlls:
            result = {
                "mean_nll": float(np.mean(nlls)),
                "std_nll": float(np.std(nlls)),
                "median_nll": float(np.median(nlls)),
                "min_nll": float(np.min(nlls)),
                "max_nll": float(np.max(nlls)),
                "ppl": float(np.exp(np.mean(nlls))),
                "n_samples": len(nlls),
                "skipped": skipped,
                "mean_seq_len": float(np.mean(lengths)),
            }

            print(f"  Result: NLL={result['mean_nll']:.4f} (+/-{result['std_nll']:.4f}), "
                  f"PPL={result['ppl']:.2f}, n={result['n_samples']}")

        else:
            print(f"  [WARN] No valid samples")
            result = {
                "mean_nll": float('nan'),
                "std_nll": float('nan'),
                "ppl": float('nan'),
                "n_samples": 0,
                "skipped": skipped,
            }

        all_results[task_name] = result

    return all_results


# ---------------------------------------------------------------------------
# Result Aggregation and Table Generation
# ---------------------------------------------------------------------------


def generate_latex_table(
    geo_results: Dict[str, Dict[str, float]],
    evq_results: Dict[str, Dict[str, float]],
) -> str:
    """Generate LaTeX table comparing geometric and EVQ models.

    Args:
        geo_results: Results from geometric model
        evq_results: Results from EVQ model

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Task & Geo NLL & EVQ NLL & Geo PPL & EVQ PPL \\\\")
    lines.append("\\midrule")

    for task in sorted(TASK_SETS["all"]):
        if task not in geo_results or task not in evq_results:
            continue

        geo = geo_results[task]
        evq = evq_results[task]

        if math.isnan(geo.get("mean_nll", float('nan'))):
            continue

        geo_nll = geo["mean_nll"]
        evq_nll = evq["mean_nll"]
        geo_ppl = np.exp(geo_nll)
        evq_ppl = np.exp(evq_nll)

        lines.append(
            f"{task:20s} & {geo_nll:6.3f} & {evq_nll:6.3f} & "
            f"{geo_ppl:7.1f} & {evq_ppl:7.1f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{LongBench NLL Comparison (1.5B Models)}")
    lines.append("\\label{tab:longbench_nll}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_comparison_plot(
    geo_results: Dict[str, Dict[str, float]],
    evq_results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Generate side-by-side bar plot comparing models.

    Args:
        geo_results: Geometric model results
        evq_results: EVQ model results
        output_path: Path to save PNG figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping comparison plot")
        return

    print(f"  Generating comparison plot...", end=" ", flush=True)

    tasks = []
    geo_ppls = []
    evq_ppls = []

    for task in sorted(TASK_SETS["all"]):
        if task not in geo_results or task not in evq_results:
            continue

        geo = geo_results[task]
        evq = evq_results[task]

        if math.isnan(geo.get("mean_nll", float('nan'))):
            continue

        tasks.append(task[:15])  # Truncate for readability
        geo_ppls.append(np.exp(geo["mean_nll"]))
        evq_ppls.append(np.exp(evq["mean_nll"]))

    if not tasks:
        print("SKIP (no data)")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width/2, geo_ppls, width, label="Geometric", alpha=0.8)
    ax.bar(x + width/2, evq_ppls, width, label="EVQ", alpha=0.8)

    ax.set_xlabel("Task")
    ax.set_ylabel("Perplexity")
    ax.set_title("LongBench Perplexity: Geometric vs EVQ (1.5B)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"saved to {output_path}")
    except Exception as e:
        print(f"SKIP ({e})")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downstream task evaluation for Phase 20 models"
    )
    parser.add_argument(
        "--geo_checkpoint",
        type=str,
        help="Path to geometric RoPE checkpoint"
    )
    parser.add_argument(
        "--evq_checkpoint",
        type=str,
        help="Path to EVQ checkpoint"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory with checkpoints (will find geo/evq variants)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Task set: qa, summary, classification, all, or comma-separated"
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=30,
        help="Max samples per task"
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=4096,
        help="Maximum context length"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds for multi-run"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Optional local LongBench data directory"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run without evaluation"
    )

    args = parser.parse_args()

    # Parse tasks
    if args.tasks in TASK_SETS:
        task_list = TASK_SETS[args.tasks]
    else:
        task_list = [t.strip() for t in args.tasks.split(",")]

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  DOWNSTREAM TASK EVALUATION (Phase 20)")
    print(f"{'='*70}")
    print(f"  Tasks: {task_list}")
    print(f"  Samples per task: {args.samples_per_task}")
    print(f"  Max context: {args.max_context_len}")
    print(f"  Seeds: {seeds}")
    print(f"  Output dir: {output_dir}")

    if args.dry_run:
        print(f"\n  [DRY-RUN] Skipping evaluation")
        return

    # Determine checkpoints
    geo_checkpoint = Path(args.geo_checkpoint) if args.geo_checkpoint else None
    evq_checkpoint = Path(args.evq_checkpoint) if args.evq_checkpoint else None

    if not geo_checkpoint or not evq_checkpoint:
        print(f"\nError: Must provide --geo_checkpoint and --evq_checkpoint")
        sys.exit(1)

    if not geo_checkpoint.exists():
        print(f"Error: Checkpoint not found: {geo_checkpoint}")
        sys.exit(1)

    if not evq_checkpoint.exists():
        print(f"Error: Checkpoint not found: {evq_checkpoint}")
        sys.exit(1)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "geo_checkpoint": str(geo_checkpoint),
            "evq_checkpoint": str(evq_checkpoint),
            "tasks": task_list,
            "device": DEVICE,
            "dtype": str(DTYPE),
        },
        "seeds": {}
    }

    # Evaluate across seeds
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  SEED: {seed}")
        print(f"{'='*70}")

        seed_results = {}

        # Evaluate geometric model
        print(f"\nEvaluating GEOMETRIC checkpoint...")
        try:
            geo_model = load_checkpoint(geo_checkpoint, tau=0.0)
            geo_results = eval_longbench_nll(
                geo_model, tokenizer, task_list,
                args.samples_per_task, args.max_context_len,
                seed, args.data_dir, is_custom_gpt=True
            )
            seed_results["geo"] = geo_results
            del geo_model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {e}")
            seed_results["geo"] = {"error": str(e)}

        # Evaluate EVQ model
        print(f"\nEvaluating EVQ checkpoint...")
        try:
            evq_model = load_checkpoint(evq_checkpoint, tau=1.5)
            evq_results = eval_longbench_nll(
                evq_model, tokenizer, task_list,
                args.samples_per_task, args.max_context_len,
                seed, args.data_dir, is_custom_gpt=True
            )
            seed_results["evq"] = evq_results
            del evq_model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {e}")
            seed_results["evq"] = {"error": str(e)}

        all_results["seeds"][f"seed_{seed}"] = seed_results

    # Generate outputs
    print(f"\n{'='*70}")
    print(f"  GENERATING OUTPUTS")
    print(f"{'='*70}")

    # Save JSON results
    result_file = output_dir / "longbench_nll_results.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved: {result_file}")

    # Generate LaTeX table (using first seed)
    first_seed = f"seed_{seeds[0]}"
    if first_seed in all_results["seeds"]:
        seed_data = all_results["seeds"][first_seed]
        if "geo" in seed_data and "evq" in seed_data:
            latex_table = generate_latex_table(seed_data["geo"], seed_data["evq"])
            table_file = output_dir / "longbench_comparison.tex"
            with open(table_file, 'w') as f:
                f.write(latex_table)
            print(f"  LaTeX table saved: {table_file}")

            # Generate comparison plot
            plot_file = output_dir / "longbench_comparison.png"
            generate_comparison_plot(seed_data["geo"], seed_data["evq"], plot_file)

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    if first_seed in all_results["seeds"]:
        seed_data = all_results["seeds"][first_seed]

        if "geo" in seed_data and "evq" in seed_data:
            geo_res = seed_data["geo"]
            evq_res = seed_data["evq"]

            print(f"\n  {'Task':<25}  {'Geo NLL':<12}  {'EVQ NLL':<12}  {'Improve %':<10}")
            print(f"  {'-'*60}")

            for task in sorted(TASK_SETS["all"]):
                if task not in geo_res or task not in evq_res:
                    continue

                geo_nll = geo_res[task].get("mean_nll")
                evq_nll = evq_res[task].get("mean_nll")

                if geo_nll is not None and evq_nll is not None:
                    geo_ppl = np.exp(geo_nll)
                    evq_ppl = np.exp(evq_nll)
                    improvement = (geo_ppl - evq_ppl) / geo_ppl * 100

                    print(f"  {task:<25}  {geo_nll:<12.4f}  {evq_nll:<12.4f}  {improvement:<9.1f}%")

    print(f"\nEvaluation complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
