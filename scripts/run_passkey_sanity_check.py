#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _bootstrap_phase4_imports(project_root: Path) -> None:
    phase4_dir = project_root / "sigmoid_rope_experiments"
    if str(phase4_dir) not in sys.path:
        sys.path.insert(0, str(phase4_dir))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_bootstrap_phase4_imports(PROJECT_ROOT)

import run_phase4 as p4  # noqa: E402
from src.rope import RoPEFrequencyAllocator  # noqa: E402
from src.utils import cleanup_cuda, env_info, get_device, save_json, set_seed  # noqa: E402


DEFAULT_LOCAL_MODEL_CANDIDATES = (
    "/root/autodl-tmp/dfrope/ms_models/EleutherAI/gpt-neox-20b,"
    "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct,"
    "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    level: str
    context_length: int
    depth_name: str
    depth_ratio: Optional[float]
    near_gap_tokens: Optional[int]


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    display: str
    aliases: Tuple[str, ...]
    inv_freq: torch.Tensor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Passkey sanity check with teacher-forcing true-vs-false logprob."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--tokenizer_mode", type=str, default="auto", choices=["auto", "hf", "byte"])
    ap.add_argument("--tokenizer_path", type=str, default="")
    ap.add_argument("--local_model_candidates", type=str, default=DEFAULT_LOCAL_MODEL_CANDIDATES)
    ap.add_argument("--model_root", type=str, default="tmp_phase4_compare")
    ap.add_argument("--fallback_model_root", type=str, default="sigmoid_rope_experiments/checkpoints")
    ap.add_argument("--summary_json", type=str, default="sigmoid_rope_experiments/data/phase4_corrected_summary.json")
    ap.add_argument("--max_seq_len_train", type=int, default=8192)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--rope_base", type=float, default=10000.0)
    ap.add_argument("--output_dir", type=str, default="results/phase4_passkey_sanity")
    ap.add_argument(
        "--allow_missing_models",
        action="store_true",
        help="Skip missing model checkpoints instead of raising an error.",
    )
    return ap.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def repeat_to_length(tokens: Sequence[int], target_len: int) -> List[int]:
    if target_len <= 0:
        return []
    base = list(tokens)
    if not base:
        return [0] * target_len
    n_rep = target_len // len(base) + 1
    return (base * n_rep)[:target_len]


def build_prompt_ids(
    tokenizer: p4.TokenizerBase,
    context_length: int,
    passkey: str,
    depth_ratio: Optional[float],
    near_gap_tokens: Optional[int],
) -> List[int]:
    filler = "The quick brown fox jumps over the lazy dog. "
    needle = f"The special magic number is {passkey}. "
    prefix = "Remember, earlier in this document, it was stated that the special magic number is "

    filler_ids = tokenizer.encode(filler)
    needle_ids = tokenizer.encode(needle)
    prefix_ids = tokenizer.encode(prefix)

    if len(prefix_ids) + len(needle_ids) >= context_length:
        raise RuntimeError(
            f"context_length={context_length} too small; need > prefix({len(prefix_ids)})+needle({len(needle_ids)})"
        )

    doc_budget = context_length - len(prefix_ids)
    if near_gap_tokens is not None:
        gap = int(max(0, near_gap_tokens))
        before_len = doc_budget - len(needle_ids) - gap
        if before_len < 0:
            raise RuntimeError(
                f"Near-copy setting invalid for context_length={context_length}, gap={gap}, needle={len(needle_ids)}"
            )
        before_ids = repeat_to_length(filler_ids, before_len)
        gap_ids = repeat_to_length(filler_ids, gap)
        doc_ids = before_ids + needle_ids + gap_ids
    else:
        filler_budget = doc_budget - len(needle_ids)
        if filler_budget < 0:
            raise RuntimeError(
                f"Not enough budget for needle: context_length={context_length}, doc_budget={doc_budget}"
            )
        base_ids = repeat_to_length(filler_ids, filler_budget)
        ratio = float(depth_ratio if depth_ratio is not None else 0.5)
        ratio = min(max(ratio, 0.0), 1.0)
        pos = int(round(filler_budget * ratio))
        pos = min(max(pos, 0), filler_budget)
        doc_ids = base_ids[:pos] + needle_ids + base_ids[pos:]

    prompt_ids = doc_ids + prefix_ids
    if len(prompt_ids) != context_length:
        prompt_ids = prompt_ids[:context_length]
        if len(prompt_ids) < context_length:
            prompt_ids += repeat_to_length(filler_ids, context_length - len(prompt_ids))

    if tokenizer.bos_token_id is not None:
        prompt_ids = [int(tokenizer.bos_token_id)] + prompt_ids
    return prompt_ids


def sample_false_passkey(
    rng: random.Random,
    tokenizer: p4.TokenizerBase,
    true_passkey: str,
    true_token_len: int,
    max_trials: int = 256,
) -> str:
    fallback = true_passkey
    for _ in range(max_trials):
        cand = f"{rng.randint(10000, 99999)}"
        if cand == true_passkey:
            continue
        fallback = cand
        if len(tokenizer.encode(cand)) == true_token_len:
            return cand
    if fallback == true_passkey:
        fallback = f"{(int(true_passkey) + 1) % 100000:05d}"
    return fallback


def key_only_ce_loss(
    model: p4.GPTSmall,
    prompt_ids: Sequence[int],
    key_ids: Sequence[int],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    if len(key_ids) <= 0:
        return float("nan")
    x_ids = list(prompt_ids) + list(key_ids)
    x = torch.tensor([x_ids], dtype=torch.long, device=device)
    key_len = len(key_ids)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
    pred = logits[:, -key_len - 1 : -1, :].contiguous()
    labels = x[:, -key_len:].contiguous()
    loss = F.cross_entropy(pred.view(-1, pred.size(-1)), labels.view(-1), reduction="mean")
    return float(loss.detach().item())


def build_case_specs() -> List[CaseSpec]:
    cases: List[CaseSpec] = []
    cases.append(
        CaseSpec(
            case_id="L1_256_near20",
            level="L1",
            context_length=256,
            depth_name="near20",
            depth_ratio=None,
            near_gap_tokens=20,
        )
    )
    for ctx in [4096, 8192]:
        cases.append(CaseSpec(f"L2_{ctx}_hard", "L2", ctx, "hard", 0.10, None))
        cases.append(CaseSpec(f"L2_{ctx}_medium", "L2", ctx, "medium", 0.50, None))
        cases.append(CaseSpec(f"L2_{ctx}_easy", "L2", ctx, "easy", 0.90, None))
    cases.append(CaseSpec("L3_12000_hard", "L3", 12000, "hard", 0.10, None))
    cases.append(CaseSpec("L3_12000_easy", "L3", 12000, "easy", 0.90, None))
    return cases


def load_rope_params(
    summary_path: Path,
    head_dim: int,
    base: float,
    max_seq_len_train: int,
) -> Dict[str, float]:
    n_pairs = head_dim // 2
    params = {
        "k": 16.05 / float(head_dim),
        "x0": 0.47 * float(n_pairs),
        "anchor20": 20.0,
    }
    inv_std = RoPEFrequencyAllocator(d=head_dim, base=base).standard()
    theta_min_std = float(inv_std[-1].item())
    theta_target = float(2.0 * math.pi / float(max_seq_len_train))
    params["anchor_alpha"] = max(1.0, theta_target / max(theta_min_std, 1e-12))

    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8", errors="ignore"))
            rope_params = payload.get("rope_params", {})
            sig = rope_params.get("sigmoid", {})
            anc20 = rope_params.get("anchored20", {})
            anc_star = rope_params.get("anchored_alpha", {})
            params["k"] = float(sig.get("k", params["k"]))
            params["x0"] = float(sig.get("x0", params["x0"]))
            params["anchor20"] = float(anc20.get("alpha", params["anchor20"]))
            params["anchor_alpha"] = float(anc_star.get("alpha", params["anchor_alpha"]))
        except Exception:
            pass
    return params


def build_model_specs(
    head_dim: int,
    base: float,
    max_seq_len_train: int,
    summary_path: Path,
) -> List[ModelSpec]:
    rope_params = load_rope_params(summary_path, head_dim, base, max_seq_len_train)
    allocator = RoPEFrequencyAllocator(d=head_dim, base=base)
    inv_std = allocator.standard()
    inv_sig = allocator.sigmoid(k=rope_params["k"], x0=rope_params["x0"])
    inv_a20 = allocator.anchored_sigmoid(k=rope_params["k"], j0=rope_params["x0"], anchor_factor=rope_params["anchor20"])
    inv_astar = allocator.anchored_sigmoid(
        k=rope_params["k"],
        j0=rope_params["x0"],
        anchor_factor=rope_params["anchor_alpha"],
    )

    return [
        ModelSpec("sigmoid", "Sigmoid", ("sigmoid",), inv_sig),
        ModelSpec("standard", "Standard", ("standard",), inv_std),
        ModelSpec("anchored_alpha", "Anchored-alpha", ("anchored_alpha", "anchored-alpha", "alpha"), inv_astar),
        ModelSpec("anchored20", "Anchored-20", ("anchored20", "anchored_20", "anchored-20"), inv_a20),
    ]


def build_search_roots(project_root: Path, model_root: Path, fallback_root: Path) -> List[Path]:
    roots: List[Path] = []
    for p in [model_root, fallback_root, project_root / "sigmoid_rope_experiments" / "checkpoints"]:
        if not p.exists():
            continue
        if p not in roots:
            roots.append(p)
        ck = p / "checkpoints"
        if ck.exists() and ck not in roots:
            roots.append(ck)
    return roots


def find_checkpoint_for_model(spec: ModelSpec, roots: Sequence[Path]) -> Optional[Path]:
    rel_priority = {
        "standard": ["standard_best/checkpoint.pt", "standard_final/model.pt"],
        "sigmoid": ["sigmoid_best/checkpoint.pt", "sigmoid_final/model.pt"],
        "anchored20": ["anchored20_best/checkpoint.pt", "anchored20_final/model.pt"],
        "anchored_alpha": ["anchored_alpha_best/checkpoint.pt", "anchored_alpha_final/model.pt"],
    }
    for root in roots:
        for rel in rel_priority.get(spec.tag, []):
            p = root / rel
            if p.exists():
                return p

    for root in roots:
        for fname in ["checkpoint.pt", "model.pt"]:
            for p in root.rglob(fname):
                low = str(p).lower()
                if any(alias in low for alias in spec.aliases):
                    return p
    return None


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint payload type at {ckpt_path}")


def instantiate_model(
    state: Dict[str, torch.Tensor],
    spec: ModelSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> p4.GPTSmall:
    if "tok_emb.weight" not in state:
        raise RuntimeError("State dict missing tok_emb.weight; cannot infer vocab size.")
    vocab_size = int(state["tok_emb.weight"].shape[0])
    model = p4.GPTSmall(
        vocab_size=vocab_size,
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        inv_freq=spec.inv_freq,
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_model(
    model: p4.GPTSmall,
    model_name: str,
    tokenizer: p4.TokenizerBase,
    cases: Sequence[CaseSpec],
    repeats: int,
    rng: random.Random,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pbar = tqdm(total=len(cases) * repeats, desc=f"sanity-{model_name}", dynamic_ncols=True)
    for case in cases:
        for rep in range(repeats):
            passkey = f"{rng.randint(10000, 99999)}"
            prompt_ids = build_prompt_ids(
                tokenizer=tokenizer,
                context_length=case.context_length,
                passkey=passkey,
                depth_ratio=case.depth_ratio,
                near_gap_tokens=case.near_gap_tokens,
            )
            true_ids = tokenizer.encode(passkey)
            false_key = sample_false_passkey(rng, tokenizer, passkey, len(true_ids))
            false_ids = tokenizer.encode(false_key)

            true_loss = key_only_ce_loss(model, prompt_ids, true_ids, device, use_amp, amp_dtype)
            false_loss = key_only_ce_loss(model, prompt_ids, false_ids, device, use_amp, amp_dtype)
            hit = int(true_loss < false_loss)

            rows.append(
                {
                    "model": model_name,
                    "case_id": case.case_id,
                    "level": case.level,
                    "context_length": case.context_length,
                    "depth_name": case.depth_name,
                    "depth_ratio": case.depth_ratio,
                    "near_gap_tokens": case.near_gap_tokens,
                    "repeat": rep,
                    "passkey_true": passkey,
                    "passkey_false": false_key,
                    "true_key_token_len": len(true_ids),
                    "false_key_token_len": len(false_ids),
                    "true_loss": true_loss,
                    "false_loss": false_loss,
                    "margin_false_minus_true": (false_loss - true_loss),
                    "hit": hit,
                }
            )
            pbar.update(1)
    pbar.close()
    return rows


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "case_id",
                "level",
                "context_length",
                "depth_name",
                "hit_rate",
                "mean_true_loss",
                "mean_false_loss",
                "mean_margin",
                "n",
            ]
        )
    agg = (
        df.groupby(["model", "case_id", "level", "context_length", "depth_name"], as_index=False)
        .agg(
            hit_rate=("hit", "mean"),
            mean_true_loss=("true_loss", "mean"),
            mean_false_loss=("false_loss", "mean"),
            mean_margin=("margin_false_minus_true", "mean"),
            n=("hit", "count"),
        )
        .sort_values(["model", "case_id"])
    )
    return agg


def build_markdown_table(agg: pd.DataFrame) -> str:
    col_defs = [
        ("L1_256_near20", "L1"),
        ("L2_4096_hard", "L2-4K-H"),
        ("L2_4096_medium", "L2-4K-M"),
        ("L2_4096_easy", "L2-4K-E"),
        ("L2_8192_hard", "L2-8K-H"),
        ("L2_8192_medium", "L2-8K-M"),
        ("L2_8192_easy", "L2-8K-E"),
        ("L3_12000_hard", "L3-12K-H"),
        ("L3_12000_easy", "L3-12K-E"),
    ]
    model_order = ["Sigmoid", "Standard", "Anchored-alpha", "Anchored-20"]

    header = "| Model | " + " | ".join([name for _, name in col_defs]) + " |"
    sep = "|---" * (len(col_defs) + 1) + "|"
    lines = [header, sep]

    for model in model_order:
        row = [model]
        sub = agg[agg["model"] == model]
        for case_id, _ in col_defs:
            hit = sub[sub["case_id"] == case_id]["hit_rate"]
            if hit.empty:
                row.append("NA")
            else:
                row.append(f"{100.0 * float(hit.iloc[0]):.1f}%")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    project_root = PROJECT_ROOT
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_root = resolve_path(project_root, args.model_root)
    fallback_root = resolve_path(project_root, args.fallback_model_root)
    summary_path = resolve_path(project_root, args.summary_json)

    device = get_device(prefer_cuda=not args.cpu)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer, tok_name = p4.resolve_tokenizer(
        mode=args.tokenizer_mode,
        tokenizer_path=args.tokenizer_path,
        local_model_candidates=args.local_model_candidates,
    )

    model_specs = build_model_specs(
        head_dim=int(args.head_dim),
        base=float(args.rope_base),
        max_seq_len_train=int(args.max_seq_len_train),
        summary_path=summary_path,
    )
    search_roots = build_search_roots(project_root, model_root, fallback_root)
    if not search_roots:
        raise RuntimeError(
            "No valid model roots found. Expected checkpoints under tmp_phase4_compare or sigmoid_rope_experiments/checkpoints."
        )

    found_ckpts: Dict[str, str] = {}
    available_specs: List[ModelSpec] = []
    missing: List[str] = []
    for spec in model_specs:
        ckpt = find_checkpoint_for_model(spec, search_roots)
        if ckpt is None:
            missing.append(spec.display)
        else:
            found_ckpts[spec.tag] = str(ckpt)
            available_specs.append(spec)
    if missing and not args.allow_missing_models:
        raise RuntimeError(
            "Missing checkpoints for models: "
            + ", ".join(missing)
            + ". Search roots: "
            + ", ".join([str(p) for p in search_roots])
            + ". Re-run with --allow_missing_models to skip them."
        )
    if missing and args.allow_missing_models:
        print(f"[passkey-sanity] warning: skipping missing checkpoints: {', '.join(missing)}")
    if not available_specs:
        raise RuntimeError("No available model checkpoints found after filtering.")

    cases = build_case_specs()
    rng = random.Random(int(args.seed) + 1337)
    rows: List[Dict[str, object]] = []
    started = time.time()

    print("[passkey-sanity] env:", env_info())
    print(f"[passkey-sanity] device={device}, amp_dtype={amp_dtype}, tokenizer={tok_name}, vocab_size={tokenizer.vocab_size}")
    print(f"[passkey-sanity] search_roots={', '.join(str(x) for x in search_roots)}")
    for spec in available_specs:
        print(f"[passkey-sanity] {spec.display} checkpoint -> {found_ckpts[spec.tag]}")

    for spec in available_specs:
        ckpt_path = Path(found_ckpts[spec.tag])
        state = load_state_dict(ckpt_path)
        model = instantiate_model(state, spec, args, device)
        model_rows = evaluate_model(
            model=model,
            model_name=spec.display,
            tokenizer=tokenizer,
            cases=cases,
            repeats=int(args.repeats),
            rng=rng,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        rows.extend(model_rows)
        del model
        cleanup_cuda()

    df = pd.DataFrame(rows)
    agg = aggregate_results(df)
    markdown = build_markdown_table(agg)

    csv_path = output_dir / "results.csv"
    agg_path = output_dir / "aggregated.csv"
    md_path = output_dir / "summary.md"
    json_path = output_dir / "results.json"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    agg.to_csv(agg_path, index=False, encoding="utf-8")
    md_path.write_text(markdown + "\n", encoding="utf-8")

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": float(time.time() - started),
        "config": {
            "seed": int(args.seed),
            "repeats": int(args.repeats),
            "tokenizer_mode": args.tokenizer_mode,
            "tokenizer_path": args.tokenizer_path,
            "local_model_candidates": args.local_model_candidates,
            "model_root": str(model_root),
            "fallback_model_root": str(fallback_root),
            "summary_json": str(summary_path),
            "max_seq_len_train": int(args.max_seq_len_train),
            "n_layers": int(args.n_layers),
            "n_heads": int(args.n_heads),
            "d_model": int(args.d_model),
            "d_ff": int(args.d_ff),
            "head_dim": int(args.head_dim),
            "rope_base": float(args.rope_base),
        },
        "env": env_info(),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "tokenizer": {
            "name": tok_name,
            "vocab_size": int(tokenizer.vocab_size),
        },
        "model_checkpoints": found_ckpts,
        "cases": [asdict(c) for c in cases],
        "summary_markdown": markdown,
        "aggregated": agg.to_dict(orient="records"),
        "raw": df.to_dict(orient="records"),
        "artifacts": {
            "results_csv": str(csv_path),
            "aggregated_csv": str(agg_path),
            "summary_md": str(md_path),
            "results_json": str(json_path),
        },
    }
    save_json(json_path, payload)

    print("\n[passkey-sanity] Markdown summary:")
    print(markdown)
    print(f"\n[passkey-sanity] saved -> {json_path}")


if __name__ == "__main__":
    main()
