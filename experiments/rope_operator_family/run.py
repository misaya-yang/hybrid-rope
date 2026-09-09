"""One method, one configuration, one requested experiment per invocation.

Examples and the complete preparation/run sequence are in README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import CompactAttention, cache_bytes, install_factors
from .operator import OperatorFactors, Shape
from .prepare import prepare_text
from .study import (FitConfig, capture, diagnose, digest, fit_layer, generator_diagnostics, load_record,
                    load_rows, records_for, write_json)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def runtime_record(args):
    return {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "device": str(args.device), "dtype": getattr(args, "dtype", "float32")}


def load_model(path, device, dtype):
    from transformers import AutoModelForCausalLM
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this machine may still be in no-GPU mode. Code preparation and CPU tests do not require enabling it.")
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                torch_dtype=getattr(torch, dtype), attn_implementation="sdpa").to(device).eval()
    return model


def fingerprint(path):
    """Hash actual model inputs; called at capture/evaluation boundaries, not each row."""
    path = Path(path)
    files = sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"no model weight files at {path}")
    return {"config_sha256": digest(path / "config.json"), "weights": {p.name: digest(p) for p in files}}


def save_factors(path, factors, **extra):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({**factors.metadata(), "state": {k: v.detach().cpu() for k, v in factors.state_dict().items()}, **extra}, temporary)
    temporary.replace(path)


def check_tokenizer(data_manifest, model_path):
    for name, expected in data_manifest.get("tokenizer_files", {}).items():
        if digest(Path(model_path) / name) != expected:
            raise ValueError(f"prepared token IDs belong to a different tokenizer: {name}")


def read_factors(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    factors = OperatorFactors(Shape(**saved["shape"]), saved["phase_scale"])
    factors.load_state_dict(saved["state"])
    return factors.to(device)


def prepare_command(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    result = prepare_text(args.source, tokenizer, args.out, args.calibration_documents,
                          args.validation_documents, args.calibration_length, args.evaluation_length, args.seed)
    print(json.dumps(result, indent=2))


def capture_command(args):
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    check_tokenizer(manifest, args.model)
    if digest(data / "capture.jsonl") != manifest["files"]["capture.jsonl"]:
        raise ValueError("prepared capture input has changed")
    model = load_model(args.model, args.device, args.dtype)
    receipt = capture(model, load_rows(data / "capture.jsonl"), args.out, args.queries)
    receipt.update(base_model=fingerprint(args.model), data_manifest_sha256=digest(data / "manifest.json"), runtime=runtime_record(args))
    write_json(Path(args.out) / "manifest.json", receipt)


def fit_command(args):
    source, output = Path(args.capture), Path(args.out)
    manifest = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    config = FitConfig(steps=args.steps, learning_rate=args.learning_rate,
                       phase_learning_rate=args.phase_learning_rate, max_position_scale=args.max_position_scale,
                       learn_projections=not args.freeze_projections, learn_frequency=not args.freeze_frequency,
                       learn_content=not args.freeze_content, value_weight=args.value_weight,
                       output_weight=args.output_weight, seed=args.seed, score_weight=args.score_weight)
    if args.initialize_only:
        config.steps = 0
    specification = {"capture_sha256": digest(source / "manifest.json"), "shape": {**manifest["model_shape"],
                      "content_rank": args.content_rank, "rotary_dim": args.rotary_dim},
                      "fold": args.fold, "fit": asdict(config)}
    if args.initialize_only:
        specification["initialize_only"] = True
    initialization = None
    if args.init_from:
        if args.initialize_only:
            raise ValueError("--init-from is for fitting from a shared initialization")
        initialization = Path(args.init_from)
        initial_manifest = json.loads((initialization / "manifest.json").read_text())
        initial_spec = initial_manifest["specification"]
        if initial_manifest["status"] != "complete" or initial_manifest.get("checkpoint_role") != "unoptimized_initialization":
            raise ValueError("--init-from needs a completed unoptimized initialization")
        if any(initial_spec[key] != specification[key] for key in ("capture_sha256", "shape", "fold")):
            raise ValueError("shared initialization must match capture, shape and fold")
        specification["initialization_sha256"] = {f"layer_{layer:03d}.pt": digest(initialization / f"layer_{layer:03d}.pt")
                                                  for layer in range(manifest["layers"])}
    spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous["specification_sha256"] != spec_hash:
            raise ValueError("this output belongs to a different configuration; choose a new directory")
    receipt = dict(status="running", specification=specification, specification_sha256=spec_hash,
                   base_model=manifest.get("base_model"), layers=manifest["layers"], completed_layers=[], runtime=runtime_record(args),
                   checkpoint_role="unoptimized_initialization" if args.initialize_only else "fitted_operator")
    started = time.monotonic()
    for layer in range(manifest["layers"]):
        target = output / f"layer_{layer:03d}.pt"
        if target.exists():
            saved = torch.load(target, map_location="cpu", weights_only=True)
            if saved.get("specification_sha256") != spec_hash:
                raise ValueError(f"incompatible saved layer: {target}")
            receipt["completed_layers"].append(layer)
            continue
        paths = records_for(source, layer, "calibration")
        if not paths:
            raise ValueError("no calibration records")
        if initialization is not None:
            factors = read_factors(initialization / f"layer_{layer:03d}.pt", args.device)
        else:
            # One layer at a time; full-model activation stores are never loaded at once.
            keys, values = [], []
            for path in paths:
                record = load_record(path)
                keys.append(record["k"])
                values.append(record["v"])
            shape = Shape(**specification["shape"])
            factors = OperatorFactors.from_freqfold(torch.cat(keys).to(args.device), torch.cat(values).to(args.device), shape, args.fold)
            del keys, values
        try:
            if args.initialize_only:
                fit_receipt = {"configuration": asdict(config), "optimizer_updates": 0, "checkpoint_role": "unoptimized_initialization"}
            else:
                fit_receipt = fit_layer(factors, paths, config, output / f"layer_{layer:03d}")
        except Exception as error:
            receipt.update(status="failed", failed_layer=layer, error=f"{type(error).__name__}: {error}")
            write_json(output / "manifest.json", receipt)
            save_factors(output / f"failed_layer_{layer:03d}_{time.time_ns()}.pt", factors,
                         specification_sha256=spec_hash)
            raise
        save_factors(target, factors, specification_sha256=spec_hash)
        write_json(output / f"layer_{layer:03d}" / "result.json", fit_receipt)
        receipt["completed_layers"].append(layer)
        receipt["seconds_this_invocation"] = time.monotonic() - started
        write_json(output / "manifest.json", receipt)
    receipt["status"] = "complete"
    receipt["seconds_this_invocation"] = time.monotonic() - started
    write_json(output / "manifest.json", receipt)
    print(json.dumps(receipt, indent=2))


def diagnose_command(args):
    factors = read_factors(Path(args.factors) / f"layer_{args.layer:03d}.pt", args.device)
    prediction = diagnose(factors, records_for(args.capture, args.layer, "calibration"), args.position_scale)
    observed = diagnose(factors, records_for(args.capture, args.layer, "validation"), args.position_scale)
    write_json(args.out, {"layer": args.layer, "factor_sha256": digest(Path(args.factors) / f"layer_{args.layer:03d}.pt"),
                          "capture_sha256": digest(Path(args.capture) / "manifest.json"),
                          "generator_diagnostics": generator_diagnostics(factors),
                          "calibration_prediction": prediction, "held_out_observation": observed})
    print(json.dumps({"calibration": prediction["means"], "held_out": observed["means"]}, indent=2))


def _install_checked(model, factor_path, base_identity, backend):
    metadata = json.loads((Path(factor_path) / "manifest.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("operator fitting has not completed all layers")
    if metadata.get("base_model") is not None and metadata["base_model"] != base_identity:
        raise ValueError("factors were fitted for different base model weights")
    install_factors(model, factor_path, backend)
    return metadata


@torch.inference_mode()
def nll_rows(model, rows, device, target_tokens):
    if not rows or target_tokens < 1:
        raise ValueError("NLL evaluation needs rows and a positive target-token count")
    results = []
    for row in rows:
        tokens = row["input_ids"]
        ids = torch.tensor(tokens[:-1], device=device).unsqueeze(0)
        if ids.shape[1] > model.config.max_position_embeddings:
            raise ValueError("NLL context exceeds the base model position range")
        labels = torch.tensor(tokens[1:], device=device)
        if len(labels) == 0:
            raise ValueError("NLL input needs at least two tokens")
        count = min(target_tokens, len(labels))
        synchronize(device)
        started = time.monotonic()
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -count:]
        logits = model.lm_head(hidden).float()
        loss = F.cross_entropy(logits[0], labels[-count:])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite model NLL on document {row['id']}")
        synchronize(device)
        results.append(dict(id=row["id"], source_id=row.get("source_id", row["id"]),
                            input_tokens=ids.shape[1], prediction_tokens=count,
                            nll=float(loss), seconds=time.monotonic() - started))
    return results


def evaluate_command(args):
    data, output = Path(args.data), Path(args.out)
    metadata = json.loads((data / "manifest.json").read_text())
    check_tokenizer(metadata, args.model)
    if digest(data / "evaluate.jsonl") != metadata["files"]["evaluate.jsonl"]:
        raise ValueError("prepared evaluation input has changed")
    identity = fingerprint(args.model)
    model = load_model(args.model, args.device, args.dtype)
    factor_metadata = _install_checked(model, args.factors, identity, args.backend)
    rows = nll_rows(model, load_rows(data / "evaluate.jsonl"), args.device, args.target_tokens)
    write_json(output, dict(status="complete", method="operator_family", base_model=identity,
                            factors_sha256=digest(Path(args.factors) / "manifest.json"),
                            backend=args.backend, runtime=runtime_record(args), rows=rows,
                            checkpoint_role=factor_metadata.get("checkpoint_role", "fitted_operator"),
                            mean_document_nll=sum(r["nll"] for r in rows) / len(rows)))
    print(json.dumps({"documents": len(rows), "mean_document_nll": sum(r["nll"] for r in rows) / len(rows)}))


@torch.inference_mode()
def generate_command(args):
    from transformers import AutoTokenizer
    identity = fingerprint(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_model(args.model, args.device, args.dtype)
    factor_metadata = _install_checked(model, args.factors, identity, args.backend)
    text = Path(args.prompt).read_text()
    if args.chat:
        text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(args.device)
    if ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus answer budget exceeds the base model context")
    synchronize(args.device)
    started = time.monotonic()
    generated = model.generate(ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                               logits_to_keep=1,
                               pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    synchronize(args.device)
    answer_ids = generated[0, ids.shape[1]:].tolist()
    result = dict(status="complete", method="operator_family", input_tokens=ids.shape[1],
                  generated_ids=answer_ids, output=tokenizer.decode(answer_ids, skip_special_tokens=True),
                  seconds=time.monotonic() - started, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  input_ids_sha256=hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
                  generation_settings={"do_sample": False, "max_new_tokens": args.max_new_tokens, "backend": args.backend},
                  factors_sha256=digest(Path(args.factors) / "manifest.json"), base_model=identity, runtime=runtime_record(args))
    result["checkpoint_role"] = factor_metadata.get("checkpoint_role", "fitted_operator")
    if args.expected_answer is not None:
        answer = tokenizer(args.expected_answer, add_special_tokens=False)["input_ids"]
        if not answer or len(answer) > args.max_new_tokens:
            raise ValueError("expected answer must be nonempty and fit in the generation budget")
        scored = nll_rows(model, [{"id": "expected_answer", "input_ids": ids[0].tolist() + answer}],
                          args.device, len(answer))[0]
        result.update(expected_answer=args.expected_answer, expected_answer_ids=answer,
                      answer_exact=result["output"].strip() == args.expected_answer.strip(),
                      answer_nll=scored["nll"], answer_scoring_seconds=scored["seconds"])
    write_json(args.out, result)
    print(result["output"])


@torch.inference_mode()
def profile_command(args):
    identity = fingerprint(args.model)
    model = load_model(args.model, args.device, args.dtype)
    _install_checked(model, args.factors, identity, args.backend)
    ids = torch.randint(0, model.config.vocab_size, (1, args.length), device=args.device)
    if args.length + args.decode_tokens > model.config.max_position_embeddings:
        raise ValueError("profile workload exceeds model context")
    samples = []
    for repeat in range(args.repeats + 1):
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        synchronize(args.device)
        started = time.monotonic()
        state = model.model(input_ids=ids, use_cache=True)
        synchronize(args.device)
        prefill_seconds = time.monotonic() - started
        resident = cache_bytes(state.past_key_values)
        token = model.lm_head(state.last_hidden_state[:, -1:]).argmax(-1)
        synchronize(args.device)
        started = time.monotonic()
        for _ in range(args.decode_tokens):
            state = model.model(input_ids=token, past_key_values=state.past_key_values, use_cache=True)
            token = model.lm_head(state.last_hidden_state[:, -1:]).argmax(-1)
        synchronize(args.device)
        item = dict(prefill_seconds=prefill_seconds, decode_seconds=time.monotonic() - started,
                    cached_bytes_at_prefill=resident, cached_bytes_after_decode=cache_bytes(state.past_key_values),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") else None)
        if repeat:
            samples.append(item)
        del state
    write_json(args.out, dict(status="complete", input_tokens=args.length, decode_tokens=args.decode_tokens,
                             samples=samples, runtime=runtime_record(args), backend=[layer.self_attn.last_backend for layer in model.model.layers],
                             note="Synthetic tokens measure this implementation's cost, not model quality."))
    print(json.dumps(samples, indent=2))


def report_command(args):
    from .report import make_report
    make_report(args.factors, args.out, args.diagnostic, args.evaluation, args.generation, args.profile)


def compare_objectives_command(args):
    from .report import compare_objectives
    result = compare_objectives(args.method, args.control,
                                args.method_diagnostic, args.control_diagnostic,
                                args.method_generation, args.control_generation)
    write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    def command(name, fn, model=False, device=False):
        c = commands.add_parser(name)
        c.set_defaults(function=fn)
        if model:
            c.add_argument("--model", required=True)
        if device:
            c.add_argument("--device", default="cuda")
        return c
    def runtime(c):
        c.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    def fitted(c):
        c.add_argument("--factors", required=True)
        c.add_argument("--backend", choices=("auto", "sdpa", "chunked"), default="auto")
        c.add_argument("--out", required=True)
        runtime(c)
    c = command("prepare", prepare_command, model=True)
    c.add_argument("--source", required=True); c.add_argument("--out", required=True)
    c.add_argument("--calibration-documents", type=int, default=32)
    c.add_argument("--validation-documents", type=int, default=8)
    c.add_argument("--calibration-length", type=int, default=2048)
    c.add_argument("--evaluation-length", type=int, default=8192)
    c.add_argument("--seed", type=int, default=42)
    c = command("capture", capture_command, model=True, device=True); runtime(c)
    c.add_argument("--data", required=True); c.add_argument("--out", required=True)
    c.add_argument("--queries", type=int, default=64)
    c = command("fit", fit_command, device=True)
    c.add_argument("--capture", required=True); c.add_argument("--out", required=True)
    c.add_argument("--content-rank", type=int, default=192); c.add_argument("--rotary-dim", type=int, default=64)
    c.add_argument("--fold", type=int)
    c.add_argument("--initialize-only", action="store_true", help="Save the same unoptimized initializer as one optional simple control; no optimizer runs")
    c.add_argument("--init-from", help="Use the exact saved unoptimized factors shared by the method and output-distillation control")
    c.add_argument("--steps", type=int, default=500)
    c.add_argument("--learning-rate", type=float, default=1e-3)
    c.add_argument("--phase-learning-rate", type=float, default=0.01)
    c.add_argument("--max-position-scale", type=float, default=8.0)
    c.add_argument("--score-weight", type=float, default=1.0, help="Operator-score objective weight; set 0 with --output-weight 1 for matched output distillation")
    c.add_argument("--value-weight", type=float, default=1.0); c.add_argument("--output-weight", type=float, default=0.0)
    c.add_argument("--freeze-projections", action="store_true"); c.add_argument("--freeze-frequency", action="store_true")
    c.add_argument("--freeze-content", action="store_true"); c.add_argument("--seed", type=int, default=42)
    c = command("diagnose", diagnose_command, device=True)
    c.add_argument("--capture", required=True); c.add_argument("--factors", required=True); c.add_argument("--out", required=True)
    c.add_argument("--layer", type=int, required=True); c.add_argument("--position-scale", type=float, default=1.0)
    c = command("evaluate", evaluate_command, model=True, device=True); fitted(c)
    c.add_argument("--data", required=True); c.add_argument("--target-tokens", type=int, default=256)
    c = command("generate", generate_command, model=True, device=True); fitted(c)
    c.add_argument("--prompt", required=True); c.add_argument("--chat", action="store_true")
    c.add_argument("--expected-answer", help="Known answer for this one prompt: report exact match and conditional answer NLL")
    c.add_argument("--max-new-tokens", type=int, default=128)
    c = command("profile", profile_command, model=True, device=True); fitted(c)
    c.add_argument("--length", type=int, default=8192); c.add_argument("--decode-tokens", type=int, default=64)
    c.add_argument("--repeats", type=int, default=3)
    c = command("report", report_command)
    c.add_argument("--factors", required=True); c.add_argument("--out", required=True)
    c.add_argument("--diagnostic"); c.add_argument("--evaluation")
    c.add_argument("--generation"); c.add_argument("--profile")
    c = command("compare-objectives", compare_objectives_command)
    c.add_argument("--method", required=True); c.add_argument("--control", required=True)
    c.add_argument("--method-diagnostic"); c.add_argument("--control-diagnostic")
    c.add_argument("--method-generation"); c.add_argument("--control-generation")
    c.add_argument("--out", required=True)
    return p


def main():
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
