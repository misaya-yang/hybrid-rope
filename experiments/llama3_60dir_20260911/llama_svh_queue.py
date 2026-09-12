"""Llama-only continuation queue: P_L0 -> S controls/candidates.

This controller is deliberately separate from ``llama_planb_queue.py``.  It
does not touch the bootstrap queue or the runner, and it never dispatches an
OLMo/Qwen command.  One unit means one runner arm, so a failed arm can be
retried once without rerunning a completed arm.  The runner owns per-row
resume; this controller owns unit ordering and durable queue state.

The queue includes every candidate marked RUN in the corrected 16-arm ledger
and the matched controls that have a frozen task-level construction.  D03's
gauge is a required algebraic parity artifact.  D07's activation-calibrated
matched-global control remains deferred to M-dev; this limits interpretation
but does not suppress the preregistered S screening arm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


DEFAULT_ROOT = "/root/autodl-tmp/llama3_planb_20260911"
DEFAULT_CODE = "/root/autodl-tmp/llama3_60dir_20260911"
DEFAULT_MODEL = "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
DEFAULT_SCORER = None  # require the deployment to name its frozen evaluator
STAGE = "S"
LENGTHS = "8192,16384,32768"

# The matched-control inventory is intentional.  A missing required control is
# a scope decision, not permission to run a weaker comparison.
CONTROL_ORDER = (
    "Native", "MR", "OfficialYaRN", "BM", "UNI", "ResonanceYaRN",
    "DROP_D05c", "SIGN_D04b", "AREA_D06b",
    "D13a_STATIC_ENDPOINT", "D13b_STATIC_ENDPOINT", "D13c_STATIC_ENDPOINT",
)
CONTROL_UNIT_IDS = {
    "Native": "S_NATIVE",
    "MR": "S_MR",
    "OfficialYaRN": "S_YARN",
    "BM": "S_BM",
    "UNI": "S_UNI",
    "ResonanceYaRN": "S_RESONANCE_YARN",
    "DROP_D05c": "S_DROP_D05C",
    "SIGN_D04b": "S_SIGN_D04B",
    "AREA_D06b": "S_AREA_D06B",
    "D13a_STATIC_ENDPOINT": "S_D13A_STATIC_ENDPOINT",
    "D13b_STATIC_ENDPOINT": "S_D13B_STATIC_ENDPOINT",
    "D13c_STATIC_ENDPOINT": "S_D13C_STATIC_ENDPOINT",
}
CONTROLLED_MODULE_ORDER = (
    "L1_MR_A087", "L1_BM_A087", "L1_MR_A113", "L1_BM_A113",
    "L1_MR_A130", "L1_BM_A130",
    "L2_MR_BF32_BS05", "L2_BM_BF32_BS05",
    "L2_MR_BF64_BS1", "L2_BM_BF64_BS1",
    "L2_MR_BF64_BS05", "L2_BM_BF64_BS05",
)


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    """Write a small state/plan document atomically and durably."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_candidates(code):
    """Return the corrected Llama arm set in original group/policy order."""
    code = Path(code)
    arms_path = code / "arms_llama3.csv"
    correction_path = code / "candidates_corrected.csv"
    if not arms_path.exists() or not correction_path.exists():
        raise RuntimeError("candidate manifests are missing; refusing to invent an S set")

    with arms_path.open(encoding="utf-8", newline="") as stream:
        arms = list(csv.DictReader(stream))
    with correction_path.open(encoding="utf-8", newline="") as stream:
        correction = {row["config"]: row for row in csv.DictReader(stream)}

    candidates = [row for row in arms if row.get("role") == "candidate"]
    rank = {"a": 0, "b": 1, "c": 2}
    # The original review plan asks for direction fairness: run every available
    # ``a`` before the ``b`` arms, then every ``c``.  This prevents one family
    # consuming its full three-arm budget before another family is sampled.
    candidates.sort(key=lambda row: (rank.get(row["config"][-1], 99),
                                     int(row["config"][1:3])))
    for row in candidates:
        cfg = row["config"]
        if cfg not in correction:
            raise RuntimeError(f"{cfg}: absent from corrected candidate ledger")
        row["direction"] = correction[cfg]["direction"]
        row["decision"] = correction[cfg]["decision"]
        row["rationale"] = correction[cfg].get("rationale", "")
    return candidates


def matched_registry(code):
    """Load the Plan B parent registry; absence is a hard planning error."""
    code = Path(code)
    sys.path.insert(0, str(code))
    try:
        import planb_matched_controls as registry
    except ImportError as exc:
        raise RuntimeError("Plan B matched-control registry is missing") from exc
    return registry.REGISTRY


def make_unit(unit_id, kind, arm, requires, parent, statistic, decision,
              on_fail, next_unit, status="PENDING", reason=None,
              required_scope="control", reusable_artifacts=()):
    return {
        "unit_id": unit_id,
        "kind": kind,
        "arm": arm,
        "requires": list(requires),
        "parent": parent,
        "statistic": statistic,
        "decision": decision,
        "on_fail": on_fail,
        "next": next_unit,
        "status": status,
        "reason": reason,
        "required_scope": required_scope,
        "reusable_artifacts": list(reusable_artifacts),
    }


def build_units(code):
    """Build a deterministic, explicit S plan from the corrected arm ledger."""
    registry = matched_registry(code)
    units = []
    previous = "P_L0"
    for arm in CONTROL_ORDER:
        unit_id = CONTROL_UNIT_IDS[arm]
        next_control = (CONTROL_UNIT_IDS[CONTROL_ORDER[CONTROL_ORDER.index(arm) + 1]]
                        if arm != CONTROL_ORDER[-1] else "S_L0_CONTROLS")
        units.append(make_unit(
            unit_id, "control", arm, ["P_L0"], "P_L0",
            f"S control {arm}: task x length macro, 8K native guard",
            "required control; no candidate selection from this unit",
            "retry once with the same panel/scorer/operator; then BLOCKED_ENGINEERING",
            next_control))
        previous = unit_id

    modules = []
    for arm in CONTROLLED_MODULE_ORDER:
        family = arm.split("_", 1)[0]
        modules.append(make_unit(
            f"S_{arm}", "controlled_module", arm,
            ["S_L0_CONTROLS"], "S_L0_CONTROLS",
            ("L1 profile x compression-amplitude interaction" if family == "L1" else
             "L2 beta-fast x beta-slow boundary-policy interaction"),
            "development module; compare only after its MR/BM pair is complete",
            "retry once with the same contract; then BLOCKED_ENGINEERING",
            None, required_scope="frequency",
            reusable_artifacts=("S_MR/S_BM supply the omitted a=1 or base-band cells",)))

    for row in load_candidates(code):
        cfg = row["config"]
        direction = row["direction"]
        decision = row["decision"]
        # The join is the explicit S-control gate.  Keep the individual
        # control dependencies too, so a state reader can see exactly which
        # arm is missing instead of only seeing a failed aggregate gate.
        required = ["S_L0_CONTROLS"] + [CONTROL_UNIT_IDS[x] for x in CONTROL_ORDER]
        spec = registry.get(cfg)
        if spec is None:
            raise RuntimeError(f"{cfg}: absent from Plan B matched-control registry")
        required_controls = tuple(dict.fromkeys(
            spec["parent_ids"] + spec["required_controls"]))
        cfg_control_ids = []
        if cfg.startswith("D01"):
            cfg_control_ids.append(CONTROL_UNIT_IDS["ResonanceYaRN"])
        if cfg.startswith("D05"):
            cfg_control_ids.extend([
                CONTROL_UNIT_IDS["DROP_D05c"], CONTROL_UNIT_IDS["SIGN_D04b"]])
        if cfg.startswith("D06"):
            cfg_control_ids.append(CONTROL_UNIT_IDS["AREA_D06b"])
        if cfg.startswith("D13"):
            cfg_control_ids.append(CONTROL_UNIT_IDS[f"{cfg}_STATIC_ENDPOINT"])
        if decision != "RUN":
            status = "HOLD_PLAN" if decision == "HOLD" else "DROP_PLAN"
            reason = f"corrected ledger decision={decision}; not scheduled"
        else:
            status = "PENDING"
            reason = ("D07 matched-global is deferred until frozen unlabeled M-dev "
                      "activations; S score is screening-only" if cfg.startswith("D07")
                      else None)
        unit_id = f"S_{cfg}"
        required = list(dict.fromkeys(required + cfg_control_ids))
        units.append(make_unit(
            unit_id, "candidate", cfg, required, "S_L0_CONTROLS",
            f"S candidate {cfg} vs matched controls {', '.join(required_controls)}; "
            "equal-task/length macro plus 8K guard",
            ("run only after all S controls and matched controls are complete"
             if status == "PENDING" else reason),
            "retry once with the same contract; then BLOCKED_ENGINEERING",
            None, status=status, reason=reason))
        units[-1]["parent_controls"] = list(required_controls)
        units[-1]["registry_construction_status"] = spec["construction_status"]
        units[-1]["interpretation_limit"] = reason
        units[-1]["required_scope"] = row["scope"]
        units[-1]["reusable_artifacts"] = [
            "S_L0 control outputs", *cfg_control_ids,
            *( ["D03 gauge parity; full MR/BM permutation behavior remains M-H"]
               if cfg.startswith("D03") else []),
            *( ["D07 matched-global remains blocked until frozen M-dev activations"]
               if cfg.startswith("D07") else []),
        ]

    # Add a non-GPU join unit to make the dependency boundary explicit.  It is
    # a state transition only, never a placeholder workload.
    units.append(make_unit(
        "S_L0_CONTROLS", "join", None,
        list(CONTROL_UNIT_IDS.values()), "P_L0",
        "all S controls complete before candidate interpretation",
        "unlock only when every S control has a COMPLETE run_summary",
        "record BLOCKED_CONTROL_FAILURE; do not run candidates",
        f"S_{CONTROLLED_MODULE_ORDER[0]}" if CONTROLLED_MODULE_ORDER else
        next((u["unit_id"] for u in units if u["kind"] == "candidate"), None)))

    # Put the join immediately before candidates while retaining candidate
    # group order.  The explicit list is what is persisted and audited.
    controls = [u for u in units if u["kind"] == "control"]
    candidates = [u for u in units if u["kind"] == "candidate"]
    join = [u for u in units if u["kind"] == "join"]
    return controls + join + modules + candidates


def validate_units(units):
    ids = [u["unit_id"] for u in units]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate unit_id")
    known = set(ids) | {"P_L0"}
    for unit in units:
        required_fields = {"unit_id", "requires", "parent", "statistic",
                           "decision", "on_fail", "next", "required_scope",
                           "reusable_artifacts"}
        if not required_fields <= unit.keys():
            raise ValueError(f"{unit['unit_id']}: incomplete unit contract")
        unknown = set(unit["requires"]) - known
        if unknown:
            raise ValueError(f"{unit['unit_id']}: unknown dependency {sorted(unknown)}")
        if unit["next"] is not None and unit["next"] not in known:
            raise ValueError(f"{unit['unit_id']}: unknown next unit {unit['next']}")
    return True


def model_identity(model):
    model = Path(model)
    forbidden = ("olmo", "qwen")
    if any(token in str(model).lower() for token in forbidden):
        raise RuntimeError("REFUSING: queue is Llama-only; model path names OLMo/Qwen")
    cfg_path = model / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"model config missing: {cfg_path}")
    cfg = read_json(cfg_path)
    got = (cfg.get("model_type"), cfg.get("hidden_size"),
           cfg.get("num_hidden_layers"), cfg.get("num_attention_heads"),
           cfg.get("num_key_value_heads"), cfg.get("rope_theta"),
           cfg.get("max_position_embeddings"), cfg.get("rope_scaling"))
    expected = ("llama", 4096, 32, 32, 8, 500000.0, 8192, None)
    if got != expected:
        raise RuntimeError(f"REFUSING: checkpoint identity {got!r} != {expected!r}")
    tokenizer = model / "tokenizer.json"
    weight_index = model / "model.safetensors.index.json"
    return {"config_sha256": file_sha(cfg_path),
            "tokenizer_sha256": file_sha(tokenizer),
            "weight_index_sha256": file_sha(weight_index),
            "identity": got}


def manifest_complete(path, stage=None):
    try:
        manifest = read_json(Path(path) / "manifest.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (manifest.get("status") == "COMPLETE"
            and (stage is None or manifest.get("stage") == stage))


def unit_complete(root, unit, artifact_identity=None):
    if unit["kind"] not in {"control", "controlled_module", "candidate"}:
        return False
    output = Path(root) / "results" / STAGE / unit["arm"]
    try:
        summary = read_json(output / "run_summary.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if summary.get("status") != "COMPLETE":
        return False
    if artifact_identity is not None:
        expected = {
            "panel_sha256": artifact_identity["panel_sha256"],
            "scorer_sha256": artifact_identity["scorer_sha256"],
            "scoring_contract_sha256": artifact_identity["scoring_contract_sha256"],
            "model_config_sha256": artifact_identity["model_config_sha256"],
            "tokenizer_sha256": artifact_identity["tokenizer_sha256"],
            "weight_index_sha256": artifact_identity["weight_index_sha256"],
            "checkpoint_manifest_sha256": artifact_identity["checkpoint_manifest_sha256"],
            "runner_sha256": artifact_identity["runner_sha256"],
            "operators_sha256": artifact_identity["operators_sha256"],
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            return False
    return (output / f"{unit['arm']}.jsonl").is_file()


def acquire_lock(root):
    lock = Path(root) / "sv_queue.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            live = live_s_runners(root)
            if live:
                raise RuntimeError(
                    f"stale controller lock but live S runner(s) remain: {live}")
            lock.unlink(missing_ok=True)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f"queue already running under pid {pid}")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def live_s_runners(root):
    """Find orphan-capable runner processes belonging to this exact S root."""
    marker = f"{Path(root) / 'results' / STAGE}/"
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "llama_runner.py" in command and marker in command:
            found.append(int(proc.name))
    return sorted(found)


def gpu_ready(python, env):
    probe = subprocess.run(
        [python, "-c", "import torch; print(torch.cuda.device_count())"],
        capture_output=True, text=True, env=env)
    try:
        return probe.returncode == 0 and int(probe.stdout.strip()) >= 1
    except ValueError:
        return False


def build_command(python, runner, model, panel, output, arm, native, scorer, scopes,
                  checkpoint_manifest=None, lengths=LENGTHS):
    command = [python, str(runner), "--model", str(model), "--panel", str(panel),
            "--expected-data", str(panel), "--out", str(output), "--arms", arm,
            "--lengths", lengths, "--native-npy", str(native), "--scorer", scorer,
            "--authorized-scopes", scopes]
    if checkpoint_manifest is not None:
        command.extend(["--checkpoint-manifest", str(checkpoint_manifest)])
    return command


def update_unit(state, state_path, unit_id, **fields):
    state["units"][unit_id].update(fields)
    state["updated_at"] = time.time()
    atomic_json(state_path, state)


def run_unit(root, state, state_path, unit, command, env, artifact_identity=None):
    log_path = Path(root) / "logs" / f"{unit['unit_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prior = state["units"].get(unit["unit_id"], {})
    # A controller crash after persisting RUNNING must consume that attempt;
    # otherwise repeated restarts turn "retry once" into an unbounded retry.
    first_attempt = (int(prior.get("attempt", 0)) + 1
                     if prior.get("status") == "RUNNING" and prior.get("attempt")
                     else 1)
    if first_attempt > 2:
        update_unit(state, state_path, unit["unit_id"],
                    status="BLOCKED_ENGINEERING", completed_at=None)
        return False
    for attempt in range(first_attempt, 3):
        update_unit(state, state_path, unit["unit_id"], status="RUNNING",
                    attempt=attempt, command=command, started_at=time.time())
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt} ===\n")
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env,
                start_new_session=True)
            update_unit(state, state_path, unit["unit_id"],
                        child_pid=process.pid, child_pgid=os.getpgid(process.pid),
                        last_progress_at=time.time())
            last_signature = None
            last_progress = time.time()
            output_dir = Path(root) / "results" / STAGE / unit["arm"]
            timed_out = False
            while process.poll() is None:
                sizes = []
                for path in [log_path, *output_dir.glob("*.partial.jsonl")]:
                    try:
                        stat = path.stat()
                        sizes.append((str(path), stat.st_size, stat.st_mtime_ns))
                    except FileNotFoundError:
                        pass
                signature = tuple(sizes)
                if signature != last_signature:
                    last_signature = signature
                    last_progress = time.time()
                    update_unit(state, state_path, unit["unit_id"],
                                last_progress_at=last_progress)
                elif time.time() - last_progress > 600:
                    timed_out = True
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(2)
            returncode = process.returncode
        if not timed_out and returncode == 0 and unit_complete(
                root, unit, artifact_identity):
            update_unit(state, state_path, unit["unit_id"], status="COMPLETE",
                        returncode=0, child_pid=None, child_pgid=None,
                        completed_at=time.time())
            return True
        update_unit(state, state_path, unit["unit_id"],
                    last_returncode=returncode, timed_out=timed_out,
                    child_pid=None, child_pgid=None, last_failed_at=time.time())
    update_unit(state, state_path, unit["unit_id"], status="BLOCKED_ENGINEERING",
                completed_at=None)
    return False


def run_queue(args, units):
    root, code, model = map(Path, (args.root, args.code, args.model))
    scorer = Path(args.scorer)
    identity = model_identity(model)
    runner = code / "llama_runner.py"
    native = root / "native_inv_freq.npy"
    panel = root / "data" / STAGE / "rows.jsonl"
    checkpoint_manifest = root / "validation" / "checkpoint_sha256.json"
    if not runner.exists() or not native.exists() or not scorer.exists():
        raise RuntimeError("runner/native/scorer prerequisite missing")
    checkpoint_identity = read_json(checkpoint_manifest)
    if checkpoint_identity.get("status") != "COMPLETE" or \
            Path(checkpoint_identity.get("model", "")).resolve() != model.resolve():
        raise RuntimeError("full checkpoint SHA-256 manifest is missing or stale")
    if not manifest_complete(root / "data" / STAGE, STAGE):
        raise RuntimeError("S panel manifest is not COMPLETE")
    data_validation_path = root / "validation" / "P_S_data_validation.json"
    data_validation = read_json(data_validation_path)
    if not data_validation.get("all_pass") or \
            data_validation.get("rows_sha256", {}).get("S") != file_sha(panel):
        raise RuntimeError("P/S data isolation validation is missing, failed, or stale")
    p_readiness_path = root / "validation" / "P_readiness_report.json"
    p_readiness = read_json(p_readiness_path)
    if p_readiness.get("status") != "READY_FOR_S":
        raise RuntimeError("P strong-control instrument has not been cleared for S")
    gauge_path = root / "validation" / "planb_matched_controls_selftest.json"
    if not gauge_path.exists():
        raise RuntimeError("D03 gauge/matched-control selftest artifact is missing")
    gauge = read_json(gauge_path)
    current_matched_sha = hashlib.sha256(
        (code / "planb_matched_controls.py").read_bytes()).hexdigest()
    if not gauge.get("all_pass") or \
            gauge.get("registry", {}).get("module_sha256") != current_matched_sha:
        raise RuntimeError("D03 gauge/matched-control selftest is failed or stale")
    p_state_path = root / "queue_state.json"
    if not p_state_path.exists() or read_json(p_state_path).get("status") != "COMPLETE_P_L0":
        raise RuntimeError("P_L0 is not COMPLETE; S must not start")
    if "olmo" in str(model).lower() or "qwen" in str(model).lower():
        raise RuntimeError("REFUSING: non-Llama model")
    authorization_path = code / "runtime_authorization.json"
    authorization = read_json(authorization_path)
    if authorization.get("status") != "AUTHORIZED" or \
            authorization.get("model") != "Meta-Llama-3-8B-Instruct":
        raise RuntimeError("runtime authorization is missing or targets another model")
    approved_scopes = tuple(authorization.get("approved_scopes", ()))
    if not approved_scopes:
        raise RuntimeError("runtime authorization grants no operator scopes")
    if set(x.lower() for x in authorization.get("forbidden_tonight", ())) != \
            {"olmo", "qwen"}:
        raise RuntimeError("runtime authorization does not preserve tonight's model exclusions")

    artifact_identity = {
        "panel_sha256": file_sha(panel),
        "panel_manifest_sha256": file_sha(root / "data" / STAGE / "manifest.json"),
        "scorer_sha256": file_sha(scorer),
        "scoring_contract_sha256": file_sha(code / "SCORING_CONTRACT.md"),
        "model_config_sha256": identity["config_sha256"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "weight_index_sha256": identity["weight_index_sha256"],
        "checkpoint_manifest_sha256": file_sha(checkpoint_manifest),
        "native_inv_freq_sha256": file_sha(native),
        "runner_sha256": file_sha(runner),
        "operators_sha256": file_sha(code / "operators.py"),
        "matched_controls_sha256": current_matched_sha,
        "runtime_authorization_sha256": file_sha(authorization_path),
        "data_validation_sha256": file_sha(data_validation_path),
        "p_readiness_sha256": file_sha(p_readiness_path),
    }

    lock = acquire_lock(root)
    try:
        plan = {"scope": "Llama-3-only S continuation", "model": str(model),
                "stage": STAGE, "lengths": LENGTHS,
                "units": units, "model_identity": identity,
                "artifact_identity": artifact_identity,
                "reusable_mappings": {
                    "L1_MR_A100": "S_MR", "L1_BM_A100": "S_BM",
                    "L2_MR_BF32_BS1": "S_MR", "L2_BM_BF32_BS1": "S_BM",
                    "D05_DC_D05c": "S_D05c",
                    "D03_GAUGE": "validation/planb_matched_controls_selftest.json",
                },
                "deferred_controls": {
                    "D07_MATCHED_GLOBAL": "requires frozen unlabeled M-dev activations",
                    "D03_FULL_BEHAVIOR": "MR/BM x identity/permutation belongs to M-H",
                },
                "no_olmo_qwen": True}
        plan_sha = canonical_sha(plan)
        plan_path = root / "sv_queue_plan.json"
        if plan_path.exists() and canonical_sha(read_json(plan_path)) != plan_sha:
            raise RuntimeError("existing S plan differs; refusing to mix queue contracts")
        atomic_json(plan_path, plan)
        state_path = root / "sv_queue_state.json"
        state = read_json(state_path) if state_path.exists() else {
            "status": "STARTING", "plan_sha256": plan_sha,
            "scope": "Llama-3-only S continuation", "units": {
                u["unit_id"]: dict(u) for u in units}, "completed_units": []}
        if state.get("plan_sha256") != plan_sha:
            raise RuntimeError("existing S state belongs to another plan")
        atomic_json(state_path, state)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        if not gpu_ready(args.python, env):
            state.update(status="BLOCKED_NO_GPU", updated_at=time.time())
            atomic_json(state_path, state)
            return 3

        for unit in units:
            current = state["units"][unit["unit_id"]]
            if current["status"] in {"COMPLETE", "SCOPE_BLOCKED", "HOLD_PLAN", "DROP_PLAN"}:
                continue
            if unit["kind"] == "join":
                if all(state["units"][x]["status"] == "COMPLETE"
                       for x in unit["requires"]):
                    update_unit(state, state_path, unit["unit_id"], status="COMPLETE",
                                completed_at=time.time())
                else:
                    update_unit(state, state_path, unit["unit_id"],
                                status="BLOCKED_CONTROL_FAILURE")
                continue
            required_scope = unit.get("required_scope", "control")
            if required_scope != "control" and required_scope not in approved_scopes:
                update_unit(state, state_path, unit["unit_id"], status="SCOPE_BLOCKED",
                            reason=f"runtime authorization does not grant {required_scope}")
                continue
            if not all((x == "P_L0" and
                        read_json(p_state_path).get("status") == "COMPLETE_P_L0") or
                       state["units"][x]["status"] == "COMPLETE"
                       for x in unit["requires"]):
                update_unit(state, state_path, unit["unit_id"],
                            status="BLOCKED_DEPENDENCY")
                continue
            if unit_complete(root, unit, artifact_identity):
                update_unit(state, state_path, unit["unit_id"], status="COMPLETE",
                            resumed_from_existing=True, completed_at=time.time())
                continue
            output = root / "results" / STAGE / unit["arm"]
            command = build_command(args.python, runner, model, panel, output,
                                    unit["arm"], native, str(scorer),
                                    ",".join(approved_scopes),
                                    checkpoint_manifest=checkpoint_manifest,
                                    lengths="8192" if unit["arm"] == "Native" else LENGTHS)
            if not run_unit(root, state, state_path, unit, command, env,
                            artifact_identity):
                # Do not start a dependent candidate after an engineering
                # failure; independent later units may still be considered.
                continue
            state["completed_units"].append(unit["unit_id"])
            atomic_json(state_path, state)

        statuses = {u["status"] for u in state["units"].values()}
        state["status"] = ("COMPLETE_S" if all(s in {"COMPLETE", "SCOPE_BLOCKED",
                                                       "HOLD_PLAN", "DROP_PLAN"}
                                                 for s in statuses)
                            else "BLOCKED")
        state["updated_at"] = time.time()
        atomic_json(state_path, state)
        return 0 if state["status"] == "COMPLETE_S" else 3
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--code", default=DEFAULT_CODE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scorer", default=DEFAULT_SCORER,
                    help="required frozen scorer path; evaluator only")
    ap.add_argument("--python", default="/root/miniconda3/bin/python")
    ap.add_argument("--cuda-device", default="0")
    ap.add_argument("--dry-run", action="store_true",
                    help="print and validate the S plan; never checks GPU or starts a child")
    ap.add_argument("--execute", action="store_true",
                    help="run the resumable queue in the foreground")
    a = ap.parse_args(argv)
    units = build_units(a.code)
    validate_units(units)
    if not a.execute:
        print(json.dumps({"scope": "Llama-3-only S continuation", "units": units},
                         indent=2, ensure_ascii=False))
        return 0
    if not a.scorer:
        raise SystemExit("REFUSING: --scorer is required; do not inherit an unverified evaluator")
    return run_queue(a, units)


if __name__ == "__main__":
    raise SystemExit(main())
