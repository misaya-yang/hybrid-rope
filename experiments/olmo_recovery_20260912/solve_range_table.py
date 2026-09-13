#!/usr/bin/env python3
"""First-order trust-region solver for one frozen OLMo Native-relative table."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np


BASELINE_ARMS = ("Native", "BM_g4", "MrPro_g4", "C42V24_g4")
FIT_SPLIT = "fit"
CF_MARGIN = 0.5
CF_REGRET_LIMIT = 0.05
FEASIBILITY_MAX_WORSEN = 0.05


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def round_rows(rows: list[dict], round_index: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["length_cap"], row["task"])].append(row)
    offset = 2 * (round_index % 4)
    selected = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda row: row["row_id"])
        if len(values) != 8:
            raise ValueError(f"fit cell {key} does not contain eight rows")
        selected.extend(values[offset : offset + 2])
    return selected


def linear_epigraph_step(
    values: dict[str, float],
    gradients: dict[str, np.ndarray],
    exponent_jacobian: np.ndarray,
    hard_constraints: list[tuple[float, np.ndarray, float]],
) -> np.ndarray:
    from scipy.optimize import linprog

    names = sorted(values)
    dimension = next(iter(gradients.values())).size
    mean_gradient = np.mean([gradients[name] for name in names], axis=0)
    rows, bounds = [], []
    for name in names:
        rows.append(np.concatenate((gradients[name], [-1.0])))
        bounds.append(-values[name])
    for current, gradient, limit in hard_constraints:
        rows.append(np.concatenate((gradient, [0.0])))
        bounds.append(limit - current)
    for jacobian_row in exponent_jacobian:
        rows.append(np.concatenate((jacobian_row, [0.0, 0.0])))
        bounds.append(0.03)
        rows.append(np.concatenate((-jacobian_row, [0.0, 0.0])))
        bounds.append(0.03)
    equality = np.zeros((1, dimension + 1), dtype=np.float64)
    equality[0, : dimension - 1] = 1.0
    primary = linprog(
        np.concatenate((np.zeros(dimension), [1.0])),
        A_ub=np.asarray(rows),
        b_ub=np.asarray(bounds),
        A_eq=equality,
        b_eq=np.zeros(1),
        bounds=[(-0.5, 0.5)] * (dimension - 1) + [(-0.02, 0.02), (None, None)],
        method="highs",
    )
    if not primary.success:
        raise RuntimeError(f"linear trust-region subproblem failed: {primary.message}")
    tie_rows = rows + [np.concatenate((np.zeros(dimension), [1.0]))]
    tie_bounds = bounds + [float(primary.x[-1]) + 1e-8]
    result = linprog(
        np.concatenate((mean_gradient, [0.0])),
        A_ub=np.asarray(tie_rows),
        b_ub=np.asarray(tie_bounds),
        A_eq=equality,
        b_eq=np.zeros(1),
        bounds=[(-0.5, 0.5)] * (dimension - 1) + [(-0.02, 0.02), (None, None)],
        method="highs",
    )
    if not result.success:
        result = primary
    return np.asarray(result.x[:dimension], dtype=np.float64)


def linear_feasibility_step(
    boundary_constraints: list[tuple[float, np.ndarray, float]],
    kl_constraints: list[tuple[float, np.ndarray, float]],
    exponent_jacobian: np.ndarray,
) -> np.ndarray:
    """Reduce the worst boundary violation while keeping Native KL hard."""
    from scipy.optimize import linprog

    dimension = boundary_constraints[0][1].size
    mean_gradient = np.mean([gradient for _, gradient, _ in boundary_constraints], axis=0)
    rows, bounds = [], []
    for current, gradient, limit in boundary_constraints:
        rows.append(np.concatenate((gradient, [-1.0])))
        bounds.append(limit - current)
        rows.append(np.concatenate((gradient, [0.0])))
        bounds.append(FEASIBILITY_MAX_WORSEN)
    for current, gradient, limit in kl_constraints:
        rows.append(np.concatenate((gradient, [0.0])))
        bounds.append(limit - current)
    for jacobian_row in exponent_jacobian:
        rows.append(np.concatenate((jacobian_row, [0.0, 0.0])))
        bounds.append(0.03)
        rows.append(np.concatenate((-jacobian_row, [0.0, 0.0])))
        bounds.append(0.03)
    equality = np.zeros((1, dimension + 1), dtype=np.float64)
    equality[0, : dimension - 1] = 1.0
    primary = linprog(
        np.concatenate((np.zeros(dimension), [1.0])),
        A_ub=np.asarray(rows),
        b_ub=np.asarray(bounds),
        A_eq=equality,
        b_eq=np.zeros(1),
        bounds=[(-0.5, 0.5)] * (dimension - 1) + [(-0.02, 0.02), (0.0, None)],
        method="highs",
    )
    if not primary.success:
        raise RuntimeError(f"linear feasibility subproblem failed: {primary.message}")
    tie_rows = rows + [np.concatenate((np.zeros(dimension), [1.0]))]
    tie_bounds = bounds + [float(primary.x[-1]) + 1e-8]
    result = linprog(
        np.concatenate((mean_gradient, [0.0])),
        A_ub=np.asarray(tie_rows),
        b_ub=np.asarray(tie_bounds),
        A_eq=equality,
        b_eq=np.zeros(1),
        bounds=[(-0.5, 0.5)] * (dimension - 1) + [(-0.02, 0.02), (0.0, None)],
        method="highs",
    )
    if not result.success:
        result = primary
    return np.asarray(result.x[:dimension], dtype=np.float64)


def lexicographic_objective(values: dict[str, float]) -> tuple[float, float]:
    ordered = list(values.values())
    return max(ordered), sum(ordered) / len(ordered)


def feasibility_objective(
    boundary_constraints: list[tuple[float, np.ndarray | None, float]],
) -> tuple[float, float]:
    violations = [current - limit for current, _, limit in boundary_constraints]
    return max(violations), sum(max(0.0, value) for value in violations) / len(violations)


def no_boundary_spike(
    current: list[tuple[float, np.ndarray | None, float]],
    proposed: list[tuple[float, np.ndarray | None, float]],
) -> bool:
    if len(current) != len(proposed):
        raise ValueError("boundary constraint sets differ")
    return all(
        after - before <= FEASIBILITY_MAX_WORSEN + 1e-6
        for (before, _, _), (after, _, _) in zip(current, proposed)
    )


def max_boundary_increase(
    current: list[tuple[float, np.ndarray | None, float]],
    proposed: list[tuple[float, np.ndarray | None, float]],
) -> float:
    if len(current) != len(proposed):
        raise ValueError("boundary constraint sets differ")
    return max(
        after - before
        for (before, _, _), (after, _, _) in zip(current, proposed)
    )


def combine_full_values_with_stochastic_gradients(full: dict, directional: dict) -> dict:
    """Use full-fit values for decisions and rotating mini-batches only for directions."""
    if set(full["values"]) != set(directional["gradients"]):
        task_gradient_keys = {key for key in directional["gradients"] if key.startswith("task/")}
        if set(full["values"]) != task_gradient_keys:
            raise ValueError("full task cells and stochastic task gradients differ")
    if set(full["cf_regrets"]) != {
        key for key in directional["gradients"] if key.startswith("cf/")
    }:
        raise ValueError("full source cells and stochastic source gradients differ")
    combined = dict(full)
    combined["gradients"] = directional["gradients"]
    combined["native_kl_gradients"] = directional["native_kl_gradients"]
    combined["cf_seed"] = directional["cf_seed"]
    combined["directional_row_ids"] = directional["selected_row_ids"]
    return combined


class Solver:
    def __init__(self, args) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from experiments.olmo_recovery_20260912.native_relative_allocation import install_native_relative_allocation
        from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config
        from experiments.olmo_recovery_20260912.runtime import validate_cuda
        from scripts.experiments.cross_audit.tables import native_table

        self.torch = torch
        self.args = args
        self.environment = validate_cuda()
        self.rows = [row for row in read_rows(args.data / "rows.jsonl") if row["split"] == FIT_SPLIT]
        self.cf_rows = [row for row in read_rows(args.source_cf / "rows.jsonl") if row["split"] == FIT_SPLIT]
        self.baselines = {
            arm: {row["row_id"]: row["nll"] for row in read_rows(args.baselines / f"{arm}.jsonl")}
            for arm in BASELINE_ARMS
        }
        self.cf_baselines = {
            arm: {
                row["row_id"]: row["margin_half_hinge"]
                for row in read_rows(args.baselines / f"source_cf_{arm}.jsonl")
            }
            for arm in BASELINE_ARMS
        }
        if len(self.rows) != 168 or len(self.cf_rows) != 128:
            raise ValueError("fit task or source-counterfactual data is incomplete")
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cuda"}, attn_implementation="sdpa",
        )
        if any(isinstance(module, torch.nn.Dropout) and module.p for module in self.model.modules()):
            raise ValueError("solver requires deterministic zero-dropout execution")
        self.model.eval().requires_grad_(False)
        self.native_teachers = self.build_native_teachers()
        table = table_for_config(config, args.initial_arm)
        base = getattr(config, "rope_theta", None) or getattr(config, "rope_parameters", {}).get("rope_theta")
        native = native_table(config.hidden_size // config.num_attention_heads, base).astype(np.float64)
        values = np.asarray(table["values_float32"], dtype=np.float64)
        exponents = -np.log(values / native) / math.log(4.0)
        exponents[:15] = 0.0
        exponents[32:] = 1.0
        self.allocation = install_native_relative_allocation(
            self.model, low=14, high=32, scale=4.0,
            initial_exponents=exponents, initial_gain=float(table["gain"]),
            initial_inv_freq=values.astype(np.float32),
        )
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.model.train()
        self.initial_table = table
        self.kl_limits = self.load_or_measure_kl_limits()

    @property
    def parameters(self):
        return self.allocation.increment_logits, self.allocation.log_gain

    def build_native_teachers(self):
        torch = self.torch
        teachers = []
        paths = sorted(self.args.native_docs.glob("doc_*.npy"))[:4]
        if len(paths) != 4:
            raise ValueError("need four retained natural-text documents")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for path in paths:
                values = np.load(path, allow_pickle=False)[:4096]
                ids = torch.tensor(values, dtype=torch.long, device="cuda")[None, :]
                positions = torch.linspace(0, 4094, 32, device="cuda").round().long()
                hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]
                probabilities = torch.nn.functional.linear(hidden, self.model.lm_head.weight).float().softmax(-1).cpu()
                teachers.append((values.astype(np.int64), positions.cpu().numpy(), probabilities))
        return teachers

    def load_or_measure_kl_limits(self):
        path = self.args.out.parent / "bm_native_kl_limits.json"
        if path.exists():
            return json.loads(path.read_text())["limits"]
        if self.args.initial_arm != "BM_g4":
            raise FileNotFoundError("BM start must create the shared Native-KL limits first")
        limits = []
        for index in range(4):
            value, _ = self.native_kl(index, grad=False)
            limits.append(value + 1e-4)
        write(path, {"status": "COMPLETE", "source": "BM_g4 incumbent plus 1e-4 numerical allowance", "limits": limits})
        return limits

    def task_loss(self, row, grad: bool):
        torch = self.torch
        target = row["target_ids"]
        ids = torch.tensor([row["prompt_ids"] + target[:-1]], dtype=torch.long, device="cuda")
        labels = torch.tensor(target, dtype=torch.long, device="cuda")
        context = torch.enable_grad() if grad else torch.no_grad()
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=len(target), return_dict=True).logits[0].float()
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def cf_group_loss(self, rows: list[dict], grad: bool):
        torch = self.torch
        inputs, targets = [], []
        for row in rows:
            correct, wrong = row["correct_target_ids"], row["wrong_target_ids"]
            if len(correct) != len(wrong):
                raise ValueError("counterfactual target lengths differ")
            inputs.extend((row["prompt_ids"] + correct[:-1], row["prompt_ids"] + wrong[:-1]))
            targets.extend((correct, wrong))
        if not rows or len({len(value) for value in inputs}) != 1 or len({len(value) for value in targets}) != 1:
            raise ValueError("counterfactual group is not batch-compatible")
        inputs = torch.tensor(inputs, dtype=torch.long, device="cuda")
        targets = torch.tensor(targets, dtype=torch.long, device="cuda")
        context = torch.enable_grad() if grad else torch.no_grad()
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(
                input_ids=inputs, use_cache=False,
                logits_to_keep=targets.shape[1], return_dict=True,
            ).logits.float()
            logprob = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).sum(-1)
            margins = logprob.reshape(len(rows), 2)[:, 0] - logprob.reshape(len(rows), 2)[:, 1]
            loss = torch.relu(torch.tensor(CF_MARGIN, device="cuda") - margins).mean()
        return loss, margins

    def native_kl(self, index: int, grad: bool):
        torch = self.torch
        from scripts.experiments.cross_audit.training import native_kl
        values, positions, probabilities = self.native_teachers[index]
        ids = torch.tensor(values, dtype=torch.long, device="cuda")[None, :]
        pos = torch.tensor(positions, dtype=torch.long, device="cuda")
        context = torch.enable_grad() if grad else torch.no_grad()
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            loss, count = native_kl(self.model, ids, pos, probabilities)
        return float(loss.detach()), loss if grad else count

    def gradient_vector(self, loss) -> np.ndarray:
        gradients = self.torch.autograd.grad(loss, self.parameters, allow_unused=False)
        return np.concatenate((gradients[0].detach().cpu().double().numpy(), [float(gradients[1].detach())]))

    def measure(self, round_index: int, grad: bool, *, full_fit: bool = False):
        selected = self.rows if full_fit else round_rows(self.rows, round_index)
        task_values, task_gradients = defaultdict(list), defaultdict(list)
        for row in selected:
            loss = self.task_loss(row, grad)
            key = f"task/{row['length_cap']}/{row['task']}"
            task_values[key].append(float(loss.detach()))
            if grad:
                task_gradients[key].append(self.gradient_vector(loss))
        cf_values, cf_gradients, cf_row_ids = defaultdict(list), defaultdict(list), defaultdict(list)
        cf_seed = "all_fit" if full_fit else sorted({row["source_seed"] for row in self.cf_rows})[round_index % 8]
        cf_batches = defaultdict(list)
        for row in self.cf_rows:
            if not full_fit and row["source_seed"] != cf_seed:
                continue
            cf_batches[(row["length_cap"], row["family"], row["source_seed"])].append(row)
        for (length_cap, family, _), batch in sorted(cf_batches.items()):
            if len(batch) != 2 or {row["world"] for row in batch} != {0, 1}:
                raise ValueError("source-counterfactual fit group must contain both worlds")
            loss, _ = self.cf_group_loss(batch, grad)
            key = f"cf/{length_cap}/{family}"
            cf_values[key].append(float(loss.detach()))
            cf_row_ids[key].extend(row["row_id"] for row in batch)
            if grad:
                cf_gradients[key].append(self.gradient_vector(loss))
        values = {key: sum(items) / len(items) for key, items in task_values.items()}
        gradients = {key: np.mean(task_gradients[key], axis=0) for key in task_gradients} if grad else {}
        cf_values = {key: sum(items) / len(items) for key, items in cf_values.items()}
        cf_gradients = {key: np.mean(cf_gradients[key], axis=0) for key in cf_gradients} if grad else {}
        row_ids_by_group = defaultdict(list)
        for row in selected:
            row_ids_by_group[f"task/{row['length_cap']}/{row['task']}"].append(row["row_id"])
        native_deltas, endpoint_regrets, cf_regrets = {}, {}, {}
        for key, ids in row_ids_by_group.items():
            _, cap, _ = key.split("/", 2)
            candidate = values[key]
            arm_means = {arm: sum(self.baselines[arm][row_id] for row_id in ids) / len(ids) for arm in BASELINE_ARMS}
            values[key] = candidate - min(arm_means.values())
            if cap == "4096":
                native_deltas[key] = candidate - arm_means["Native"]
            if cap == "16384":
                endpoint_regrets[key] = candidate - min(arm_means[arm] for arm in ("BM_g4", "MrPro_g4", "C42V24_g4"))
        for key, ids in cf_row_ids.items():
            arm_means = {
                arm: sum(self.cf_baselines[arm][row_id] for row_id in ids) / len(ids)
                for arm in BASELINE_ARMS
            }
            cf_regrets[key] = cf_values[key] - min(arm_means.values())
            if grad:
                gradients[key] = cf_gradients[key]
        kl_values, kl_gradients = {}, {}
        for index in range(4):
            kl_value, kl_loss = self.native_kl(index, grad)
            kl_values[str(index)] = kl_value
            kl_gradients[str(index)] = self.gradient_vector(kl_loss) if grad else None
        return {
            "values": values,
            "gradients": gradients,
            "native_deltas": native_deltas,
            "endpoint_regrets": endpoint_regrets,
            "cf_regrets": cf_regrets,
            "native_kl_by_document": kl_values,
            "native_kl_gradients": kl_gradients,
            "selected_row_ids": [row["row_id"] for row in selected],
            "cf_seed": cf_seed,
        }

    def exponent_jacobian(self) -> np.ndarray:
        torch = self.torch
        logits = self.allocation.increment_logits.detach().clone().requires_grad_(True)
        def function(value):
            increments = torch.softmax(value - value.mean(), dim=0)
            return torch.cumsum(increments, dim=0)
        return torch.autograd.functional.jacobian(function, logits).detach().cpu().double().numpy()

    def boundary_constraints(self, measurement) -> list[tuple[float, np.ndarray | None, float]]:
        result = []
        for key, value in measurement["native_deltas"].items():
            result.append((value, measurement["gradients"].get(key), 0.03))
        for key, value in measurement["endpoint_regrets"].items():
            result.append((value, measurement["gradients"].get(key), 0.05))
        for key, value in measurement["cf_regrets"].items():
            result.append((value, measurement["gradients"].get(key), CF_REGRET_LIMIT))
        return result

    def kl_constraints(self, measurement) -> list[tuple[float, np.ndarray | None, float]]:
        return [
            (
                measurement["native_kl_by_document"][str(index)],
                measurement["native_kl_gradients"][str(index)],
                self.kl_limits[index],
            )
            for index in range(4)
        ]

    def hard_constraints(self, measurement) -> list[tuple[float, np.ndarray, float]]:
        constraints = self.boundary_constraints(measurement) + self.kl_constraints(measurement)
        if any(gradient is None for _, gradient, _ in constraints):
            raise ValueError("hard constraints require gradients")
        return constraints

    def feasible(self, measurement) -> bool:
        return (
            max(measurement["native_deltas"].values()) <= 0.03 + 1e-6
            and max(measurement["endpoint_regrets"].values()) <= 0.05 + 1e-6
            and max(measurement["cf_regrets"].values()) <= CF_REGRET_LIMIT + 1e-6
            and all(
                measurement["native_kl_by_document"][str(index)] <= self.kl_limits[index] + 1e-7
                for index in range(4)
            )
        )

    def state(self):
        return self.allocation.increment_logits.detach().clone(), self.allocation.log_gain.detach().clone()

    def table_receipt(self) -> dict:
        allocation = self.allocation.receipt()
        values = self.allocation.realized_inv_freq().detach().cpu().float().numpy()
        return {
            "values_float32": values.tolist(),
            "gain": allocation["gain"],
            "construction": allocation,
        }

    def run(self):
        torch = self.torch
        self.args.out.mkdir(parents=True, exist_ok=False)
        events = []
        accepted = 0
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        full_current = self.measure(0, grad=False, full_fit=True)
        for round_index in range(self.args.max_accepted_steps):
            current_state = self.state()
            directional = self.measure(round_index, grad=True)
            current = combine_full_values_with_stochastic_gradients(full_current, directional)
            current_objective = lexicographic_objective(current["values"])
            current_feasibility = feasibility_objective(self.boundary_constraints(current))
            jacobian = self.exponent_jacobian()
            try:
                if current_feasibility[0] > 1e-6:
                    phase = "feasibility_restoration"
                    boundary = self.boundary_constraints(current)
                    if any(gradient is None for _, gradient, _ in boundary):
                        raise ValueError("boundary restoration requires gradients")
                    kl_constraints = self.kl_constraints(current)
                    if any(gradient is None for _, gradient, _ in kl_constraints):
                        raise ValueError("Native KL restoration constraint requires a gradient")
                    step = linear_feasibility_step(boundary, kl_constraints, jacobian)
                else:
                    phase = "range_objective"
                    step = linear_epigraph_step(
                        current["values"], current["gradients"], jacobian,
                        self.hard_constraints(current),
                    )
            except RuntimeError as error:
                events.append({
                    "round": round_index, "status": "NO_LINEAR_SOLUTION", "phase": phase,
                    "error": str(error), "current_objective": current_objective,
                    "current_feasibility_objective": current_feasibility,
                    "current_native_deltas": current["native_deltas"],
                    "current_endpoint_regrets": current["endpoint_regrets"],
                    "current_cf_regrets": current["cf_regrets"],
                    "current_native_kl_by_document": current["native_kl_by_document"],
                })
                break
            accepted_measurement = None
            accepted_alpha = None
            attempts = []
            for alpha in (1.0, 0.5, 0.25, 0.125):
                self.allocation.set_state_(
                    current_state[0] + alpha * torch.tensor(step[:-1], device="cuda", dtype=torch.float32),
                    current_state[1] + alpha * float(step[-1]),
                )
                proposed = self.measure(0, grad=False, full_fit=True)
                proposed_objective = lexicographic_objective(proposed["values"])
                proposed_feasibility = feasibility_objective(self.boundary_constraints(proposed))
                if phase == "feasibility_restoration":
                    improves = proposed_feasibility[0] < current_feasibility[0] - 1e-5 or (
                        proposed_feasibility[0] <= current_feasibility[0] + 1e-5
                        and proposed_feasibility[1] < current_feasibility[1] - 1e-5
                    )
                    kl_holds = all(
                        value <= limit + 1e-7
                        for value, _, limit in self.kl_constraints(proposed)
                    )
                    spike_holds = no_boundary_spike(
                        self.boundary_constraints(current), self.boundary_constraints(proposed)
                    )
                    constraints_hold = (
                        kl_holds and spike_holds
                        if self.args.acceptance_policy == "strict"
                        else True
                    )
                else:
                    improves = proposed_objective[0] < current_objective[0] - 1e-5 or (
                        proposed_objective[0] <= current_objective[0] + 1e-5
                        and proposed_objective[1] < current_objective[1] - 1e-5
                    )
                    constraints_hold = (
                        self.feasible(proposed)
                        if self.args.acceptance_policy == "strict"
                        else True
                    )
                    kl_holds = constraints_hold
                    spike_holds = constraints_hold
                attempts.append({
                    "alpha": alpha,
                    "improves": improves,
                    "constraints_hold": constraints_hold,
                    "native_kl_holds": kl_holds,
                    "boundary_spike_holds": spike_holds,
                    "max_boundary_increase": max_boundary_increase(
                        self.boundary_constraints(current), self.boundary_constraints(proposed)
                    ),
                    "objective": proposed_objective,
                    "feasibility_objective": proposed_feasibility,
                    "native_deltas": proposed["native_deltas"],
                    "endpoint_regrets": proposed["endpoint_regrets"],
                    "cf_regrets": proposed["cf_regrets"],
                    "native_kl_by_document": proposed["native_kl_by_document"],
                    "table": self.table_receipt(),
                })
                if improves and constraints_hold:
                    accepted_measurement = proposed
                    accepted_alpha = alpha
                    break
            if accepted_measurement is None:
                self.allocation.set_state_(*current_state)
                events.append({
                    "round": round_index, "status": "NO_ACCEPTED_BACKTRACK",
                    "phase": phase,
                    "current_objective": current_objective,
                    "current_feasibility_objective": current_feasibility,
                    "current_native_deltas": current["native_deltas"],
                    "current_endpoint_regrets": current["endpoint_regrets"],
                    "current_cf_regrets": current["cf_regrets"],
                    "current_native_kl_by_document": current["native_kl_by_document"],
                    "current_feasible": self.feasible(current),
                    "backtrack_attempts": attempts,
                    "step_parameter_vector": step.tolist(),
                    "linearized_exponent_step": (jacobian @ step[:-1]).tolist(),
                    "gain_step": float(step[-1]),
                })
                break
            accepted += 1
            full_current = accepted_measurement
            gradient_matrix = np.stack(list(current["gradients"].values()))
            linearized_exponent_step = jacobian @ step[:-1]
            event = {
                "round": round_index,
                "status": "ACCEPTED",
                "phase": phase,
                "alpha": accepted_alpha,
                "before_objective": current_objective,
                "after_objective": lexicographic_objective(accepted_measurement["values"]),
                "before_values": current["values"],
                "after_values": accepted_measurement["values"],
                "before_feasibility_objective": current_feasibility,
                "after_feasibility_objective": feasibility_objective(
                    self.boundary_constraints(accepted_measurement)
                ),
                "after_native_kl_by_document": accepted_measurement["native_kl_by_document"],
                "before_native_kl_by_document": current["native_kl_by_document"],
                "before_native_deltas": current["native_deltas"],
                "after_native_deltas": accepted_measurement["native_deltas"],
                "before_endpoint_regrets": current["endpoint_regrets"],
                "after_endpoint_regrets": accepted_measurement["endpoint_regrets"],
                "before_cf_regrets": current["cf_regrets"],
                "after_cf_regrets": accepted_measurement["cf_regrets"],
                "gradient_rank": int(np.linalg.matrix_rank(gradient_matrix)),
                "gradient_singular_values": np.linalg.svd(gradient_matrix, compute_uv=False).tolist(),
                "group_gradient_norms": {
                    key: float(np.linalg.norm(gradient)) for key, gradient in current["gradients"].items()
                },
                "linearized_group_changes": {
                    key: float(gradient @ step) for key, gradient in current["gradients"].items()
                },
                "step_norm": float(np.linalg.norm(step)),
                "step_parameter_vector": step.tolist(),
                "linearized_exponent_step": linearized_exponent_step.tolist(),
                "max_linearized_exponent_step": float(np.max(np.abs(linearized_exponent_step))),
                "gain_step": float(step[-1]),
                "cf_seed": current["cf_seed"],
                "backtrack_attempts": attempts,
                "allocation": self.allocation.receipt(),
                "table": self.table_receipt(),
            }
            events.append(event)
            write(self.args.out / "live.json", event)
            write(self.args.out / f"step_{accepted:02d}.json", event)
        receipt = self.allocation.receipt()
        full_fit = full_current
        full_fit_validation = {
            "values": full_fit["values"],
            "range_objective": lexicographic_objective(full_fit["values"]),
            "native_deltas": full_fit["native_deltas"],
            "endpoint_regrets": full_fit["endpoint_regrets"],
            "cf_regrets": full_fit["cf_regrets"],
            "native_kl_by_document": full_fit["native_kl_by_document"],
            "feasibility_objective": feasibility_objective(self.boundary_constraints(full_fit)),
            "all_hard_constraints_hold": self.feasible(full_fit),
            "task_rows": len(full_fit["selected_row_ids"]),
            "source_cf_rows": len(self.cf_rows),
        }
        result = {
            "status": "COMPLETE",
            "initial_arm": self.args.initial_arm,
            "acceptance_policy": self.args.acceptance_policy,
            "accepted_steps": accepted,
            "events": events,
            "allocation": receipt,
            "table": self.table_receipt(),
            "full_fit_validation": full_fit_validation,
            "environment": self.environment,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "fit optimization and full-fit forward validation only; select/internal-confirm free generation not yet run",
        }
        write(self.args.out / "result.json", result)
        write(self.args.out / "status.json", {"status": "COMPLETE", "accepted_steps": accepted})
        print(json.dumps({
            "status": result["status"], "initial_arm": self.args.initial_arm,
            "accepted_steps": accepted, "elapsed_seconds": result["elapsed_seconds"],
            "peak_memory_allocated_bytes": result["peak_memory_allocated_bytes"],
        }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-cf", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--native-docs", type=Path, required=True)
    parser.add_argument(
        "--initial-arm",
        choices=("BM_g4", "C42V24_g4", "BetaSym_gamma3_g4"),
        required=True,
    )
    parser.add_argument("--max-accepted-steps", type=int, default=8)
    parser.add_argument(
        "--acceptance-policy", choices=("strict", "exploratory"), default="strict",
        help="strict enforces proxy constraints; exploratory retains improving Pareto directions for later generation",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY", "initial_arm": args.initial_arm,
            "max_accepted_steps": args.max_accepted_steps,
            "acceptance_policy": args.acceptance_policy,
        }))
        return
    if args.max_accepted_steps < 1 or args.out.exists():
        raise ValueError("positive step count and a new output directory are required")
    Solver(args).run()


if __name__ == "__main__":
    main()
