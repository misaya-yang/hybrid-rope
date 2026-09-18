"""OLMo recovery-v2 model, table, optimizer, and family-loss runtime."""
from __future__ import annotations

import contextlib
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from experiments.evq_recovery.tables import anchored_cosh
from scripts.experiments.cross_audit.tables import install_static, native_table
from scripts.experiments.cross_audit.training import causal_loss, native_kl

ARMS = ("Native", "Cosh_tau1", "Cosh_tau2")
TABLE_ARMS = (
    "Native", "Cosh_tau_sqrt2", "Cosh_tau1", "Cosh_tau2", "C42V24_g4",
    "CoshDeploy_tau1_g4", "BM_g4", "BM_g8", "MrPro_g4", "MrPro_g8", "MrUni_g4",
    "BetaSym_gamma1p5_g4", "BetaSym_gamma3_g4", "BetaSym_gamma3_g8", "RangeBridge50_g8",
    "BM_g8_RangeGain",
)
QK = ("q_proj", "k_proj")
VO = ("v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")
ALL_MODULES = QK + VO + FFN


def seed_all(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def table_for_config(config, arm: str) -> dict:
    if arm not in TABLE_ARMS: raise ValueError(arm)
    attention_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    parameters = getattr(config, "rope_parameters", {}) or {}
    partial = float(
        getattr(config, "partial_rotary_factor", None)
        or parameters.get("partial_rotary_factor", 1.0)
    )
    dim = int(attention_dim * partial)
    if dim != attention_dim * partial or dim % 2: raise ValueError("invalid partial RoPE dimension")
    base = getattr(config, "rope_theta", None) or getattr(config, "rope_parameters", {}).get("rope_theta")
    if config.model_type == "olmo2" and dim == 128 and config.num_hidden_layers == 16 and base == 500000:
        reference_length = 4096
    elif (config.model_type == "llama" and dim == 128
          and config.num_hidden_layers in (32, 80)
          and config.num_attention_heads in (32, 64)
          and config.num_key_value_heads == 8
          and config.max_position_embeddings == 8192 and base == 500000):
        reference_length = 8192
        if arm not in ("Native", "BM_g4", "BM_g8", "MrPro_g4", "MrPro_g8", "BetaSym_gamma3_g8", "RangeBridge50_g8", "BM_g8_RangeGain"):
            raise ValueError("unsupported Llama transfer table")
    elif (config.model_type == "llama" and dim == 128
          and config.num_hidden_layers == 32
          and config.num_attention_heads == 32
          and config.num_key_value_heads == 8
          and config.max_position_embeddings == 32768 and base == 8000000):
        # Kanana-1.5-8B uses the standard Llama rotary module with a public
        # 32K/8M geometry.  The generic evaluator always loads its exact Native
        # table first and then installs one explicit frozen receipt.  Do not
        # infer any of the Meta-Llama-specific built-in transfer tables here.
        reference_length = 32768
        if arm != "Native":
            raise ValueError("Kanana screens require an explicit static table")
    elif (config.model_type == "qwen2" and dim == 128
          and config.max_position_embeddings == 32768 and base == 1000000):
        # The generic evaluator loads Native first and then installs an explicit
        # frozen JSON table.  Keep built-in recovery-v2 arms unavailable so a
        # Qwen run cannot silently inherit OLMo/Llama-specific definitions.
        reference_length = 32768
        if arm != "Native":
            raise ValueError("Qwen band screens require an explicit static table")
    elif (config.model_type == "glm4" and attention_dim == 128 and dim == 64
          and config.max_position_embeddings == 32768 and base == 10000):
        reference_length = 32768
        if arm != "Native":
            raise ValueError("GLM partial-RoPE screens require an explicit static table")
    elif (config.model_type == "phi3" and attention_dim == 96 and dim == 96
          and config.num_hidden_layers == 32 and config.num_attention_heads == 32
          and config.num_key_value_heads == 32
          and config.max_position_embeddings == 4096 and base == 10000):
        # The 4K checkpoint is the unscaled Phi-3 source model.  The generic
        # evaluator loads its exact Native table first and installs only an
        # explicit frozen receipt afterwards; no built-in Phi arm is inferred.
        reference_length = 4096
        if arm != "Native":
            raise ValueError("Phi-3 screens require an explicit static table")
    else:
        raise ValueError("unsupported recovery-v2 model geometry")
    native = native_table(dim, base).astype(np.float32)
    if arm == "Native": values, gain, construction = native, 1.0, {"method": "identity"}
    elif arm == "C42V24_g4":
        from experiments.rope_fast_5090_20260912.e3_tables import tables
        record = tables()["C42V24"]
        values = np.asarray(record["values_float32"], dtype=np.float32)
        gain = float(record["gain"])
        construction = {**record["construction"], "source": "E3 exact reconstructed C42V24"}
    elif arm == "CoshDeploy_tau1_g4":
        scale = 4.0
        u = (np.arange(dim // 2, dtype=np.float64) + 0.5) / (dim // 2)
        exponents = 1.0 - np.arcsinh((1.0 - u) * math.sinh(1.0))
        native64 = np.power(float(base), -np.arange(dim // 2, dtype=np.float64) / (dim // 2))
        values = (native64 * np.power(scale, -exponents)).astype(np.float32)
        gain = 1.0 + 0.1 * math.log(scale)
        construction = {
            "method": "native-relative deployed midpoint Cosh",
            "tau": 1.0,
            "scale": scale,
            "gain": gain,
            "source": "pre-existing evq_deploy_t1 construction",
        }
    elif arm in ("BM_g4", "BM_g8", "BM_g8_RangeGain"):
        from scripts.lib.rope.boundary_matched import boundary_matched_inv_freq
        scale = 4.0 if arm == "BM_g4" else 8.0
        installed, gain, construction = boundary_matched_inv_freq(
            torch.from_numpy(native.copy()),
            base=float(base),
            reference_length=reference_length,
            scale=scale,
        )
        if config.model_type == "llama":
            # Preserve the exact float64 construction used by the completed
            # frozen Llama BM campaign before installing the FP32 table.
            native64 = np.power(float(base), -np.arange(dim // 2, dtype=np.float64) / (dim // 2))
            exponent = np.asarray(construction["exponents"], dtype=np.float64)
            values = (native64 * np.power(scale, -exponent)).astype(np.float32)
            source = "exact frozen Llama BM construction"
        else:
            values = installed.detach().cpu().float().numpy()
            source = "existing E2 bm_g4 construction"
        if arm == "BM_g8_RangeGain":
            gain = 1.0 + 0.05 * math.log(scale)
            source = "exact BM_g8 frequencies with log-length interval-average fixed gain"
        construction = {**construction, "source": source}
        if arm == "BM_g8_RangeGain":
            construction.update(
                gain_rule="mean of 1 + 0.1*ln(r) under uniform log-length measure on r in [1,S]",
                frequency_table_equal_to="BM_g8",
            )
    elif arm in ("MrPro_g4", "MrPro_g8"):
        from scripts.experiments.cross_audit.tables import transform
        scale = 4.0 if arm == "MrPro_g4" else 8.0
        values, gain, construction = transform(
            native, dim=dim, base=float(base), reference_length=reference_length,
            scale=scale, method="mrpro",
        )
        construction = {**construction, "source": f"MrRoPE-Pro target-{int(reference_length * scale) // 1024}K diagnostic"}
    elif arm == "RangeBridge50_g8":
        scale = 8.0
        turns = np.power(float(base), -np.arange(dim // 2, dtype=np.float64) / (dim // 2))
        turns *= reference_length / (2.0 * math.pi)
        low = int(np.flatnonzero(turns > 32.0)[-1])
        high = int(np.flatnonzero(turns < 1.0)[0])
        n = high - low
        q = np.clip(np.arange(dim // 2) - low, 0, n).astype(np.float64)
        mrpro = q * (q + 1.0) / (n * (n + 1.0))
        bm = q * (q + 1.0) * (3.0 * n + 2.0 - 2.0 * q) / (n * (n + 1.0) * (n + 2.0))
        exponent = 0.5 * (mrpro + bm)
        native64 = np.power(float(base), -np.arange(dim // 2, dtype=np.float64) / (dim // 2))
        values = (native64 * np.power(scale, -exponent)).astype(np.float32)
        gain = 1.0 + 0.1 * math.log(scale)
        construction = {
            "method": "equal minimax midpoint of BM and MrRoPE-Pro cumulative exponents",
            "mixture_weight_BM": 0.5,
            "mixture_weight_MrPro": 0.5,
            "low": low,
            "high": high,
            "N": n,
            "scale": scale,
            "reference_length": reference_length,
            "source": "parameter-free exponent-space bridge between complementary fixed-table incumbents",
        }
    elif arm == "MrUni_g4":
        from scripts.experiments.cross_audit.tables import transform
        values, gain, construction = transform(
            native, dim=dim, base=float(base), reference_length=reference_length,
            scale=4.0, method="mruni",
        )
        construction = {**construction, "source": "existing E2 MrRoPE-Uni construction"}
    elif arm in ("BetaSym_gamma1p5_g4", "BetaSym_gamma3_g4", "BetaSym_gamma3_g8"):
        gamma = 1.5 if arm == "BetaSym_gamma1p5_g4" else 3.0
        scale = 8.0 if arm.endswith("_g8") else 4.0
        turns = np.power(float(base), -np.arange(dim // 2, dtype=np.float64) / (dim // 2))
        turns *= reference_length / (2.0 * math.pi)
        fast = np.flatnonzero(turns > 32.0)
        slow = np.flatnonzero(turns < 1.0)
        if not len(fast) or not len(slow) or int(slow[0]) <= int(fast[-1]):
            raise ValueError("native grid has no valid 32/1-turn transition")
        low, high = int(fast[-1]), int(slow[0])
        n = high - low
        kk = np.arange(1, n + 1, dtype=np.float64)
        weights = np.power(kk, gamma - 1.0) * np.power(n + 1 - kk, gamma - 1.0)
        exponent = np.zeros(dim // 2, dtype=np.float64)
        exponent[low:low + n + 1] = np.concatenate([[0.0], np.cumsum(weights / weights.sum())])
        exponent[low + n + 1:] = 1.0
        native64 = np.power(float(base), -np.arange(dim // 2, dtype=np.float64) / (dim // 2))
        values = (native64 * np.power(scale, -exponent)).astype(np.float32)
        gain = 1.0 + 0.1 * math.log(scale)
        construction = {
            "method": "symmetric beta increments",
            "gamma": gamma,
            "shape_a": gamma - 1.0,
            "shape_b": gamma - 1.0,
            "low": low,
            "high": high,
            "N": n,
            "scale": scale,
            "reference_length": reference_length,
            "sum_exponents": float(exponent.sum()),
            "source": "OLMo interval-selected gamma=3 transfer; fixed table for every runtime length",
        }
    elif arm.startswith("Cosh_tau"):
        tau = math.sqrt(2.0) if arm == "Cosh_tau_sqrt2" else float(arm.removeprefix("Cosh_tau"))
        z = anchored_cosh(dim // 2, tau)
        logfreq = -np.log(native.astype(np.float64))
        values = np.exp(-(logfreq[0] + (logfreq[-1] - logfreq[0]) * z)).astype(np.float32)
        values[[0, -1]] = native[[0, -1]]; gain = 1.0
        construction = {"method": "anchored midpoint Cosh", "tau": tau, "same_native_endpoints": True}
    else: raise AssertionError("unreachable arm")
    if values.shape != (dim // 2,) or not np.isfinite(values).all() or not np.all(values[:-1] > values[1:]):
        raise ValueError("invalid static frequency table")
    return {"values_float32": values.tolist(), "gain": float(gain), "construction": construction}


def configure_lora(model):
    from peft import LoraConfig, get_peft_model
    ranks = {name: 64 for name in QK + VO} | {name: 16 for name in FFN}
    alphas = {name: 128 for name in QK + VO} | {name: 32 for name in FFN}
    wrapper = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=.05,
        target_modules=list(ALL_MODULES), rank_pattern=ranks, alpha_pattern=alphas,
        bias="none", task_type="CAUSAL_LM"))
    base_model = wrapper.get_base_model()
    trainable = [(name, p) for name, p in base_model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("base/embed/norm/head/rope must remain frozen")
    for module in ALL_MODULES:
        matches = [(name, p) for name, p in trainable if f".{module}." in name]
        if not matches: raise ValueError("missing LoRA module " + module)
    return base_model, wrapper


def load_model(model_path, arm, checkpoint=None, training=True):
    from transformers import AutoConfig, AutoModelForCausalLM
    seed_all(42); model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    table = table_for_config(config, arm)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa")
    if checkpoint is None and training:
        model, wrapper = configure_lora(model)
    elif checkpoint is None:
        # A freshly configured LoRA is output-equivalent to the base model but
        # needlessly materializes its projections during long frozen inference.
        wrapper = None
    else:
        from peft import PeftModel
        wrapper = PeftModel.from_pretrained(model, checkpoint, is_trainable=training)
        model = wrapper.get_base_model()
    values = np.asarray(table["values_float32"], dtype=np.float32)
    install_static(model, values, table["gain"])
    rotary = model.model.rotary_emb
    if rotary.inv_freq.shape != values.shape or not torch.isfinite(rotary.inv_freq).all() or float(rotary.attention_scaling) != table["gain"]:
        raise RuntimeError("static table installation failed")
    model.config.use_cache = not training
    if training:
        checkpointing = os.environ.get("OLMO_ACTIVATION_CHECKPOINTING", "0") == "1"
        if checkpointing: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else: model.gradient_checkpointing_disable()
        model.enable_input_require_grads(); model.train()
        model._activation_checkpointing = checkpointing
        compile_mode = os.environ.get("OLMO_COMPILE_MODE", "max-autotune-no-cudagraphs")
        if compile_mode.lower() not in ("0", "off", "none"):
            from .runtime import register_blackwell_attention
            register_blackwell_attention(model)
            compiled = torch.compile(model.model, mode=compile_mode, dynamic=False, fullgraph=False)
            model.__dict__["_compiled_long_lm_backbone"] = compiled
            model.__dict__["_variable_checkpointing"] = os.environ.get("OLMO_VARIABLE_CHECKPOINTING", "1") == "1"
        model.__dict__["_compile_mode"] = compile_mode
    else: model.eval()
    return model, wrapper, table


def group_optimizer(model):
    groups = {"qk": [], "vo": [], "ffn": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if any(f".{module}." in name for module in QK): groups["qk"].append(parameter)
        elif any(f".{module}." in name for module in VO): groups["vo"].append(parameter)
        elif any(f".{module}." in name for module in FFN): groups["ffn"].append(parameter)
        else: raise ValueError("unclassified trainable parameter: " + name)
    if any(not values for values in groups.values()): raise ValueError("empty optimizer group")
    return torch.optim.AdamW([
        {"name": "qk", "params": groups["qk"], "lr": 5e-5, "peak_lr": 5e-5},
        {"name": "vo", "params": groups["vo"], "lr": 2.5e-5, "peak_lr": 2.5e-5},
        {"name": "ffn", "params": groups["ffn"], "lr": 1.25e-5, "peak_lr": 1.25e-5},
    ], betas=(.9, .95), weight_decay=0., fused=True)


def ids_labels(example, family: str, device):
    ids = example if isinstance(example, np.ndarray) else example["input_ids"]
    ids = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    labels = ids.clone()
    if "sft" in family or "synthetic" in family:
        start = int(example["target_start"])
        if not 0 < start < ids.shape[1]: raise ValueError("invalid SFT target boundary")
        labels[:, :start] = -100
    return ids, labels


def selected_positions(ids, labels, limit=128):
    eligible = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if not eligible.numel(): raise ValueError("short KL has no legal prediction position")
    if eligible.numel() > limit:
        take = torch.linspace(0, eligible.numel() - 1, limit, device=ids.device).round().long()
        eligible = eligible[take]
    return eligible  # hidden/logit p predicts labels[p+1]


def teacher_probabilities(model, wrapper, ids, positions, native_values):
    current = model.model.rotary_emb.inv_freq.detach().cpu().numpy().copy()
    gain = float(model.model.rotary_emb.attention_scaling); was_training = model.training
    try:
        model.eval(); install_static(model, native_values, 1.0)
        with torch.no_grad(), wrapper.disable_adapter():
            hidden = model.model(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0, positions]
            probabilities = torch.nn.functional.linear(hidden, model.lm_head.weight).float().softmax(-1).detach()
    finally:
        install_static(model, current, gain); model.train(was_training)
    return probabilities


def backward_family(model, wrapper, examples, family: str, native_values, *, chunk_size=128):
    """Backward one >=65K-token same-family update; caller steps optimizer."""
    if not examples: raise ValueError("empty family update")
    device = next(model.parameters()).device
    compiled = model.__dict__.get("_compiled_long_lm_backbone")
    backbone = compiled if family == "long_lm" else None
    variable_checkpointing = bool(compiled is not None and family != "long_lm" and model.__dict__.get("_variable_checkpointing", True))
    if variable_checkpointing: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prepared = [ids_labels(example, family, device) for example in examples]
    prediction_counts = [int((labels[:, 1:] != -100).sum()) for _, labels in prepared]
    primary_backbone_tokens = sum(int(ids.shape[1] - 1) for ids, _ in prepared)
    total_predictions = sum(prediction_counts)
    if not total_predictions: raise ValueError("family has no prediction targets")
    ce_sum = 0.0; kl_sum = 0.0; kl_positions = 0; kl_extra_forward_tokens = 0
    for (ids, labels), count in zip(prepared, prediction_counts):
        with torch.autocast("cuda", dtype=torch.bfloat16): loss, actual = causal_loss(model, ids, labels, chunk_size=chunk_size, backbone=backbone)
        if actual != count or not torch.isfinite(loss): raise RuntimeError("nonfinite or miscounted CE")
        # LM is token-normalized over the update. SFT is example-normalized.
        weight = count / total_predictions if family.endswith("lm") else 1.0 / len(prepared)
        (loss * weight).backward(); ce_sum += float(loss.detach()) * weight
        if family.startswith("short_"):
            positions = selected_positions(ids, labels)
            teacher = teacher_probabilities(model, wrapper, ids, positions, native_values)
            kl_input = ids[:, :-1]
            with torch.autocast("cuda", dtype=torch.bfloat16): kl, nk = native_kl(model, kl_input, positions, teacher)
            if not torch.isfinite(kl): raise RuntimeError("nonfinite short teacher KL")
            (0.2 * kl / len(prepared)).backward(); kl_sum += float(kl.detach()) / len(prepared); kl_positions += nk
            kl_extra_forward_tokens += 2 * int(kl_input.shape[1])  # Native teacher + student KL backbones
            del teacher, kl, positions, kl_input
        del loss, ids, labels
    if variable_checkpointing: model.gradient_checkpointing_disable()
    return {"family": family, "examples": len(examples), "ce": ce_sum,
            "short_fullvocab_kl": kl_sum if family.startswith("short_") else None,
            "kl_weight": .2 if family.startswith("short_") else 0.,
            "prediction_tokens": total_predictions, "primary_ce_backbone_input_tokens": primary_backbone_tokens,
            "short_kl_teacher_student_extra_forward_input_tokens": kl_extra_forward_tokens,
            "token_accounting": "schedule uses primary CE backbone input tokens only; KL extra forwards are separately reported compute, not added to the schedule budget",
            "kl_positions": kl_positions}
