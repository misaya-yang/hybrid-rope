from __future__ import annotations

import copy
import math

import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evq_attention_restoration import (
    RelationCapture,
    configure_relation_capture_attention,
    dense_relation_forward_kl,
    install_qkv_lora,
    normalized_context_mse,
)


def _orthogonal(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(dimension, dimension, generator=generator)
    return torch.linalg.qr(value).Q


def test_dense_relation_kl_is_zero_for_identical_inputs() -> None:
    torch.manual_seed(1)
    value = torch.randn(2, 3, 7, 4, requires_grad=True)
    loss = dense_relation_forward_kl(value, value, value.detach(), value.detach())
    assert abs(float(loss.detach())) < 1e-5
    loss.backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_qk_loss_detects_self_relation_gauge_counterexample() -> None:
    torch.manual_seed(2)
    query = torch.randn(1, 2, 8, 6)
    key = torch.randn(1, 2, 8, 6)
    left_rotation = _orthogonal(6, 3)
    right_rotation = _orthogonal(6, 4)
    student_query = query @ left_rotation
    student_key = key @ right_rotation

    qq = dense_relation_forward_kl(
        student_query,
        student_query,
        query,
        query,
    )
    kk = dense_relation_forward_kl(
        student_key,
        student_key,
        key,
        key,
    )
    qk = dense_relation_forward_kl(
        student_query,
        student_key,
        query,
        key,
    )
    assert float(qq) < 1e-5
    assert float(kk) < 1e-5
    assert float(qk) > 1e-3


def test_dense_relation_kl_respects_padding_and_causality() -> None:
    torch.manual_seed(5)
    teacher = torch.randn(1, 2, 6, 4)
    student = teacher.clone()
    student[:, :, 4:, :] += 100.0
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    loss = dense_relation_forward_kl(
        student,
        student,
        teacher,
        teacher,
        attention_mask=mask,
        causal=True,
    )
    assert math.isclose(float(loss), 0.0, abs_tol=1e-5)


def test_context_loss_is_normalized_and_identifying() -> None:
    torch.manual_seed(6)
    teacher = torch.randn(1, 2, 8, 4)
    assert float(normalized_context_mse(teacher, teacher)) == 0.0
    changed = teacher @ _orthogonal(4, 7)
    assert float(normalized_context_mse(changed, teacher)) > 1e-3


def test_olmo2_attention_interface_captures_post_rope_gradients() -> None:
    torch.manual_seed(8)
    config = Olmo2Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=500_000,
        attention_dropout=0.0,
    )
    teacher = Olmo2ForCausalLM(config)
    student = copy.deepcopy(teacher)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainable = install_qkv_lora(student, rank=8, alpha=16.0)
    assert trainable > 0
    student.model.rotary_emb.inv_freq.mul_(0.9)
    configure_relation_capture_attention(teacher)
    configure_relation_capture_attention(student)
    capture = RelationCapture(layer_index=1)
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        teacher.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
            relation_capture=capture,
            relation_mode="teacher",
        )
    student.model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=False,
        relation_capture=capture,
        relation_mode="student",
    )
    teacher_value, student_value = capture.require_pair()
    assert teacher_value.query.shape == (1, 2, 32, 64)
    assert student_value.query.requires_grad
    loss = dense_relation_forward_kl(
        student_value.query,
        student_value.key,
        teacher_value.query,
        teacher_value.key,
    ) / (1 * 2 * 32)
    loss = loss + normalized_context_mse(
        student_value.context, teacher_value.context
    )
    assert float(loss.detach()) > 1e-5
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in student.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert any(
        bool(torch.count_nonzero(gradient))
        for gradient in gradients
        if gradient is not None
    )
