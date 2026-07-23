from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch


def _load(module_name: str):
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, f"missing experiment module: {module_name}"
    return importlib.import_module(module_name)


def test_default_phases_hold_physical_tokens_per_step_constant() -> None:
    curriculum = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.curriculum")

    expected = {
        "warmup": (4096, 8, 64, 512),
        "transition": (4096, 8, 128, 1024),
        "exact_8k": (8192, 4, 128, 512),
        "exact_16k": (16384, 2, 96, 192),
    }
    for name, (seq_len, effective_batch, steps, examples) in expected.items():
        phase = curriculum.get_phase(name)
        assert phase.seq_len == seq_len
        assert phase.effective_batch == effective_batch
        assert phase.steps == steps
        assert phase.tokens_per_step == 32768
        assert phase.training_examples == examples


def test_log_frequency_homotopy_has_exact_endpoints_and_geometric_midpoint() -> None:
    curriculum = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.curriculum")
    native = torch.tensor([1.0, 0.1, 0.01], dtype=torch.float64)
    target = torch.tensor([0.8, 0.2, 0.04], dtype=torch.float64)

    assert torch.equal(curriculum.log_frequency_homotopy(native, target, 0.0), native)
    assert torch.equal(curriculum.log_frequency_homotopy(native, target, 1.0), target)
    midpoint = curriculum.log_frequency_homotopy(native, target, 0.5)
    assert torch.allclose(midpoint, torch.sqrt(native * target), rtol=0.0, atol=1e-12)
    assert torch.all(midpoint > 0)


@pytest.mark.parametrize(
    ("native", "target", "message"),
    [
        (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.5]), "positive"),
        (torch.tensor([1.0, 0.5]), torch.tensor([1.0]), "shape"),
        (torch.tensor([[1.0, 0.5]]), torch.tensor([[1.0, 0.5]]), "one-dimensional"),
    ],
)
def test_log_frequency_homotopy_fails_closed(native, target, message: str) -> None:
    curriculum = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.curriculum")
    with pytest.raises(ValueError, match=message):
        curriculum.log_frequency_homotopy(native, target, 0.5)


def test_answer_only_labels_mask_prompt_and_supervise_only_answer_span() -> None:
    curriculum = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.curriculum")

    labels = curriculum.answer_only_labels(
        torch.tensor([10, 11, 12, 13, 14, 15]),
        answer_start=4,
        answer_end=6,
    )

    assert labels.tolist() == [-100, -100, -100, -100, 14, 15]


def test_answer_only_labels_reject_empty_or_out_of_bounds_answer() -> None:
    curriculum = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.curriculum")
    tokens = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError, match="answer span"):
        curriculum.answer_only_labels(tokens, answer_start=2, answer_end=2)
    with pytest.raises(ValueError, match="answer span"):
        curriculum.answer_only_labels(tokens, answer_start=2, answer_end=4)


def test_rotary_pair_energy_uses_llama_half_split_pairing() -> None:
    curriculum = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.curriculum")
    row_energy = torch.tensor([1, 2, 3, 4, 10, 20, 30, 40], dtype=torch.float64)

    pair_energy = curriculum.half_split_pair_energy(row_energy, head_dim=4)

    # Head 1: (1+3, 2+4); head 2: (10+30, 20+40).
    assert pair_energy.tolist() == pytest.approx([44.0, 66.0])


def test_exact_distance_example_is_fixed_length_and_answer_only() -> None:
    prepare_data = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.prepare_data")
    template = prepare_data.RetrievalTemplate(
        instruction=(1, 2),
        source_prefix=(3,),
        source_infix=(4,),
        source_suffix=(5,),
        query_prefix=(6, 7),
        query_suffix=(8,),
    )

    example = prepare_data.build_retrieval_example(
        template=template,
        key_ids=(20, 21),
        value_ids=(30, 31, 32),
        filler_ids=tuple(range(100, 300)),
        seq_len=64,
        target_distance=20,
        eos_token_id=9,
        task_type="kv",
    )

    assert example.input_ids.dtype == torch.int32
    assert example.input_ids.numel() == 64
    assert example.answer_start - 1 - example.source_value_start == 20
    assert example.query_key_start - example.source_value_start == example.query_key_distance
    assert example.query_key_distance < example.distance
    assert example.distance == 20
    labels = example.labels()
    assert torch.all(labels[: example.answer_start] == -100)
    assert labels[example.answer_start : example.answer_end].tolist() == [30, 31, 32, 9]


def test_update_example_keeps_old_value_before_relevant_source() -> None:
    prepare_data = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.prepare_data")
    template = prepare_data.RetrievalTemplate(
        instruction=(1,),
        source_prefix=(2,),
        source_infix=(3,),
        source_suffix=(4,),
        query_prefix=(5,),
        query_suffix=(6,),
    )
    old_source = (70, 71, 72, 73, 74)

    example = prepare_data.build_retrieval_example(
        template=template,
        key_ids=(20,),
        value_ids=(30, 31),
        filler_ids=tuple(range(100, 300)),
        seq_len=48,
        target_distance=16,
        eos_token_id=9,
        task_type="update",
        pre_source_ids=old_source,
    )

    source = example.input_ids.tolist()
    assert source.index(old_source[0]) < example.source_start
    assert example.answer_start - 1 - example.source_value_start == 16


def test_build_template_uses_real_chat_generation_boundary() -> None:
    prepare_data = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.prepare_data")

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is True
            assert messages == [{"role": "user", "content": ""}]
            no_generation = [1, 2, 99]
            return no_generation + ([3, 4] if add_generation_prompt else [])

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [10 + (ord(char) % 20) for char in text]}

    template = prepare_data.build_template(Tokenizer(), "train")

    assert template.instruction[:2] == (1, 2)
    assert template.query_suffix[-3:] == (99, 3, 4)


def test_build_template_accepts_batch_encoding_chat_template_output() -> None:
    prepare_data = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.prepare_data")

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is True
            tokens = [1, 2, 99] + ([3, 4] if add_generation_prompt else [])
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [10 + (ord(char) % 20) for char in text]}

    template = prepare_data.build_template(Tokenizer(), "train")

    assert template.instruction[:2] == (1, 2)
    assert template.query_suffix[-3:] == (99, 3, 4)


def test_counterfactual_triplet_preserves_positions_and_changes_only_contract_spans() -> None:
    prepare_data = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.prepare_data")
    template = prepare_data.RetrievalTemplate(
        instruction=(1,),
        source_prefix=(2,),
        source_infix=(3,),
        source_suffix=(4,),
        query_prefix=(5,),
        query_suffix=(6,),
    )
    example = prepare_data.build_retrieval_example(
        template=template,
        key_ids=(20,),
        value_ids=(30, 31),
        filler_ids=tuple(range(100, 300)),
        seq_len=48,
        target_distance=16,
        eos_token_id=9,
        task_type="kv",
    )

    triplet = prepare_data.build_counterfactual_triplet(
        example,
        swapped_value_ids=(40, 41),
        removal_fill_ids=tuple(range(500, 500 + example.source_end - example.source_start)),
        group_id="g0",
    )

    assert [record.variant for record in triplet] == ["original", "swapped", "source_removed"]
    assert {record.input_ids.numel() for record in triplet} == {48}
    assert {record.answer_start for record in triplet} == {example.answer_start}
    assert {record.query_key_start for record in triplet} == {example.query_key_start}
    swapped = triplet[1]
    assert swapped.input_ids[example.source_value_start : example.source_value_end].tolist() == [40, 41]
    assert swapped.input_ids[example.answer_start : example.answer_end - 1].tolist() == [40, 41]
    removed = triplet[2]
    assert (
        removed.input_ids[example.source_start : example.source_end].tolist()
        != example.input_ids[example.source_start : example.source_end].tolist()
    )
    assert (
        removed.input_ids[example.answer_start : example.answer_end].tolist()
        == example.input_ids[example.answer_start : example.answer_end].tolist()
    )


def test_phase_frequency_endpoints_keep_geo_matched_and_evq_exact() -> None:
    train = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.train")
    native = torch.tensor([1.0, 0.1], dtype=torch.float64)
    evq = torch.tensor([0.8, 0.2], dtype=torch.float64)

    start, end = train.phase_frequency_endpoints(native, evq, "transition", "geo")
    assert torch.equal(start, native)
    assert torch.equal(end, native)
    start, end = train.phase_frequency_endpoints(native, evq, "transition", "evq")
    assert torch.equal(start, native)
    assert torch.equal(end, evq)
    start, end = train.phase_frequency_endpoints(native, evq, "exact_8k", "evq")
    assert torch.equal(start, evq)
    assert torch.equal(end, evq)
    with pytest.raises(ValueError, match="warmup"):
        train.phase_frequency_endpoints(native, evq, "warmup", "evq")


def test_frequency_transition_reaches_exact_target_on_last_optimizer_step() -> None:
    train = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.train")
    native = torch.tensor([1.0, 0.1], dtype=torch.float64)
    evq = torch.tensor([0.8, 0.2], dtype=torch.float64)
    transition = train.FrequencyTransition(native, evq, steps=5)

    assert torch.equal(transition.at_step(0), native)
    assert torch.equal(transition.at_step(4), evq)
    assert torch.allclose(transition.at_step(2), torch.sqrt(native * evq), atol=1e-12)


def test_effective_lora_update_energy_uses_output_rotary_rows() -> None:
    train = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.train")
    lora_a = torch.eye(2, dtype=torch.float64)
    lora_b = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=torch.float64,
    )

    energy = train.effective_lora_pair_energy(
        lora_a,
        lora_b,
        scaling=0.5,
        head_dim=4,
    )

    # Delta-row energies after scaling: [.25, 1, 2.25, 4].
    assert energy.tolist() == pytest.approx([2.5, 5.0])


def test_tensor_answer_dataset_materializes_only_answer_labels() -> None:
    train = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.train")
    bundle = {
        "format_version": 1,
        "phase": "warmup",
        "split": "train",
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
        "answer_start": torch.tensor([2, 3], dtype=torch.int32),
        "answer_end": torch.tensor([4, 4], dtype=torch.int32),
        "metadata": [{"task_type": "kv"}, {"task_type": "update"}],
    }

    dataset = train.TensorAnswerDataset(bundle)

    assert dataset[0]["labels"].tolist() == [-100, -100, 3, 4]
    assert dataset[1]["labels"].tolist() == [-100, -100, -100, 8]
    assert dataset[0]["attention_mask"].tolist() == [1, 1, 1, 1]


def test_tail_answer_cross_entropy_uses_only_causal_answer_predictors() -> None:
    train = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.train")
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    # The retained tail has hidden positions [1, 2, 3]. Positions 1 and 2
    # predict answer tokens 3 and 4; the final logit is intentionally unused.
    tail_logits = torch.full((1, 3, 8), -10.0)
    tail_logits[0, 0, 3] = 10.0
    tail_logits[0, 1, 4] = 10.0

    loss = train.tail_answer_cross_entropy(tail_logits, input_ids, labels)

    assert loss.item() < 1e-6


def test_tail_answer_cross_entropy_rejects_non_tail_supervision() -> None:
    train = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.train")
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, -100, 4]])
    tail_logits = torch.zeros((1, 3, 8))

    with pytest.raises(ValueError, match="contiguous tail"):
        train.tail_answer_cross_entropy(tail_logits, input_ids, labels)


def test_stack_and_validate_bundle_preserve_exact_phase_contract() -> None:
    prepare_data = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.prepare_data")
    template = prepare_data.RetrievalTemplate(
        instruction=(1,),
        source_prefix=(2,),
        source_infix=(3,),
        source_suffix=(4,),
        query_prefix=(5,),
        query_suffix=(6,),
    )
    examples = [
        prepare_data.build_retrieval_example(
            template=template,
            key_ids=(20 + index,),
            value_ids=(30 + index, 40 + index),
            filler_ids=tuple(range(100, 500)),
            seq_len=48,
            target_distance=16 + index,
            eos_token_id=9,
            task_type="kv",
        )
        for index in range(2)
    ]

    bundle = prepare_data.stack_examples(
        examples,
        phase="fixture",
        split="train",
        seed=42,
        protocol={"source": "fixture"},
    )
    validated = prepare_data.validate_bundle(
        bundle,
        expected_phase="fixture",
        expected_split="train",
        expected_seq_len=48,
    )

    assert validated["input_ids"].dtype == torch.int32
    assert tuple(validated["input_ids"].shape) == (2, 48)
    assert validated["metadata"][1]["distance"] == 17


def test_teacher_forced_answer_metrics_use_causal_shift() -> None:
    evaluate = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.evaluate")
    input_ids = torch.tensor([1, 2, 3, 4])
    logits = torch.full((4, 8), -10.0)
    logits[1, 3] = 10.0  # token at position 1 predicts answer token 3 at position 2
    logits[2, 4] = 10.0  # token at position 2 predicts answer token 4 at position 3

    metrics = evaluate.teacher_forced_answer_metrics(
        logits,
        input_ids,
        answer_start=2,
        answer_end=4,
    )

    assert metrics["exact"] is True
    assert metrics["nll"] < 1e-6


def test_counterfactual_summary_is_group_paired() -> None:
    evaluate = _load("rebuttal.pre_rebuttal.frequency_adaptation_8b.evaluate")
    records = [
        evaluate.ScoredRecord("g0", "original", "kv", 1000, 0.2, True),
        evaluate.ScoredRecord("g0", "swapped", "kv", 1000, 0.3, True),
        evaluate.ScoredRecord("g0", "source_removed", "kv", 1000, 1.0, False),
        evaluate.ScoredRecord("g1", "original", "update", 2000, 0.4, True),
        evaluate.ScoredRecord("g1", "swapped", "update", 2000, 0.5, False),
        evaluate.ScoredRecord("g1", "source_removed", "update", 2000, 0.2, False),
    ]

    summary = evaluate.summarize_counterfactual_scores(records)

    assert summary["groups"] == 2
    assert summary["original_exact"] == pytest.approx(1.0)
    assert summary["swapped_exact"] == pytest.approx(0.5)
    assert summary["pair_consistency"] == pytest.approx(0.5)
    assert summary["removal_nll_increase_mean"] == pytest.approx(0.3)
    assert summary["removal_positive_fraction"] == pytest.approx(0.5)
    assert set(summary["by_task"]) == {"kv", "update"}
