"""CPU checks for the MrRoPE-parity PPL and NIAH artifacts."""

from experiments.iclr2027_three_track_sprint_20260915.mrrope_niah_heatmap_report import (
    rouge1_recall,
)
from experiments.iclr2027_three_track_sprint_20260915.prepare_mrrope_niah_heatmap import (
    chat_shell,
    cyclic_slice,
    grid_cells,
)


def test_niah_grid_has_four_lengths_nine_depths_three_repeats():
    cells = grid_cells()
    assert len(cells) == 108
    assert len(set(cells)) == 108
    assert len(grid_cells(20)) == 720


def test_cyclic_slice_wraps_without_changing_requested_count():
    assert cyclic_slice([1, 2, 3], start=2, count=5) == [3, 1, 2, 3, 1]


def test_single_number_rouge1_recall_accepts_answer_inside_short_text():
    assert rouge1_recall("The answer is 12345678.", "12345678") == 1.0
    assert rouge1_recall("87654321", "12345678") == 0.0


def test_chat_shell_accepts_batch_encoding_style_result():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1, 2, 99, 3], "attention_mask": [1, 1, 1, 1]}

        def convert_tokens_to_ids(self, value):
            assert value == "<|eot_id|>"
            return 99

    assert chat_shell(Tokenizer()) == ([1, 2], [99, 3])
