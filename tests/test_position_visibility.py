"""Input-coordinate invariants; these do not qualify a real model runtime."""
import pytest

from scripts.experiments.scale_transport.position_visibility import layout


def test_visibility_changes_neither_token_identity_nor_decoder_history():
    prompt = [10, 11, 12, 13, 14, 15, 16]
    keep = [0, 2, 3, 6]
    views = {mode: layout(prompt, keep, mode) for mode in "OLPC"}
    assert views["O"]["prefill_ids"] == views["L"]["prefill_ids"] == prompt[:-1]
    assert views["L"]["cache_keep_positions"] == keep[:-1]
    assert views["P"]["prefill_ids"] == views["C"]["prefill_ids"] == [10, 12, 13]
    assert views["P"]["prefill_positions"] == [0, 2, 3]
    assert views["C"]["prefill_positions"] == [0, 1, 2]
    assert views["P"]["query_position"] == views["L"]["query_position"] == 6
    assert views["C"]["query_position"] == 3
    for view in views.values():
        assert view["processor_prompt_ids"] == prompt
        assert view["query_id"] == prompt[-1]


@pytest.mark.parametrize("keep", [[0, 2], [0, 4, 4], [0, 4, 2], [-1, 4], [0, 5], [0, True, 4]])
def test_invalid_or_reordered_masks_are_rejected(keep):
    with pytest.raises(ValueError):
        layout([1, 2, 3, 4, 5], keep, "P")
