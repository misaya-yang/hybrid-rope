import pytest
import torch
from scripts.experiments.olmo_fast_screen.chunked_mlp import chunk_forward


def test_positionwise_gated_mlp_with_partial_final_chunk():
    torch.manual_seed(7)
    gate = torch.nn.Linear(13, 31, bias=False).double()
    up = torch.nn.Linear(13, 31, bias=False).double()
    down = torch.nn.Linear(31, 13, bias=False).double()
    def forward(x):
        return down(torch.nn.functional.silu(gate(x))*up(x))
    x = torch.randn(2, 19, 13, dtype=torch.float64)
    before = x.clone()
    with torch.inference_mode():
        expected = forward(x)
        actual = chunk_forward(forward, x, 7)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    assert torch.equal(x, before)


def test_chunking_refuses_training():
    with pytest.raises(RuntimeError, match='inference-only'):
        chunk_forward(lambda x:x, torch.ones(1, 4, 2), 2)
