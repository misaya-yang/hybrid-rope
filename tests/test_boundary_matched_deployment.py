import json
from pathlib import Path

import pytest
import torch

from scripts.lib.rope.boundary_matched import boundary_matched_inv_freq
from scripts.experiments.cross_audit.tables import tensor_sha, transform, native_table
from scripts.analysis.build_boundary_matched_mrpro import build


def test_deployable_function_matches_the_actual_frozen_olmo_tensor():
    root=Path(__file__).resolve().parents[1]
    source=json.loads((root/'docs/research/ROPE_OLMO_MRPRO_SOURCE_20260908.json').read_text())
    candidate=json.loads((root/'docs/research/ROPE_MRPRO_BM_CANDIDATE_20260908.json').read_text())['target_olmo']
    native=torch.tensor(source['tables']['Native']['values_float32'],dtype=torch.float32)
    result,gain,meta=boundary_matched_inv_freq(native,base=500000,reference_length=4096,scale=4)
    assert tensor_sha(result.numpy()) == candidate['tensor_sha256']
    assert result.tolist()==candidate['values_float32']
    assert gain==candidate['gain'] and (meta['low'],meta['high'])==(14,32)
    assert torch.equal(result[:15],native[:15])
    assert torch.equal(result[32:],native[32:]/4)


@pytest.mark.parametrize('base,length,scale',[(500000,4096,8),(1000000,32768,4),(500000,4096,2.5)])
def test_matches_independently_solved_discrete_objective(base,length,scale):
    native=native_table(128,base)
    mr,gain,meta=transform(native,dim=128,base=base,reference_length=length,scale=scale,method='mrpro')
    reference=build(native.tolist(),mr.tolist(),gain,scale,meta['low'],meta['high'])
    result,actual_gain,_=boundary_matched_inv_freq(torch.from_numpy(native),base=base,reference_length=length,scale=scale)
    assert tensor_sha(result.numpy())==reference['tensor_sha256']
    assert actual_gain==gain


def test_native_identity_and_reject_already_scaled_or_reduced_precision():
    native=torch.from_numpy(native_table(128,500000))
    result,gain,_=boundary_matched_inv_freq(native,base=500000,reference_length=4096,scale=1)
    assert torch.equal(result,native) and gain==1
    with pytest.raises(ValueError):boundary_matched_inv_freq(native/4,base=500000,reference_length=4096,scale=4)
    with pytest.raises(ValueError):boundary_matched_inv_freq(native.bfloat16(),base=500000,reference_length=4096,scale=4)
