from pathlib import Path

import pytest

from scripts.experiments.olmo_fast_screen.prepare import verify_weight_stats
from scripts.experiments.olmo_fast_screen.ruler_bench import TASKS, verdict


def test_sharded_weights_are_checked_individually(tmp_path):
    stats={}
    for name in ('model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'):
        p=tmp_path/name;p.write_bytes(b'unchanged shard')
        s=p.stat();stats[name]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
    manifest=dict(model_path=str(tmp_path),weight_stats=stats)
    verify_weight_stats(manifest)
    (tmp_path/'model-00002-of-00002.safetensors').write_bytes(b'changed second shard')
    with pytest.raises(ValueError,match='00002-of-00002'):verify_weight_stats(manifest)


def test_legacy_single_weight_contract_still_works(tmp_path):
    p=tmp_path/'model.safetensors';p.write_bytes(b'weights');s=p.stat()
    verify_weight_stats(dict(model_path=str(tmp_path),weight_stat=dict(size=s.st_size,mtime_ns=s.st_mtime_ns)))


def test_native_32k_to_128k_transfer_uses_correct_endpoints():
    baseline=[];candidate=[]
    for cap in (32768,131072):
        for task in TASKS:
            row=dict(row_id=f'{task}_{cap}',task=task,length_cap=cap,references=['gold'],
                     prompt_sha256=f'{task}_{cap}',ended_eos=True,correct=.5)
            baseline.append(row)
            candidate.append(dict(row,correct=.5 if cap==32768 else .75))
    result=verdict(candidate,baseline)
    assert result['status']=='DEVELOPMENT_WIN'
    assert result['macro_delta_by_length']=={'32768':0,'131072':.25}
    candidate[0]['correct']=0
    assert verdict(candidate,baseline)['status']=='TRADEOFF'
