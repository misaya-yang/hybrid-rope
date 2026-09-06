"""Confirmation must stay separate from selection and deployment drift."""
from copy import deepcopy
import pytest
from scripts.analysis.review_native_confirmation import EXPECTED,DEPLOYMENT,validate_identity


def fixture():
    baseline={key:'fixed' for key in DEPLOYMENT}
    baseline.update(status='FRESH_STRATIFIED_NATIVE_ENDPOINTS_V1',fold='confirmation',
        table_is_native=True,gain=1.,adapter_sha256=None,adapter_config_sha256=None,
        code_sha256='code',evaluation_engine_sha256='engine',data_sha256='data')
    candidate={**baseline,'adapter_sha256':'adapter','adapter_config_sha256':'adapter-config'}
    locks=[{**{key:r[key] for key in DEPLOYMENT},'status':'FIXED_DEPLOYMENT_NATIVE_CONFIRMATION_V1',
            'retention_threshold':.88,'expected_rows':EXPECTED.copy()} for r in (baseline,candidate)]
    rows=[{'row_id':f'{task}-{i}','asset_sha256':f'{task}-asset-{i}','task':task,'group':f'source-{i//2}'}
          for task,count in EXPECTED.items() for i in range(count)]
    return baseline,candidate,locks,(rows,deepcopy(rows))


def test_fixed_confirmation_admitted_but_selection_and_partial_pool_rejected():
    b,c,locks,data=fixture();validate_identity(b,c,locks,data)
    c['fold']='selection'
    with pytest.raises(ValueError,match='confirmation receipts'):validate_identity(b,c,locks,data)
    c['fold']='confirmation'
    with pytest.raises(ValueError,match='strata'):validate_identity(b,c,locks,(data[0],data[1][:-1]))


def test_locked_deployment_and_paired_source_ownership_cannot_drift():
    b,c,locks,data=fixture();c['gain']=1.1
    with pytest.raises(ValueError,match='deployment'):validate_identity(b,c,locks,data)
    c['gain']=1.;data[1][0]['group']='another-source'
    with pytest.raises(ValueError,match='source groups'):validate_identity(b,c,locks,data)
