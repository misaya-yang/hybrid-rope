#!/usr/bin/env python3
"""Freeze N/Z/G/official-Transformers-YaRN controls; no LM load or outcome search."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.export_log_p2_factor_frontier import tensor_sha256,file_sha256

P2_SHA='56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b'


def same_support_geometric(table):
    if table.dtype!=np.float32 or table.ndim!=1 or len(table)<2 or not np.all(table[:-1]>table[1:]) or not np.all(table>0) or not np.isfinite(table).all():
        raise ValueError('positive ordered float32 table required')
    result=np.exp(np.linspace(np.log(float(table[0])),np.log(float(table[-1])),len(table))).astype(np.float32)
    result[[0,-1]]=table[[0,-1]]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--native-table',type=Path,required=True)
    p.add_argument('--reference-log-s4',type=Path,required=True)
    p.add_argument('--profile-source-native',type=Path,help='transport the pinned K64 movement to another Native basis')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    native=np.load(a.native_table,allow_pickle=False); z=np.load(a.reference_log_s4,allow_pickle=False)
    if tensor_sha256(z)!=P2_SHA: raise ValueError('full-p2 reference identity drift')
    cfg=AutoConfig.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False)
    if 'default' in ROPE_INIT_FUNCTIONS:
        calculated,_=ROPE_INIT_FUNCTIONS['default'](cfg,torch.device('cpu'))
    else:
        # Transformers 5 moved default RoPE construction into the model class.
        if cfg.model_type=='olmo2':
            from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding as Rotary
        elif cfg.model_type=='qwen2':
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as Rotary
        else: raise ValueError('unregistered model family for fixed controls')
        calculated,_=Rotary.compute_default_rope_parameters(cfg,torch.device('cpu'))
    if native.dtype!=np.float32 or tensor_sha256(native)!=tensor_sha256(calculated.float().numpy()):
        raise ValueError('Native tensor does not match original config under this runtime')
    if a.profile_source_native:
        from scripts.analysis.export_log_p2_factor_frontier import derive_movement,realize
        source=np.load(a.profile_source_native,allow_pickle=False)
        movement=derive_movement(source,z,4.)
        if native.shape!=source.shape: raise ValueError('only exact K64 profile transport is registered')
        z=realize(native,movement,4.)
    yarn_cfg=copy.deepcopy(cfg)
    yarn_cfg.rope_scaling={'rope_type':'yarn','factor':4.,'original_max_position_embeddings':cfg.max_position_embeddings}
    if hasattr(yarn_cfg,'rope_parameters'):
        yarn_cfg.rope_parameters={**cfg.rope_parameters,**yarn_cfg.rope_scaling}
    yarn,yarn_gain=ROPE_INIT_FUNCTIONS['yarn'](yarn_cfg,torch.device('cpu'))
    controls={'N':(native,1.),'Z':(z,1.102585782722872),
              'G':(same_support_geometric(z),1.102585782722872),'Y':(yarn.float().numpy(),float(yarn_gain))}
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    arms={}
    for name,(table,gain) in controls.items():
        path=a.output/f'{name}.npy'; np.save(path,table,allow_pickle=False)
        arms[name]={'path':path.name,'float32_sha256':tensor_sha256(table),'file_sha256':file_sha256(path),
                    'rotary_amplitude':gain,'effective_logit_multiplier':gain**2}
    result={'status':'FIXED_NZGY_CONTROLS_FROZEN_V1','arms':arms,'transformers':transformers.__version__,
            'profile_source_Z_sha256':P2_SHA if a.profile_source_native else None,
            'profile_source_native_sha256':file_sha256(a.profile_source_native) if a.profile_source_native else None,
            'actual_native_context_length':cfg.max_position_embeddings,
            'checkpoint_config_sha256':file_sha256(a.checkpoint/'config.json'),'native_sha256':tensor_sha256(native),
            'scope':'G versus Z isolates interiors at matched sampled support/gain; N/Y are practical controls'}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
