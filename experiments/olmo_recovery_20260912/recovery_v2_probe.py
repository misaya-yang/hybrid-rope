#!/usr/bin/env python3
"""One disposable v2 family/8K/16K probe; no checkpoint is reused for science."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--arm',choices=['Native','Cosh_tau1','Cosh_tau2'],default='Cosh_tau1')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    names=['short_lm','short_sft','long_lm_8192','long_lm_16384','long_sft_8192','long_sft_16384','long_synthetic_8192','long_synthetic_16384']
    if not args.execute:
        print(json.dumps({'status':'PLAN_ONLY','discarded':True,'pools':names,'micro_updates':8}));return
    import numpy as np
    import torch
    from .recovery_v2_runtime import load_model,group_optimizer,backward_family,table_for_config
    from .recovery_v2_train import Pool,validate_cuda
    validate_cuda();manifest=json.loads(args.data.read_text())
    pools={name:Pool(manifest['pools'][name]) for name in names}
    model,wrapper,table=load_model(args.model,args.arm,training=True);optimizer=group_optimizer(model)
    counts={name:0 for name in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']}
    for name,module in model.named_modules():
        if not hasattr(module,'lora_A'):continue
        projection=name.split('.')[-1]
        if projection not in counts:raise ValueError('unexpected adapted projection '+name)
        rank=16 if projection in ['gate_proj','up_proj','down_proj'] else 64
        if module.lora_A['default'].weight.shape[0]!=rank or module.scaling['default']!=2.0:
            raise ValueError('rank/scaling pattern did not reach '+name)
        counts[projection]+=1
    if any(count!=16 for count in counts.values()):raise ValueError('LoRA does not cover all 16 layers')
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters!=24641536:raise ValueError('unexpected trainable parameter count: '+str(parameters))
    native=np.asarray(table_for_config(model.config,'Native')['values_float32'],dtype=np.float32)
    records=[];started=time.monotonic()
    for name,pool in pools.items():
        family=name.removesuffix('_8192').removesuffix('_16384')
        optimizer.zero_grad(set_to_none=True)
        record=backward_family(model,wrapper,[pool.get(0,42)],family,native)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();record.update(pool=name,gradient_norm=float(norm));records.append(record)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'status':'GPU_PROBE_PASS','discarded':True,'records':records,
                                  'trainable_parameters':parameters,'layers_per_projection':counts,
                                  'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-started,
                                  'activation_checkpointing':bool(getattr(model,'_activation_checkpointing',False)),
                                  'compile_mode':model.__dict__.get('_compile_mode'),'optimizer_fused':True,
                                  'scope':'engineering forward/backward only; not learning or capability evidence'},indent=2)+'\n')
    print(args.out.read_text())


if __name__=='__main__':main()
