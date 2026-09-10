"""Create a concrete execution plan from locally prepared inputs and cached model metadata."""
import argparse
import json
from pathlib import Path

from .data import write_json
from .tables import construct


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-metadata',type=Path,required=True)
    p.add_argument('--runtime-model',required=True)
    a=p.parse_args()
    from transformers import AutoConfig
    from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding
    config=AutoConfig.from_pretrained(a.model_metadata,local_files_only=True)
    if config.model_type!='olmo2' or config.max_position_embeddings!=4096:
        raise ValueError('this registered preparation is for the selected native-4K OLMo checkpoint')
    native=Olmo2RotaryEmbedding(config).inv_freq.detach().cpu().numpy()
    tables=construct(config.to_dict(),native,tau=2.,scale=4.)
    a.root.mkdir(parents=True,exist_ok=True)
    write_json(a.root/'tables.json',tables)
    plan=dict(model_id='allenai/OLMo-2-0425-1B-Instruct',
        model_revision='48d788eca847d4d7548f375ad03d3c9312f6139e',
        model_weight_sha256='36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f',
        model_path=a.runtime_model,native_length=4096,cpt_length=16384,
        evaluation_lengths=[4096,8192,16384,32768],
        arms=list(tables),seed=20260910,rank=32,learning_rate=2e-5,
        sft_weight=1.,native_weight=.25,schedule_cpt_tokens=134217728,
        checkpoint_cpt_tokens=[8388608,33554432,67108864,134217728],
        first_training_endpoint=33554432,
        losses='Dense causal CE on every real 16K text token + full assistant answer/EOS CE on long instruction + .25 short replay CE. Each term is separately mean-normalized and separately logged.',
        trainable='Q/K/V/O and gate/up/down FFN LoRA; optional same-data full-parameter recovery comparison',
        order=['Cosh and YaRN recovery comparison first','Exponential, Hybrid, Native matched-budget shape comparisons after recipe has informative capability results'],
        interpretation='A lower loss is not recovered capability. A finite training endpoint is not a universal LoRA limit. Cosh superiority can only refer to these actual comparators and matched conditions.',
        teacher='No teacher model or teacher-generated supervision is invented. Short replay uses the existing verified reference pool; LongAlign assistant labels retain their synthetic origin.',
        hardware_note='Real 16K GPU smoke is required to measure memory/time; CPU preparation cannot certify GPU execution. No automatic deadline or shutdown.')
    write_json(a.root/'plan.json',plan)
    print(json.dumps({name:dict(parameters=t['parameters'],rms=t.get('rms_normalized_deformation'),gain=t['gain'])
                      for name,t in tables.items()},indent=2))


if __name__=='__main__':main()
