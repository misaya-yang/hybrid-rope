# Core Text Phases

This directory is the main Phase 8–15 text experiment chain.

## Phase Map

### Phase 8: raw EVQ scaling-law foundation
- `run_evq_sweep.py`: base sweep entrypoint for the early EVQ from-scratch regime
- `phase8d_scaling_law.py`: direct scaling-law verification sweep
- `phase8f_multi_seed.py`: multi-seed statistical verification

### Phase 11: PE-dominant regime
- `phase11_L256_extrap.py`: L=256 raw extrapolation chain
- `phase11_yarn_eval.py`: L=256 YaRN interaction evaluation
- `phase11b_125m_dape.py`: 125M DAPE-style compatibility / comparison
- `phase11c_454m_scaling.py`: token-scaling follow-up
- `phase11e_continued_pretrain.py`: Geo-to-EVQ continued-pretraining retrofit experiment

### Phase 13: downstream NLL probe
- `phase13a_longbench_nll.py`: 750M historical LongBench-NLL support run

### Phase 14: EVQ × YaRN systems story
- `phase14c_multiscale_evq_yarn.py`: multiscale EVQ+YaRN validation
- `phase14d_125m_tinystories_10pct.py`: 125M TinyStories 10% passkey mix follow-up

### Phase 15: larger-scale continued pretraining
- `phase15_750m_2k_to_4k_continue_ckpt_eval.py`: 750M `2K -> 4K` continuation with checkpoint evals

## Shared Evaluation Helpers

- `eval_passkey.py`
- `eval_multi_needle.py`
- `eval_longbench_nll.py`
- `eval_pe_baselines.py`
- `eval_super_extrap.py`
- `eval_dsr.py`

## Practical Reading Order

1. `phase8d_scaling_law.py`
2. `phase11_L256_extrap.py`
3. `phase11_yarn_eval.py`
4. `phase14c_multiscale_evq_yarn.py`
5. `phase15_750m_2k_to_4k_continue_ckpt_eval.py`
