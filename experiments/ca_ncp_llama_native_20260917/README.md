# Llama CA-NCP Native transfer

This package prepares the contingent Llama-3-8B Native-8K transfer of CA-NCP.
It is not launched unless the frozen OLMo five-arm gate supports the method.

The frequency rule is not retuned: the existing public-input Llama NCP table is
used directly, the external carrier ratio remains `2.205`, and the resulting
carrier is slot 38 with active coordinates 35--62. The task panel is the first
10 source-order rows per task from the existing clean Native-8K Full-13x50
asset. Existing Native outputs are reused as N0; C0/P0/N1/P1 are the only new
formal arms.

Llama has no OLMo-style `q_norm/k_norm`. The shared runtime therefore installs
the same unitary map on `q_proj/k_proj` outputs, before reshape and RoPE. Parity
must prove that an identity map is token exact before formal generation.

CPU preparation:

```bash
bash experiments/ca_ncp_llama_native_20260917/prepare_server_cpu.sh
```

Contingent GPU sequence:

```bash
REPO=/root/autodl-tmp/hybrid-rope
ROOT=/root/autodl-tmp/today_rope_plan_20260914/ca_ncp_llama_native_20260917
MODEL=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
DATA=/root/autodl-tmp/today_rope_plan_20260914/strong_evidence/llama_s4_clean_native_x5/assets/manifest.json

cd "$REPO"
python -m experiments.ca_ncp_native_20260917.capture_statistics \
  --model "$MODEL" --assets "$ROOT/assets/statistics" \
  --construction "$ROOT/construction" --out "$ROOT/statistics" --execute
python -m experiments.ca_ncp_native_20260917.build_alignment \
  --statistics "$ROOT/statistics" --construction "$ROOT/construction" \
  --out "$ROOT/alignment"
python -m experiments.ca_ncp_native_20260917.run \
  --data "$DATA" --model "$MODEL" --root "$ROOT" \
  --parallel-workers 1 --execute
```

Use one worker on the 32GB 4080. Three Llama-8B workers are reserved for a
96GB host; the OLMo three-way setting must not be copied onto the 4080.
