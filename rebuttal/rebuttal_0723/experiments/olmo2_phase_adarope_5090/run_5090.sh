#!/usr/bin/env bash
set -euo pipefail
action="${1:-}"
root="${PHASE_ADAROPE_ROOT:-/root/autodl-tmp/iclr_next_runs/phase_adarope_20260822}"
asset_root="${PHASE_ADAROPE_ASSET_ROOT:-$root/assets}"
data_root="${PHASE_ADAROPE_DATA_ROOT:-$root/data}"
checkpoint="${PHASE_ADAROPE_CHECKPOINT:-$asset_root/checkpoint}"
ready="${PHASE_ADAROPE_READY:-$asset_root/checkpoint_ready.json}"
target_manifest="${PHASE_ADAROPE_TARGET_MANIFEST:-$asset_root/target_manifest_derived.json}"
python="${PYTHON:-/root/miniconda3/bin/python}"
code_root="${CODE_ROOT:-/root/autodl-tmp/hybrid-rope}"
module="rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.train_phase_adarope"
downstream_module="rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.evaluate_downstream"
longbench_zip="${PHASE_ADAROPE_LONGBENCH_ZIP:-$asset_root/LongBench-data.zip}"
ruler_data="${RULER_DATA:-$root/ruler}"
export PYTHONPATH="$code_root" TOKENIZERS_PARALLELISM=false

# Scientific pause: this static whole-table tournament is a falsifiable
# empirical hypothesis, not the structurally Native-preserving retrofit the
# current task requires.  Keep only CPU preparation/audit actions callable.
case "$action" in
  derive-target|preflight|prepare-ruler|"") ;;
  *)
    echo "PAUSED BEFORE GPU: static whole-table AdaRoPE does not structurally preserve the Native window." >&2
    echo "Use the Native-anchored suffix-only retrofit after its protocol is frozen." >&2
    exit 65
    ;;
esac

gpu() { export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PHASE_ADAROPE_GPU_AUTHORIZED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$root/torchinductor_cache}"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"; }
assets() { [[ -d "$checkpoint" && -f "$ready" && -f "$target_manifest" ]] || { echo 'checkpoint/READY/derived target manifest missing' >&2; exit 2; }; "$python" - "$target_manifest" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
if p.get("derivation_receipt", {}).get("parent_manifest_sha256") != "cf03385431df1084508055232853d341a903aa2d8b99fca087bc45d513d34af9":
    raise SystemExit("derived target parent identity drift")
c=p.get("candidates", p)
for key in ("phase_chord_olmo_r0_lambda_0p1", "matched_exponential_control", "moment_matched_same_sign_control"):
    if key not in c:
        raise SystemExit(f"missing derived target: {key}")
PY
}
run() { "$python" -m "$module" "$@"; }

case "$action" in
  derive-target)
    CUDA_VISIBLE_DEVICES="" "$python" -m rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.derive_target_manifest --parent-manifest "$asset_root/target_manifest.json" --output "$target_manifest" --expected-parent-sha cf03385431df1084508055232853d341a903aa2d8b99fca087bc45d513d34af9; ;;
  preflight)
    assets; CUDA_VISIBLE_DEVICES="" run preflight --checkpoint "$checkpoint" --checkpoint-ready "$ready" --target-manifest "$target_manifest" --data-root "$data_root" --output "$root/PREFLIGHT_READY.json"; ;;
  smoke-stage0)
    assets; gpu; run smoke --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --compile --data "$data_root/train4k" --raw-replay "$data_root/warmup4k_clm" --output "$root/smoke_stage0"; ;;
  stage0)
    assets; gpu; run stage0 --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --compile --data "$data_root/train4k" --raw-replay "$data_root/warmup4k_clm" --eos-data "$data_root/train_eos4k" --output "$root/stage0"; ;;
  eval-stage0)
    assets; gpu; run eval --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --parent "$root/stage0" --data "$data_root/component_gate4k" --retention "$data_root/retention4k_raw" --output "$root/stage0_eval"; ;;
  gate-stage0)
    CUDA_VISIBLE_DEVICES="" run gate --metrics "$root/stage0_eval" --final-manifest "$data_root/final_validation4k/manifest.json" --gate-kind stage0_gate --output "$root/stage0_gate.json"; ;;
  smoke-stage1-null|smoke-stage1-native-scale|smoke-stage1-exp-negative|smoke-stage1-phase|smoke-stage1-moment)
    assets; gpu; arm="${action#smoke-stage1-}"; key=phase_chord_olmo_r0_lambda_0p1; [[ "$arm" == exp-negative ]] && key=matched_exponential_control; [[ "$arm" == moment ]] && key=moment_matched_same_sign_control; extra=(); [[ "$arm" == moment ]] && extra+=(--gate "$root/selection.json"); run "$action" --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --compile --parent "$root/stage0" --data "$data_root/train16k" --raw-replay "$data_root/warmup4k_clm" --target-manifest "$target_manifest" --target-key "$key" --stage0-gate "$root/stage0_gate.json" "${extra[@]}" --output "$root/smoke_stage1_$arm"; ;;
  stage1-null|stage1-native-scale|stage1-exp-negative|stage1-phase|stage1-moment)
    assets; gpu; arm="${action#stage1-}"; key=phase_chord_olmo_r0_lambda_0p1; [[ "$arm" == exp-negative ]] && key=matched_exponential_control; [[ "$arm" == moment ]] && key=moment_matched_same_sign_control; extra=(); [[ "$arm" == moment ]] && extra+=(--gate "$root/selection.json"); run "$action" --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --compile --parent "$root/stage0" --data "$data_root/train16k" --raw-replay "$data_root/warmup4k_clm" --target-manifest "$target_manifest" --target-key "$key" --stage0-gate "$root/stage0_gate.json" "${extra[@]}" --output "$root/stage1_$arm"; ;;
  eval-stage1-*)
    assets; gpu; arm="${action#eval-stage1-}"; key=phase_chord_olmo_r0_lambda_0p1; [[ "$arm" == exp-negative ]] && key=matched_exponential_control; [[ "$arm" == moment ]] && key=moment_matched_same_sign_control; comparison="$root/stage1_null"; [[ "$arm" == null ]] && comparison="$root/stage0"; run eval --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --parent "$root/stage1_$arm" --comparison "$comparison" --stage0-gate "$root/stage0_gate.json" --data "$data_root/component_gate16k" --retention "$data_root/retention4k_raw" --target-manifest "$target_manifest" --target-key "$key" --output "$root/stage1_${arm}_eval"; ;;
  select-winner)
    "$python" - "$root" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); out={}
for arm in ("native-scale", "exp-negative", "phase"):
    payload=json.loads((r/f"stage1_{arm}_eval").read_text())
    rows=payload.get("per_document_rows", [])
    if len(rows) != 32 or len({str(x.get("document_id")) for x in rows}) != 32:
        raise SystemExit(f"{arm}: expected exactly 32 component documents")
    out[{"native-scale":"native_scale","exp-negative":"context_stretch_exp_negative","phase":"phase_chord"}[arm]]=payload
tmp=r/"tournament_evaluations.json.incomplete"; tmp.write_text(json.dumps(out, sort_keys=True)); tmp.replace(r/"tournament_evaluations.json")
PY
    CUDA_VISIBLE_DEVICES="" run select-winner --metrics "$root/tournament_evaluations.json" --output "$root/selection.json"; ;;
  eval-attribution-phase)
    assets; gpu; run eval --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --parent "$root/stage1_phase" --comparison "$root/stage1_moment" --data "$data_root/component_gate16k" --retention "$data_root/retention4k_raw" --target-manifest "$target_manifest" --target-key phase_chord_olmo_r0_lambda_0p1 --output "$root/phase_vs_moment_eval"; ;;
  gate-attribution)
    CUDA_VISIBLE_DEVICES="" run gate-attribution --metrics "$root/phase_vs_moment_eval" --comparison-metrics "$root/stage1_moment_eval" --output "$root/moment_gate.json"; ;;
  smoke-stage2-null|smoke-stage2-native-scale|smoke-stage2-exp-negative|smoke-stage2-phase|smoke-stage2-moment)
    assets; gpu; arm="${action#smoke-stage2-}"; key=phase_chord_olmo_r0_lambda_0p1; [[ "$arm" == exp-negative ]] && key=matched_exponential_control; [[ "$arm" == moment ]] && key=moment_matched_same_sign_control; extra=(); [[ "$arm" == moment ]] && extra+=(--attribution-gate "$root/moment_gate.json"); run "$action" --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --compile --parent "$root/stage1_$arm" --gate "$root/selection.json" "${extra[@]}" --data "$data_root/train16k" --raw-replay "$data_root/warmup4k_clm" --target-manifest "$target_manifest" --target-key "$key" --output "$root/smoke_stage2_$arm"; ;;
  stage2-null|stage2-native-scale|stage2-exp-negative|stage2-phase|stage2-moment)
    assets; gpu; arm="${action#stage2-}"; key=phase_chord_olmo_r0_lambda_0p1; [[ "$arm" == exp-negative ]] && key=matched_exponential_control; [[ "$arm" == moment ]] && key=moment_matched_same_sign_control; extra=(); [[ "$arm" == moment ]] && extra+=(--attribution-gate "$root/moment_gate.json"); run "$action" --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --compile --parent "$root/stage1_$arm" --gate "$root/selection.json" "${extra[@]}" --data "$data_root/train16k" --raw-replay "$data_root/warmup4k_clm" --eos-data "$data_root/train_eos16k" --target-manifest "$target_manifest" --target-key "$key" --output "$root/stage2_$arm"; ;;
  eval-stage2-*-4k|eval-stage2-*-8k|eval-stage2-*-16k)
    assets; gpu; stem="${action%-*}"; arm="${stem#eval-stage2-}"; length="${action##*-}"; key=phase_chord_olmo_r0_lambda_0p1; [[ "$arm" == exp-negative ]] && key=matched_exponential_control; [[ "$arm" == moment ]] && key=moment_matched_same_sign_control; run eval --authorize --checkpoint "$checkpoint" --checkpoint-ready "$ready" --parent "$root/stage2_$arm" --data "$data_root/final_validation${length}" --retention "$data_root/retention4k_raw" --target-manifest "$target_manifest" --target-key "$key" --output "$root/stage2_${arm}_eval_${length}"; ;;
  downstream-null|downstream-native-scale|downstream-exp-negative|downstream-phase|downstream-moment)
    assets; [[ -d "$ruler_data" && -f "$longbench_zip" ]] || { echo 'prepared RULER root and LongBench zip are required' >&2; exit 2; }; gpu; arm="${action#downstream-}"; external_arm=lora_only_null; [[ "$arm" == native-scale ]] && external_arm=native_scale; [[ "$arm" == exp-negative ]] && external_arm=context_stretch_exp_negative; [[ "$arm" == phase ]] && external_arm=phase_chord; [[ "$arm" == moment ]] && external_arm=moment_matched_same_sign_control; "$python" -m "$downstream_module" --authorize --arm "$external_arm" --checkpoint "$checkpoint" --checkpoint-ready-receipt "$ready" --target-manifest "$target_manifest" --adapter-bundle "$root/stage2_$arm" --natural-gate "$root/stage2_${arm}_eval_4k" "$root/stage2_${arm}_eval_8k" "$root/stage2_${arm}_eval_16k" --natural-data-root "$data_root/final_validation4k" "$data_root/final_validation8k" "$data_root/final_validation16k" --natural-parent-receipt "$root/stage2_$arm/receipt.json" --ruler-data "$ruler_data" --longbench-zip "$longbench_zip" --output "$root/downstream_$arm.json"; ;;
  prepare-ruler)
    assets; [[ -n "${RULER_SOURCE:-}" ]] || exit 2; nohup "$python" -m rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer --ruler-root "$RULER_SOURCE" --checkpoint "$checkpoint" --output "$ruler_data" --lengths 4096 8192 16384 --samples-per-cell 20 --seed 20260822 >"$root/ruler_prepare.log" 2>&1 & echo "ruler preparation pid=$!"; ;;
  *) echo "usage: $0 {derive-target|preflight|smoke-stage0|stage0|eval-stage0|gate-stage0|stage1-*|eval-stage1-*|select-winner|eval-attribution-phase|gate-attribution|stage2-*|eval-stage2-*-{4k,8k,16k}|downstream-*|prepare-ruler}" >&2; exit 2; ;;
esac
