#!/usr/bin/env bash
# Driver for the success-first zero-training tournament.
#
#   freeze     CPU: bind R0 + learned/budgeted tables, freeze the portfolio and
#              the selection rule.  No authorization.
#   splits     CPU/tokenizer: materialise the firewall-disjoint D/S/T splits.
#   manifest   CPU: assemble the evaluator candidate manifest (F1, plus any
#              development representatives passed via DEV_RECEIPTS).
#   contract   CPU: identity-check the candidate manifest, tables and rows.
#   parity     GPU: 1x-vs-4x-prefix parity smoke on the Native table.
#   dev        GPU: per-family development on D (f1|f2|f3|f4).
#   select     GPU: score the frozen representatives on S and pick one winner.
#   confirm    GPU: open T once for the global winner and issue one verdict.
#   capability GPU: RULER capability confirmation for a winning table (R3).
#
# Every GPU stage needs BOTH the stage authorization variable and
# ZERO_TRAINING_TOURNAMENT_GPU_AUTHORIZED=YES, so no stage can start paid
# compute by accident and each stage is authorized separately.
set -euo pipefail

MODE="${1:-}"; shift || true

ROOT="${EVQ_REPO_ROOT:?set EVQ_REPO_ROOT}"
PYTHON="${EVQ_PYTHON:-/root/miniconda3/bin/python}"
ZT_ROOT="${EVQ_ZT_ROOT:?set EVQ_ZT_ROOT}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PORTFOLIO="$ZT_ROOT/portfolio"
SPLITS="$ZT_ROOT/splits"
mkdir -p "$ZT_ROOT"

require_gpu_stage() {
  local stage_var="$1"
  if [[ "${!stage_var:-}" != "YES" ]]; then
    echo "set ${stage_var}=YES after explicit ${stage_var%_AUTHORIZED} authorization" >&2
    exit 3
  fi
  export ZERO_TRAINING_TOURNAMENT_GPU_AUTHORIZED=YES
}

case "$MODE" in
  freeze)
    exec "$PYTHON" "$ROOT/scripts/analysis/freeze_success_first_portfolio.py" \
      --r0-collection "${EVQ_R0_COLLECTION:?set EVQ_R0_COLLECTION}" \
      ${EVQ_R0_SHA256:+--r0-expected-sha256 "$EVQ_R0_SHA256"} \
      --learned-table "${EVQ_LEARNED_TABLE:?set EVQ_LEARNED_TABLE}" \
      --budgeted-table "${EVQ_BUDGETED_TABLE:?set EVQ_BUDGETED_TABLE}" \
      --output "$PORTFOLIO"
    ;;

  splits)
    PRIOR_ARGS=()
    for m in ${EVQ_PRIOR_MANIFESTS:-}; do PRIOR_ARGS+=(--prior-manifest "$m"); done
    EXCL_ARGS=()
    for f in ${EVQ_EXCLUDE_FILES:-}; do EXCL_ARGS+=(--exclude-text-sha256-file "$f"); done
    exec "$PYTHON" "$ROOT/scripts/data/build_success_first_splits.py" \
      --source "${EVQ_FINEWEB_SOURCE:?set EVQ_FINEWEB_SOURCE}" \
      --expected-source-sha256 "${EVQ_FINEWEB_SHA256:?set EVQ_FINEWEB_SHA256}" \
      --checkpoint "${EVQ_OLMO_CHECKPOINT:?set EVQ_OLMO_CHECKPOINT}" \
      "${PRIOR_ARGS[@]}" "${EXCL_ARGS[@]}" \
      ${EVQ_SKIP_ELIGIBLE:+--skip-eligible-documents "$EVQ_SKIP_ELIGIBLE"} \
      --output "$SPLITS"
    ;;

  manifest)
    DEV_ARGS=()
    for r in ${DEV_RECEIPTS:-}; do DEV_ARGS+=(--dev-receipt "$r"); done
    exec "$PYTHON" "$ROOT/scripts/analysis/build_candidate_manifest.py" \
      --portfolio-manifest "$PORTFOLIO/portfolio_manifest.json" \
      "${DEV_ARGS[@]}" \
      ${MANIFEST_FAMILIES:+--families $MANIFEST_FAMILIES} \
      --output "${MANIFEST_OUT:-$ZT_ROOT/candidates.json}"
    ;;

  contract)
    exec "$PYTHON" "$ROOT/scripts/eval/eval_zero_training_tournament.py" \
      --contract \
      --candidates "${CANDIDATES:-$ZT_ROOT/candidates.json}" \
      --rows "${ROWS:-$SPLITS/rows_D.jsonl}" \
      --output "${OUT:-$ZT_ROOT/contract}"
    ;;

  parity)
    require_gpu_stage ZT_PARITY_AUTHORIZED
    exec "$PYTHON" "$ROOT/scripts/eval/eval_zero_training_tournament.py" \
      --parity-smoke --authorize \
      --candidates "${CANDIDATES:-$ZT_ROOT/candidates.json}" \
      --rows "${ROWS:-$SPLITS/rows_D.jsonl}" \
      --checkpoint "${EVQ_OLMO_CHECKPOINT:?set EVQ_OLMO_CHECKPOINT}" \
      --output "${OUT:-$ZT_ROOT/parity}"
    ;;

  dev)
    require_gpu_stage ZT_DEV_AUTHORIZED
    FAMILY="${1:?usage: $0 dev f1|f2|f3|f4 [--initialize-from <f1_winner.npy>]}"
    shift || true
    case "$FAMILY" in
      f1)
        # F1 is a frozen grid: score it on D, then reduce by the selection rule.
        exec "$PYTHON" "$ROOT/scripts/eval/eval_zero_training_tournament.py" \
          --evaluate --authorize \
          --candidates "${CANDIDATES:-$ZT_ROOT/candidates.json}" \
          --rows "$SPLITS/rows_D.jsonl" \
          --checkpoint "$EVQ_OLMO_CHECKPOINT" \
          --output "$ZT_ROOT/dev_F1"
        ;;
      f2|f3|f4)
        INIT_ARGS=()
        if [[ "${1:-}" == "--initialize-from" ]]; then INIT_ARGS+=(--initialize-from "$2"); fi
        exec "$PYTHON" "$ROOT/scripts/eval/develop_zero_training_family.py" \
          --family "$(echo "$FAMILY" | tr '[:lower:]' '[:upper:]')" --authorize \
          --portfolio-manifest "$PORTFOLIO/portfolio_manifest.json" \
          --rows "$SPLITS/rows_D.jsonl" \
          --checkpoint "$EVQ_OLMO_CHECKPOINT" \
          "${INIT_ARGS[@]}" \
          --output "$ZT_ROOT/dev_$(echo "$FAMILY" | tr '[:lower:]' '[:upper:]')"
        ;;
      *) echo "unknown family: $FAMILY" >&2; exit 2 ;;
    esac
    ;;

  select)
    require_gpu_stage ZT_SELECT_AUTHORIZED
    exec "$PYTHON" "$ROOT/scripts/eval/eval_zero_training_tournament.py" \
      --evaluate --authorize \
      --candidates "${CANDIDATES:-$ZT_ROOT/candidates_full.json}" \
      --rows "$SPLITS/rows_S.jsonl" \
      --checkpoint "$EVQ_OLMO_CHECKPOINT" \
      --output "$ZT_ROOT/select"
    ;;

  confirm)
    require_gpu_stage ZT_CONFIRM_AUTHORIZED
    exec "$PYTHON" "$ROOT/scripts/eval/eval_zero_training_tournament.py" \
      --evaluate --authorize \
      --candidates "${CANDIDATES:-$ZT_ROOT/candidates_winner.json}" \
      --rows "$SPLITS/rows_T.jsonl" \
      --checkpoint "$EVQ_OLMO_CHECKPOINT" \
      --output "$ZT_ROOT/confirm"
    ;;

  capability)
    require_gpu_stage ZT_CAPABILITY_AUTHORIZED
    TABLE="${1:?usage: $0 capability <table.npy> <name>}"; NAME="${2:?usage: $0 capability <table.npy> <name>}"
    [[ -n "${EVQ_RULER_DATA:-}" ]] || { echo "set EVQ_RULER_DATA" >&2; exit 5; }
    for LENGTH in 8192 16384; do
      "$PYTHON" "$ROOT/scripts/eval/target_free_ruler_smoke.py" \
        --checkpoint "$EVQ_OLMO_CHECKPOINT" \
        --data-root "$EVQ_RULER_DATA" \
        --method external_table_static \
        --table "$TABLE" \
        --table-name "$NAME" \
        --table-support native \
        --table-factor 4.0 \
        --long-attention-scaling 1.0 \
        --lengths "$LENGTH" \
        --limit-per-cell 20 \
        --native-context-length 4096 \
        --output "$ZT_ROOT/capability_${NAME}_${LENGTH}"
    done
    ;;

  *)
    echo "usage: $0 {freeze|splits|manifest|contract|parity|dev|select|confirm|capability}" >&2
    exit 2
    ;;
esac
