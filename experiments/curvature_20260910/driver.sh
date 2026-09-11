#!/usr/bin/env bash
# Staged runner for the frequency-allocation solve.
#
# Every stage writes one file under runs/ and is skipped if that file already
# exists, so a re-run after an interruption costs nothing and never silently
# redoes a measurement with a different configuration.  Delete the file to
# force a stage.
#
# Gates stop the chain.  The expensive one is s4: if real forwards do not
# reproduce the local model's own prediction, the local model of the frequency
# response is wrong and no later GPU work is worth releasing.  That is the
# pre-registered stopping rule, not a judgement call made after seeing results.
#
#   bash experiments/curvature_20260910/driver.sh s0        # free, no GPU
#   bash experiments/curvature_20260910/driver.sh s1        # ~5 min
#   bash experiments/curvature_20260910/driver.sh s1b       # free: the veto test
#   bash experiments/curvature_20260910/driver.sh s2        # ~20 min
#   bash experiments/curvature_20260910/driver.sh s3        # seconds, CPU only
#   bash experiments/curvature_20260910/driver.sh s4        # ~6 min, GATE
#   bash experiments/curvature_20260910/driver.sh s5        # seconds, queues a panel job
#   bash experiments/curvature_20260910/driver.sh all       # s0..s5 with gates
#   bash experiments/curvature_20260910/driver.sh status
#
# Cost is estimated from this host's own 640 archived rows (32K ~= 4.1 s,
# 128K ~= 33.9 s median, peak 21-27 GiB); s2 dominates and is the only stage
# that needs the long corpus.  Nothing here trains, and no weights change.

set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

export PATH=/root/miniconda3/bin:$PATH
PY="${PY:-python}"

# ---- paths: override any of these by exporting before invocation -----------
MODEL="${MODEL:-/root/autodl-tmp/rope_qwen_baseline_20260907/model}"
BM="${BM:-/root/autodl-tmp/bm_transfer_20260908}"
HARNESS="${HARNESS:-/root/autodl-tmp/nongeometric_screen_20260909}"
TABLES="${TABLES:-$BM/prepared_qwen3_01/tables.json}"
GT="${GT:-analysis/unify_20260910/tables/ground_truth_tables.json}"

# The probe corpus is the long_document corpus, NOT prepared_nll_01 -- those
# documents are 32769 tokens, so a 131072 request would silently truncate to the
# document length and score an empty tail.  long_inputs/ documents are 131073
# tokens, one contiguous real document prefix with no concatenated filler.
#
# It is also deliberately NOT one of the documents the panel is scored on.  The
# panel's long_eval takes the first docs_per_dataset per source, i.e. pg19
# 28988/30312 and proofpile 001364/001901; this is pg19 37702, held out.  A
# gradient measured on the evaluation documents would be fitting the test set
# one forward at a time.
NPY="${NPY:-$HARNESS/long_inputs/pg19_test_37702.npy}"

RUNS="$HERE/runs"
mkdir -p "$RUNS"
LOG="$RUNS/driver.log"

BASE_TABLE="${BASE_TABLE:-mrpro_n17}"
SLOTS="${SLOTS:-wide}"
EPS="${EPS:-1e-3}"
# NOT 0450.  The queue is FROZEN and pre-registered:
#
#   0446 StackFrontBack -> 0448 MrProN16 -> 0449 MrProN15
#     -> 0450 E1 holdout16 -> 0451 P2 long-end confirmation
#
# (UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md:143; NEXT_DERIVATION_KKT_PROBLEM.md:113
# pins "holdout (0450/0451)").  Those five are the Core-A/B/C verification set and
# 0450/0451 are the HOLDOUT -- the derivation line is explicit that F may not claim
# per-row predictive power before them.  Queueing this package's job under 0450 would
# spend the holdout's namespace on a development-panel job.
#
# 0500 sorts after all five, so if the frozen queue is still in `queue/` when the card
# comes back, those five run first and this package follows them.  That is the correct
# priority: they are already committed, and this package is not.
QUEUE_ID="${QUEUE_ID:-0500}"

log() { printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }

stop_requested() { [[ -f "$RUNS/STOP" ]] && { log "STOP file present; halting before $1"; return 0; }; return 1; }

# a stage is "done" when its output exists and is non-empty valid json
done_already() { [[ -s "$1" ]] && $PY -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null; }

# -1, not 0: SECONDS is 0 at the first stage, so 0 cannot mean "nothing has run
# yet" without also meaning "the last stage ended at t=0".  The sentinel has to
# be outside the range SECONDS can take.
LAST_STAGE_END=-1
run_stage() {
  local name="$1" out="$2"; shift 2
  if done_already "$out"; then log "$name: already done ($out)"; return 0; fi
  stop_requested "$name" && return 2
  # Standing rule: the GPU must not sit idle.  Between stages the only idle is
  # the model load (transformers import alone is ~13 s, plus ~6 GB of weights
  # from disk), which is unavoidable while the stages are separate resumable
  # processes.  What must never happen is a *silent* gap -- an idle card while
  # something is being decided.  So every boundary is timed and logged, and
  # anything over 120 s is called out rather than passing unnoticed.
  if [[ $LAST_STAGE_END -ge 0 ]]; then
    local gap=$((SECONDS - LAST_STAGE_END))
    if [[ $gap -gt 120 ]]; then
      log "WARNING: ${gap}s of wall clock since the previous stage ended -- check"
      log "WARNING: nothing was holding the GPU for that long."
    else
      log "  (${gap}s since previous stage)"
    fi
  fi
  log "$name: starting -> $out"
  local t0=$SECONDS
  if "$@"; then
    LAST_STAGE_END=$SECONDS
    log "$name: done in $((SECONDS - t0))s"
  else
    LAST_STAGE_END=$SECONDS
    log "$name: FAILED after $((SECONDS - t0))s; see the output above"
    return 1
  fi
}

require() { for f in "$@"; do [[ -e "$f" ]] || { log "missing: $f"; return 1; }; done; }

gate() {
  local out="$1" expr="$2" why="$3"
  if $PY -c "
import json,sys
rec=json.load(open(sys.argv[1]))
ok=bool($expr)
print(sys.argv[2], 'PASS' if ok else 'FAIL', json.dumps(rec.get('gates', {})))
sys.exit(0 if ok else 1)" "$out" "$1" ; then
    log "GATE $1: PASS"
  else
    log "GATE $1: FAIL -- $why"
    return 1
  fi
}

# ---------------------------------------------------------------------------
s0() {
  require "$MODEL/config.json" || return 1
  run_stage s0 "$RUNS/preflight.json" \
    $PY -m experiments.curvature_20260910.preflight \
      --model "$MODEL" --gt "$GT" --npy "$NPY" --harness "$HARNESS" \
      --tables "$TABLES" --out "$RUNS/preflight.json"
}

s1() {
  require "$MODEL/config.json" "$NPY" || return 1
  # 64 one-sided forwards at the native length plus 8 pair forwards: no backward
  # pass is taken at any length, which is why this is minutes and not hours.
  run_stage s1 "$RUNS/fisher_${BASE_TABLE}_32k.json" \
    $PY -m experiments.curvature_20260910.local_probe \
      --model "$MODEL" --npy "$NPY" --length 32768 --keep 512 \
      --base-table "$BASE_TABLE" --fisher cross --slots all \
      --out "$RUNS/fisher_${BASE_TABLE}_32k.json"
}

s1b() {
  # Free.  No forwards: this only reads the Fisher s1 just measured plus the
  # panel's own published tables and scores.
  #
  # It exists because the constraint side of this package can be falsified
  # without touching the card, and because `eps` needs a measured scale rather
  # than a taste.  Two things it decides:
  #
  #   * the veto test -- E1_s28_less is a measured pure Pareto point (+5.21pp at
  #     128K, no 32K cost).  If it is INSIDE the budget and the solve later
  #     returns G ~= 0, the linear model missed a step the panel can see and the
  #     solve is falsified by a number measured before it existed.
  #   * cost(MrPro) = D_N(MrPro - native), which is the budget at which "MrPro
  #     is a KKT point" is a claim with content.  An eps far below it does not
  #     constrain around MrPro at all -- it excludes MrPro from the feasible
  #     set, and then the EXPLAINS branch would be asserting optimality about a
  #     point the problem does not contain.
  require "$RUNS/fisher_${BASE_TABLE}_32k.json" "$GT" || return 1
  run_stage s1b "$RUNS/arms_${BASE_TABLE}.json" \
    $PY -m experiments.curvature_20260910.arms \
      --gt "$GT" --fisher "$RUNS/fisher_${BASE_TABLE}_32k.json" \
      --base MrPro --eps "$EPS" --out "$RUNS/arms_${BASE_TABLE}.json" || return 1
  # print the calibrated budget next to the pre-registered one, and say plainly
  # which one s3 is about to use -- the step must be taken at a budget named
  # BEFORE the long gradient is measured, so this is a report, not a re-choice.
  $PY -c "
import json,sys
r=json.load(open(sys.argv[1]))
c=r['base_cost']; e=r['eps']
print()
print(f'  pre-registered eps      = {e:.6g} nats/token (used by s3)')
print(f'  measured cost(MrPro)    = {c:.6g} nats/token = {c/e:.4g} x eps')
if c > 2*e:
    print('  NOTE: eps sits far below what MrRoPE itself spends on native drift, so s3')
    print('        solves inside a ball that does not contain MrRoPE.  Read the solve')
    print('        as \"what is the best direction at THIS budget\", NOT as \"MrRoPE is')
    print('        optimal\".  For the KKT/EXPLAINS reading, re-solve at --eps %.6g' % c)
    print('        and report both; the KKT residual lambda_j is the budget-free half.')
" "$RUNS/arms_${BASE_TABLE}.json"
  return 0
}

s2() {
  # the long side, twice: the bridge band first (asMrRoPE's own support, the
  # highest-information region) and then the wider sample of both bands.
  require "$MODEL/config.json" "$NPY" || return 1
  run_stage s2a "$RUNS/gl_${BASE_TABLE}_128k_bridge.json" \
    $PY -m experiments.curvature_20260910.long_grad \
      --model "$MODEL" --npy "$NPY" --length 131072 --base-table "$BASE_TABLE" \
      --slots bridge --eps 2e-2 --out "$RUNS/gl_${BASE_TABLE}_128k_bridge.json" || return 1
  s2c || return 1
  run_stage s2b "$RUNS/gl_${BASE_TABLE}_128k_wide.json" \
    $PY -m experiments.curvature_20260910.long_grad \
      --model "$MODEL" --npy "$NPY" --length 131072 --base-table "$BASE_TABLE" \
      --slots wide --eps 2e-2 --out "$RUNS/gl_${BASE_TABLE}_128k_wide.json"
}

s2c() {
  # the in-window gradient, for the lambda diagnostic only.  Same code, short
  # prefix: this is d(NLL)/d(log freq) on the distribution the model was
  # trained for, and lambda_j = -g_Lj / g_Nj is the KKT reading of the table.
  require "$MODEL/config.json" "$NPY" || return 1
  run_stage s2c "$RUNS/gn_${BASE_TABLE}_32k.json" \
    $PY -m experiments.curvature_20260910.long_grad \
      --model "$MODEL" --npy "$NPY" --length 32768 --base-table "$BASE_TABLE" \
      --slots "$SLOTS" --eps 2e-2 --out "$RUNS/gn_${BASE_TABLE}_32k.json"
}

s3() {
  local long="$RUNS/gl_${BASE_TABLE}_128k_wide.json"
  [[ -s "$long" ]] || long="$RUNS/gl_${BASE_TABLE}_128k_bridge.json"
  require "$RUNS/fisher_${BASE_TABLE}_32k.json" "$long" || return 1
  run_stage s3 "$RUNS/kkt_${BASE_TABLE}.json" \
    $PY -m experiments.curvature_20260910.solve_kkt \
      --fisher "$RUNS/fisher_${BASE_TABLE}_32k.json" --long "$long" \
      --native-grad "$RUNS/gn_${BASE_TABLE}_32k.json" \
      --base-table "$BASE_TABLE" --slots "$SLOTS" --eps "$EPS" \
      --out "$RUNS/kkt_${BASE_TABLE}.json"
}

s4() {
  require "$RUNS/kkt_${BASE_TABLE}.json" || return 1
  if $PY -c "
import json,sys
sys.exit(0 if json.load(open(sys.argv[1])).get('degenerate') else 1)" "$RUNS/kkt_${BASE_TABLE}.json"; then
    log "s4: SKIPPED -- the KKT solve returned G <= 0. There is no step to check."
    log "s4: this is the EXPLAINS branch: the frozen checkpoint's Fisher admits no"
    log "s4: payable long-range direction at this budget, so MrRoPE's table is a"
    log "s4: KKT point of the stated problem. Write that up; do not force a step."
    return 2
  fi
  run_stage s4 "$RUNS/forward_check_${BASE_TABLE}.json" \
    $PY -m experiments.curvature_20260910.forward_check \
      --model "$MODEL" --kkt "$RUNS/kkt_${BASE_TABLE}.json" --npy "$NPY" \
      --native-length 32768 --long-length 131072 \
      --screen-lengths 8192,32768,65536,131072 \
      --out "$RUNS/forward_check_${BASE_TABLE}.json" || return 1
  gate "$RUNS/forward_check_${BASE_TABLE}.json" \
       "rec['gates']['trust_ratio']['passed'] and rec['gates'].get('native_kl_exponent',{}).get('passed',True)" \
       "the local model does not reproduce its own forward passes. STOP here: the local model of the frequency response is wrong, so no later GPU work is worth releasing. Diagnose in this order: (a) is the native KL exponent off? then F_N is measured outside its quadratic region -- shrink eps. (b) is the ratio low but the exponent fine? then g_L is the problem -- re-measure with --one-sided off and a larger eps. (c) is the ratio high? suspect the baseline loss, not the step." || return 1
  # beats_best_control is deliberately NOT a stopping gate -- a solver that only
  # matches the controls is a positive finding, not a modelling failure: it would
  # mean budget alone decides the outcome and the interior profile is unreadable.
  # It is still worth seeing before committing 25 minutes to the panel, so print it.
  $PY -c "
import json,sys
g=json.load(open(sys.argv[1]))['gates']
c=g.get('beats_best_control',{})
print('  controls:', json.dumps(c))
print('  ' + ('solver beats the best budget-matched control' if c.get('passed')
      else 'SOLVER DOES NOT BEAT THE CONTROLS -- budget alone may decide the outcome; '
           'that is a finding in its own right, and the panel is still worth running '
           'because it tests whether shape matters at the downstream level at all'))
" "$RUNS/forward_check_${BASE_TABLE}.json"
  # explicit: the gate is what decides this stage, not the diagnostics printed
  # after it.  Without this the function would return the exit status of a
  # report-only command and a passing gate could still stop the chain.
  return 0
}

s5() {
  require "$RUNS/kkt_${BASE_TABLE}.json" "$TABLES" || return 1
  stop_requested s5 && return 2
  log "s5: queueing TWO jobs under queue id $QUEUE_ID"
  log "s5:   <id>_<name>.json       -> registers the table, runs the RULER panel,"
  log "s5:                            scores NLL at 8192/16384/32768 against the"
  log "s5:                            archived MrPro rows"
  log "s5:   <id>_<name>_long.json  -> module long_eval: 65536/131072 on the held-out"
  log "s5:                            long corpus, paired against MrPro in one load"
  log "s5: they must run in that order; the worker takes sorted(queue/*.json)[0]."
  $PY -m experiments.curvature_20260910.panel_jobs \
    --kkt "$RUNS/kkt_${BASE_TABLE}.json" --root "$HARNESS" --tables "$TABLES" \
    --queue-id "$QUEUE_ID" --gt "$GT" --receipts "$RUNS/panel_jobs" \
    --panel full --nll-docs 16 --nll-lengths 8192,16384,32768 \
    --long-lengths 65536,131072 --docs-per-dataset 2
}

s6() {
  # escalation, only when the diagonal Fisher was not defensible.  One forward
  # plus T backwards on the retained graph, giving the exact 64x64 PSD matrix
  # so the solver keeps the cross-frequency coupling instead of assuming it away.
  require "$MODEL/config.json" "$NPY" || return 1
  run_stage s6 "$RUNS/fisher_mc_${BASE_TABLE}_32k.json" \
    $PY -m experiments.curvature_20260910.local_probe \
      --model "$MODEL" --npy "$NPY" --length 32768 --keep 512 \
      --base-table "$BASE_TABLE" --fisher mc --samples 128 \
      --out "$RUNS/fisher_mc_${BASE_TABLE}_32k.json"
}

status() {
  echo "runs/ under $RUNS"
  for f in "$RUNS"/*.json; do
    [[ -e "$f" ]] || continue
    $PY - "$f" <<'EOF' 2>/dev/null || echo "  $(basename "$f") (unreadable)"
import json,sys
r=json.load(open(sys.argv[1])); n=sys.argv[1].split('/')[-1]
if 'ALL_PASS' in r: print(f"  {n:44s} gates ALL_PASS={r['ALL_PASS']} trust={r.get('trust_ratio')}")
elif 'G' in r:      print(f"  {n:44s} G={r['G']:.4g} degenerate={r['degenerate']} pred_gain={r['pred_long_gain']:.4g}")
elif 'grad' in r:   print(f"  {n:44s} {len(r['grad'])} slots, base_loss={r.get('base_loss'):.4f}, {r.get('elapsed_seconds',0):.0f}s")
elif 'fisher_diag' in r: print(f"  {n:44s} {len(r['fisher_diag'])} slots, all_at_once={r.get('all_at_once')}")
else:               print(f"  {n:44s} ok")
EOF
  done
  echo
  [[ -f "$RUNS/live.json" ]] && { echo "harness live:"; cat "$RUNS/live.json"; echo; }
  echo "disk: $(df -h "$ROOT" | tail -1)"
}

case "${1:-all}" in
  s0) s0 ;;
  s1) s1 ;;
  s1b) s1b ;;
  s2) s2 ;;
  s2c) s2c ;;
  s3) s3 ;;
  s4) s4 ;;
  s5) s5 ;;
  s6) s6 ;;
  status) status ;;
  all)
    s0 || { log "s0 failed; nothing downstream is interpretable. Fix and re-run."; exit 1; }
    s2c || exit 1
    s1 || exit 1
    s1b || exit 1
    s2 || exit 1
    s3 || exit 1
    if s4; then
      s5 || exit 1
      log "chain complete: the panel job is queued. Start the harness worker with"
      log "  python -m experiments.nongeometric_screen.worker --root $HARNESS --history $BM"
    else
      rc=$?
      if [[ $rc -eq 2 ]]; then log "chain ended at the EXPLAINS branch (see above)"; exit 0; fi
      log "chain stopped at the s4 gate (see above)"; exit 1
    fi ;;
  *) echo "usage: $0 {s0|s1|s1b|s2|s2c|s3|s4|s5|s6|status|all}"; exit 2 ;;
esac
