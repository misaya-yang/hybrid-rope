#!/bin/bash
# Local watchdog: reconnect, verify, restart what died, read what finished.
#
# WHY.  The instance has now been disrupted twice in one day (11:29 killed every
# runner within 2 s; 16:0x closed SSH during kex and left the card empty), with no
# OOM, no reboot and no dmesg trace.  `supervise.sh` only heals stages that are
# already running -- if the whole process tree dies, or the host is unreachable
# when a stage would have been restarted, nothing recovers.  This runs LOCALLY,
# outside the instance, so a dead instance cannot take the watchdog with it.
#
# It is idempotent by construction: every action is gated on a row count, and the
# runners skip rows already present, so a restart resumes rather than repeats.
#
# Stages it protects, and what "done" means for each:
#   lb_hold      olmo_lb_h/nu_{p,m}*.jsonl      >= 180 rows each   -> run lb_read.py
#   qwen4x_bm    qwen4x_power/rows.jsonl        has beta_b1_BM    -> run qwen4x_power_read.py
#   gsweep       olmo_gsweep/gain_bm_g1p20.jsonl >= 350 rows      -> run gsweep_read.py
#   amp8x        s8_amp8x/rows.jsonl            >= 288 rows       -> run amp8x_read.py
#                (4 arms x 72; needs tables + reader copied first -- see the sync step)
#
# Usage:  nohup ./keepalive.sh > keepalive.log 2>&1 &
set -u
H="ssh -o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p 27741 [REDACTED_EMAIL]"
D=/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
HOLD=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl
NEWT=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl
ARCH=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01
TBL=/Users/yang/projects/hybrid-rope/ds_workspace/recon_20260910/code/tables

alive=0
for i in $(seq 1 240); do
  if ! $H "echo UP" >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] ssh down (attempt $i)"; sleep 60; continue
  fi
  if [ $alive -eq 0 ]; then echo "[$(date +%H:%M:%S)] === SSH BACK ==="; alive=1; fi

  # GPU GATE.  The instance lost its GPU entirely at 16:38 (nvidia-smi: "No
  # devices were found", /dev/nvidia0 gone, torch.cuda.device_count()==0).
  # Restarting stages in that state only produces immediate CUDA errors, so
  # check first and just report while the card is absent.
  # tr -dc keeps only digits: ssh can emit banners/warnings that would make the
  # value non-numeric and trip `[: integer expression expected`.
  # `nvidia-smi -L` prints "No devices found." on STDOUT (not stderr) when the
  # card is gone, so `wc -l` returns 1 and a naive gate reports the GPU present.
  # Count actual device lines instead.
  GPU=$($H "nvidia-smi -L 2>/dev/null | grep -c '^GPU '" 2>/dev/null | tr -dc '0-9' | head -c 4)
  [ -z "$GPU" ] && GPU=0
  if [ "$GPU" -lt 1 ]; then
    echo "[$(date +%H:%M:%S)] ssh up but NO GPU on the instance -- waiting"
    alive=1; sleep 120; continue
  fi

  # AMPLITUDE-SWEEP TABLES.  The four amp8x tables are literal m-arrays that the
  # runner reaches only via --m-file, so they must exist on the instance before
  # the stage can start.  Gate on a marker so this happens once per session, and
  # verify the file landed rather than trusting scp's exit code (a silent
  # no-write has already cost this campaign a day -- see index lesson 1).
  if ! $H "test -f $D/tables/amp8x_s2p0.json" 2>/dev/null; then
    $H "mkdir -p $D/tables" 2>/dev/null
    scp -P 27741 -q "$TBL"/amp8x_s*.json \
      [REDACTED_EMAIL]:"$D/tables/" 2>&1 | tail -2
    scp -P 27741 -q "$(dirname "$TBL")/../../experiments/zerotrain_20260910/amp8x_read.py" \
      [REDACTED_EMAIL]:"$D/" 2>&1 | tail -2
    # The deployed qwen4x_power_read.py prints the OPPOSITE verdict: its negative
    # branch says "MrRoPE genuinely beats BM" while dd<0 in NLL means BM is better.
    # Verified by diffing against the frozen audit snapshot
    # (audit/pro_decision_20260911/qwen4x_power_read.py), which differs from the
    # fixed copy ONLY in the corrected lines -- so pushing this clobbers no other
    # patch.  Without this, the watchdog auto-prints a wrong verdict.
    scp -P 27741 -q "$(dirname "$TBL")/../../experiments/zerotrain_20260910/qwen4x_power_read.py" \
      [REDACTED_EMAIL]:"$D/" 2>&1 | tail -2
    if $H "test -f $D/tables/amp8x_s2p0.json && test -f $D/amp8x_read.py && grep -q 'LOWER IS BETTER' $D/qwen4x_power_read.py" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] amp8x tables + readers synced (verified: marker line present)"
    else
      echo "[$(date +%H:%M:%S)] amp8x table sync FAILED -- stage will refuse to start"
    fi
  fi

  $H "cd $D
NP=\$(cat olmo_lb_h/nu_p6p104em05.jsonl 2>/dev/null | wc -l | tr -dc '0-9')
NM=\$(cat olmo_lb_h/nu_m6p104em05.jsonl 2>/dev/null | wc -l | tr -dc '0-9')
BM=\$(grep -c 'beta_b1_BM' qwen4x_power/rows.jsonl 2>/dev/null | head -1)
GS=\$(cat olmo_gsweep/gain_bm_g1p20.jsonl 2>/dev/null | wc -l | tr -dc '0-9')
AMP=\$(grep -c 'amp8x' s8_amp8x/rows.jsonl 2>/dev/null | head -1)
PLB=\$(pgrep -fc '^/root/miniconda3/bin/python .*olmo_beta.py' || echo 0)
PQW=\$(pgrep -fc '^/root/miniconda3/bin/python .*qwen_longnll.py' || echo 0)
PGS=\$(pgrep -fc '^/root/miniconda3/bin/python .*olmo_beta.py' || echo 0)
PAMP=\$(pgrep -fc '^/root/miniconda3/bin/python .*olmo_beta.py.*s8_amp8x' || echo 0)
echo \"lb_hold=\$NP/\$NM  qwenBM=\$BM  gsweep=\$GS  amp8x=\$AMP  procs(lb,qw,gs,amp)=\$PLB,\$PQW,\$PGS,\$PAMP\"

# --- lb_hold: the decisive held-out verdict -------------------------------
if [ \"\$NP\" -ge 180 ] && [ \"\$NM\" -ge 180 ]; then
  if [ ! -f olmo_lb_h/.read ]; then
    echo '=== LB HELD-OUT DONE ==='; $PY lb_read.py | tee olmo_lb_h/read.txt; touch olmo_lb_h/.read
  fi
elif [ \"\$PLB\" -eq 0 ]; then
  echo '=== RESTART lb_hold ==='
  PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:$D/repoharness:$D \
  setsid nohup $PY olmo_beta.py --root $D/olmo_lb_h --model $MODEL --panel $HOLD \
    --archive $D/empty_archive --betas '' --turns '' \
    --nu-shift '6.103515625e-05:28,29,30,31;-6.103515625e-05:28,29,30,31' \
    >> lb_hold2.log 2>&1 < /dev/null & disown
fi

# --- qwen 4x BM arm: the last underpowered cell in the leverage table ------
if [ \"\$BM\" -ge 1 ]; then
  if [ ! -f qwen4x_power/.read ]; then
    echo '=== QWEN 4x BM DONE ==='; $PY qwen4x_power_read.py | tee qwen4x_power/read.txt; touch qwen4x_power/.read
  fi
elif [ \"\$PQW\" -eq 0 ]; then
  echo '=== RESTART qwen BM arm ==='
  PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:$D/repoharness:$D \
  setsid nohup $PY qwen_longnll.py --root $D/qwen4x_power \
    --model /root/autodl-tmp/qwen25_1p5b_32k \
    --nll-dir /root/autodl-tmp/longtext/prepared_pg19_4x \
    --far 131073 --tail 512 --only native,beta_b1_BM \
    >> qwen4x_bm.log 2>&1 < /dev/null & disown
fi

# --- gsweep 1.20: the upper bracket of the gain curve ---------------------
if [ \"\$GS\" -ge 350 ]; then
  if [ ! -f olmo_gsweep/.read ]; then
    echo '=== GSWEEP DONE ==='; $PY gsweep_read.py | tee olmo_gsweep/read.txt; touch olmo_gsweep/.read
  fi
elif [ \"\$PGS\" -eq 0 ]; then
  echo '=== RESTART gsweep 1.20 ==='
  PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:$D/repoharness:$D \
  setsid nohup $PY olmo_beta.py --root $D/olmo_gsweep --model $MODEL --panel $NEWT \
    --archive $ARCH --betas '' --turns '' --gain-tables bm --gains 1.20 \
    >> gsweep2.log 2>&1 < /dev/null & disown
fi

# --- amp8x: the amplitude sweep (AMP8X_SWEEP_PREREG) ----------------------
# Decides between three contradictory predictions about where the optimal
# compression amplitude sits at 8x.  4 new arms x 72 rows; s=1.00 is the
# already-measured anchor and is NOT re-run.
if [ \"\$AMP\" -ge 288 ]; then
  if [ ! -f s8_amp8x/.read ]; then
    echo '=== AMP8X DONE ==='; AMP_ROOT=s8_amp8x $PY amp8x_read.py | tee s8_amp8x/read.txt; touch s8_amp8x/.read
  fi
elif [ \"\$PAMP\" -eq 0 ]; then
  if [ ! -f tables/amp8x_s1p5.json ]; then
    echo '=== AMP8X: tables missing in $D/tables -- sync step did not run ==='
  else
  echo '=== START amp8x (4 arms) ==='
  PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:$D/repoharness:$D \
  setsid nohup $PY olmo_beta.py --root $D/s8_amp8x --model $MODEL \
    --panel /root/autodl-tmp/olmo_fast_screen_20260908/prepared_s8_01/screen.jsonl \
    --archive $D/empty_archive --betas '' --turns '' \
    --m-file $D/tables/amp8x_s1p25.json:amp8x_s1p25 \
    --m-file $D/tables/amp8x_s1p5.json:amp8x_s1p5 \
    --m-file $D/tables/amp8x_s1p75.json:amp8x_s1p75 \
    --m-file $D/tables/amp8x_s2p0.json:amp8x_s2p0 \
    >> amp8x.log 2>&1 < /dev/null & disown
  fi
fi
" 2>&1 | grep -v '^$'

  # all three read -> stop
  if $H "test -f $D/olmo_lb_h/.read && test -f $D/qwen4x_power/.read && test -f $D/olmo_gsweep/.read && test -f $D/s8_amp8x/.read" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] === ALL FOUR READ, watch complete ==="; break
  fi
  sleep 120
done
echo "[$(date +%H:%M:%S)] keepalive exiting"
