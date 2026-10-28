#!/bin/bash
# π0.5 SFT for the nine main-table tasks on the relit native60 release.
#
# Same recipe as the official v3 runs (launch_v3.sh / saved train_config):
#   base lerobot/pi05_base, full finetune, seed 1000,
#   AdamW 2.5e-5, warmup 1000, cosine to 2.5e-6 over STEPS,
#   delta v3 (arm relative, base/gripper absolute),
#   n_action_steps=8, tokenizer_max_length=256, gradient_checkpointing=false.
# This run uses GPUs 0-5 only (6 x batch 8 = global 48) and STEPS=30000.
#
# Checkpoints kept: 10000 20000 25000 26000 27000 28000 29000 30000.
# Training uses GPUs 0-5 only. Eval of the last five checkpoints (26000-30000,
# 50 episodes) uses GPUs 6 and 7 while the next task trains. The last task's
# eval waits until training has exited, then uses all 8 GPUs.
# Eval workers on one GPU are staggered: several MuJoCo EGL contexts created
# in the same second raise Xid 31/109 and SIGABRT (rc=134). A failed shard
# is retried on the other eval GPU.
#
# Source pixels are the HF release (relit), not g1_native60.
# Does not write lerobot_v3/ or pi05_runs_v3/.
#
#   PI05_WORK_ROOT=/root/workspace/zecheng bash train_nine_relight_30k.sh
#   DRY_RUN=1 bash train_nine_relight_30k.sh
#
# save_freq stays 1000 so those steps exist; other checkpoints are removed
# once a newer one has been written. KEEP_ALL=1 disables that prune.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -n "${PI05_WORK_ROOT:-}" ]; then
  WORK=$PI05_WORK_ROOT
elif [ -d "$HERE/../lerobot_v3" ]; then
  WORK=$(cd "$HERE/.." && pwd)
else
  WORK=$(cd "$HERE/../.." && pwd)
fi
export PI05_WORK_ROOT=$WORK
PY=${PI05_PYTHON:-${PI05_PY:-$WORK/.venv-pi05/bin/python}}
LAUNCH=$HERE/launch_v3.sh
EXPORT=$HERE/export_v3.py
DELTA=$HERE/make_delta_v3.py

NATIVE=${PI05_NATIVE_ROOT:-$WORK/hf_datasets/bigym-g1-native60}
LEROBOT=${PI05_LEROBOT_ROOT:-$WORK/lerobot_v3_relight}
RUNROOT=${RUNROOT:-$WORK/pi05_runs_v3_relight30k}
export PI05_NATIVE_ROOT=$NATIVE
export PI05_LEROBOT_ROOT=$LEROBOT

STEPS=${STEPS:-30000}
SAVE=${SAVE:-1000}
GPUS=${GPUS:-0,1,2,3,4,5}
BATCH=${BATCH:-8}
KEEP_STEPS=${KEEP_STEPS:-"10000 20000 25000 26000 27000 28000 29000 30000"}
KEEP_ALL=${KEEP_ALL:-0}
DRY_RUN=${DRY_RUN:-0}
EVAL_EPS=${EVAL_EPS:-50}
EVAL_WPC=${EVAL_WPC:-4}
EVAL_PER_GPU=${EVAL_PER_GPU:-2}
EVAL_RETRIES=${EVAL_RETRIES:-3}
EVAL_STAGGER_S=${EVAL_STAGGER_S:-20}
# Eval never shares a GPU with a live training job. 6 and 7 are the eval pair
# until the last task, which uses every GPU after training has exited.
EVAL_GPUS=${EVAL_GPUS:-6,7}
LAST_EVAL_GPUS=${LAST_EVAL_GPUS:-0,1,2,3,4,5,6,7}
# Rolling prune keeps about 8 checkpoints. One extra in-flight save is ~23GB.
MIN_FREE_GB=${MIN_FREE_GB:-200}

# Shortest demos first so an export/layout failure shows up before the long tasks.
TASKS=${TASKS:-"reach_target_multi_modal reach_target_dual drawer_top_close drawer_top_open move_plate dishwasher_load_cups move_two_plates pick_box dishwasher_close"}

LOGDIR=$WORK/pi05_logs
mkdir -p "$LOGDIR" "$LEROBOT" "$RUNROOT"
LOG=$LOGDIR/train_nine_relight_30k.log
LOCK=$LOGDIR/train_nine_relight_30k.lock
EVAL_ROOT=${EVAL_ROOT:-$WORK/pi05_eval_v3/relight30k_last5}
EVAL=$HERE/eval_v3.py

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[relight30k] already running (lock $LOCK)"
  exit 0
fi

stamp () { date -u +%FT%TZ; }
log () { echo "[relight30k $(stamp)] $*" | tee -a "$LOG"; }
fail () { log "FAIL: $*"; exit 1; }

free_gb () {
  df -BG --output=avail "$WORK" | awk 'NR==2 {gsub(/G/,"",$1); print $1}'
}

final_ckpt () { printf '%06d' "$STEPS"; }

dataset_ready () {
  local task=$1
  "$PY" - "$NATIVE" "$task" << 'PY'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
native, task = Path(sys.argv[1]), sys.argv[2]
info_p = native / task / "meta" / "info.json"
if not info_p.is_file():
    raise SystemExit(f"missing {info_p}")
info = json.loads(info_p.read_text())
want = int(info["total_episodes"]), int(info["total_frames"])
if want[0] != 60:
    raise SystemExit(f"{task}: total_episodes={want[0]}, expected 60")
got = 0
for p in sorted((native / task / "data").rglob("*.parquet")):
    got += pq.ParquetFile(p).metadata.num_rows
if got != want[1]:
    raise SystemExit(f"{task}: parquet rows {got} != total_frames {want[1]}")
print(f"{task} frames={got}")
PY
}

export_task () {
  local task=$1
  local marker=$LEROBOT/${task}_delta/EXPORT_OK
  local frames
  frames=$("$PY" -c "import json; print(json.load(open('$NATIVE/$task/meta/info.json'))['total_frames'])")
  if [ -f "$marker" ] && [ "$(cat "$marker")" = "$frames" ]; then
    log "export SKIP $task ($frames frames already at $LEROBOT/${task}_delta)"
    return 0
  fi
  log "export $task from $NATIVE"
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN export PI05_NATIVE_ROOT=$NATIVE PI05_LEROBOT_ROOT=$LEROBOT $PY $EXPORT $task && $PY $DELTA $task"
    return 0
  fi
  "$PY" -u "$EXPORT" "$task" >> "$LOG" 2>&1 || fail "export_v3 $task"
  "$PY" -u "$DELTA" "$task" >> "$LOG" 2>&1 || fail "make_delta_v3 $task"
  echo "$frames" > "$marker"
}

prune_ckpts () {
  local out=$1
  local final_pass=${2:-0}
  [ "$KEEP_ALL" = "1" ] && return 0
  [ -d "$out/checkpoints" ] || return 0
  "$PY" - "$out" "$KEEP_STEPS" "$final_pass" << 'PY'
import shutil, sys
from pathlib import Path
root = Path(sys.argv[1]) / "checkpoints"
keep = {f"{int(x):06d}" for x in sys.argv[2].split()}
final_pass = sys.argv[3] == "1"
complete = []
for p in root.iterdir():
    if p.is_dir() and p.name.isdigit() and (p / "pretrained_model").is_dir():
        complete.append(p)
complete.sort(key=lambda p: int(p.name))
protect = set()
if not final_pass and complete:
    protect.add(complete[-1].name)
last = root / "last"
if last.is_symlink() or last.exists():
    try:
        protect.add(last.resolve().name)
    except Exception:
        pass
dropped = []
for p in complete:
    if p.name in keep or p.name in protect:
        continue
    shutil.rmtree(p)
    dropped.append(p.name)
left = sorted(x.name for x in root.iterdir() if x.is_dir() and x.name.isdigit())
print(f"pruned {dropped or '-'} kept {left}")
PY
}

wait_train () {
  local out=$1
  local final
  final=$(final_ckpt)
  sleep 90
  local quiet=0
  while true; do
    prune_ckpts "$out" 0 >> "$LOG" || true
    if pgrep -f 'lerobot_trai[n]' >/dev/null; then
      quiet=0
    else
      quiet=$((quiet + 1))
      if [ -d "$out/checkpoints/$final/pretrained_model" ] && [ "$quiet" -ge 2 ]; then
        return 0
      fi
      if [ "$quiet" -ge 3 ]; then
        return 1
      fi
    fi
    sleep 60
  done
}

train_task () {
  local task=$1
  local dataset=$LEROBOT/${task}_delta
  local out=$RUNROOT/${task}_delta
  local final
  final=$(final_ckpt)
  if [ -d "$out/checkpoints/$final/pretrained_model" ]; then
    log "train SKIP $task already has $final"
    prune_ckpts "$out" 1 | tee -a "$LOG"
    return 0
  fi
  if [ ! -d "$dataset" ]; then
    if [ "$DRY_RUN" = "1" ]; then
      log "DRY_RUN dataset not built yet: $dataset"
      return 0
    fi
    fail "missing dataset $dataset"
  fi
  if pgrep -af 'lerobot.scripts.lerobot_train' | grep -F -- "$out" >/dev/null; then
    log "train ADOPT $task already running on $GPUS"
    wait_train "$out" || fail "adopted train died before checkpoint $final ($out)"
    [ -d "$out/checkpoints/$final/pretrained_model" ] || fail "missing $out/checkpoints/$final"
    log "train DONE $task $final"
    prune_ckpts "$out" 1 | tee -a "$LOG"
    return 0
  fi
  while pgrep -f 'lerobot.scripts.lerobot_trai[n]' >/dev/null; do
    log "waiting for another lerobot_train to exit before $task"
    sleep 30
  done
  local resume=0
  if [ -d "$out/checkpoints/last/pretrained_model" ] || [ -L "$out/checkpoints/last" ]; then
    resume=1
  fi
  local free
  free=$(free_gb)
  log "train $task resume=$resume steps=$STEPS save=$SAVE gpus=$GPUS batch=$BATCH free=${free}G out=$out"
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    fail "only ${free}G free, need ${MIN_FREE_GB}G before a 30k run"
  fi
  if [ "$DRY_RUN" = "1" ]; then
    DATASET=$dataset OUT=$out JOB=pi05_v3_${task}_relight30k \
      TASK=$task VARIANT=delta GPUS=$GPUS BATCH=$BATCH STEPS=$STEPS SAVE=$SAVE \
      GRAD_CKPT=false RESUME=$resume DRY_RUN=1 \
      bash -c 'exec 9>&-; exec "$0" "$@"' "$LAUNCH" | tee -a "$LOG"
    return 0
  fi
  DATASET=$dataset OUT=$out JOB=pi05_v3_${task}_relight30k \
    TASK=$task VARIANT=delta GPUS=$GPUS BATCH=$BATCH STEPS=$STEPS SAVE=$SAVE \
    GRAD_CKPT=false RESUME=$resume \
    bash -c 'exec 9>&-; exec "$0" "$@"' "$LAUNCH" | tee -a "$LOG" || fail "launch_v3 $task"
  wait_train "$out" || fail "train died before checkpoint $final ($out)"
  [ -d "$out/checkpoints/$final/pretrained_model" ] || fail "missing $out/checkpoints/$final"
  log "train DONE $task $final"
  prune_ckpts "$out" 1 | tee -a "$LOG"
}

eval_last5 () {
  local task=$1
  local use_gpus=$EVAL_GPUS
  local allow_train_gpus=0
  if [ "$task" = "${TASKS##* }" ]; then
    use_gpus=$LAST_EVAL_GPUS
    allow_train_gpus=1
  fi
  local run=$RUNROOT/${task}_delta
  local dest=$EVAL_ROOT/$task
  local prompt
  prompt=$("$PY" - "$LEROBOT/${task}_delta/meta/tasks.parquet" << 'PY'
import sys
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
col = "task" if "task" in t.column_names else t.column_names[0]
print(t[col][0].as_py())
PY
)
  [ -n "$prompt" ] || fail "empty prompt for $task"
  case "$prompt" in
    "Move the plate between two draining racks.")
      if [ "$task" != "move_plate" ]; then
        fail "wrong prompt for $task: $prompt"
      fi
      ;;
  esac
  local steps=()
  local s
  for s in $KEEP_STEPS; do steps+=("$(printf '%06d' "$s")"); done
  local n=${#steps[@]}
  local last5=("${steps[@]:$((n-5)):5}")
  local step
  for step in "${last5[@]}"; do
    if [ ! -d "$run/checkpoints/$step/pretrained_model" ]; then
      log "FAIL: eval missing $run/checkpoints/$step"
      return 1
    fi
  done
  log "eval $task last5=${last5[*]} eps=$EVAL_EPS gpus=$use_gpus prompt=$(printf '%s' "$prompt" | head -c 80)"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  mkdir -p "$EVAL_ROOT" "$dest"
  local blfile=$EVAL_ROOT/blacklist_gpus.txt
  touch "$blfile"
  shard_ok () {
    "$PY" - "$1" "$task" << 'PY'
import json, sys
p, task = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p))
except Exception:
    raise SystemExit(1)
if not d.get("episodes"):
    raise SystemExit(1)
if task != "move_plate" and d.get("prompt") == "Move the plate between two draining racks.":
    raise SystemExit(1)
PY
  }
  run_eval_job () {
    exec 9>&-
    set -- $1
    local step=$1 wid=$2 gpu=$3
    local outdir="$dest/step_$step"
    mkdir -p "$outdir"
    if [ -s "$outdir/w$wid.json" ] && shard_ok "$outdir/w$wid.json"; then
      return 0
    fi
    rm -f "$outdir/w$wid.json"
    local delay=${4:-0}
    if [ "${ALLOW_TRAIN_GPUS:-0}" != "1" ] || pgrep -f 'lerobot.scripts.lerobot_trai[n]' >/dev/null; then
      case ",${GPUS}," in
        *,"$gpu",*) echo "REFUSE eval on train gpu $gpu while training" >> "$outdir/w$wid.log"; return 0 ;;
      esac
    fi
    sleep "$delay"
    env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" \
      OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PI05_WORK_ROOT="$WORK" \
      "$PY" -u "$EVAL" --task "$task" \
      --ckpt "$run/checkpoints/$step/pretrained_model" \
      --out "$outdir/w$wid.json" --episodes "$EVAL_EPS" --num-workers "$EVAL_WPC" \
      --worker-id "$wid" --n-action-steps 16 --seed-base 620000 \
      --prompt "$prompt" > "$outdir/w$wid.log" 2>&1
    local rc=$?
    echo "EXITSTATUS step_$step/w$wid gpu$gpu rc=$rc $(date -u +%FT%TZ)" >> "$outdir/w$wid.log"
    if [ "$rc" != "0" ]; then
      rm -f "$outdir/w$wid.json"
    fi
    return 0
  }
  export -f shard_ok run_eval_job
  export dest run EVAL EVAL_EPS EVAL_WPC PY WORK task prompt GPUS ALLOW_TRAIN_GPUS=$allow_train_gpus
  local pass missing=0
  for pass in $(seq 1 "$EVAL_RETRIES"); do
    local -a pool=()
    local gpu
    IFS=',' read -ra G <<< "$use_gpus"
    for gpu in "${G[@]}"; do
      if grep -qx "$gpu" "$blfile"; then
        continue
      fi
      pool+=("$gpu")
    done
    [ "${#pool[@]}" -gt 0 ] || fail "no eval GPUs left after blacklist ($blfile)"
    local q=$LOGDIR/relight30k_eval_$task
    rm -rf "$q"
    mkdir -p "$q"
    local i njob=0 step
    for step in "${last5[@]}"; do
      for i in $(seq 0 $((EVAL_WPC-1))); do
        if shard_ok "$dest/step_$step/w$i.json"; then
          continue
        fi
        gpu=${pool[$((njob % ${#pool[@]}))]}
        local wave=$((njob / ${#pool[@]}))
        echo "$step $i $gpu $((wave * EVAL_STAGGER_S))" >> "$q/jobs.txt"
        njob=$((njob+1))
      done
    done
    if [ "$njob" = "0" ]; then
      missing=0
      break
    fi
    log "eval pass $pass/$EVAL_RETRIES $task shards=$njob gpus=${pool[*]}"
    xargs -a "$q/jobs.txt" -d "\n" -P $((${#pool[@]} * EVAL_PER_GPU)) -I{} bash -c 'run_eval_job "{}"'
    local bad gpu hits
    bad=$(grep -h 'rc=134' "$dest"/step_*/w*.log 2>/dev/null | sed -n 's/.* gpu\([0-9][0-9]*\) rc=134.*/\1/p' | sort -u || true)
    for gpu in $bad; do
      echo "$gpu" >> "$EVAL_ROOT/rc134_count.txt"
      hits=$(grep -cx "$gpu" "$EVAL_ROOT/rc134_count.txt" || true)
      if [ "$hits" -ge 2 ] && ! grep -qx "$gpu" "$blfile"; then
        echo "$gpu" >> "$blfile"
        log "eval blacklist gpu $gpu after $hits rc=134 (EGL/Xid)"
      else
        log "eval gpu $gpu rc=134 ($hits); shard will move to another eval GPU"
      fi
    done
    missing=0
    for step in "${last5[@]}"; do
      for i in $(seq 0 $((EVAL_WPC-1))); do
        if shard_ok "$dest/step_$step/w$i.json"; then
          continue
        fi
        log "eval incomplete $task step_$step/w$i"
        missing=$((missing+1))
      done
    done
    [ "$missing" = "0" ] && break
    sleep 20
  done
  [ "$missing" = "0" ] || { log "FAIL: eval $task missing $missing shards after $EVAL_RETRIES passes"; return 1; }
  "$PY" - "$dest" "${last5[*]}" << 'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
dest = Path(sys.argv[1])
steps = sys.argv[2].split()
ok = n = 0
for step in steps:
    sok = sn = 0
    for p in sorted((dest / f"step_{step}").glob("w*.json")):
        d = json.loads(p.read_text())
        for ep in d.get("episodes") or []:
            sn += 1
            sok += int(bool(ep.get("success")))
    ok += sok
    n += sn
    rate = 100.0 * sok / sn if sn else 0
    print(f"  {step} {sok}/{sn} = {rate:.1f}%")
rate = 100.0 * ok / n if n else 0
print(f"EVAL_LAST5 {ok}/{n} = {rate:.1f}%")
PY
}

# Eval GPUs and train GPUs must be disjoint. A shared device is how the
# EGL abort landed on a card that training also needs.
overlap=$("$PY" - "$GPUS" "$EVAL_GPUS" << 'PY'
import sys
train = {x for x in sys.argv[1].split(",") if x != ""}
ev = {x for x in sys.argv[2].split(",") if x != ""}
print(" ".join(sorted(train & ev, key=int)))
PY
)
[ -z "$overlap" ] || fail "eval GPUs overlap train GPUs: $overlap"

EVAL_QUEUE=$LOGDIR/relight30k_eval.queue
EVAL_QLOCK=$LOGDIR/relight30k_eval.qlock
EVAL_STOP=$LOGDIR/relight30k_eval.stop
rm -f "$EVAL_QUEUE" "$EVAL_STOP"
touch "$EVAL_QLOCK" "$EVAL_ROOT/rc134_count.txt"
: > "$EVAL_ROOT/rc134_count.txt"

enqueue_eval () {
  flock "$EVAL_QLOCK" bash -c 'printf "%s\n" "$1" >> "$2"' _ "$1" "$EVAL_QUEUE"
}

eval_consumer () {
  while true; do
    local task=""
    task=$(flock "$EVAL_QLOCK" bash -c '
      f="$1"
      if [ ! -s "$f" ]; then exit 0; fi
      head -n 1 "$f"
      tail -n +2 "$f" > "$f.next"
      mv "$f.next" "$f"
    ' _ "$EVAL_QUEUE" || true)
    if [ -n "$task" ]; then
      if [ "$task" = "${TASKS##* }" ]; then
        while pgrep -f 'lerobot.scripts.lerobot_trai[n]' >/dev/null; do
          log "last eval $task waits until training releases GPUs 0-5"
          sleep 30
        done
      fi
      eval_last5 "$task" || log "eval $task returned failure; training continues"
      continue
    fi
    if [ -f "$EVAL_STOP" ]; then
      break
    fi
    sleep 10
  done
}

log "START native=$NATIVE lerobot=$LEROBOT runs=$RUNROOT steps=$STEPS save=$SAVE train_gpus=$GPUS batch=$BATCH keep='$KEEP_STEPS' eval_gpus=$EVAL_GPUS eval_per_gpu=$EVAL_PER_GPU stagger=${EVAL_STAGGER_S}s dry_run=$DRY_RUN"
log "TASKS=$TASKS"
if [ "$DRY_RUN" != "1" ]; then
  eval_consumer &
  EVAL_CONSUMER_PID=$!
  log "eval consumer pid=$EVAL_CONSUMER_PID gpus=$EVAL_GPUS"
fi
for task in $TASKS; do
  dataset_ready "$task" | tee -a "$LOG" || fail "dataset not ready: $task"
  export_task "$task"
  train_task "$task"
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN would enqueue eval $task on $EVAL_GPUS"
  else
    enqueue_eval "$task"
    if [ "$task" = "${TASKS##* }" ]; then
      log "eval queued $task on $LAST_EVAL_GPUS after training exits"
    else
      log "eval queued $task on $EVAL_GPUS (training does not wait)"
    fi
  fi
done
if [ "$DRY_RUN" != "1" ]; then
  touch "$EVAL_STOP"
  log "training finished; waiting for eval consumer $EVAL_CONSUMER_PID"
  wait "$EVAL_CONSUMER_PID" || log "eval consumer exited nonzero"
fi
log "ALL_DONE steps=$STEPS tasks=$TASKS"
echo "RELIGHT30K_DONE $(stamp)" | tee -a "$LOG"
