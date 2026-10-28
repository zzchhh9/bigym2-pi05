#!/bin/bash
# pi0.5 SFT on a v3 task, OpenPI-style relative (delta) or absolute actions.
# The defaults are the paper's relight30k recipe: 6 GPUs x batch 8 = global 48,
# 30000 steps, no gradient checkpointing.
#   TASK=move_plate bash scripts/launch_v3.sh
#
# RESUME=1  continue OUT/checkpoints/last up to STEPS (optimizer + scheduler).
# DRY_RUN=1 print the argv and exit 0; does not start torchrun.
#
# PI05_WORK_ROOT  data/run root (lerobot_v3/, pi05_runs_v3/, pi05_logs/);
#                 default <repo>/work.
# PI05_PYTHON     interpreter with lerobot and bigym; default `python`.
# PI05_BASE       base checkpoint; default lerobot/pi05_base from the Hub.
set -euo pipefail
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${PI05_WORK_ROOT:-$(cd "$HERE/.." && pwd)/work}
PY=${PI05_PYTHON:-python}
BASE=${PI05_BASE:-lerobot/pi05_base}

TASK=${TASK:?set TASK}; VARIANT=${VARIANT:-delta}
GPUS=${GPUS:-0,1,2,3,4,5}; NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
BATCH=${BATCH:-8}                 # per GPU; 6 x 8 = global 48 in the paper
STEPS=${STEPS:-30000}; SAVE=${SAVE:-1000}; LOGF=${LOGF:-200}
GRAD_CKPT=${GRAD_CKPT:-false}; WORKERS=${WORKERS:-6}
RESUME=${RESUME:-0}; DRY_RUN=${DRY_RUN:-0}
DATASET=${DATASET:-$WORK/lerobot_v3/${TASK}_${VARIANT}}
OUT=${OUT:-$WORK/pi05_runs_v3/${TASK}_${VARIANT}}
JOB=${JOB:-pi05_v3_${TASK}_${VARIANT}}

[ -d "$DATASET" ] || { echo "no dataset $DATASET" >&2; exit 1; }
mkdir -p "$WORK/pi05_logs" "$(dirname "$OUT")"
LOG=$WORK/pi05_logs/${JOB}_$(date -u +%Y%m%dT%H%M%SZ).log
if [ "${DRY_RUN:-0}" != "1" ]; then
  echo "$LOG" > "$WORK/pi05_logs/latest_${JOB}"
fi

if [ "$RESUME" = "1" ]; then
  CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"
  if [ ! -f "$CFG" ]; then
    last=$(ls -d "$OUT"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1 || true)
    CFG="${last:-}/pretrained_model/train_config.json"
  fi
  [ -f "$CFG" ] || { echo "no checkpoint to resume: $CFG" >&2; exit 1; }
  # Config from the checkpoint; --steps is the TARGET total (not remaining).
  # dp_replicate must match nproc: a 1-GPU ckpt's train_config keeps dp=1,
  # which would spawn 8 ranks as one process.
  ARGS=(--config_path="$CFG" --resume=true --steps="$STEPS" --save_freq="$SAVE"
        --parallelism.dp_replicate="$NPROC")
else
  ARGS=(
    --dataset.repo_id="local/$JOB" --dataset.root="$DATASET"
    --policy.type=pi05 --policy.pretrained_path=$BASE
    # Kept as run for the paper, but it has NO effect here: with a
    # pretrained_path, lerobot_train loads the tokenizer step from the base
    # checkpoint's policy_preprocessor.json (max_length 200) and does not
    # override it. Every paper checkpoint saved max_length 200, so the
    # discretised 50-D/56-D state prompt was truncated in training and eval.
    --policy.tokenizer_max_length=256
    --policy.n_action_steps=8
    --policy.optimizer_lr=2.5e-05
    --policy.scheduler_warmup_steps=1000
    --policy.scheduler_decay_steps=$STEPS
    --policy.scheduler_decay_lr=2.5e-06
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false
    --policy.gradient_checkpointing=$GRAD_CKPT --policy.dtype=bfloat16
    --policy.device=cuda --policy.push_to_hub=false
    --parallelism.dp_replicate=$NPROC --parallelism.dp_shard=1
    --output_dir="$OUT" --job_name="$JOB"
    --batch_size=$BATCH --num_workers=$WORKERS --steps=$STEPS
    --save_freq=$SAVE --log_freq=$LOGF --seed=1000 --wandb.enable=false
  )
  if [ "$VARIANT" = "delta" ]; then
    ARGS+=(--policy.use_relative_actions=true
           --policy.relative_exclude_joints='["base_","gripper"]')
  fi
fi

echo "pi0.5 v3 $VARIANT -> $OUT"
echo "  gpus=$GPUS nproc=$NPROC batch=$BATCH global=$((NPROC*BATCH)) steps=$STEPS ckpt=$GRAD_CKPT resume=$RESUME dry_run=$DRY_RUN"
echo "  log: $LOG"
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN argv: $PY -m torch.distributed.run --standalone --nproc_per_node=$NPROC -m lerobot.scripts.lerobot_train ${ARGS[*]}"
  exit 0
fi
setsid env CUDA_VISIBLE_DEVICES="$GPUS" nohup "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC" -m lerobot.scripts.lerobot_train "${ARGS[@]}" \
  > "$LOG" 2>&1 < /dev/null &
echo "launched pid $!"
