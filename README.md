# π0.5 on BiGym 2.0: training and evaluation scripts

These scripts produced the π0.5 numbers in the
[BiGym 2.0 paper](https://arxiv.org/abs/2610.07594) (relight30k runs, Tables A4 and A5).
They are ported to run on the public release
([swirl-uk/BiGym2](https://github.com/swirl-uk/BiGym2)) and the public demonstrations
([SWIRL-Lab/bigym-g1-native60](https://huggingface.co/datasets/SWIRL-Lab/bigym-g1-native60)).

`as_run/` holds the files that actually ran, byte for byte. They have the paths of our
cluster and import a pre-release BiGym module. Read them; run `scripts/`.

## Settings the paper does not state

Tables A4 and A5 give the hyperparameters. A stock LeRobot run with those values does not
train the same model. These are the differences. We have not measured the effect of each
one on the success rate.

1. **Relative actions need a permuted state.** LeRobot's `use_relative_actions` computes
   `action[i] - state[i]` by index. It picks the channels to keep absolute from
   `action_feature_names`; with no names, it converts every channel. In the BiGym state,
   the arm joints are not at the same indices as in the action (the action starts with 4 or 5
   base commands). `export_v3.py` permutes `observation.state` so that the 14 arm joints sit
   at the action's arm indices, and writes the action names. Base commands and grippers stay
   absolute (`relative_exclude_joints=["base_","gripper"]`). The layout of each task is in
   `scripts/v3_layout.json`.
2. **Normalisation stats are recomputed on the relative actions.** LeRobot's π0.5 pipeline
   is raw → relative → normalise (QUANTILES, no clip), and the normaliser reads
   `meta/stats.json`, which holds the quantiles of the *absolute* actions. Arm deltas
   (std ≈ 0.03 rad) would be scaled by absolute ranges (std ≈ 0.3 rad). `make_delta_v3.py`
   rewrites the arm-channel stats from the deltas (OpenPI computes its norm stats after
   `DeltaActions`; this is the same thing). It also widens near-constant dimensions whose
   q01 equals q99, which otherwise give targets of order 1e7.
3. **The evaluation fixes the chunk anchor.** In training, every action of a chunk is
   relative to the state at the first step of the chunk. LeRobot's `select_action` queues the
   chunk and its post-processor adds the state of the *current* step back, so the anchor
   moves with the robot. In our tests that made the arm targets travel several times the demo
   distance and scored 0. `eval_v3.py` removes the current state and adds the chunk's first
   state (see `(5) CHUNK ANCHOR` in the code).
4. **Evaluation protocol.** `n_action_steps=16` at evaluation (8 in training),
   10 flow-matching steps (default), deterministic reset, seeds `620000 + i` for
   `i = 0..49`. Each of the last five checkpoints (26k, 27k, 28k, 29k, 30k steps) is
   evaluated on the same 50 seeds and the 250 episodes are pooled. The 250 episodes are
   therefore not independent.
5. **Hardware and batch.** 6 × NVIDIA H20 (97 GB), DDP, batch 8 per GPU (global 48),
   no gradient accumulation, bf16, no gradient checkpointing, seed 1000, image augmentation
   off, ImageNet stats for SigLIP. Full train configs: `configs/`.
6. **Tokenizer length: no difference from a stock run.** `launch_v3.sh` passes
   `--policy.tokenizer_max_length=256`, but it has no effect. When training starts from
   `pretrained_path`, LeRobot takes the tokenizer step from the base checkpoint's
   `policy_preprocessor.json` (`max_length: 200`) and does not override it. All our
   checkpoints saved `max_length: 200`. The prompt with a 50-D or 56-D discretised state
   is about 205-240 Gemma tokens, so its end is truncated in both training and evaluation.
   This applies to our runs as much as to yours.

## Versions

| Component | Version |
|---|---|
| BiGym | public `swirl-uk/BiGym2` (validated at `d715013`). The paper ran the pre-release code (substrate `bigym2-mj381-v3`); see [Substrate](#substrate). |
| LeRobot | `huggingface/lerobot@b1bf24f5` (2026-08-06). The four π0.5 files we ran are byte-identical to this commit. |
| Base model | `lerobot/pi05_base` |
| Python / torch / transformers | 3.12 / 2.9.0 / 5.5.x |

`as_run/requirements_pi05_freeze.txt` is the full `pip freeze` of the training environment.

## Install

```bash
git clone https://github.com/swirl-uk/BiGym2.git
git clone https://github.com/huggingface/lerobot.git && git -C lerobot checkout b1bf24f5
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch==2.9.0 torchvision==0.24.0
uv pip install -p .venv/bin/python -e ./BiGym2 -e "./lerobot[pi,dataset,training]"
```

The π0.5 tokenizer is `google/paligemma-3b-pt-224`, a gated model on the Hugging Face Hub:
accept its licence and log in before training.

## Run one task

All scripts read `PI05_WORK_ROOT` (default `./work`), `PI05_NATIVE_ROOT`
(default `$PI05_WORK_ROOT/g1_native60`) and `PI05_LEROBOT_ROOT`
(default `$PI05_WORK_ROOT/lerobot_v3`).

```bash
export PI05_WORK_ROOT=$PWD/work PI05_PYTHON=$PWD/.venv/bin/python MUJOCO_GL=egl
TASK=move_plate

# 1. Download the task's demonstrations (LeRobot v3, 84x84, 60 episodes).
hf download SWIRL-Lab/bigym-g1-native60 --repo-type dataset \
  --include "$TASK/*" --local-dir work/g1_native60

# 2. Absolute dataset (physical actions, permuted state, action names),
#    then the delta variant (same frames, stats of the relative actions).
python scripts/export_v3.py $TASK
python scripts/make_delta_v3.py $TASK

# 3. Gate: replay the demos through the eval env. Every episode must succeed.
#    max|state-dataset| is the largest gap between the env state and the dataset
#    state; a wrong state layout gives gaps of order 1 from the first steps.
python scripts/eval_v3.py --task $TASK --replay --episodes 10 --out work/replay_$TASK.json

# 4. Train (6 GPUs x batch 8, 30000 steps, a checkpoint every 1000 steps).
TASK=$TASK bash scripts/launch_v3.sh

# 5. Evaluate one checkpoint on 50 seeds; split over workers with
#    --num-workers/--worker-id. Repeat for steps 26000..30000 and pool.
python scripts/eval_v3.py --task $TASK --episodes 50 --n-action-steps 16 --seed-base 620000 \
  --ckpt work/pi05_runs_v3/${TASK}_delta/checkpoints/030000/pretrained_model \
  --out work/eval/$TASK/030000.json
```

Run every worker with `CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_EGL_DEVICE_ID=<gpu>`: the EGL
device index counts all GPUs and ignores `CUDA_VISIBLE_DEVICES`. Start workers on one GPU
at least a few seconds apart. Several MuJoCo EGL contexts on one GPU, or created in the same
second, crashed our workers (`rc=134`, Xid 31).

## Our numbers

Per-checkpoint successes out of 50 (seeds 620000-620049) of the paper runs, on the
pre-release substrate:

| Task | 26k | 27k | 28k | 29k | 30k | Pooled /250 |
|---|---|---|---|---|---|---|
| reach_target_multi_modal | 47 | 46 | 45 | 43 | 33 | 214 (85.6%) |
| reach_target_dual | 26 | 32 | 29 | 29 | 28 | 144 (57.6%) |
| drawer_top_open | 50 | 50 | 49 | 50 | 48 | 247 (98.8%) |
| drawer_top_close | 50 | 50 | 48 | 38 | 38 | 224 (89.6%) |
| move_plate | 31 | 37 | 29 | 34 | 32 | 163 (65.2%) |
| move_two_plates | 34 | 33 | 37 | 35 | 32 | 171 (68.4%) |
| dishwasher_load_cups | 45 | 46 | 49 | 46 | 45 | 231 (92.4%) |
| dishwasher_close | 37 | 38 | 46 | 36 | 40 | 197 (78.8%) |
| pick_box | 9 | 10 | 9 | 13 | 7 | 48 (19.2%) |

Each row comes from one training run. We have not measured the spread over training
seeds. `eval_v3.py` does not seed torch, so the flow-matching noise differs between runs
and the same checkpoint does not give the same per-seed outcomes twice.

### Check on the public release

We ran `scripts/` from a clean environment (public BiGym2 `d715013`, LeRobot `b1bf24f5`,
the public dataset), on 8 × H20, on 2026-10-28:

- `export_v3.py` + `make_delta_v3.py` on `move_plate`: the action and state quantiles, mean
  and std equal the normaliser stats saved in our 30k-step checkpoint exactly (difference 0,
  37705 frames in both).
- Replay gate: 10 of 10 `move_plate` demos succeed.
- Our `move_plate` 30k-step checkpoint, evaluated with `eval_v3.py` on the public env:
  **33/50** (paper run: 32/50 on the pre-release substrate).
- `launch_v3.sh` trains 20 steps on 2 GPUs; the saved preprocessor has the 20 action names
  and `max_length: 200`.

## Substrate

The paper's evaluations ran on the pre-release BiGym (substrate `bigym2-mj381-v3`).
Before the public release, the env started to run `mj_forward` after every control step
and reset (the public substrate is named `bigym2-mj381-v1`). The BiGym changelog lists the
tasks whose rollouts and camera frames this changes; of the nine paper tasks they are
`drawer_top_open`, `drawer_top_close`, `dishwasher_close`, `dishwasher_load_cups` and
`pick_box`. `move_plate`, `move_two_plates` and the two reach tasks are not affected. The
task budgets and official configurations are the same. `eval_v3.py` checks the official
configuration against the demonstrations' `source_metadata.json` and stops on any
difference.

The public demonstrations do not replay bit-exactly on the public env. For `move_plate`,
`substrate_fingerprint()["contact_signature"]` is `36156d4845f2db32` on BiGym2 `d715013`
and `a3972624f79d3e98` in the demos' `source_metadata.json`. The replayed `qpos` is off by
about 3e-3 after the first step, and 10 of 10 replayed demos still succeed.

## Files

| File | Role |
|---|---|
| `scripts/v3_layout.json`, `scripts/delta_spec_v3.py` | Per-task action/state layout (20/50 or 21/56), state permutation, relative mask |
| `scripts/export_v3.py` | Native dataset → absolute LeRobot dataset |
| `scripts/make_delta_v3.py` | Absolute → delta dataset (hard-linked frames, new stats) |
| `scripts/launch_v3.sh` | `torchrun lerobot_train` with the paper's arguments |
| `scripts/eval_v3.py` | Closed-loop evaluation and the replay gate |
| `configs/*_train_config.json` | Saved `train_config.json` of the nine paper runs at 30k steps (paths are our cluster's) |
| `as_run/` | The files that ran, unchanged, incl. the nine-task orchestrator |

## Licence

Apache-2.0, as BiGym 2.0.
