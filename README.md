# π0.5 on BiGym 2.0: training and evaluation scripts

These scripts produced the π0.5 numbers in the
[BiGym 2.0 paper](https://arxiv.org/abs/2610.07594) (relight30k runs, Tables A4 and A5).
They are ported to run on the public release
([swirl-uk/BiGym2](https://github.com/swirl-uk/BiGym2)) and the public demonstrations
([SWIRL-Lab/bigym-g1-native60](https://huggingface.co/datasets/SWIRL-Lab/bigym-g1-native60)).

`scripts/` uses only the public release: the env is the task's official configuration
(`bigym.loco.make(task)`, no overrides, `protocol_violations` empty), and seeds and
success follow `bigym.loco.eval` (`eval_seeds`, `is_success`). `as_run/` holds the files
that actually ran for the paper, byte for byte. They have the paths of our cluster and
import a pre-release BiGym module. Read them; run `scripts/`.

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
   absolute (`relative_exclude_joints=["base_","gripper"]`). The layout of every task in
   the public dataset (20 tasks) is in `scripts/v3_layout.json`.
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
   therefore not independent. An episode counts as a success by `bigym.loco.eval.is_success`
   (the task latched success and the robot never fell). The policy's physical action is
   clipped to the demonstrations' action envelope (`action_stats` in
   `source_metadata.json`) before the env's own `normalize_action`; the envelope is
   narrower than the env's bounds only on the base velocity commands (x 0.35, y 0.25,
   yaw 0.5 instead of 1.0). This is the action path of the paper's runs.
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
`requirements_public_validated.txt` is the environment that ran the checks under
"Check on the public release".

## Install

```bash
git clone https://github.com/swirl-uk/BiGym2.git
# Skip the LFS test artefacts; a full LFS checkout of lerobot can hang.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout b1bf24f5
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

# 3. Check: replay the demos through the eval env, as the public
#    bigym.loco.demos.success_hold.latch_step does. Every step must land on the
#    recorded qpos and observation bit for bit, and every demo must succeed.
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
| reach_target_multi_modal | 47 | 46 | 45 | 43 | 45 | 226 (90.4%) |
| reach_target_dual | 26 | 32 | 29 | 29 | 28 | 144 (57.6%) |
| drawer_top_open | 50 | 50 | 49 | 50 | 48 | 247 (98.8%) |
| drawer_top_close | 50 | 50 | 50 | 50 | 50 | 250 (100%) |
| move_plate | 31 | 37 | 29 | 34 | 32 | 163 (65.2%) |
| move_two_plates | 34 | 33 | 37 | 35 | 32 | 171 (68.4%) |
| dishwasher_load_cups | 45 | 46 | 49 | 46 | 45 | 231 (92.4%) |
| dishwasher_close | 37 | 38 | 46 | 36 | 40 | 197 (78.8%) |
| pick_box | 9 | 10 | 9 | 13 | 7 | 48 (19.2%) |

These are the numbers of the paper's main table (pooled, rounded). Each row comes from one
training run. We have not measured the spread over training
seeds. `eval_v3.py` does not seed torch, so the flow-matching noise differs between runs
and the same checkpoint does not give the same per-seed outcomes twice.

### Check on the public release

We ran `scripts/` from a clean environment (public BiGym2 `d715013`, LeRobot `b1bf24f5`,
the public dataset at revision `4c3e441`) on H20 GPUs, on 2026-10-28 and 2026-10-29. For each
of the nine tasks: export and delta stats against the stats saved in our checkpoint, 10
replayed demos, and our 30k-step checkpoint on seeds 620000-620049. The raw outputs are in
`results/public_release/` (stats, policy evaluation) and `results/replay/` (replay). The
evaluation files come from `eval_v3.py` at commit `3a2906f`, which counted success as
reward 1.0 and clipped through the demo envelope by setting the env's action stats. The
current version uses `is_success` and leaves the env's stats alone; the actions it sends
are the same, and none of the 450 episodes had a success with a fall.

| Task | Stats vs our checkpoint | Replay, bit-exact | 30k ckpt, public env | 30k ckpt, paper run |
|---|---|---|---|---|
| reach_target_multi_modal | equal | 10/10 | 47/50 | 45/50 |
| reach_target_dual | equal | 10/10 | 28/50 | 28/50 |
| drawer_top_close | equal | 10/10 | 50/50 | 50/50 |
| drawer_top_open | equal | 10/10 | 48/50 | 48/50 |
| move_plate | equal | 10/10 | 33/50 | 32/50 |
| move_two_plates | equal | 10/10 | 34/50 | 32/50 |
| dishwasher_load_cups | differ (45998 vs 46027 frames) | 10/10 | 48/50 | 45/50 |
| dishwasher_close | equal | 10/10 | 36/50 | 40/50 |
| pick_box | differ (100928 vs 101290 frames) | 10/10 | 12/50 | 7/50 |

- "equal" means the action and state quantiles, mean, std, min and max are identical to the
  normaliser stats in our checkpoint (difference 0). For `dishwasher_load_cups` and
  `pick_box` the public demonstrations have fewer frames than the ones we trained on (we
  think the episode ends were re-cut after our runs; we have not checked this). A model trained on the public data for these two tasks is not
  trained on exactly our data.
- Every public-env score is inside the range of our five paper checkpoints (26k-30k) for
  that task, in the per-checkpoint table under "Our numbers". With 50 episodes, one standard error is about 3 episodes.
- `launch_v3.sh` trains 20 steps on 2 GPUs; the saved preprocessor has the action names
  and `max_length: 200`.

## The other eleven tasks of the dataset

The public dataset has 20 tasks; the paper reports π0.5 on nine. The other eleven run with
the same recipe and the same scripts, unchanged:

| Task | Layout | Frames (60 demos) | Episode budget | Replay, bit-exact |
|---|---|---|---|---|
| reach_target_single | 20/50 | 12085 | 9000 | 10/10 |
| wall_cupboard_open | 21/56 | 35650 | 14500 | 10/10 |
| wall_cupboard_close | 21/56 | 35402 | 17000 | 10/10 |
| flip_cup | 21/56 | 36443 | 18500 | 10/10 |
| flip_cutlery | 21/56 | 44158 | 19500 | 10/10 |
| dishwasher_load_cutlery | 21/56 | 57541 | 26500 | 10/10 |
| put_cups | 21/56 | 61124 | 30000 | 10/10 |
| dishwasher_load_plates | 21/56 | 74885 | 31500 | 10/10 |
| sandwich_remove | 21/56 | 65863 | 31500 | 10/10 |
| saucepan_to_hob | 21/56 | 91205 | 46500 | 10/10 |
| stack_blocks | 21/56 | 231636 | 87500 | 10/10 |

- Layout: the ten new entries of `v3_layout.json` come from `infer_layout.py`. All ten
  pitch tasks have the same layout as `dishwasher_close`. `infer_layout.py --check`
  re-derives all 20 entries on the public env and matches each one.
- Every budget is `data_derived` and equals the demos' `recommended_episode_length`, so
  `export_v3.py` accepts every task. Raw outputs: `results/replay/`, `results/release_check/`.
- The same recipe means 30k steps for every task, as in the paper. A task with more frames
  sees fewer epochs: about 6 for `stack_blocks`, against about 38 for `move_plate`.

To run them on one 8-GPU machine:

```bash
python scripts/run_tasks.py reach_target_single wall_cupboard_open wall_cupboard_close \
  flip_cup flip_cutlery dishwasher_load_cutlery put_cups dishwasher_load_plates \
  sandwich_remove saucepan_to_hob stack_blocks \
  --train-gpus 0,1,2,3,4,5 --eval-gpus 6,7
```

Cost on 8 × H20: training runs at 1.4 s/step (measured), about 11.7 h per task and 129 h
for the eleven. Evaluation (250 episodes per task) runs at about 9 env steps/s per worker;
when every episode times out it takes from 7 worker-hours (`reach_target_single`) to
68 worker-hours (`stack_blocks`), 256 worker-hours for the eleven. With 4 eval workers it
fits inside the training time. We have run each step of `run_tasks.py` separately (export,
`launch_v3.sh`, `eval_v3.py`); we have not run the script end to end.

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

The public demonstrations replay bit for bit on the public env: on all 20 tasks, 10 of 10
demos land on the recorded `qpos` and observation at every step and latch success on the
recorded last step (`results/replay/`).

A correction: an earlier version of this README said that the demos drift on the public
env and that 1 to 10 of 10 replay successfully. That was a bug in our replay, not in the
release. The export stores, in frame k, the action taken *from* observation k (action
alignment v2, `meta/alignment.json`); our replay read it as the action that *produced*
observation k, so it applied every action one step early. Training and policy evaluation
were not affected: they pair observation k with the action taken from it, and our training
data equals the public data (see the stats column above).

## Files

| File | Role |
|---|---|
| `scripts/v3_layout.json`, `scripts/delta_spec_v3.py` | Per-task action/state layout (20/50 or 21/56), state permutation, relative mask |
| `scripts/export_v3.py` | Native dataset → absolute LeRobot dataset |
| `scripts/make_delta_v3.py` | Absolute → delta dataset (hard-linked frames, new stats) |
| `scripts/launch_v3.sh` | `torchrun lerobot_train` with the paper's arguments |
| `scripts/eval_v3.py` | Closed-loop evaluation and the replay check |
| `scripts/infer_layout.py` | Derives a task's layout from its demonstrations and the official env; `--check` re-derives every entry of `v3_layout.json` |
| `scripts/run_tasks.py` | Train and evaluate a list of tasks with the paper's recipe on one 8-GPU machine (resumable) |
| `configs/*_train_config.json` | Saved `train_config.json` of the nine paper runs at 30k steps (paths are our cluster's) |
| `as_run/` | The files that ran, unchanged, incl. the nine-task orchestrator |
| `results/public_release/` | Raw output of the public-release check: `<task>_prep.json` (stats comparison), `<task>_030000_w<k>.json` (one shard of the 50 seeds each) |
| `results/replay/` | Bit-exact replay check of all 20 tasks, 10 demos each |
| `requirements_public_validated.txt` | Package versions of the public-release check |
| `results/release_check/` | Release-only check of the current `eval_v3.py`: one 5-episode shard of each paper task with `protocol_violations` recorded |

## Licence

Apache-2.0, as BiGym 2.0.
