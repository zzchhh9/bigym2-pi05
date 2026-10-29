"""v3 evaluation for pi0.5 checkpoints (absolute OR OpenPI-delta).

Runs on the public BiGym 2.0 release (github.com/swirl-uk/BiGym2) and its
evaluation protocol: the env is the task's official configuration,
``bigym.loco.make(task)`` with no overrides (``protocol_violations`` must be
empty), seeds come from ``bigym.loco.eval.eval_seeds`` (620000 + i), and an
episode counts with ``bigym.loco.eval.is_success`` (success latched, robot never
fell). build_env also checks the official config against the demos'
source_metadata.json and stops on any difference. The paper's runs used the
pre-release substrate; see the README.

Four things must line up or the score is a silent zero (all four cost us a full
sweep once already):

 1. ENV CONFIG must be the one the demos were collected with — the native60
    batch was collected with enable_all_floating_dof=True (20-D action, 50-D
    state); a hand-written config silently builds a 19-D/47-D env.
 2. STATE SPACE is RAW (normalize_low_dim_obs=False): the policy normalises
    observation.state itself from the stats baked into its preprocessor.
 3. STATE IS PERMUTED exactly as the training dataset was, so the arm joints sit
    where lerobot's positional `action[i] -= state[i]` expects them. For the
    delta variant AbsoluteActionsProcessorStep adds this same permuted state
    back, so the permutation must match bit-for-bit.
 4. The policy emits PHYSICAL actions (delta already integrated by the
    postprocessor). They are clipped to the demonstrations' action envelope
    (source_metadata.json action_stats; it is narrower than the env's bounds
    only on the base velocity commands) and converted to [-1, 1] with the env's
    own ``normalize_action``. This is the paper's action path; the env itself
    keeps its official action stats.

The *episode budget* is NOT taken from source_metadata.json (that is the
collection cap, e.g. 60000). Formal eval uses
``bigym.loco.tasks.task_spec(task).episode_length`` (move_plate = 17000) and
lets the env terminate. ``--max-steps`` is only a diagnostic
override. Reset is ``deterministic_reset=True``: a fixed 620000+i seed is
not the same as aligned controller state.

--replay ignores the policy and pushes the demo's own recorded actions through
this exact env. On the public release 7-10 of 10 demos per task succeed (the
contact model differs from the one the demos were recorded with); a wrong state
layout shows as max|state-dataset| of order 1 from the first steps.

The language instruction is the SFT dataset string (tasks.parquet): the
policy must see the sentence it was trained on.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                        # <repo>/scripts -> repo root
WORK = Path(os.environ.get("PI05_WORK_ROOT", REPO / "work"))

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from delta_spec_v3 import spec

NATIVE = Path(os.environ.get("PI05_NATIVE_ROOT", str(WORK / "g1_native60")))
LEROBOT = Path(os.environ.get("PI05_LEROBOT_ROOT", str(WORK / "lerobot_v3")))

# The prompts the paper's relight30k checkpoints were trained and evaluated
# with (tasks.parquet of the HF release). Eval prefers the live parquet; this
# map is the fallback when no exported dataset is present.
SFT_PROMPTS = {
    "reach_target_single": "Reach the target with the left wrist.",
    "reach_target_dual": "Reach the two targets, one with each wrist.",
    "reach_target_multi_modal": "Reach the target with either wrist.",
    "drawer_top_open": "Open the top drawer of the kitchen cabinet.",
    "drawer_top_close": "Close the top drawer of the kitchen cabinet.",
    "move_plate": "Move the plate between two draining racks.",
    "move_two_plates": "Move two plates simultaneously from one draining rack to the other.",
    "dishwasher_close": "Push back all trays and close the door of the dishwasher.",
    "dishwasher_load_cups": "Move the cups from the table into the dishwasher's upper tray.",
    "pick_box": "Pick up the box from the side table and place it on the counter.",
}


def _dataset_instruction(task):
    for variant in (f"{task}_delta", f"{task}_abs", task):
        p = LEROBOT / variant / "meta" / "tasks.parquet"
        if not p.is_file():
            continue
        import pyarrow.parquet as pq
        table = pq.read_table(p)
        if "task" not in table.column_names:
            continue
        vals = [x for x in table["task"].to_pylist() if x]
        if vals:
            return str(vals[0])
    return None


def _yaml_instruction(task):
    try:
        import yaml
    except ImportError:
        return None
    base = task[:-3] if task.endswith("_g1") else task
    base = base.removesuffix("_delta").removesuffix("_abs")
    import bigym.loco.demos
    cands = [Path(bigym.loco.demos.__file__).with_name("task_instructions.yaml")]
    for fn in cands:
        if not fn.is_file():
            continue
        table = yaml.safe_load(fn.read_text()) or {}
        text = table.get(task) or table.get(base)
        if text:
            return str(text)
    return None


def resolve_prompt(task, meta=None, override=None):
    """Return (prompt, source). Never silently fall back to move_plate.

    Priority: --prompt, LeRobot tasks.parquet (what SFT trained on),
    compiled SFT_PROMPTS map, task_instructions.yaml. source_metadata.json
    has no task_text (collection metadata) and is not consulted.
    """
    # `meta` is source_metadata.json (collection). It has never carried
    # task_text; do not prefer it over the SFT dataset string.
    if override:
        return str(override), "cli"
    ds = _dataset_instruction(task)
    if ds:
        return ds, "lerobot_tasks_parquet"
    mapped = SFT_PROMPTS.get(task) or SFT_PROMPTS.get(str(task).removesuffix("_delta"))
    if mapped:
        return mapped, "sft_prompt_map"
    yaml_text = _yaml_instruction(task)
    if yaml_text:
        return yaml_text, "task_instructions.yaml"
    raise SystemExit(
        f"no eval prompt for {task!r}: pass --prompt or stamp "
        f"lerobot_v3/{task}_delta/meta/tasks.parquet"
    )


def protocol_budget(task):
    """Official inner-step budget + provenance. Collection metadata is not this."""
    from bigym.loco.tasks import budget_provenance, task_spec
    return int(task_spec(task).episode_length), budget_provenance(task)


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def _outcome(env):
    """(success by the public protocol, termination, fell)."""
    from bigym.loco.eval import is_success
    return bool(is_success(env)), env.episode_termination(), bool(env.episode_fell())


def build_env(task):
    SPEC = spec(task)
    from bigym.loco import make
    from bigym.loco.eval import protocol_violations
    from bigym.loco.tasks import task_config
    budget, provenance = protocol_budget(task)
    episode_length = budget
    m = json.loads((NATIVE / task / "meta" / "source_metadata.json").read_text())
    t = m["task"]
    collected = int(t.get("episode_length") or 0)
    if collected and collected != episode_length:
        print(
            f"[env] collection episode_length={collected} vs protocol "
            f"{budget} ({provenance}); using protocol budget",
            flush=True,
        )
    # The official config must be the collection config on every field that
    # shapes the observation or the action. Pitch-enabled tasks observe the
    # waist joints (56-D state); the top-drawer tasks use g1_id_v1.
    cfg = task_config(task)
    want = {
        "robot_model": t["robot_model"], "action_mode": t["action_mode"],
        "control_pelvis": bool(t["control_pelvis"]),
        "enable_all_floating_dof": bool(t["enable_all_floating_dof"]),
        "demo_down_sample_rate": int(t["demo_down_sample_rate"]),
        "camera_shape": tuple(t["camera_shape"]), "camera_keys": tuple(t["camera_keys"]),
        "state_keys": tuple(t["state_keys"]),
        "initialization_profile": t.get("initialization_profile") or "upstream",
        "success_hold_seconds": float(t.get("success_hold_seconds", 1.0)),
        "reach_tolerance": t.get("reach_tolerance"),
    }
    have = {k: getattr(cfg, k) for k in want}
    have = {k: tuple(v) if isinstance(v, list) else v for k, v in have.items()}
    diff = {k: (have[k], want[k]) for k in want if have[k] != want[k]}
    assert not diff, f"official config differs from the demos: {diff}"
    env = make(task)                                           # official; (2) RAW state
    violations = protocol_violations(env)
    assert not violations, f"env is not the official configuration: {violations}"
    assert env.config.controller.deterministic_reset
    assert int(env.config.episode_length) == budget, (env.config.episode_length, budget)
    fp = m["substrate_fingerprint"]
    live = len(env.action_stats["min"])
    assert live == int(fp["action_dim"]) == SPEC.action_dim, (live, fp["action_dim"])
    st = m["action_stats"]
    env.demo_envelope = (np.asarray(st["min"], np.float32),
                         np.asarray(st["max"], np.float32))     # (4)
    return env, m, {
        "episode_length": int(episode_length),
        "episode_length_source": "task_protocol",
        "episode_length_protocol": int(budget),
        "episode_length_collected": collected or None,
        "budget_provenance": provenance,
        "deterministic_reset": True,
        "protocol_violations": violations,
    }


def to_env_action(env, phys):
    """(4) physical action -> [-1, 1]: demo envelope clip, then the env's own stats."""
    lo, hi = env.demo_envelope
    return np.clip(env.normalize_action(np.clip(phys, lo, hi)), -1, 1)


def guard_state_space(env, ckpt, seed, SPEC):
    """Probe obs must fall inside the envelope the checkpoint recorded."""
    from safetensors.torch import load_file
    ts = env.reset(seed=seed)
    s = SPEC.permute_state(np.asarray(ts.low_dim_obs, np.float32))            # (3)
    assert s.shape[0] == SPEC.state_dim, s.shape
    norm = None
    for fn in sorted(Path(ckpt).glob("*normalizer*.safetensors")):
        b = load_file(str(fn))
        if any(k.startswith("observation.state.") for k in b): norm = b; break
    if norm is None:
        print("WARNING: no state normaliser in checkpoint"); return ts, 0
    lo = norm["observation.state.min"].float().numpy().ravel()
    hi = norm["observation.state.max"].float().numpy().ravel()
    assert lo.shape[0] == s.shape[0], (lo.shape, s.shape)
    span = np.maximum(hi - lo, 1e-6)
    inside = int(((s >= lo - .5 * span) & (s <= hi + .5 * span)).sum())
    if inside / s.size < 0.9:
        raise RuntimeError(f"state-space mismatch: {inside}/{s.size} inside envelope")
    return ts, inside


def obs_to_batch(ts, cams, prompt, device, SPEC):
    import torch
    rgb = np.asarray(ts.rgb_obs)
    b = {"observation.state": torch.from_numpy(
             SPEC.permute_state(np.asarray(ts.low_dim_obs, np.float32)))[None].to(device),
         "task": [prompt]}
    for i, c in enumerate(cams):
        b[f"observation.images.{c}"] = torch.from_numpy(rgb[i].astype(np.float32) / 255.0)[None].to(device)
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True); p.add_argument("--ckpt")
    p.add_argument("--out", default=None); p.add_argument("--prompt", default=None)
    p.add_argument(
        "--print-prompt", action="store_true",
        help="resolve and print the eval instruction, then exit (no env/policy)",
    )
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed-base", type=int, default=620000)
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="diagnostic outer-step cap; default is the env time limit from "
             "task_protocol (not the collection episode_length)",
    )
    p.add_argument("--n-action-steps", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=1); p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--device", default="cuda"); p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    if a.print_prompt:
        prompt, src = resolve_prompt(a.task, override=a.prompt)
        print(json.dumps({"task": a.task, "prompt": prompt, "prompt_source": src}))
        return
    if not a.out:
        p.error("--out is required unless --print-prompt")

    SPEC = spec(a.task)
    env, meta, budget = build_env(a.task)
    cams = tuple(meta["task"]["camera_keys"])
    prompt, prompt_source = resolve_prompt(a.task, meta, a.prompt)
    tag = f"[w{a.worker_id}]"
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    downsample = int(meta["task"]["demo_down_sample_rate"])
    env_outer = max(1, budget["episode_length"] // downsample)
    step_cap = int(a.max_steps) if a.max_steps is not None else env_outer
    print(
        f"{tag} budget inner={budget['episode_length']} outer={env_outer} "
        f"max_steps={a.max_steps} step_cap={step_cap} "
        f"deterministic_reset={budget['deterministic_reset']}",
        flush=True,
    )
    print(f"{tag} prompt={prompt!r} source={prompt_source}", flush=True)

    if a.replay:                                   # (gate) demo actions, no policy
        # Demos start at the VR ENGAGE moment (post reset-warmup), not at a bare
        # reset, so the recorded engage snapshot must be restored first or the
        # actions are replayed from the wrong state and score 0 — which says
        # nothing about the eval path.
        import pyarrow.parquet as pq, glob
        from bigym.loco.demos.success_hold import restore_engage_state
        root = LEROBOT / f"{a.task}_abs"
        snaps = json.loads((root / "meta" / "episode_init_states.json").read_text())["episodes"]

        def arr(d, key, dt):
            v = d[key]
            return np.asarray(v["data"], dt).reshape(v["shape"])

        acts, states, eidx = [], [], []
        for pf in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
            t = pq.read_table(pf)
            acts.append(np.asarray(t["action"].to_pylist(), np.float32))
            states.append(np.asarray(t["observation.state"].to_pylist(), np.float32))
            eidx.append(np.asarray(t["episode_index"].to_pylist()))
        acts, states, eidx = np.concatenate(acts), np.concatenate(states), np.concatenate(eidx)

        eps = []
        for ep in sorted({int(k) for k in snaps})[:a.episodes]:
            sn = snaps[str(ep)]["arrays"]
            A, S = acts[eidx == ep], states[eidx == ep]
            dtype = lambda k: (np.int64 if k == "seed" else
                               np.float32 if k.startswith("lb_state.") else np.float64)
            restore_engage_state(env, {k: arr(sn, k, dtype(k)) for k in sn})
            mx, k, worst = 0.0, 0, 0.0
            for k in range(1, len(A)):
                ts = env.step(to_env_action(env, A[k]))
                # observation.state of row k is the observation the action of
                # row k was taken from, so the state after that action is row
                # k+1. Matching it checks the v3 layout on this BiGym. The
                # first steps after the engage restore differ by up to 0.1
                # (grippers); report the last 90% of the episode.
                if k + 1 < len(S) and k >= len(A) // 10:
                    s_env = SPEC.permute_state(np.asarray(ts.low_dim_obs, np.float32))
                    worst = max(worst, float(np.abs(s_env - S[k + 1]).max()))
                mx = max(mx, float(ts.reward or 0.0))
                if ts.last(): break
            success, termination, fell = _outcome(env)
            eps.append({"episode": ep, "steps": k, "max_reward": mx,
                        "success": success, "termination": termination, "fell": fell,
                        "max_state_err": worst})
            print(f"{tag} replay ep{ep}: steps={k} reward={mx} end={termination} "
                  f"max|state-dataset|={worst:.2e}", flush=True)
            out.write_text(json.dumps({
                "task": a.task, "mode": "replay", "episodes": eps,
                "prompt": prompt, "prompt_source": prompt_source,
                "success_rate": sum(e["success"] for e in eps)/len(eps),
                **budget,
                "substrate_fingerprint": _jsonable(env.substrate_fingerprint()),
            }, indent=1))
        env.close()
        return

    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from bigym.loco.eval import eval_seeds
    seeds = eval_seeds(a.episodes, seed_base=a.seed_base)[a.worker_id::a.num_workers]
    print(f"{tag} {len(seeds)} episodes: {seeds[:3]}...", flush=True)
    ts, inside = guard_state_space(env, a.ckpt, seeds[0], SPEC)
    print(f"{tag} state-space verified: {inside}/{SPEC.state_dim} dims inside envelope", flush=True)

    policy = PI05Policy.from_pretrained(a.ckpt); policy.eval(); policy.to(a.device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=a.ckpt)
    policy.config.n_action_steps = int(a.n_action_steps)
    rel = getattr(policy.config, "use_relative_actions", False)
    print(f"{tag} policy ready (n_action_steps={policy.config.n_action_steps}, relative={rel})", flush=True)

    # (5) CHUNK ANCHOR. Training builds the chunk with ONE anchor:
    #   delta[k] = action[t0+k] - s_t0   (to_relative_actions broadcasts s_t0
    #   across the whole horizon). But select_action queues the chunk in delta
    #   space and AbsoluteActionsProcessorStep adds whatever state pre() cached
    #   THIS step, so the anchor drifts with the robot's own motion — a positive
    #   feedback that made arm targets travel 3.8x the demo and scored 0/55.
    #   Undo the per-step state and add the frozen chunk anchor instead.
    mask = SPEC.relative_mask() if rel else None
    H = int(policy.config.n_action_steps)

    eps = []
    for seed in seeds:
        policy.reset(); ts = env.reset(seed=seed)
        mx, k, t0 = 0.0, 0, time.time()
        anchor = None
        for k in range(1, step_cap + 1):
            s_now = SPEC.permute_state(np.asarray(ts.low_dim_obs, np.float32))
            if rel and (k - 1) % H == 0:
                anchor = s_now.copy()          # queue refills on these steps
            with torch.no_grad():
                phys = post(policy.select_action(pre(obs_to_batch(ts, cams, prompt, a.device, SPEC))))
            phys = np.asarray(phys.squeeze(0).float().cpu().numpy(), np.float32)
            if rel:
                phys[mask] += anchor[:phys.shape[0]][mask] - s_now[:phys.shape[0]][mask]
            ts = env.step(to_env_action(env, phys))                     # (4)
            mx = max(mx, float(ts.reward or 0.0))
            if ts.last(): break
        success, termination, fell = _outcome(env)
        eps.append({"seed": int(seed), "steps": k, "max_reward": mx,
                    "success": success, "termination": termination, "fell": fell,
                    "wall_s": round(time.time() - t0, 1)})
        print(f"{tag} seed {seed}: success={success} steps={k} end={termination} | "
              f"running {sum(e['success'] for e in eps)}/{len(eps)}", flush=True)
        out.write_text(json.dumps({
            "task": a.task, "ckpt": a.ckpt, "relative": bool(rel),
            "n_action_steps": a.n_action_steps,
            "max_steps_override": a.max_steps,
            "step_cap": step_cap,
            "prompt": prompt, "prompt_source": prompt_source,
            **budget,
            "substrate_fingerprint": _jsonable(env.substrate_fingerprint()),
            "episodes": eps,
            "success_rate": sum(e["success"] for e in eps)/len(eps),
        }, indent=1))
    env.close()


if __name__ == "__main__":
    main()
