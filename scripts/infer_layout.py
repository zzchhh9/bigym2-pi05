"""Infer a task's v3 layout (the v3_layout.json entry) from its native demonstrations.

Every frame stores full_qpos. The task's official env (bigym.loco.make) gives each
joint's qpos address. An arm joint's state dim is the one equal to its full_qpos
column in every frame; exactly one dim must match, or the script stops.

    python scripts/infer_layout.py move_plate pick_box ...      # print entries
    python scripts/infer_layout.py --check                      # re-derive v3_layout.json
    python scripts/infer_layout.py --write flip_cup ...         # add entries to v3_layout.json
"""
import json, os, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from bigym.loco.fingerprint import SUBSTRATE_VERSION

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("PI05_WORK_ROOT", HERE.parent / "work"))
NATIVE = Path(os.environ.get("PI05_NATIVE_ROOT", str(WORK / "g1_native60")))
LAYOUT = HERE / "v3_layout.json"
TOL = 1e-6


def qpos_address(task):
    """{joint name: qpos index} of the task's official env."""
    import mujoco
    from bigym.loco import make
    env = make(task)
    m = env.inner_env.simulation.model
    out = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1]: int(m.jnt_qposadr[j])
           for j in range(m.njnt)}
    env.close()
    return out


def infer(task):
    src = NATIVE / task
    meta = json.loads((src / "meta" / "source_metadata.json").read_text())
    floating = [d.removeprefix("pelvis_") for d in meta["outer_action_floating_dofs"]]
    limbs = list(meta["outer_limb_actuator_names"])
    adr = qpos_address(task)
    A, S, Q = [], [], []
    for pf in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(pf, columns=["raw_outer_action", "observation.state", "full_qpos"])
        A.append(np.asarray(t["raw_outer_action"].to_pylist(), np.float32))
        S.append(np.asarray(t["observation.state"].to_pylist(), np.float64))
        Q.append(np.asarray(t["full_qpos"].to_pylist(), np.float64))
    A, S, Q = np.concatenate(A), np.concatenate(S), np.concatenate(Q)
    arm = {}
    for name in limbs:
        col = Q[:, adr[name], None]
        hits = np.flatnonzero(np.abs(S - col).max(0) < TOL)
        if len(hits) != 1:
            raise SystemExit(f"{task}: {name} matches state dims {hits.tolist()}")
        arm[name] = int(hits[0])
    return {
        "action_dim": int(A.shape[1]), "state_dim": int(S.shape[1]),
        "floating_dofs": floating, "n_grippers": int(A.shape[1] - len(floating) - len(limbs)),
        "limb_names": limbs, "arm_action_to_state": arm,
        "meta_action_dim": int(A.shape[1]),
        "substrate": SUBSTRATE_VERSION,
    }


KEYS = ("action_dim", "state_dim", "floating_dofs", "n_grippers", "limb_names", "arm_action_to_state")

if __name__ == "__main__":
    args = sys.argv[1:]
    layout = json.loads(LAYOUT.read_text())
    if args == ["--check"]:
        bad = 0
        for task, entry in layout.items():
            got = infer(task)
            diff = [k for k in KEYS if got[k] != entry[k]]
            bad += bool(diff)
            print(f"{task}: {'OK' if not diff else 'DIFF ' + str(diff)}")
        raise SystemExit(1 if bad else 0)
    write = args[:1] == ["--write"]
    for task in args[write:]:
        got = infer(task)
        print(task, json.dumps({k: got[k] for k in ("action_dim", "state_dim", "floating_dofs")}),
              "arm->state", list(got["arm_action_to_state"].values()))
        if write:
            if task in layout and any(layout[task][k] != got[k] for k in KEYS):
                raise SystemExit(f"{task}: inferred layout differs from the existing entry")
            layout[task] = got
    if write:
        LAYOUT.write_text(json.dumps(layout, indent=2) + "\n")
