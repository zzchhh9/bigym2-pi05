"""v3 delta spec — every index MEASURED on the live env, never assumed.

Source of truth: delta_v3/v3_layout.json, produced by probe_v3_layout.py, which
builds each task's env from its own source_metadata.json and perturbs one joint
at a time (asserting exactly one state dim moves).

THE LAYOUT IS PER TASK. An earlier version of this file exposed one global
ACTION_DIM/STATE_DIM/permute_state derived from move_plate, on the strength of a
gate asserting all five native60 tasks shared a layout. dishwasher_close breaks
that: it was collected with groot_wbc_g1.enable_pitch_cmd=True, which adds a
pelvis-pitch (ry) command, and on a bigym that lets pitch-enabled tasks observe
the waist joints -- 21 action / 56 state against the others' 20/50. Applying the
20/50 spec to it would not raise; it would silently permute the wrong dims and
mask the wrong action block. So there are no module-level dims any more: every
consumer must name its task and get a TaskSpec.

v3 vs v2: action 19->20 and state 47->50 for the 4-base-DOF group (collection ran
enable_all_floating_dof=True, so pelvis_z is an outer action dim). The arm->state
map happens to be unchanged, but the ACTION offsets are not: the base block is 4
dims, not 3. Reusing the v2 spec would shift the delta mask by one.
"""
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

_LAYOUT_PATH = Path(__file__).resolve().parent / "v3_layout.json"
_L = json.loads(_LAYOUT_PATH.read_text())

TASKS = list(_L)
EXCLUDE_JOINTS = ["base_", "gripper"]      # kept absolute (OpenPI keeps gripper absolute)


@dataclass(frozen=True)
class TaskSpec:
    """One task's measured action/state layout."""
    task: str
    action_dim: int
    state_dim: int
    n_base: int
    n_grip: int
    limb_names: Tuple[str, ...]
    arm_state_idx: Tuple[int, ...]
    action_names: Tuple[str, ...]

    @property
    def arm_slice(self) -> slice:
        return slice(self.n_base, self.n_base + len(self.limb_names))

    def state_permutation(self) -> np.ndarray:
        """p with permuted = state[p]: arm joints land at arm_slice so lerobot's
        positional `action[i] -= state[i]` lines up. Pure permutation, nothing lost."""
        p = np.empty(self.state_dim, dtype=np.int64)
        arm_slots = list(range(self.arm_slice.start, self.arm_slice.stop))
        p[arm_slots] = self.arm_state_idx
        rest = [i for i in range(self.state_dim) if i not in self.arm_state_idx]
        other = [i for i in range(self.state_dim) if i not in arm_slots]
        assert len(rest) == len(other)
        p[other] = rest
        return p

    def permute_state(self, state) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32)
        assert s.shape[-1] == self.state_dim, (self.task, s.shape, self.state_dim)
        return s[..., self.state_permutation()]

    def relative_mask(self) -> np.ndarray:
        toks = [t.lower() for t in EXCLUDE_JOINTS]
        return np.array([not any(t in n.lower() for t in toks) for n in self.action_names])


@lru_cache(maxsize=None)
def spec(task: str) -> TaskSpec:
    if task not in _L:
        raise KeyError(
            f"{task!r} has no measured layout in {_LAYOUT_PATH.name}. Run "
            f"probe_v3_layout.py for it first -- do NOT reuse another task's spec."
        )
    e = _L[task]
    limbs = tuple(e["limb_names"])
    s = TaskSpec(
        task=task,
        action_dim=int(e["action_dim"]),
        state_dim=int(e["state_dim"]),
        n_base=len(e["floating_dofs"]),
        n_grip=int(e["n_grippers"]),
        limb_names=limbs,
        arm_state_idx=tuple(int(e["arm_action_to_state"][n]) for n in limbs),
        action_names=tuple(
            [f"base_{d}" for d in e["floating_dofs"]]
            + [n.replace("_joint", "") for n in limbs]
            + ["left_gripper", "right_gripper"]
        ),
    )
    # Per-task self-consistency. These are the assertions that used to live at
    # import time as a cross-task equality gate; equality across tasks is no
    # longer true, but each task must still add up on its own.
    assert s.action_dim == s.n_base + len(s.limb_names) + s.n_grip, (task, s.action_dim)
    assert len(s.action_names) == s.action_dim, (task, len(s.action_names))
    assert len(set(s.arm_state_idx)) == len(s.arm_state_idx), (task, "arm->state not injective")
    assert max(s.arm_state_idx) < s.state_dim, (task, "arm index outside state")
    return s


def layout_groups() -> dict:
    """Tasks bucketed by identical layout — what the old Gate B used to assert."""
    g = {}
    for t in TASKS:
        s = spec(t)
        g.setdefault((s.action_dim, s.state_dim, s.n_base), []).append(t)
    return g


if __name__ == "__main__":
    from bigym.loco.tasks import budget_provenance, task_spec
    work = Path(os.environ.get("PI05_WORK_ROOT", Path(__file__).resolve().parents[1] / "work"))
    for (ad, sd, nb), tasks in layout_groups().items():
        print(f"{ad}/{sd} base={nb}: {', '.join(tasks)}")
        for t in tasks:
            spec(t)
            print(
                f"  protocol {t}: episode_length={task_spec(t).episode_length} "
                f"({budget_provenance(t)})"
            )
            meta = work / "g1_native60" / t / "meta" / "source_metadata.json"
            if meta.is_file():
                collected = (json.loads(meta.read_text()).get("task") or {}).get(
                    "episode_length"
                )
                print(f"  metadata {t}: collected episode_length={collected}")
    print("SPEC_OK")
