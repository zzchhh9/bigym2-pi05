"""STEP 2: build the ABSOLUTE-physical dataset for one v3 task.

Rewrites only two parquet columns and leaves everything else — crucially the
PNG image bytes — byte-identical, so this is a column edit, not a re-encode.

  action            -> raw_outer_action (PHYSICAL units)
  observation.state -> permuted so state[ARM_SLICE] are the arm joints

Both are required by lerobot's relative-action steps, which align
`action[i] -= state[i]` POSITIONALLY and assume shared units. The delta variant
reuses this same data; only meta/stats.json differs (see make_delta_v3.py).
"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                    # <repo>/scripts/delta_v3 -> repo root
WORK = Path(os.environ.get("PI05_WORK_ROOT", REPO.parent))

sys.path.insert(0, str(HERE))
from delta_spec_v3 import spec


def _check_protocol_budget(task: str, src: Path) -> None:
    """Fail export if demo recommended length is not registered in protocol.

    Demo packs stamp recommended_episode_length = ceil(2x longest demo...).
    eval_v3 only reads G1_EPISODE_BUDGETS / budget_provenance — it does NOT
    auto-import metadata. New tasks must update protocol.py before export.
    """
    meta_paths = [
        src / "metadata.json",
        src / "meta" / "source_metadata.json",
    ]
    recommended = None
    meta_used = None
    for mp in meta_paths:
        if not mp.is_file():
            continue
        meta = json.loads(mp.read_text())
        if "recommended_episode_length" in meta:
            recommended = int(meta["recommended_episode_length"])
            meta_used = mp
            break
    if recommended is None:
        print(
            f"WARN: {task}: no recommended_episode_length in metadata; "
            "skipping protocol budget gate"
        )
        return

    bigym_roots = [
        WORK / "bigym",
        REPO / "bigym",
        Path(os.environ.get("BIGYM_ROOT", "")),
    ]
    for root in bigym_roots:
        if root and (root / "bigym" / "loco" / "protocol.py").is_file():
            sys.path.insert(0, str(root))
            break
    try:
        from bigym.loco.protocol import budget_provenance, task_protocol
    except Exception as e:
        raise SystemExit(
            f"FAIL: {task}: cannot import bigym.loco.protocol for budget gate ({e}). "
            "Set PI05_WORK_ROOT/bigym or BIGYM_ROOT."
        ) from e

    row = task_protocol(task)
    protocol_len = int(row["episode_length"])
    prov = budget_provenance(task)
    msgs = []
    if protocol_len != recommended:
        msgs.append(
            f"G1_EPISODE_BUDGETS[{task!r}]={protocol_len} != "
            f"recommended_episode_length={recommended} (from {meta_used})"
        )
    if not str(prov).startswith("data_derived"):
        msgs.append(
            f"budget_provenance({task!r})={prov!r}; expected data_derived ... "
            "(add to G1_BUDGET_PROVENANCE in bigym/loco/protocol.py)"
        )
    if msgs:
        raise SystemExit(
            "FAIL: protocol budget not registered for export of new/updated task.\n  - "
            + "\n  - ".join(msgs)
            + "\nFix: run scripts/recommend_demo_episode_length.py, set "
            f"G1_EPISODE_BUDGETS[{task!r}]={recommended}, add G1_BUDGET_PROVENANCE "
            "entry, then re-export."
        )
    print(
        f"{task}: protocol budget gate OK "
        f"(episode_length={protocol_len} == recommended; provenance={prov})"
    )


SRC_ROOT = Path(os.environ.get("PI05_NATIVE_ROOT", str(WORK / "g1_native60")))
OUT_ROOT = Path(os.environ.get("PI05_LEROBOT_ROOT", str(WORK / "lerobot_v3")))
task = sys.argv[1]
SPEC = spec(task)                              # layout is PER TASK; see delta_spec_v3
src, out = SRC_ROOT / task, OUT_ROOT / f"{task}_abs"
_check_protocol_budget(task, src)
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)

# meta + any non-parquet payload copied verbatim
for item in src.iterdir():
    if item.name != "data":
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, out / item.name)

meta = json.loads((src / "meta" / "source_metadata.json").read_text())
lo = np.asarray(meta["action_stats"]["min"], np.float32)
hi = np.asarray(meta["action_stats"]["max"], np.float32)

acts, states, n_rows, worst_unnorm = [], [], 0, 0.0
for pf in sorted((src / "data").rglob("*.parquet")):
    t = pq.read_table(pf)
    a_norm = np.asarray(t["action"].to_pylist(), np.float32)
    raw = np.asarray(t["raw_outer_action"].to_pylist(), np.float32)
    st = np.asarray(t["observation.state"].to_pylist(), np.float32)
    assert raw.shape[1] == SPEC.action_dim and st.shape[1] == SPEC.state_dim, (pf, raw.shape, st.shape)
    # GATE: raw_outer_action must be exactly the un-normalised action
    un = (a_norm + 1) / 2 * (hi - lo + 1e-8) + lo
    worst_unnorm = max(worst_unnorm, float(np.abs(un - raw).max()))
    stp = SPEC.permute_state(st)
    t = t.set_column(t.schema.get_field_index("action"), "action",
                     pa.FixedSizeListArray.from_arrays(pa.array(raw.ravel(), pa.float32()), SPEC.action_dim))
    t = t.set_column(t.schema.get_field_index("observation.state"), "observation.state",
                     pa.FixedSizeListArray.from_arrays(pa.array(stp.ravel(), pa.float32()), SPEC.state_dim))
    dst = out / pf.relative_to(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst)
    acts.append(raw); states.append(stp); n_rows += raw.shape[0]

A, S = np.concatenate(acts), np.concatenate(states)
info = json.loads((out / "meta" / "info.json").read_text())
info["features"]["action"]["names"] = list(SPEC.action_names)
(out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

QK = {"q01": 1, "q10": 10, "q50": 50, "q90": 90, "q99": 99}
stats = json.loads((out / "meta" / "stats.json").read_text())
for key, arr in (("action", A), ("observation.state", S)):
    e = stats[key]
    for k, q in QK.items(): e[k] = np.percentile(arr, q, axis=0).tolist()
    e["mean"], e["std"] = arr.mean(0).tolist(), arr.std(0).tolist()
    e["min"], e["max"] = arr.min(0).tolist(), arr.max(0).tolist()
    e["count"] = [int(arr.shape[0])]
(out / "meta" / "stats.json").write_text(json.dumps(stats))

print(f"{task}: rows {n_rows} | max|unnorm(action)-raw| {worst_unnorm:.3e} "
      f"| action {A.shape} state {S.shape} -> {out}")
assert worst_unnorm < 1e-5, "raw_outer_action is NOT the un-normalised action"
print("EXPORT_OK")
