"""STEP 2b: derive the DELTA variant of a v3 task from its _abs dataset.

The frame data is IDENTICAL — lerobot's RelativeActionsProcessorStep does
`action -= state` at train time. The only thing that must change is
meta/stats.json: `normalize` runs AFTER the relative step but reads stats
computed on ABSOLUTE actions, so without this the delta (std ~0.03) is scaled
by absolute quantiles (std ~0.3) under pi0.5's QUANTILES mode. OpenPI computes
norm stats after DeltaActions; this is the equivalent.

Parquet files are HARD-LINKED, so the variant costs no extra disk.
"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                        # <repo>/scripts -> repo root
WORK = Path(os.environ.get("PI05_WORK_ROOT", REPO / "work"))

sys.path.insert(0, str(HERE))
from delta_spec_v3 import spec

ROOT = Path(os.environ.get("PI05_LEROBOT_ROOT", str(WORK / "lerobot_v3")))

# pi0.5 QUANTILES is (x-q01)/(q99-q01) with NO clip (lerobot NormalizerProcessorStep).
# When a dim is almost-constant, q01==q99 so the denom becomes eps=1e-8, and any
# leftover outlier (e.g. a rare pelvis-vy on dishwasher_load_cups) becomes a
# 1e7-scale action target → flow-matching MSE ~4e10. Floor those dims to min/max.
_Q_EPS = 1e-8


def floor_degenerate_action_quantiles(stats: dict) -> int:
    """Widen q01/q99 (and collapsed q10/q90) when percentiles pin but min!=max."""
    e = stats["action"]
    q01 = np.asarray(e["q01"], np.float64)
    q99 = np.asarray(e["q99"], np.float64)
    mn = np.asarray(e["min"], np.float64)
    mx = np.asarray(e["max"], np.float64)
    bad = (np.abs(q99 - q01) < _Q_EPS) & (np.abs(mx - mn) > _Q_EPS)
    n = int(bad.sum())
    if n:
        q01 = q01.copy(); q99 = q99.copy()
        q01[bad] = mn[bad]
        q99[bad] = mx[bad]
        e["q01"] = q01.tolist()
        e["q99"] = q99.tolist()
        for lo, hi in (("q10", "q90"),):
            if lo not in e or hi not in e:
                continue
            a = np.asarray(e[lo], np.float64).copy()
            b = np.asarray(e[hi], np.float64).copy()
            collapsed = (np.abs(b - a) < _Q_EPS) & bad
            a[collapsed] = mn[collapsed]
            b[collapsed] = mx[collapsed]
            e[lo] = a.tolist()
            e[hi] = b.tolist()
        print(f"  floored {n} degenerate QUANTILES dim(s): {np.where(bad)[0].tolist()}")
    return n


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--repair":
        task = sys.argv[2]
        fn = ROOT / f"{task}_delta" / "meta" / "stats.json"
        if not fn.is_file():
            raise SystemExit(f"no stats at {fn}")
        bak = Path(str(fn) + ".pre_quantile_floor")
        if not bak.exists():
            shutil.copy(fn, bak)
        stats = json.loads(fn.read_text())
        n = floor_degenerate_action_quantiles(stats)
        fn.write_text(json.dumps(stats))
        print(f"REPAIR_OK {task} floored={n} -> {fn}")
        raise SystemExit(0)

    task = sys.argv[1]
    SPEC = spec(task)                              # layout is PER TASK; see delta_spec_v3
    src, out = ROOT / f"{task}_abs", ROOT / f"{task}_delta"
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    for item in src.rglob("*"):
        dst = out / item.relative_to(src)
        if item.is_dir(): dst.mkdir(parents=True, exist_ok=True)
        elif item.suffix == ".parquet": dst.parent.mkdir(parents=True, exist_ok=True); os.link(item, dst)
        else: dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(item, dst)

    mask = SPEC.relative_mask()
    deltas = []
    for pf in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(pf)
        a = np.asarray(t["action"].to_pylist(), np.float32)
        s = np.asarray(t["observation.state"].to_pylist(), np.float32)[:, :SPEC.action_dim]
        d = a.copy(); d[:, mask] -= s[:, mask]
        # GATE: the inverse must restore the absolute action exactly (float32)
        back = d.copy(); back[:, mask] += s[:, mask]
        assert np.abs(back - a).max() < 1e-5, (pf, float(np.abs(back - a).max()))
        deltas.append(d)
    D = np.concatenate(deltas)

    QK = {"q01": 1, "q10": 10, "q50": 50, "q90": 90, "q99": 99}
    fn = out / "meta" / "stats.json"
    shutil.copy(fn, str(fn) + ".absolute_backup")
    stats = json.loads(fn.read_text()); e = stats["action"]
    for key, q in QK.items():
        v = np.asarray(e[key], np.float64); v[mask] = np.percentile(D[:, mask], q, axis=0); e[key] = v.tolist()
    for key, f in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        v = np.asarray(e[key], np.float64); v[mask] = f(D[:, mask], axis=0); e[key] = v.tolist()
    n_floor = floor_degenerate_action_quantiles(stats)
    fn.write_text(json.dumps(stats))

    du = sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and f.suffix != ".parquet")
    print(f"{task}: {D.shape[0]} frames | delta |max| {np.abs(D[:, mask]).max():.4f} "
          f"p99 {np.percentile(np.abs(D[:, mask]), 99):.4f} std {D[:, mask].std():.4f} "
          f"| quantile_floor={n_floor} extra disk {du/1e6:.1f}MB -> {out}")
    print("DELTA_OK")
