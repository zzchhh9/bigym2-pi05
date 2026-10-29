"""Train and score tasks with the paper's recipe on one 8-GPU machine.

Per task: export + delta dataset, train with scripts/launch_v3.sh (6 GPUs x
batch 8, 30k steps, seed 1000), then evaluate the last five checkpoints
(26k-30k) on the 50 protocol seeds each, 10 shards per checkpoint. Tasks train
one after another; a finished task is scored on the eval GPUs while the next one
trains, and on the training GPUs too once all training is done. The script can
be stopped and started again: finished checkpoints and complete eval shards are
kept, and a trainer that died resumes from its newest checkpoint.

    python scripts/run_tasks.py flip_cup put_cups --train-gpus 0,1,2,3,4,5 --eval-gpus 6,7

Training needs only CUDA. Evaluation renders with MuJoCo EGL; each eval worker
gets CUDA_VISIBLE_DEVICES=<gpu> and MUJOCO_EGL_DEVICE_ID=<gpu>, and workers on
one GPU start STAGGER seconds apart.
"""
import argparse, json, os, re, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("PI05_WORK_ROOT", HERE.parent / "work"))
PY = os.environ.get("PI05_PYTHON", sys.executable)
STEPS, KEEP_FROM = 30000, 26000
EVAL_STEPS, NSHARD, NEP, STAGGER = [26000, 27000, 28000, 29000, 30000], 10, 50, 25


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


class EvalPool:
    """GPU slots for eval workers; same-GPU starts are STAGGER seconds apart."""

    def __init__(self, gpus, per_gpu):
        self.cv, self.free, self.last = threading.Condition(), [], {}
        self.per_gpu = per_gpu
        self.add(gpus)

    def add(self, gpus):
        with self.cv:
            self.free.extend((g, i) for i in range(self.per_gpu) for g in gpus)
            self.cv.notify_all()

    def take(self):
        with self.cv:
            while not self.free:
                self.cv.wait()
            g, i = self.free.pop(0)
            t0 = max(time.time(), self.last.get(g, 0.0) + STAGGER)
            self.last[g] = t0
        time.sleep(max(0.0, t0 - time.time()))
        return g, i

    def give(self, slot):
        with self.cv:
            self.free.append(slot); self.cv.notify()


def shard_complete(path, w):
    try:
        return len(json.load(open(path))["episodes"]) == len(range(w, NEP, NSHARD))
    except Exception:
        return False


def eval_shard(pool, runs, out_dir, task, step, w):
    out = out_dir / f"{task}_{step:06d}_w{w}.json"
    if shard_complete(out, w):
        return 0
    ckpt = runs / f"{task}_delta/checkpoints/{step:06d}/pretrained_model"
    for attempt in range(3):
        slot = pool.take(); g = slot[0]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), MUJOCO_EGL_DEVICE_ID=str(g))
        out.unlink(missing_ok=True)
        with open(out.with_suffix(".log"), "w") as fh:
            rc = subprocess.run(
                [PY, "-u", str(HERE / "eval_v3.py"), "--task", task, "--ckpt", str(ckpt),
                 "--out", str(out), "--episodes", str(NEP), "--num-workers", str(NSHARD),
                 "--worker-id", str(w), "--n-action-steps", "16"],
                env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        pool.give(slot)
        if rc == 0 and shard_complete(out, w):
            return 0
        log(f"EVAL FAIL {task} {step} w{w} gpu{g} rc={rc} attempt {attempt}")
    return 1


def steps_on_disk(ckdir):
    return sorted(int(p.name) for p in ckdir.glob("[0-9]*") if p.is_dir()) if ckdir.exists() else []


def prune(ckdir):
    for s in steps_on_disk(ckdir)[:-1]:      # always keep the newest (resume point)
        if s < KEEP_FROM:
            shutil.rmtree(ckdir / f"{s:06d}", ignore_errors=True)


def prepare(task):
    delta = Path(os.environ.get("PI05_LEROBOT_ROOT", WORK / "lerobot_v3")) / f"{task}_delta"
    if (delta / "meta" / "stats.json.absolute_backup").exists():
        return
    for script in ("export_v3.py", "make_delta_v3.py"):
        subprocess.run([PY, str(HERE / script), task], check=True)


def train(runs, task, gpus):
    out = runs / f"{task}_delta"; ckdir = out / "checkpoints"
    for attempt in range(4):
        if STEPS in steps_on_disk(ckdir):
            log(f"TRAIN DONE {task}")
            return True
        resume = "1" if steps_on_disk(ckdir) else "0"
        if resume == "0" and out.exists():
            shutil.rmtree(out)                 # died before its first checkpoint
        env = dict(os.environ, TASK=task, OUT=str(out), JOB=f"pi05_{task}", GPUS=gpus,
                   STEPS=str(STEPS), SAVE="1000", RESUME=resume)
        r = subprocess.run(["bash", str(HERE / "launch_v3.sh")], env=env, capture_output=True, text=True)
        m = re.search(r"launched pid (\d+)", r.stdout)
        if not m:
            log(f"TRAIN LAUNCH FAIL {task}: {r.stdout[-400:]} {r.stderr[-400:]}")
            return False
        pid = int(m.group(1))
        log(f"TRAIN {task} pid {pid} resume={resume} attempt {attempt}")
        while Path(f"/proc/{pid}").exists():
            prune(ckdir); time.sleep(300)
        prune(ckdir)
        log(f"TRAIN EXIT {task}: checkpoints {steps_on_disk(ckdir)[-3:]}")
    return STEPS in steps_on_disk(ckdir)


def summary(out_dir, task):
    row, tot, n = [], 0, 0
    for s in EVAL_STEPS:
        eps = [e for w in range(NSHARD) if (out_dir / f"{task}_{s:06d}_w{w}.json").exists()
               for e in json.load(open(out_dir / f"{task}_{s:06d}_w{w}.json"))["episodes"]]
        k = sum(e["success"] for e in eps); tot += k; n += len(eps)
        row.append(f"{s // 1000}k:{k}/{len(eps)}")
    log(f"RESULT {task}: {' '.join(row)} pooled {tot}/{n}")
    return {"task": task, "per_step": row, "pooled": [tot, n]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", nargs="+")
    p.add_argument("--train-gpus", default="0,1,2,3,4,5")
    p.add_argument("--eval-gpus", default="6,7")
    p.add_argument("--final-eval-gpus", default=None,
                   help="GPUs added to eval once all training is done (default: --train-gpus)")
    p.add_argument("--eval-per-gpu", type=int, default=2)
    p.add_argument("--runs", default=str(WORK / "pi05_runs_v3"))
    p.add_argument("--eval-out", default=str(WORK / "eval"))
    a = p.parse_args()
    runs, out_dir = Path(a.runs), Path(a.eval_out)
    runs.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    pool = EvalPool([int(g) for g in a.eval_gpus.split(",")], a.eval_per_gpu)
    ex = ThreadPoolExecutor(64); futs = {}
    for task in a.tasks:
        prepare(task)
        if not train(runs, task, a.train_gpus):
            log(f"GIVE UP training {task}"); continue
        futs[task] = [ex.submit(eval_shard, pool, runs, out_dir, task, s, w)
                      for s in EVAL_STEPS for w in range(NSHARD)]
    pool.add([int(g) for g in (a.final_eval_gpus or a.train_gpus).split(",")])
    results = []
    for task, fs in futs.items():
        log(f"EVAL DONE {task} failed_shards={sum(f.result() for f in fs)}")
        results.append(summary(out_dir, task))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    log("ALL_DONE")


if __name__ == "__main__":
    main()
