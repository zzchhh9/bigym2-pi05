"""Train and score tasks with the paper's recipe on one 8-GPU machine.

Per run: export + delta dataset, train with scripts/launch_v3.sh (6 GPUs x
batch 8, seed 1000, cosine decay over the run), then evaluate the last five
checkpoints (1000 steps apart, ending at the final step) on the 50 protocol
seeds each, 10 shards per checkpoint. A run is a task name, trained for 30000
steps as in the paper, or ``task:steps`` for another length; a run with another
length gets its own directory (``<task>_delta_<k>k``), so one task can be
trained at two lengths.

Runs train one after another; a finished run is scored on the eval GPUs while
the next one trains, and on --final-eval-gpus too once all training is done.
The script can be stopped and started again: finished checkpoints and complete
eval shards are kept, a trainer that is still running is waited for (not started
twice), and a trainer that died resumes from its newest checkpoint. A file named
STOP in --eval-out stops the script from starting further training; evaluation
of the runs already trained goes on.

    python scripts/run_tasks.py flip_cup stack_blocks stack_blocks:72000 \\
        --train-gpus 0,1,2,3,4,5 --eval-gpus 6,7

Training needs only CUDA. Evaluation renders with MuJoCo EGL; each eval worker
gets CUDA_VISIBLE_DEVICES=<gpu> and MUJOCO_EGL_DEVICE_ID=<gpu>, and workers on
one GPU start STAGGER seconds apart.
"""
import argparse, json, os, re, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("PI05_WORK_ROOT", HERE.parent / "work"))
PY = os.environ.get("PI05_PYTHON", sys.executable)
PAPER_STEPS, LAST, NSHARD, NEP, STAGGER = 30000, 5, 10, 50, 25


@dataclass(frozen=True)
class Run:
    """One training run: a task and its number of steps."""
    task: str
    steps: int

    @property
    def name(self):
        return f"{self.task}_delta" + ("" if self.steps == PAPER_STEPS else f"_{self.steps // 1000}k")

    @property
    def eval_steps(self):
        return [self.steps - 1000 * i for i in reversed(range(LAST))]

    @classmethod
    def parse(cls, spec):
        task, _, steps = spec.partition(":")
        return cls(task, int(steps) if steps else PAPER_STEPS)


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


def eval_shard(pool, runs, out_dir, run, step, w):
    out = out_dir / f"{run.name}_{step:06d}_w{w}.json"
    if shard_complete(out, w):
        return 0
    ckpt = runs / run.name / f"checkpoints/{step:06d}/pretrained_model"
    for attempt in range(3):
        slot = pool.take(); g = slot[0]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), MUJOCO_EGL_DEVICE_ID=str(g))
        out.unlink(missing_ok=True)
        with open(out.with_suffix(".log"), "w") as fh:
            rc = subprocess.run(
                [PY, "-u", str(HERE / "eval_v3.py"), "--task", run.task, "--ckpt", str(ckpt),
                 "--out", str(out), "--episodes", str(NEP), "--num-workers", str(NSHARD),
                 "--worker-id", str(w), "--n-action-steps", "16"],
                env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        pool.give(slot)
        if rc == 0 and shard_complete(out, w):
            return 0
        log(f"EVAL FAIL {run.name} {step} w{w} gpu{g} rc={rc} attempt {attempt}")
    return 1


def steps_on_disk(ckdir):
    return sorted(int(p.name) for p in ckdir.glob("[0-9]*") if p.is_dir()) if ckdir.exists() else []


def prune(ckdir, keep_from):
    # Keep the two newest: the newest may still be being written when a trainer dies.
    for s in steps_on_disk(ckdir)[:-2]:
        if s < keep_from:
            shutil.rmtree(ckdir / f"{s:06d}", ignore_errors=True)


def prepare(task):
    delta = Path(os.environ.get("PI05_LEROBOT_ROOT", WORK / "lerobot_v3")) / f"{task}_delta"
    if (delta / "meta" / "stats.json.absolute_backup").exists():
        return
    for script in ("export_v3.py", "make_delta_v3.py"):
        subprocess.run([PY, str(HERE / script), task], check=True)


def running_trainer(out):
    """PID of a torchrun launcher already training into ``out``, else None."""
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if "torch.distributed.run" in args and any(
                a == f"--output_dir={out}" or a.startswith(f"--config_path={out}/") for a in args):
            return int(proc.name)
    return None


def wait_trainer(pid, ckdir, keep_from):
    while Path(f"/proc/{pid}").exists():
        prune(ckdir, keep_from); time.sleep(300)
    prune(ckdir, keep_from)


def train(runs, run, gpus):
    out = runs / run.name; ckdir = out / "checkpoints"
    for attempt in range(4):
        if run.steps in steps_on_disk(ckdir):
            log(f"TRAIN DONE {run.name}")
            return True
        pid = running_trainer(out)
        if pid:
            log(f"TRAIN {run.name}: waiting for the trainer already running (pid {pid})")
            wait_trainer(pid, ckdir, run.eval_steps[0])
            continue
        resume = "1" if steps_on_disk(ckdir) else "0"
        if resume == "0" and out.exists():
            shutil.rmtree(out)                 # died before its first checkpoint
        env = dict(os.environ, TASK=run.task, OUT=str(out), JOB=f"pi05_{run.name}", GPUS=gpus,
                   STEPS=str(run.steps), SAVE="1000", RESUME=resume)
        r = subprocess.run(["bash", str(HERE / "launch_v3.sh")], env=env, capture_output=True, text=True)
        m = re.search(r"launched pid (\d+)", r.stdout)
        if not m:
            log(f"TRAIN LAUNCH FAIL {run.name}: {r.stdout[-400:]} {r.stderr[-400:]}")
            return False
        pid = int(m.group(1)); logline = next((l for l in r.stdout.splitlines() if "log:" in l), "")
        log(f"TRAIN {run.name} pid {pid} resume={resume} attempt {attempt} {logline.strip()}")
        wait_trainer(pid, ckdir, run.eval_steps[0])
        log(f"TRAIN EXIT {run.name}: checkpoints {steps_on_disk(ckdir)[-3:]}")
    return run.steps in steps_on_disk(ckdir)


def summary(out_dir, run):
    row, tot, n, bad = [], 0, 0, []
    for s in run.eval_steps:
        files = [out_dir / f"{run.name}_{s:06d}_w{w}.json" for w in range(NSHARD)]
        docs = [json.load(open(f)) for f in files if f.exists()]
        eps = [e for d in docs for e in d["episodes"]]
        k = sum(e["success"] for e in eps); tot += k; n += len(eps)
        if sorted(e["seed"] for e in eps) != list(range(620000, 620000 + NEP)):
            bad.append(f"{s}: seeds incomplete")
        bad += [f"{s}: {d['protocol_violations']}" for d in docs if d.get("protocol_violations")]
        row.append(f"{s // 1000}k:{k}/{len(eps)}")
    log(f"RESULT {run.name}: {' '.join(row)} pooled {tot}/{n} problems={bad or 'none'}")
    return {"run": run.name, "task": run.task, "steps": run.steps, "per_step": row,
            "pooled": [tot, n], "problems": bad}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs_", nargs="+", metavar="task[:steps]")
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
    ex = ThreadPoolExecutor(64); reporters, done = [], []

    def report(run, fs):
        log(f"EVAL DONE {run.name} failed_shards={sum(f.result() for f in fs)}")
        res = summary(out_dir, run); done.append(res)
        (out_dir / f"{run.name}_summary.json").write_text(json.dumps(res, indent=1))

    for run in map(Run.parse, a.runs_):
        if (out_dir / "STOP").exists():
            log(f"STOP file found: no more training ({run.name} and later are skipped)"); break
        prepare(run.task)
        if not train(runs, run, a.train_gpus):
            log(f"GIVE UP training {run.name}"); continue
        fs = [ex.submit(eval_shard, pool, runs, out_dir, run, s, w)
              for s in run.eval_steps for w in range(NSHARD)]
        reporters.append(threading.Thread(target=report, args=(run, fs)))
        reporters[-1].start()
    pool.add([int(g) for g in (a.final_eval_gpus or a.train_gpus).split(",")])
    for t in reporters:
        t.join()
    (out_dir / "summary.json").write_text(json.dumps(done, indent=1))
    log("ALL_DONE")


if __name__ == "__main__":
    main()
