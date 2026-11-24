"""Evaluate the 36 missing cells of the 7x7 site-FM x evaluation-site matrix (3 original + 12 EHRSHOT tasks).

--gpus takes GPU:memory-fraction pairs; claim files keep parallel launches from repeating a job.
Usage: python scripts/run_cross_matrix.py --gpus 2:1,4:0.5 [--target SITE] [--dry-run]
"""
import argparse
import filecmp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "outputs" / "sites_disjoint"
PYTHON = str(ROOT / ".venv" / "bin" / "python")
CAPPED = str(ROOT / "scripts" / "run_capped.py")
LOGS = SD / "logs" / "cross_matrix"
CLAIMS = LOGS / "claims"
SITES = ["A", "D0_random", "D1_pediatric", "D2_er", "D3_cancer", "D4_female", "D5_elderly"]
# Measured GPU hours of one old (3-task) / new (12-task) job per evaluation site; longest jobs run first.
OLD_HOURS = {"D3_cancer": 2.2, "D2_er": 1.6, "A": 1.15, "D0_random": 1.1, "D4_female": 0.7, "D1_pediatric": 0.55, "D5_elderly": 1.67}
NEW_HOURS = {"D3_cancer": 1.9, "D2_er": 1.55, "A": 1.3, "D0_random": 1.3, "D4_female": 1.05, "D1_pediatric": 0.9, "D5_elderly": 1.55}
MAX_LOGISTIC = 4
OLD_TASKS = ["los7", "readmit30", "outpatient90"]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
sys.path.insert(0, str(ROOT / "src"))
from ausom_ehr.common import RAW_ROOT  # noqa: E402
from ausom_ehr.federated import prepare_eval  # noqa: E402

PHASE1 = [("D3_cancer", t) for t in ["A", "D0_random", "D1_pediatric", "D2_er", "D4_female"]] + \
         [("D5_elderly", t) for t in ["A", "D0_random", "D1_pediatric", "D2_er", "D3_cancer", "D4_female"]]
PHASE2 = [(s, t) for s in ["A", "D0_random", "D1_pediatric", "D2_er", "D4_female"]
          for t in ["A", "D0_random", "D1_pediatric", "D2_er", "D3_cancer", "D4_female"] if s != t]


def population(site: str) -> Path:
    return ROOT / "outputs" / "m0" if site == "A" else SD / site / "joint"


def checkpoint(site: str) -> Path:
    return population(site) / "training" / "continued" / "epoch-20" / "state.pt"


def old_out(source: str, target: str) -> Path:
    return SD / "eval" / f"{source}_epoch20@{target}"


def new_out(source: str, target: str) -> Path:
    return SD / "eval_ehrshot" / f"{source}_epoch20@{target}"


def done(kind: str, source: str, target: str) -> bool:
    if kind == "old":
        return (old_out(source, target) / "evaluation/continued/epoch20/results.json").exists()
    return (new_out(source, target) / "evaluation/continued/epoch20/test_predictions.parquet").exists()


def job_list(target: str | None = None) -> list[tuple[int, str, str, str]]:
    jobs = []
    for phase, cells in [(1, PHASE1), (2, PHASE2)]:
        items = [(phase, kind, s, t) for s, t in cells for kind in ["old", "new"] if target in (None, t)]
        items.sort(key=lambda j: -(OLD_HOURS if j[1] == "old" else NEW_HOURS)[j[3]])
        jobs += items
    if target:  # with one target site, finish the 12 new tasks first
        jobs.sort(key=lambda j: j[1] != "new")
    return jobs


def stamp() -> str:
    return time.strftime("%m-%d %H:%M:%S")


failures: list[str] = []
threads: list[threading.Thread] = []
logistic_slots = threading.Semaphore(MAX_LOGISTIC)
phase_reported = set()
report_lock = threading.Lock()


def report_phase(jobs) -> None:
    with report_lock:
        for phase in [1, 2]:
            if phase in phase_reported:
                continue
            if all(done(k, s, t) for p, k, s, t in jobs if p == phase):
                phase_reported.add(phase)
                print(f"{stamp()} phase {phase} done", flush=True)


def claim(name: str) -> bool:
    try:
        os.close(os.open(CLAIMS / f"{name}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def release(name: str) -> None:
    (CLAIMS / f"{name}.claim").unlink(missing_ok=True)


def logistic(command: list[str], name: str, jobs) -> None:
    with logistic_slots:
        print(f"{stamp()} logistic {name} start", flush=True)
        env = {**ENV, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
               "MKL_NUM_THREADS": "8", "POLARS_MAX_THREADS": "8"}
        with open(LOGS / f"logistic_{name}.log", "w") as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{stamp()} logistic {name} done (exit {code})", flush=True)
        if code:
            failures.append(f"logistic {name}")
            release(name)
        report_phase(jobs)


def worker(gpu: str, fraction: float, jobs) -> None:
    env = {**ENV, "CUDA_VISIBLE_DEVICES": gpu, "GPU_MEMORY_FRACTION": str(fraction)}
    for phase, kind, source, target in jobs:
        name = f"{kind}_{source}@{target}"
        if done(kind, source, target) or not claim(name):
            continue
        if kind == "old":
            # Same steps as federated run_eval, but embed runs under run_capped so the memory cap applies.
            out = old_out(source, target)
            config = prepare_eval(checkpoint(source), population(target).resolve(), out, RAW_ROOT)
            common = ["--config", str(config), "--initialization", "continued", "--run-name", "epoch20"]
            folder = out / "embeddings/continued/epoch20"
            embedded = (folder / "summary.json").exists() and all((folder / f"{t}.parquet").exists() for t in OLD_TASKS)
            embed_command = [PYTHON, "-u", CAPPED, "ausom_ehr.evaluate", "embed", *common]
            logistic_command = [PYTHON, "-u", "-m", "ausom_ehr.evaluate", "logistic", *common]
        else:
            out = new_out(source, target)
            embedded = (out / "embeddings/continued/epoch20/ehrshot_cutoffs.parquet").exists()
            embed_command = [PYTHON, "-u", CAPPED, "ausom_ehr.ehrshot_tasks", "embed", "--site", target,
                             "--model", str(checkpoint(source)), "--output", str(out)]
            logistic_command = [PYTHON, "-u", "-m", "ausom_ehr.ehrshot_tasks", "logistic", "--site", target, "--output", str(out)]
        if not embedded:
            print(f"{stamp()} [phase {phase}] embed {name} start (GPU {gpu})", flush=True)
            with open(LOGS / f"embed_{name}.log", "w") as log:
                code = subprocess.run(embed_command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"{stamp()} [phase {phase}] embed {name} done (exit {code}, GPU {gpu})", flush=True)
            if code:
                failures.append(f"embed {name}")
                release(name)
                continue
        thread = threading.Thread(target=logistic, args=(logistic_command, name, jobs))
        thread.start()
        threads.append(thread)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2:1,4:0.5")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", choices=SITES, help="evaluate only this evaluation-site column")
    args = parser.parse_args()
    gpus = [(item.split(":")[0], float(item.split(":")[1])) for item in args.gpus.split(",")]
    jobs = job_list(args.target)
    pending = [j for j in jobs if not done(j[1], j[2], j[3])]
    hours = sum((OLD_HOURS if k == "old" else NEW_HOURS)[t] for _, k, _, t in pending)
    print(f"{stamp()} {len(pending)} of {len(jobs)} jobs left, estimated GPU hours {hours:.1f}", flush=True)
    if args.dry_run:
        for p, k, s, t in pending:
            print(f"  phase {p} {k:3s} {s} -> {t} ({(OLD_HOURS if k == 'old' else NEW_HOURS)[t]}h)")
        return
    problems = []
    for _, kind, source, target in pending:
        if not checkpoint(source).exists():
            problems.append(f"Missing checkpoint: {checkpoint(source)}")
        if not filecmp.cmp(population(source) / "tokens/vocab.parquet", population(target) / "tokens/vocab.parquet", shallow=False):
            problems.append(f"Vocabularies differ: {source} vs {target}")
        need = (population(target) / "cohorts/inputs_summary.json") if kind == "old" else \
            (SD / "tasks_ehrshot" / target / "cohorts/inputs_summary.json")
        if not need.exists():
            problems.append(f"Missing evaluation inputs: {need}")
    busy = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, env=ENV).stdout
    used = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in busy.strip().splitlines()}
    for gpu, fraction in gpus:
        if fraction >= 1 and used.get(gpu, 0) > 1000:
            problems.append(f"GPU {gpu} in use by another process ({used[gpu]} MiB)")
        if fraction < 1 and used.get(gpu, 0) > 49140 * (1 - fraction):
            problems.append(f"GPU {gpu} has less than {fraction:.0%} memory free ({used[gpu]} MiB in use)")
    if problems:
        sys.exit("Preflight check failed:\n" + "\n".join(sorted(set(problems))))
    LOGS.mkdir(parents=True, exist_ok=True)
    CLAIMS.mkdir(parents=True, exist_ok=True)
    print(f"{stamp()} starting on GPUs {gpus}", flush=True)
    workers = [threading.Thread(target=worker, args=(gpu, fraction, jobs)) for gpu, fraction in gpus]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    for t in threads:
        t.join()
    report_phase(jobs)
    print(f"{stamp()} run finished. {len(failures)} failed: {failures}", flush=True)


if __name__ == "__main__":
    main()
