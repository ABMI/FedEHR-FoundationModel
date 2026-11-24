"""Evaluate the 12 EHRSHOT tasks for every FM x evaluation site: embed on a GPU queue, logistic on CPU.

Pairs: site FMs on their own site, FedAvg-7 and FedAvg-6 (minus D5) on all sites, other site FMs on D5.
Usage: python scripts/run_ehrshot_eval.py [--loso SITE]   (--loso: only FedAvg-6 minus SITE on all sites)
"""
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "outputs" / "sites_disjoint"
PYTHON = str(ROOT / ".venv" / "bin" / "python")
LOGS = SD / "logs" / "ehrshot"
GPUS = ["2", "3", "4"]
MAX_LOGISTIC = 6
ORDER = ["D3_cancer", "D5_elderly", "D2_er", "D0_random", "A", "D4_female", "D1_pediatric"]  # most cutoff dates first
FL7 = SD / "fedavg_7sites" / "final" / "state.pt"
FL6 = SD / "fedavg6_minus_D5_elderly" / "final" / "state.pt"


def local(site: str) -> Path:
    if site == "A":
        return ROOT / "outputs" / "m0" / "training" / "continued" / "epoch-20" / "state.pt"
    return SD / site / "joint" / "training" / "continued" / "epoch-20" / "state.pt"


JOBS = []
LOSO = sys.argv[sys.argv.index("--loso") + 1] if "--loso" in sys.argv else None
if LOSO:
    JOBS = [(f"fedavg6_minus_{LOSO}_round20", SD / f"fedavg6_minus_{LOSO}" / "final" / "state.pt", site) for site in ORDER]
else:
    for site in ORDER:
        JOBS += [(f"{site}_epoch20", local(site), site), ("fedavg7_round20", FL7, site),
                 ("fedavg6_minus_D5_elderly_round20", FL6, site)]
    for source in ["D3_cancer", "D2_er", "D0_random", "A", "D4_female", "D1_pediatric"]:
        JOBS.append((f"{source}_epoch20", local(source), "D5_elderly"))

FRACTIONS = dict(item.split(":") for item in os.environ.get("GPU_MEMORY_FRACTIONS", "").split(",") if item)
RUN_CAPPED = str(ROOT / "scripts" / "run_capped.py")
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
logistic_slots = threading.Semaphore(MAX_LOGISTIC)
threads = []
failures = []


def stamp() -> str:
    return time.strftime("%m-%d %H:%M:%S")


def logistic(name: str, site: str, output: Path) -> None:
    with logistic_slots:
        print(f"{stamp()} logistic {name}@{site} start", flush=True)
        env = {**ENV, "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "POLARS_MAX_THREADS": "8"}
        with open(LOGS / f"logistic_{name}@{site}.log", "w") as log:
            code = subprocess.run([PYTHON, "-u", "-m", "ausom_ehr.ehrshot_tasks", "logistic", "--site", site,
                                   "--output", str(output)], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{stamp()} logistic {name}@{site} done (exit {code})", flush=True)
        if code:
            failures.append(f"logistic {name}@{site}")


def gpu_worker(gpu: str, jobs: queue.Queue) -> None:
    while True:
        try:
            name, model, site = jobs.get_nowait()
        except queue.Empty:
            return
        output = SD / "eval_ehrshot" / f"{name}@{site}"
        done = output / "evaluation" / "continued" / "epoch20" / "test_predictions.parquet"
        embedded = output / "embeddings" / "continued" / "epoch20" / "ehrshot_cutoffs.parquet"
        if done.exists():
            print(f"{stamp()} {name}@{site} already done", flush=True)
            continue
        if not embedded.exists():
            print(f"{stamp()} embed {name}@{site} start (GPU {gpu})", flush=True)
            launcher, env = [PYTHON, "-u", "-m"], {**ENV, "CUDA_VISIBLE_DEVICES": gpu}
            if gpu in FRACTIONS:
                launcher, env["GPU_MEMORY_FRACTION"] = [PYTHON, "-u", RUN_CAPPED], FRACTIONS[gpu]
            with open(LOGS / f"embed_{name}@{site}.log", "w") as log:
                code = subprocess.run([*launcher, "ausom_ehr.ehrshot_tasks", "embed", "--site", site,
                                       "--model", str(model), "--output", str(output)],
                                      env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"{stamp()} embed {name}@{site} done (exit {code}, GPU {gpu})", flush=True)
            if code:
                failures.append(f"embed {name}@{site}")
                continue
        thread = threading.Thread(target=logistic, args=(name, site, output))
        thread.start()
        threads.append(thread)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    for _name, model, _site in JOBS:
        if not model.exists():
            sys.exit(f"Missing checkpoint: {model}")
    for site in ORDER:
        summary = SD / "tasks_ehrshot" / site / "cohorts" / "inputs_summary.json"
        if not summary.exists():
            sys.exit(f"Missing task inputs: {summary}")
    busy = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, env=ENV).stdout
    in_use = [line.split(",")[0] for line in busy.strip().splitlines()
              if line.split(",")[0].strip() in GPUS and line.split(",")[0].strip() not in FRACTIONS
              and int(line.split(",")[1]) > 1000]
    if in_use:
        sys.exit(f"GPUs already in use: {in_use}")
    jobs = queue.Queue()
    for job in JOBS:
        jobs.put(job)
    print(f"{stamp()} starting {len(JOBS)} jobs", flush=True)
    workers = [threading.Thread(target=gpu_worker, args=(gpu, jobs)) for gpu in GPUS]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    for thread in threads:
        thread.join()
    print(f"{stamp()} all done. {len(failures)} failed: {failures}", flush=True)


if __name__ == "__main__":
    main()
