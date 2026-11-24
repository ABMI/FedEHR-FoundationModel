"""Run a Python module with GPU memory capped at GPU_MEMORY_FRACTION (0-1), for shared GPUs.

Exceeding the cap ends the run with a PyTorch out-of-memory error.
Usage: GPU_MEMORY_FRACTION=0.5 CUDA_VISIBLE_DEVICES=4 python scripts/run_capped.py MODULE [ARGS ...]
"""
import os
import runpy
import sys

import torch

fraction = float(os.environ.get("GPU_MEMORY_FRACTION", "1"))
if fraction < 1 and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(f"GPU memory cap: {fraction:.0%}", flush=True)
module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
