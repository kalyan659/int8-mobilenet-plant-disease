"""How much of the INT8 CPU speed-up is operator fusion rather than integer arithmetic?

Reviewer question (pre-submission review, 2026-11-17): on Kaggle the FP32 CPU model was timed unfused while the INT8 models
are fused (Conv-BN(-ReLU) folded during quantization), so part of the reported speed-up may come from fusion. This script
times, on ONE local machine and with the notebook's latency protocol (batch 1, 10 warm-up runs, median of 50 timed runs,
1 and 4 threads):

  FP32 unfused (eager)      - what the notebook timed for FP32
  FP32 fused (eager)        - same weights with BatchNorm folded into the convolutions (eval-mode fusion), no quantization
  FP32 unfused / fused (TorchScript traced) - the same two, traced, to match the execution mode of the saved INT8 files
  INT8 PTQ (TorchScript)    - the model saved by the Kaggle run (models/<arch>_int8_ptq_torchscript.pt)
  INT8 PTQ outlier-aware (TorchScript)

Weights come from the v2.3.0 seed-0 runs in research/runs/. Latency does not depend on the input values, so random
inputs are used, as in the notebook. Numbers are local-machine numbers (different CPU and quantized engine from Kaggle):
use them only as within-machine ratios, not as replacements for the Kaggle latencies.

    <venv with torch + torchvision>/python -I fusion_latency_check.py > fusion_latency_check.out.txt
"""
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_small
from torchvision.models.quantization import mobilenet_v2 as quantizable_mobilenet_v2

warnings.filterwarnings("ignore")
RUNS = Path(__file__).resolve().parents[1] / "runs"
IMG, WARMUP, TIMED, ROUNDS = 224, 10, 50, 5


def load_fp32(arch, seed=0):
    m = quantizable_mobilenet_v2(weights=None, quantize=False) if arch == "mobilenet_v2" else mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 38)
    ck = torch.load(RUNS / f"{arch}_seed{seed}" / "checkpoints" / "best_fp32.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state"])
    return m.eval()


def fuse(arch, m):
    if arch == "mobilenet_v2":
        m.fuse_model(is_qat=False)          # torchvision's own eager fusion: Conv-BN(-ReLU) -> one conv with folded BN
        return m
    from torch.ao.quantization.quantize_fx import fuse_fx
    return fuse_fx(m)                       # FX fusion (eval mode folds BN), as used before FX quantization


@torch.inference_mode()
def median_ms(m, threads):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, IMG, IMG)
    for _ in range(WARMUP):
        m(x)
    t = []
    for _ in range(TIMED):
        s = time.perf_counter()
        m(x)
        t.append((time.perf_counter() - s) * 1000)
    return statistics.median(t)


@torch.inference_mode()
def max_abs_diff(a, b):
    x = torch.randn(4, 3, IMG, IMG)
    return float((a(x) - b(x)).abs().max())


def main():
    torch.backends.quantized.engine = "x86" if "x86" in torch.backends.quantized.supported_engines else "onednn"
    print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | quantized engine {torch.backends.quantized.engine}")
    print(f"{platform.processor()} | {platform.system()} {platform.release()} | logical CPUs {torch.get_num_threads()} (default)")
    print(f"protocol: batch 1, {WARMUP} warm-up + median of {TIMED} runs; {ROUNDS} interleaved rounds, median over rounds\n")
    for arch in ["mobilenet_v2", "mobilenet_v3_small"]:
        unfused = load_fp32(arch)
        fused = fuse(arch, load_fp32(arch))
        x = torch.randn(1, 3, IMG, IMG)
        with torch.inference_mode():
            models = {
                "FP32 unfused (eager)": unfused,
                "FP32 fused (eager)": fused,
                "FP32 unfused (TorchScript)": torch.jit.trace(unfused, x),
                "FP32 fused (TorchScript)": torch.jit.trace(fused, x),
            }
        for tag, name in [("int8_ptq", "INT8 PTQ (TorchScript)"), ("int8_ptq_outlier_aware", "INT8 PTQ outlier-aware (TorchScript)")]:
            p = RUNS / f"{arch}_seed0" / "models" / f"{arch}_{tag}_torchscript.pt"
            models[name] = torch.jit.load(str(p)).eval()
        print(f"== {arch} (seed 0) | fusion check: max |logit difference| fused vs unfused = {max_abs_diff(unfused, fused):.2e}")
        for threads in (1, 4):
            res = {k: [] for k in models}
            for _ in range(ROUNDS):
                for k, m in models.items():
                    res[k].append(median_ms(m, threads))
            med = {k: statistics.median(v) for k, v in res.items()}
            base = med["FP32 unfused (eager)"]
            print(f"  {threads} thread(s)")
            for k, v in med.items():
                spread = f"{min(res[k]):.2f}-{max(res[k]):.2f}"
                print(f"    {k:<40} {v:7.2f} ms  (rounds {spread})  speed-up vs FP32 unfused eager {base / v:5.2f}x")
            f_e, f_t = med["FP32 fused (eager)"], med["FP32 fused (TorchScript)"]
            q = med["INT8 PTQ (TorchScript)"]
            print(f"    -> fusion alone (eager): {base / f_e:.2f}x | INT8 PTQ vs FP32 fused (both TorchScript): {f_t / q:.2f}x | "
                  f"INT8 PTQ vs FP32 unfused (both TorchScript): {med['FP32 unfused (TorchScript)'] / q:.2f}x")
        print()


if __name__ == "__main__":
    sys.exit(main())
