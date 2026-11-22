"""Does FP32 confidence explain where INT8 flips happen? (offline, from saved logits of the 6 v2.3.0 runs)

For each run and test set: FP32 mean confidence; FP32 confidence of flipped vs unflipped images; and the flip rate per
FP32-confidence bin, pooled over runs. If flip rates within a confidence bin are similar across test sets, the higher flip
rate under shift is explained mainly by FP32 being less confident on shifted images (cf. Xia et al., 2021).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path(__file__).resolve().parents[1] / "runs"
SETS = ["pv_test", "pv_segmented", "plantdoc_all"]
VAR = {"mobilenet_v2": "int8_ptq", "mobilenet_v3_small": "int8_ptq_outlier_aware"}
BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.0001]


def conf(lg):
    z = lg - lg.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p.max(1), p.argmax(1)


rows, binrows = [], []
for run in sorted(glob.glob(str(RUNS / "mobilenet_*_seed[0-2]"))):
    run = Path(run)
    arch = run.name.rsplit("_seed", 1)[0]
    for ts in SETS:
        f = np.load(run / "predictions" / f"fp32__{ts}.npz")
        q = np.load(run / "predictions" / f"{VAR[arch]}__{ts}.npz")
        cf, pf = conf(f["logits"])
        _, pq = conf(q["logits"])
        flip = pf != pq
        rows.append({"arch": arch, "run": run.name, "test_set": ts, "fp32_mean_conf": cf.mean(), "flip_rate": flip.mean(),
                     "conf_flipped": cf[flip].mean() if flip.any() else np.nan, "conf_unflipped": cf[~flip].mean()})
        b = np.digitize(cf, BINS) - 1
        for k in range(len(BINS) - 1):
            m = b == k
            binrows.append({"arch": arch, "test_set": ts, "bin": f"{BINS[k]:.1f}-{min(BINS[k + 1], 1):.1f}",
                            "n": int(m.sum()), "flips": int(flip[m].sum())})
d = pd.DataFrame(rows)
print("Per test set (mean over 3 seeds):")
print(d.groupby(["arch", "test_set"])[["fp32_mean_conf", "flip_rate", "conf_flipped", "conf_unflipped"]].mean()
      .to_string(float_format=lambda v: f"{v:.3f}"))
bd = pd.DataFrame(binrows).groupby(["arch", "bin", "test_set"])[["n", "flips"]].sum()
bd["flip_rate"] = bd.flips / bd.n.clip(lower=1)
print("\nFlip rate by FP32-confidence bin (pooled over 3 seeds; n = images in bin):")
print(bd.unstack("test_set").to_string(float_format=lambda v: f"{v:.3f}"))
out = Path(__file__).with_name("flip_confidence_analysis.csv")
bd.reset_index().to_csv(out, index=False)
