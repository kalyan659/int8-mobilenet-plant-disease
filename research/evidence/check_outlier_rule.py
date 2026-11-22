"""Which layers does the outlier-aware rule (per-channel BN-output range: max / median > 16) flag on the two REAL
pilot checkpoints? Diagnostic only (127 test-split images from figure_assets); the threshold 16 is fixed a priori from
quantization arithmetic (7-bit x86 activations: a 16x outlier leaves typical channels 128/16 = 8 levels), not tuned here.
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torchvision.models.quantization import mobilenet_v2 as qmnv2

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
RUNS = Path(__file__).resolve().parents[1] / "runs" / "_kaggle"
CK = {"mobilenet_v2": RUNS.parent / "_pilot_v2.2.0/kaggle_download_p1/plant_disease_results/mobilenet_v2_seed0",
      "mobilenet_v3_small": RUNS / "plantvillage-int8-mobilenetv3-small-pilot-diag/plant_disease_results/mobilenet_v3_small_seed0_pilotdiag"}
THRESHOLD = 16.0
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
paths = []
for d in CK.values():
    m = pd.read_csv(d / "figure_assets" / "manifest.csv")
    m = m[m["test_set"] == "pv_test"].drop_duplicates("source_path")
    paths += [(d / "figure_assets" / a, s) for a, s in zip(m["asset"], m["source_path"])]
seen, imgs = set(), []
for p, s in paths:
    if s not in seen:
        seen.add(s)
        imgs.append(tf(Image.open(p).convert("RGB")))
X = torch.stack(imgs)
print(f"{len(X)} images; threshold {THRESHOLD}x")

for arch, d in CK.items():
    ck = torch.load(d / "checkpoints" / "best_fp32.pt", map_location="cpu", weights_only=True)
    m = qmnv2(weights=None, quantize=False) if arch == "mobilenet_v2" else mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(ck["class_names"]))
    m.load_state_dict(ck["model_state"])
    m.eval()
    stats = {}

    def hook(name):
        def f(mod, inp, out):
            ch = out.detach().transpose(0, 1).reshape(out.shape[1], -1)
            stats[name] = (ch.max(1).values - ch.min(1).values)
        return f
    hs = [mod.register_forward_hook(hook(n)) for n, mod in m.named_modules() if isinstance(mod, nn.BatchNorm2d)]
    with torch.no_grad():
        m(X)
    for h in hs:
        h.remove()
    rows = sorted(((n, float(r.max() / r.median().clamp_min(1e-8))) for n, r in stats.items()), key=lambda t: -t[1])
    flagged = [n for n, v in rows if v > THRESHOLD]
    print(f"\n{arch}: {len(rows)} BN outputs | top-5 ratios: " + ", ".join(f"{n} {v:.1f}x" for n, v in rows[:5]))
    print(f"   flagged (> {THRESHOLD}x): {flagged or 'none'}")
