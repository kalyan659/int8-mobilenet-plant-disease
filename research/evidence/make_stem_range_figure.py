"""Figure 12: per-channel activation ranges of the stem (features.0) output, for torchvision ImageNet weights and our
fine-tuned checkpoints (3 seeds), MobileNetV3-Small vs MobileNetV2. Illustration only: the input images are the
test-split figure assets of the final runs; the outlier rule itself runs on training (calibration) images in the notebook.
Writes ../paper_figures/figures/fig12_stem_activation_ranges.{png,pdf} and ../paper_figures/tables/fig12_data.csv.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.quantization import mobilenet_v2 as qmnv2

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parents[1]
RUNS, OUT = R / "runs", R / "paper_figures"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
PAL = ["#2a78d6", "#eb6834"]
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "axes.axisbelow": True, "pdf.fonttype": 42})

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
seen, imgs = set(), []
for run in sorted(RUNS.glob("mobilenet_*_seed[0-2]")):
    m = pd.read_csv(run / "figure_assets" / "manifest.csv")
    for a, s in zip(m.loc[m.test_set == "pv_test", "asset"], m.loc[m.test_set == "pv_test", "source_path"]):
        if s not in seen:
            seen.add(s)
            imgs.append(tf(Image.open(run / "figure_assets" / a).convert("RGB")))
X = torch.stack(imgs)
print(f"{len(X)} unique PlantVillage test-split images")


def stem_ranges(model):
    model.eval()
    out = {}
    h = model.features[0].register_forward_hook(lambda m_, i_, o: out.__setitem__("a", o.detach()))
    with torch.no_grad():
        model(X)
    h.remove()
    a = out["a"]
    ch = a.transpose(0, 1).reshape(a.shape[1], -1)
    return (ch.max(1).values - ch.min(1).values).numpy()


def finetuned(arch, seed):
    ck = torch.load(RUNS / f"{arch}_seed{seed}" / "checkpoints" / "best_fp32.pt", map_location="cpu", weights_only=True)
    m = qmnv2(weights=None, quantize=False) if arch == "mobilenet_v2" else mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(ck["class_names"]))
    m.load_state_dict(ck["model_state"])
    return m


rows, panels = [], {}
for arch, label, imagenet in (("mobilenet_v3_small", "MobileNetV3-Small",
                               lambda: mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)),
                              ("mobilenet_v2", "MobileNetV2",
                               lambda: qmnv2(weights=MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False))):
    series = {"ImageNet weights": stem_ranges(imagenet())}
    for s in (0, 1, 2):
        series[f"fine-tuned seed {s}"] = stem_ranges(finetuned(arch, s))
    panels[arch] = (label, series)
    for name, r in series.items():
        ratio = r.max() / np.median(r)
        rows.append({"arch": arch, "weights": name, "channels": len(r), "median_range": float(np.median(r)),
                     "max_range": float(r.max()), "max_over_median": float(ratio)})
        print(f"{label:18s} {name:20s} channels {len(r):2d} | median {np.median(r):7.3f} | max {r.max():8.3f} | ratio {ratio:6.1f}x")
pd.DataFrame(rows).to_csv(OUT / "tables" / "fig12_data.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, arch in zip(axes, ("mobilenet_v3_small", "mobilenet_v2")):
    label, series = panels[arch]
    for name, r in series.items():
        srt = np.sort(r)[::-1]
        x = np.arange(1, len(srt) + 1)
        if name == "ImageNet weights":
            ax.plot(x, srt, "o", ms=4.5, color=PAL[1], label=name, zorder=3)
        else:
            ax.plot(x, srt, "-", lw=1.2, color=PAL[0], alpha=0.85, label="fine-tuned (3 seeds)" if name.endswith("0") else None)
    med = np.median(series["fine-tuned seed 0"])
    ax.axhline(16 * med, ls="--", lw=0.9, color="#52514e")
    ax.text(1, 16 * med * 1.12, "16 × median (rule threshold)", va="bottom", ha="left", fontsize=6.5, color="#52514e")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ratios = [r.max() / np.median(r) for r in series.values()]
    ax.set_title(f"{label}: largest/median channel range {min(ratios):.0f}–{max(ratios):.0f}×", loc="left", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Stem output channel (sorted by range)")
    ax.set_ylabel("Activation range (max − min)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / "figures" / f"fig12_stem_activation_ranges.{ext}", dpi=300, bbox_inches="tight")
with open(OUT / "MANIFEST_extra.md", "w", encoding="utf-8") as f:
    f.write("# Extra figures (not produced by make_paper_figures.py)\n\n| Output | Source |\n|---|---|\n"
            "| figures/fig12_stem_activation_ranges.png/.pdf, tables/fig12_data.csv | research/evidence/make_stem_range_figure.py; "
            f"checkpoints runs/*_seed[0-2]/checkpoints/best_fp32.pt + torchvision ImageNet weights; {len(X)} test-split figure-asset images "
            "(illustration only) |\n")
print("saved fig12")
