"""Graphical abstract for Computers and Electronics in Agriculture, drawn from the paper's own tables (no hand-typed numbers).

    python research/make_graphical_abstract.py

Guide for authors: separate file, at least 1328 px wide with the longest side horizontal, readable at a width of 13 cm.
The figure is laid out at 13 cm (5.12 in) wide so the text sizes below are the printed sizes; PNG and TIFF are written at
330 dpi (1690 px wide), plus a vector PDF. Sources: paper_figures/tables/tab02_main_results.csv, tab03_quantization_accounting.csv,
tab04_shift.csv (all means over three seeds; see MANIFEST.md for provenance).
"""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
TAB = HERE / "paper_figures" / "tables"
OUT = HERE / "paper_figures" / "graphical_abstract"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#c9c8c3"      # blue/orange validated (CVD ΔE 24.7, normal ΔE 33.6)
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def rows(name):
    with open(TAB / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(s):
    return float(re.match(r"\s*(-?[\d.]+)", s).group(1))


def pick(table, **kv):
    hit = [r for r in table if all(r[k] == v for k, v in kv.items())]
    assert len(hit) == 1, (kv, len(hit))
    return hit[0]


acc, main, shift = rows("tab03_quantization_accounting.csv"), rows("tab02_main_results.csv"), rows("tab04_shift.csv")
V3, V2 = "MobileNetV3-Small", "MobileNetV2"
dyn = pick(acc, Model=V3, Variant="INT8 dynamic (Khan et al.)")
fp32_acc = pick(acc, Model=V3, Variant="FP32")
static = pick(acc, Model=V3, Variant="INT8 PTQ")
conv_fp32_dyn = int(re.search(r"still FLOAT conv (\d+)", dyn["Operator census"]).group(1))
conv_int8_static = int(re.search(r"(\d+) conv", static["Operator census"]).group(1))
params_fp32, params_dyn = int(fp32_acc["parameters()"]) / 1e6, int(dyn["parameters()"]) / 1e6
a_fp32 = mean(pick(main, Model=V3, Variant="FP32")["Accuracy"]) * 100
a_ptq = mean(pick(main, Model=V3, Variant="INT8 PTQ")["Accuracy"]) * 100
oa = pick(main, Model=V3, Variant="INT8 PTQ outlier-aware (ours)")
a_oa, size_oa = mean(oa["Accuracy"]) * 100, mean(oa["Size (MB)"])
size_fp32 = mean(pick(main, Model=V3, Variant="FP32")["Size (MB)"])
SETS = [("PlantVillage test", "Lab\ntest"), ("Background removed", "Background\nremoved"), ("PlantDoc (all), strict", "Field\n(PlantDoc)")]
flips = {lab: [mean(pick(shift, Model=m, Variant=v, **{"Test set": s})["Flip rate vs FP32"]) * 100 for s, _ in SETS]
         for lab, m, v in [("MobileNetV2 INT8", V2, "INT8 PTQ"), ("MobileNetV3-Small outlier-aware INT8", V3, "INT8 PTQ outlier-aware (ours)")]}

fig = plt.figure(figsize=(5.12, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.3], wspace=0.42, left=0.02, right=0.985, top=0.84, bottom=0.30)
TITLE = dict(loc="left", fontsize=7.5, fontweight="bold", pad=14)

# ---- 1. what "INT8" quantizes (labels above the bars, note inside the panel: nothing can be clipped)
ax = fig.add_subplot(gs[0])
ax.set_title("1  Is it really INT8?", **TITLE)
ax.barh([2.0], [conv_fp32_dyn], height=0.5, color=NEUTRAL)
ax.barh([0.6], [conv_int8_static], height=0.5, color=BLUE)
ax.text(0, 2.42, "Dynamic INT8 (widely cited)", fontsize=6.5, va="bottom")
ax.text(0, 1.02, "Static INT8", fontsize=6.5, va="bottom")
ax.text(conv_fp32_dyn / 2, 2.0, f"{conv_fp32_dyn} conv in FP32", ha="center", va="center", fontsize=6.5, color=INK)
ax.text(conv_int8_static / 2, 0.6, f"{conv_int8_static} conv in INT8", ha="center", va="center", fontsize=6.5, color=SURF)
ax.text(0, -0.15, f"Parameter count {params_fp32:.2f} M \u2192 {params_dyn:.2f} M:\na counting artifact, not compression",
        fontsize=6.2, color=INK2, va="top", linespacing=1.25)
ax.set_xlim(0, conv_int8_static)
ax.set_ylim(-1.2, 2.9)
ax.axis("off")

# ---- 2. collapse and fix
ax = fig.add_subplot(gs[1])
ax.set_title("2  Collapse and fix", **TITLE)
names, vals, cols = ["FP32", "Static\nINT8", "Outlier-\naware\nINT8"], [a_fp32, a_ptq, a_oa], [NEUTRAL, ORANGE, BLUE]
ax.bar(range(3), vals, width=0.62, color=cols)
for x, v in enumerate(vals):
    ax.text(x, v + 3, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.3)
ax.set_xticks(range(3), names, fontsize=6.3)
ax.set_ylim(0, 118)
ax.set_yticks([0, 50, 100], ["0", "50", "100%"], fontsize=6)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("MobileNetV3-Small accuracy", fontsize=6.5, labelpad=3)
ax.xaxis.set_label_position("top")
ax.annotate(f"Outlier-aware: {size_oa:.2f} MB,\n{size_fp32 / size_oa:.1f}\u00d7 smaller than FP32", xy=(0.5, -0.33),
            xycoords="axes fraction", ha="center", va="top", fontsize=6.2, color=INK2, linespacing=1.25)

# ---- 3. lab vs field
ax = fig.add_subplot(gs[2])
ax.set_title("3  Lab \u2260 field", **TITLE)
w = 0.34
for k, (lab, ys) in enumerate(flips.items()):
    xs = [i + (k - 0.5) * (w + 0.06) for i in range(len(SETS))]
    ax.bar(xs, ys, width=w, color=[BLUE, ORANGE][k], label=lab)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.8, f"{y:.1f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(len(SETS)), ["Lab", "Background\nremoved", "Field"], fontsize=6.3)
ymax = max(max(v) for v in flips.values())
ax.set_ylim(0, ymax * 1.55)
ax.set_yticks([0, 10, 20, 30], ["0", "10", "20", "30%"], fontsize=6)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("Predictions changed by INT8", fontsize=6.5, labelpad=3)
ax.xaxis.set_label_position("top")
ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.0, borderaxespad=0.2,
          labels=["MobileNetV2 INT8", "MobileNetV3-Small\noutlier-aware INT8"])
ax.annotate("Field = PlantDoc images", xy=(0.5, -0.33), xycoords="axes fraction", ha="center", va="top", fontsize=6.2, color=INK2)

fig.text(0.02, 0.025, "MobileNetV2 and MobileNetV3-Small trained on PlantVillage (38 classes); means over three seeds.",
         fontsize=6, color=INK2)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "graphical_abstract.pdf")
fig.savefig(OUT / "graphical_abstract.png", dpi=330)
Image.open(OUT / "graphical_abstract.png").convert("RGB").save(OUT / "graphical_abstract.tif", compression="tiff_lzw", dpi=(330, 330))
w_px, h_px = Image.open(OUT / "graphical_abstract.png").size
print(f"graphical abstract: {w_px} x {h_px} px | conv FP32 (dynamic) {conv_fp32_dyn}, INT8 (static) {conv_int8_static} | "
      f"params {params_fp32:.2f}->{params_dyn:.2f} M | acc FP32 {a_fp32:.1f} PTQ {a_ptq:.1f} OA {a_oa:.1f} | size {size_oa} MB | "
      f"flips {[[round(v, 2) for v in ys] for ys in flips.values()]}")
