"""Publication figures and tables from the Kaggle run folders (CPU only, no training).

    python make_paper_figures.py [runs_dir] [out_dir]

runs_dir: folder holding the unpacked <RUN_ID>/ folders (or the zips; they are unpacked once). Default: research/runs
out_dir : default research/paper_figures
Writes figures as 300-dpi PNG + vector PDF, tables as CSV + Markdown + LaTeX (booktabs), and MANIFEST.md (provenance:
which runs, notebook versions, fingerprints and source files produced every output). Refuses mixed fingerprints.
Every number is read from run outputs; nothing is typed in by hand.
"""
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "paper_figures"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style (validated categorical palette, fixed order)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
VARIANTS = ["FP32", "FP16 autocast", "FP16 converted", "INT8 dynamic (Khan2023 protocol)", "INT8 PTQ", "INT8 QAT",
            "INT8 PTQ outlier-aware", "INT8 QAT outlier-aware"]
SHORT = {"FP32": "FP32", "FP16 autocast": "FP16 (autocast)", "FP16 converted": "FP16 (converted)",
         "INT8 dynamic (Khan2023 protocol)": "INT8 dynamic (Khan et al.)", "INT8 PTQ": "INT8 PTQ", "INT8 QAT": "INT8 QAT",
         "INT8 PTQ outlier-aware": "INT8 PTQ outlier-aware (ours)", "INT8 QAT outlier-aware": "INT8 QAT outlier-aware (ours)"}
SLUG = {"FP32": "fp32", "INT8 dynamic (Khan2023 protocol)": "int8_dynamic_khan2023_protocol", "INT8 PTQ": "int8_ptq",
        "INT8 QAT": "int8_qat", "FP16 autocast": "fp16_autocast", "FP16 converted": "fp16_converted",
        "INT8 PTQ outlier-aware": "int8_ptq_outlier_aware", "INT8 QAT outlier-aware": "int8_qat_outlier_aware"}
ARCH_LABEL = {"mobilenet_v2": "MobileNetV2", "mobilenet_v3_small": "MobileNetV3-Small"}
TESTSET_LABEL = {("pv_test", "all_classes"): "PlantVillage test", ("pv_segmented", "all_classes"): "Background removed",
                 ("plantdoc_all", "strict"): "PlantDoc (all), strict", ("plantdoc_all", "restricted"): "PlantDoc (all), restricted",
                 ("plantdoc", "strict"): "PlantDoc test, strict", ("plantdoc", "restricted"): "PlantDoc test, restricted"}
MANIFEST = []
# PlantDoc images (CC BY 4.0 dataset, but web-collected) checked by eye on 2026-11-16: these carry stock-photo watermarks
# (shutterstock, dreamstime, alamy), photographer credits or printed text, so they are excluded from all figures.
WATERMARKED = {
    "Potato_leaf_early_blight/test_Potato leaf early blight_3.jpg",   # photographer credit
    "Soyabean_leaf/test_Soyabean leaf_8.jpg",                         # shutterstock
    "Corn_rust_leaf/test_Corn rust leaf_2.jpg",                       # photographer credit
    "Soyabean_leaf/test_Soyabean leaf_3.jpg",                         # dreamstime
    "Strawberry_leaf/train_Strawberry leaf_6.jpg",                    # dreamstime
    "Tomato_Early_blight_leaf/test_Tomato Early blight leaf_5.jpg",   # textbook page with printed text
    "Blueberry_leaf/train_Blueberry leaf_23.jpg",                     # shutterstock
    "Blueberry_leaf/test_Blueberry leaf_11.jpg",                      # dreamstime
    "Tomato_leaf/train_Tomato leaf_37_4.jpg",                         # dreamstime
    "Squash_Powdery_mildew_leaf/train_Squash Powdery mildew leaf_123.jpg",  # alamy
}

# ---------------------------------------------------------------- load runs
for z in sorted(RUNS.glob("*.zip")):
    if not (RUNS / z.stem).exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RUNS)
runs = []
for cfg_path in sorted(RUNS.glob("*/experiment_config.json")):
    d = cfg_path.parent
    cfg = json.loads(cfg_path.read_text())
    r = {"dir": d, "cfg": cfg, "run_id": cfg["run_id"], "arch": cfg["derived"]["arch"], "seed": cfg["derived"]["seed"],
         "bench": pd.read_csv(d / "reports" / "benchmark_results.csv", keep_default_na=False, na_values=[""]),
         "shift": pd.read_csv(d / "reports" / "shift_results.csv"),
         "per_class": pd.read_csv(d / "reports" / "per_class_f1.csv"),
         "history": pd.read_csv(d / "reports" / "training_history.csv")}
    am = d / "figure_assets" / "manifest.csv"
    r["assets"] = pd.read_csv(am) if am.exists() and am.stat().st_size > 5 else pd.DataFrame()
    if len(r["assets"]):   # never show web images that carry stock-photo watermarks or third-party credits
        tail = r["assets"]["source_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-2:].str.join("/")
        r["assets"] = r["assets"][~tail.isin(WATERMARKED)].reset_index(drop=True)
    runs.append(r)
if not runs:
    sys.exit(f"No runs in {RUNS}")
fps = {(r["cfg"]["derived"]["dataset_fingerprint"], r["cfg"]["derived"]["split_fingerprint"]) for r in runs}
if len(fps) != 1:
    sys.exit("REFUSING: runs differ in dataset/split fingerprint")
versions = sorted({r["cfg"]["notebook_version"] for r in runs})
if len(versions) > 1:
    print(f"WARNING: mixed notebook versions {versions}; tables report them in MANIFEST.md")
ARCHS = [a for a in ARCH_LABEL if any(r["arch"] == a for r in runs)]
bench = pd.concat([r["bench"] for r in runs], ignore_index=True)
bench = bench[bench["status"] == "ok"]
shift = pd.concat([r["shift"] for r in runs], ignore_index=True)
print(f"{len(runs)} runs: " + ", ".join(r["run_id"] for r in runs))


def save(fig, name, sources):
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append((f"figures/{name}.png/.pdf", sources))


def ms(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return (s.mean() if len(s) else np.nan), (s.std(ddof=1) if len(s) > 1 else np.nan), len(s)


def fmt(m, s, d=4):
    if pd.isna(m):
        return "–"
    return f"{m:.{d}f}" if pd.isna(s) else f"{m:.{d}f} ± {s:.{d}f}"


def write_table(df, name, caption, sources):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
    md = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    md += "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))
    (OUT / "tables" / f"{name}.md").write_text(f"**{caption}**\n\n{md}\n", encoding="utf-8")
    esc = lambda s: str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("±", r"$\pm$")
    tex = ["\\begin{table}[t]", "\\centering", f"\\caption{{{esc(caption)}}}", f"\\label{{tab:{name}}}", "\\small",
           "\\begin{tabular}{" + "l" * len(df.columns) + "}", "\\toprule", " & ".join(esc(c) for c in df.columns) + " \\\\",
           "\\midrule"] + [" & ".join(esc(v) for v in row) + " \\\\" for row in df.itertuples(index=False)] + \
          ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "tables" / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    MANIFEST.append((f"tables/{name}.csv/.md/.tex", sources))


def variants_present(arch):
    present = set(bench.loc[bench["arch"] == arch, "experiment"])
    return [v for v in VARIANTS if v in present]


# ================================================================ Fig 1: method overview (static diagram)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")
boxes = [(0.5, 12, 17, "PlantVillage (colour)\nMD5 de-duplication\nstratified 70/15/15\nfixed split seed"),
         (20.5, 12, 17, "Training\nfrozen backbone, then\nfull fine-tune; best\nepoch on validation"),
         (40.5, 12, 18, "Variants\nFP32 · FP16\nINT8 dynamic (Khan)\nINT8 PTQ · INT8 QAT"),
         (61.5, 12, 18, "Test sets\nPlantVillage test\nbackground removed\nPlantDoc (field)"),
         (82.5, 12, 17, "Measurements\naccuracy · F1 · CI\nflips · McNemar · ECE\nlayer census · latency")]
for k, (x, y, w, txt) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x, y - 9), w, 18, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc="#f4f7fc" if k % 2 == 0 else "#fdf4ef", ec=PAL[0] if k % 2 == 0 else PAL[1], lw=1))
    ax.text(x + w / 2, y, txt, ha="center", va="center", fontsize=6.2, linespacing=1.35)
    if k < len(boxes) - 1:
        ax.annotate("", xy=(boxes[k + 1][0] - 0.6, y), xytext=(x + w + 0.8, y),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.text(50, 26.5, "Experimental pipeline: 3 seeds × 2 architectures; no test data used for training, selection or calibration",
        ha="center", fontsize=7.5)
save(fig, "fig01_method_overview", "static diagram (no data)")

# ================================================================ Fig 2: dataset gallery + test-set examples
first = next((r for r in runs if len(r["assets"])), None)
if first is not None:
    a = first["assets"]
    gal = a[a["category"] == "gallery"].sort_values("label")
    n = len(gal)
    if n:
        cols = 8
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.05 * rows + 0.3), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, row) in zip(axes.ravel(), gal.iterrows()):
            ax.imshow(Image.open(first["dir"] / "figure_assets" / row["asset"]))
            ax.set_title(row["label"].replace("___", "\n").replace("_", " ")[:34], fontsize=4.8)
        fig.suptitle("PlantVillage classes (one held-out test image each)", fontsize=8)
        save(fig, "fig02a_dataset_gallery", f"{first['run_id']}/figure_assets (gallery)")
    examples = []
    sp = a[a["category"].str.startswith("segpair_both_right")]
    if len(sp):
        pid = sp["pair_id"].iloc[0]
        examples += [(r_, "PlantVillage (colour)" if r_["view"] == "colour" else "Background removed")
                     for _, r_ in sp[sp["pair_id"] == pid].sort_values("view").iterrows()]
    pdx = a[a["category"] == "plantdoc_fp32_right"]
    examples += [(r_, "PlantDoc (field)") for _, r_ in pdx.head(2).iterrows()]
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(1.6 * len(examples), 1.9), squeeze=False)
        for ax, (row, title) in zip(axes[0], examples):
            ax.imshow(Image.open(first["dir"] / "figure_assets" / row["asset"]))
            ax.set_title(f"{title}\n{row['label'].replace('___', ': ').replace('_', ' ')[:30]}", fontsize=5.5)
            ax.axis("off")
        save(fig, "fig02b_test_set_examples", f"{first['run_id']}/figure_assets (segpair, plantdoc)")

# ================================================================ Fig 3: clean accuracy / macro-F1 (mean ± std over seeds)
fig, axes = plt.subplots(2, len(ARCHS), figsize=(3.6 * len(ARCHS), 4.2), squeeze=False)
for j, arch in enumerate(ARCHS):
    vs = variants_present(arch)[::-1]
    for i, (col, lab) in enumerate((("accuracy", "Accuracy"), ("f1_macro", "Macro-F1"))):
        ax = axes[i, j]
        for k, v in enumerate(vs):
            vals = bench[(bench["arch"] == arch) & (bench["experiment"] == v)][col].astype(float)
            m, s, _ = ms(vals)
            ax.scatter(vals, [k] * len(vals), s=12, facecolors="none", edgecolors=PAL[0], linewidths=0.8, alpha=0.7)
            ax.errorbar(m, k, xerr=0 if pd.isna(s) else s, fmt="o", color=PAL[0], ms=5, capsize=3, lw=1.2)
            ax.annotate(f"{m:.4f}", (m, k), xytext=(6, 5), textcoords="offset points", fontsize=6)
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels([SHORT[v] for v in vs])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab)
        if i == 0:
            ax.set_title(ARCH_LABEL[arch], loc="left")
fig.suptitle("PlantVillage test set: mean ± s.d. over seeds (open circles = individual seeds)", fontsize=8)
fig.tight_layout()
save(fig, "fig03_clean_accuracy_f1", "runs/*/reports/benchmark_results.csv")

# ================================================================ Fig 4: efficiency (size, CPU latency 1 thread)
fig, axes = plt.subplots(2, len(ARCHS), figsize=(3.6 * len(ARCHS), 4.2), squeeze=False)
for j, arch in enumerate(ARCHS):
    vs = [v for v in variants_present(arch) if v != "FP16 autocast"][::-1]
    for i, (col, lab, f) in enumerate((("model_size_mb", "Model file size (MB)", "{:.2f}"),
                                       ("cpu_latency_1t_ms", "CPU latency, batch 1, 1 thread (ms)", "{:.1f}"))):
        ax = axes[i, j]
        stats = [ms(bench[(bench["arch"] == arch) & (bench["experiment"] == v)][col]) for v in vs]
        means = np.array([s_[0] for s_ in stats], float)
        sds = np.array([0 if pd.isna(s_[1]) else s_[1] for s_ in stats], float)
        ax.barh(range(len(vs)), np.nan_to_num(means), xerr=sds, color=PAL[0], height=0.55, capsize=2,
                error_kw={"lw": 0.8, "ecolor": INK2})
        for k, m in enumerate(means):
            if not np.isnan(m):
                ax.annotate(f.format(m), (m, k), xytext=(4, -6), textcoords="offset points", fontsize=6)
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels([SHORT[v] for v in vs])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab)
        if np.isfinite(np.nanmax(means)) and np.nanmax(means) > 0:
            ax.set_xlim(0, np.nanmax(means) * 1.3)
        if i == 0:
            ax.set_title(ARCH_LABEL[arch], loc="left")
fig.tight_layout()
save(fig, "fig04_size_latency", "runs/*/reports/benchmark_results.csv (FP16 rows: GPU-only, no CPU latency)")

# ================================================================ Fig 5: accuracy under shift
pd_name = "plantdoc_all" if (shift["test_set"] == "plantdoc_all").any() else "plantdoc"
SETS = [("pv_test", "all_classes"), ("pv_segmented", "all_classes"), (pd_name, "strict"), (pd_name, "restricted")]
SETS = [s_ for s_ in SETS if ((shift["test_set"] == s_[0]) & (shift["protocol"] == s_[1])).any()]
fig, axes = plt.subplots(1, len(ARCHS), figsize=(3.6 * len(ARCHS), 3.4), squeeze=False)
for j, arch in enumerate(ARCHS):
    ax = axes[0, j]
    vs = [v for v in variants_present(arch)][::-1]
    h = 0.8 / len(SETS)
    for k, s_ in enumerate(SETS):
        means, sds = [], []
        for v in vs:
            m, s, _ = ms(shift[(shift["arch"] == arch) & (shift["experiment"] == v) & (shift["test_set"] == s_[0])
                               & (shift["protocol"] == s_[1])]["accuracy"])
            means.append(m)
            sds.append(0 if pd.isna(s) else s)
        pos = np.arange(len(vs)) + (k - (len(SETS) - 1) / 2) * h
        ax.barh(pos, np.nan_to_num(means), xerr=sds, height=h * 0.9, color=PAL[k], label=TESTSET_LABEL[s_],
                error_kw={"lw": 0.6, "ecolor": INK2})
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels([SHORT[v] for v in vs])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy")
    ax.grid(axis="y", visible=False)
    ax.set_title(ARCH_LABEL[arch], loc="left")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="lower center", ncol=len(SETS), fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig05_accuracy_under_shift", "runs/*/reports/shift_results.csv")

# ================================================================ Fig 6: flip rate vs FP32 by test set (key figure)
INT8 = ["INT8 dynamic (Khan2023 protocol)", "INT8 PTQ", "INT8 QAT", "INT8 PTQ outlier-aware"]
FS = [s_ for s_ in [("pv_test", "all_classes"), ("pv_segmented", "all_classes"), (pd_name, "strict")]
      if ((shift["test_set"] == s_[0]) & (shift["protocol"] == s_[1])).any()]
fig, axes = plt.subplots(1, len(ARCHS), figsize=(3.6 * len(ARCHS), 2.8), squeeze=False)
for j, arch in enumerate(ARCHS):
    ax = axes[0, j]
    vs = [v for v in INT8 if v in variants_present(arch)]
    w = 0.8 / max(1, len(vs))
    for k, v in enumerate(vs):
        means, sds = [], []
        for s_ in FS:
            m, s, _ = ms(shift[(shift["arch"] == arch) & (shift["experiment"] == v) & (shift["test_set"] == s_[0])
                               & (shift["protocol"] == s_[1])]["flip_rate_vs_fp32"])
            means.append(m)
            sds.append(0 if pd.isna(s) else s)
        x = np.arange(len(FS)) + (k - (len(vs) - 1) / 2) * w
        mm = np.nan_to_num(means)
        lower = np.minimum(np.array(sds), mm)            # a rate cannot go below 0
        ax.bar(x, mm, yerr=[lower, sds], width=w * 0.9, color=PAL[k + 1], label=SHORT[v],
               error_kw={"lw": 0.6, "ecolor": INK2})
    ax.set_xticks(range(len(FS)))
    ax.set_xticklabels([TESTSET_LABEL[s_].replace(", strict", "") for s_ in FS], fontsize=6.5)
    ax.set_ylabel("Prediction flip rate vs FP32")
    ax.grid(axis="x", visible=False)
    ax.set_title(ARCH_LABEL[arch], loc="left")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="lower center", ncol=len(l_), fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "fig06_flip_rate_by_test_set", "runs/*/reports/shift_results.csv (flip_rate_vs_fp32)")

# ================================================================ Fig 7: reliability, pooled over seeds, clean vs field
def pooled(arch, variant, ts):
    ys, ls = [], []
    for r in runs:
        f = r["dir"] / "predictions" / f"{SLUG[variant]}__{ts}.npz"
        if r["arch"] == arch and f.exists():
            z = np.load(f)
            ys.append(z["y_true"])
            ls.append(z["logits"])
    return (np.concatenate(ys), np.concatenate(ls)) if ys else (None, None)


MIN_BIN = 20   # plotted bins need >= MIN_BIN images; ECE still uses every bin


def rel_curve(y, lg, bins=15):
    z = lg - lg.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    conf, ok = p.max(1), (p.argmax(1) == y)
    e = np.linspace(0, 1, bins + 1)
    xs, ys_, ece = [], [], 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
            if m.sum() >= MIN_BIN:
                xs.append(conf[m].mean())
                ys_.append(ok[m].mean())
    return xs, ys_, ece


rel_sets = [("pv_test", "PlantVillage test")] + ([(pd_name, "PlantDoc (field), strict")] if pd_name in set(shift["test_set"]) else [])
fig, axes = plt.subplots(len(rel_sets), len(ARCHS), figsize=(3.3 * len(ARCHS), 3.1 * len(rel_sets)), squeeze=False)
for j, arch in enumerate(ARCHS):
    for i, (ts, tlab) in enumerate(rel_sets):
        ax = axes[i, j]
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=INK2)
        for k, v in enumerate([v for v in ("FP32", "INT8 PTQ", "INT8 QAT", "INT8 PTQ outlier-aware") if v in variants_present(arch)]):
            y, lg = pooled(arch, v, ts)
            if y is None:
                continue
            xs, ys_, ece = rel_curve(y, lg)
            ax.plot(xs, ys_, marker="o", ms=3, lw=1.4, color=PAL[[0, 4, 5, 2][k]], label=f"{SHORT[v]} (ECE {ece:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ARCH_LABEL[arch]} · {tlab}", loc="left", fontsize=7.5)
        ax.legend(frameon=False, fontsize=5.8, loc="lower right")
fig.text(0.5, -0.01, f"Bins with fewer than {MIN_BIN} images are not plotted; ECE uses all bins (15 equal-width bins).",
         ha="center", fontsize=6, color=INK2)
fig.tight_layout()
save(fig, "fig07_reliability_clean_vs_field", "runs/*/predictions/*.npz (pooled over seeds)")

# ================================================================ Fig 8: per-class ΔF1 vs support
fig, axes = plt.subplots(1, len(ARCHS), figsize=(3.6 * len(ARCHS), 2.8), squeeze=False)
for j, arch in enumerate(ARCHS):
    ax = axes[0, j]
    pc = pd.concat([r["per_class"] for r in runs if r["arch"] == arch])
    g = pc.groupby("class_name").mean(numeric_only=True)
    for k, v in enumerate(("INT8 PTQ", "INT8 QAT")):
        c = f"delta_f1__{SLUG[v]}"
        if c in g:
            ax.scatter(g["support"], g[c], s=14, color=PAL[[4, 5][k]], label=SHORT[v], edgecolors=SURF, linewidths=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Class test support (log)")
    ax.set_ylabel("ΔF1 vs FP32 (mean over seeds)")
    ax.set_title(ARCH_LABEL[arch], loc="left")
    ax.legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "fig08_per_class_delta_f1", "runs/*/reports/per_class_f1.csv")

# ================================================================ Fig 9: training curves per seed
fig, axes = plt.subplots(1, len(ARCHS), figsize=(3.6 * len(ARCHS), 2.6), squeeze=False)
styles = ["-", "--", ":", "-."]
for j, arch in enumerate(ARCHS):
    ax = axes[0, j]
    for k, r in enumerate([r for r in runs if r["arch"] == arch]):
        h = r["history"]
        ax.plot(h["global_epoch"], h["val_f1_macro"], ls=styles[k % 4], color=PAL[0], lw=1.2, label=f"val macro-F1, seed {r['seed']}")
        ax.plot(h["global_epoch"], h["train_accuracy"], ls=styles[k % 4], color=PAL[1], lw=1.0, label=f"train acc., seed {r['seed']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(ARCH_LABEL[arch], loc="left")
    ax.legend(frameon=False, fontsize=5.5)
fig.tight_layout()
save(fig, "fig09_training_curves", "runs/*/reports/training_history.csv")

# ================================================================ Fig 10/11: qualitative grids
def square(path, size=320):
    """Letterbox to a square white canvas so mixed aspect ratios keep captions aligned."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def grid(rows_, run, name, title, captions):
    if not len(rows_):
        return
    n = min(len(rows_), 8)
    fig, axes = plt.subplots(1, n, figsize=(1.3 * n, 2.25), squeeze=False)
    for ax, (_, row) in zip(axes[0], rows_.head(n).iterrows()):
        ax.imshow(square(run["dir"] / "figure_assets" / row["asset"]))
        ax.set_title(captions(row), fontsize=4.4)
        ax.axis("off")
    fig.suptitle(title, fontsize=7, y=1.04)
    src = f"{run['run_id']}/figure_assets/manifest.csv"
    if rows_.head(n)["test_set"].astype(str).str.startswith("plantdoc").any():
        src += " | LICENCE CHECK: PlantDoc images are web-scraped (some carry stock-photo watermarks); do not print without rights"
    save(fig, name, src)


def short_c(s):
    crop, _, dis = str(s).partition("___")
    crop = crop.split("_(")[0].split(",")[0].replace("_", " ")
    return f"{crop}: {dis.replace('_', ' ')}"[:26]
for arch in ARCHS:
    r = next((r for r in runs if r["arch"] == arch and len(r["assets"])), None)
    if r is None:
        continue
    a = r["assets"]
    for ts in (pd_name, "pv_segmented", "pv_test"):
        fl = a[(a["category"] == "flip_lost__int8_ptq") & (a["test_set"] == ts)]
        if len(fl):
            ts_title = {"plantdoc_all": "all PlantDoc images", "plantdoc": "PlantDoc test split",
                        "pv_segmented": "background removed", "pv_test": "PlantVillage test"}.get(ts, ts)
            grid(fl, r, f"fig10_flips_{arch}_{ts}", f"{ARCH_LABEL[arch]}: FP32 correct, INT8 PTQ wrong ({ts_title})",
                 lambda row: f"true: {short_c(row['label'])}\nFP32: {short_c(row['pred__fp32'])}\nINT8: {short_c(row['pred__int8_ptq'])}")
            break
    sp = a[a["category"] == "segpair_colour_right_seg_wrong"].sort_values(["pair_id", "view"])
    if len(sp):
        grid(sp, r, f"fig11_background_pairs_{arch}", f"{ARCH_LABEL[arch]}: same leaf, colour vs background removed (FP32)",
             lambda row: f"{row['view']}\ntrue: {short_c(row['label'])}\nFP32: {short_c(row['pred__fp32'])}")

# ================================================================ Tables
cfg0 = runs[0]["cfg"]["derived"]
si = cfg0["shift_test_sets"]
t1 = pd.DataFrame([
    ("Images found (ImageFolder)", cfg0.get("images_before_dedup")),
    ("Exact duplicates removed", cfg0["duplicates_removed"]), ("Label-conflicting files removed", cfg0["label_conflicts_removed"]),
    ("Classes", cfg0["num_classes"]), ("Train / val / test", "{train} / {val} / {test}".format(**cfg0["split_sizes"])),
    ("Calibration images (from train)", cfg0["calibration_images"]),
    ("Background-removed test pairs", si.get("pv_segmented", {}).get("matched", "–")),
    ("PlantDoc test images (mapped, readable)", si.get("plantdoc", {}).get("images", "–")),
    ("PlantDoc all images (deduplicated)", si.get("plantdoc_all", {}).get("images", "–")),
    ("PlantDoc files identical to PlantVillage (removed)", si.get("plantdoc", {}).get("identical_to_plantvillage_removed", "–")),
    ("PlantDoc folders mapped / unmapped", f"{si.get('plantdoc', {}).get('mapped_folders', '–')} / "
                                           f"{len(si.get('plantdoc', {}).get('unmapped_folders', []))}"),
    ("Dataset fingerprint (SHA-256, first 16)", cfg0["dataset_fingerprint"][:16]),
], columns=["Item", "Value"])
write_table(t1, "tab01_dataset", "Dataset preparation and test sets", f"{runs[0]['run_id']}/experiment_config.json")

rows = []
for arch in ARCHS:
    for v in variants_present(arch):
        b = bench[(bench["arch"] == arch) & (bench["experiment"] == v)]
        rows.append([ARCH_LABEL[arch], SHORT[v], len(b), fmt(*ms(b["accuracy"])[:2]), fmt(*ms(b["f1_macro"])[:2]),
                     fmt(*ms(b["ece"])[:2]), fmt(*ms(b["model_size_mb"])[:2], d=2), fmt(*ms(b["cpu_latency_1t_ms"])[:2], d=1),
                     fmt(*ms(b["gpu_latency_bs1_ms"])[:2], d=2)])
write_table(pd.DataFrame(rows, columns=["Model", "Variant", "Seeds", "Accuracy", "Macro-F1", "ECE", "Size (MB)",
                                        "CPU 1-thr (ms)", "GPU bs1 (ms)"]),
            "tab02_main_results", "PlantVillage test results (mean ± s.d. over seeds)", "runs/*/reports/benchmark_results.csv")

rows = []
for arch in ARCHS:
    for v in [v for v in VARIANTS if v.startswith("INT8") or v == "FP32"]:
        b = bench[(bench["arch"] == arch) & (bench["experiment"] == v)]
        if not len(b):
            continue
        rows.append([ARCH_LABEL[arch], SHORT[v], int(b["params_reported"].iloc[0]), int(b["stored_weight_elements"].iloc[0]),
                     int(b["int8_weight_layers"].iloc[0]), int(b["fp32_weight_layers"].iloc[0]),
                     f"{b['model_size_mb'].astype(float).mean():.2f}", str(b["quantization_scope"].iloc[0])])
write_table(pd.DataFrame(rows, columns=["Model", "Variant", "parameters()", "Stored weights", "INT8 layers", "FP32 layers",
                                        "Size (MB)", "Operator census"]),
            "tab03_quantization_accounting", "What each INT8 workflow actually quantizes (first seed; identical across seeds)",
            "runs/*/reports/benchmark_results.csv")

rows = []
for arch in ARCHS:
    for v in variants_present(arch):
        for s_ in SETS:
            sh = shift[(shift["arch"] == arch) & (shift["experiment"] == v) & (shift["test_set"] == s_[0]) & (shift["protocol"] == s_[1])]
            if not len(sh):
                continue
            sig = (sh["mcnemar_p"] < 0.05).sum() if "mcnemar_p" in sh and v != "FP32" else None
            rows.append([ARCH_LABEL[arch], SHORT[v], TESTSET_LABEL[s_], int(sh["n"].iloc[0]), fmt(*ms(sh["accuracy"])[:2]),
                         fmt(*ms(sh["f1_macro"])[:2]),
                         fmt(*ms(sh["flip_rate_vs_fp32"])[:2]) if v != "FP32" else "–",
                         f"{sig}/{len(sh)}" if sig is not None else "–"])
write_table(pd.DataFrame(rows, columns=["Model", "Variant", "Test set", "n", "Accuracy", "Macro-F1", "Flip rate vs FP32",
                                        "McNemar p<0.05 (runs)"]),
            "tab04_shift", "Accuracy and prediction flips under distribution shift (mean ± s.d. over seeds)",
            "runs/*/reports/shift_results.csv")

pcols = ["run_id", "experiment", "flip_rate_vs_fp32", "mcnemar_b_fp32_right_variant_wrong", "mcnemar_c_fp32_wrong_variant_right",
         "mcnemar_p", "delta_accuracy", "delta_accuracy_ci_low", "delta_accuracy_ci_high"]
pt = bench[bench["experiment"] != "FP32"][[c for c in pcols if c in bench.columns]].copy()
for c in pt.columns[2:]:
    pt[c] = pd.to_numeric(pt[c], errors="coerce").map(lambda x: "–" if pd.isna(x) else f"{x:.4g}")
pt["experiment"] = pt["experiment"].map(SHORT)
write_table(pt, "tab05_paired_tests", "Paired comparison with FP32 on the PlantVillage test set, per run",
            "runs/*/reports/benchmark_results.csv")

# ---------------------------------------------------------------- provenance
lines = ["# Paper figures and tables: provenance (generated, do not edit)", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by research/make_paper_figures.py", "",
         f"Runs ({len(runs)}): " + ", ".join(f"{r['run_id']} (v{r['cfg']['notebook_version']}, torch {r['cfg']['environment']['torch']})" for r in runs),
         f"Dataset fingerprint {cfg0['dataset_fingerprint']}", f"Split fingerprint {cfg0['split_fingerprint']}", "",
         "| Output | Source |", "|---|---|"] + [f"| {o} | {s} |" for o, s in MANIFEST]
(OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {sum(1 for o, _ in MANIFEST if o.startswith('figures'))} figures and "
      f"{sum(1 for o, _ in MANIFEST if o.startswith('tables'))} tables to {OUT}")
