"""Build plant_disease_mobilenetv2_tinyml.ipynb programmatically with nbformat.

Usage:  python build_notebook.py            -> writes the notebook next to this file
The script validates the notebook schema and the syntax of every code cell.
"""
import ast
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_VERSION = "2.3.0"
HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "plant_disease_mobilenetv2_tinyml.ipynb"

cells = []


def md(src):
    cells.append(new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(new_code_cell(src.strip("\n").replace("__NOTEBOOK_VERSION__", NOTEBOOK_VERSION)))


# =============================================================================
md(r"""
# Faithful INT8 for edge plant-disease recognition: PlantVillage · MobileNetV2 / MobileNetV3-small

Notebook version `__V__`. This is the experiment notebook behind the paper's results. **One run = one (architecture, training
seed)**. The data split is fixed by `SPLIT_SEED`, so every run sees exactly the same train/val/test files.

**Planned runs (6):** `ARCH ∈ {mobilenet_v2, mobilenet_v3_small}` × `SEED ∈ {0, 1, 2}`. Change the two values in the configuration
cell, then *Save Version → Save & Run All*. Each run writes to `/kaggle/working/plant_disease_results/<RUN_ID>/`.

**Variants compared in every run** (all measured in this run, none pre-filled):

| Variant | What it is | Runs on |
|---|---|---|
| FP32 | best checkpoint by **validation** macro-F1 | GPU + CPU |
| FP16 autocast | FP32 weights, `torch.autocast` FP16 compute | GPU |
| FP16 converted | separate `.half()` copy (FP16 weights) | GPU |
| INT8 dynamic (Khan2023 protocol) | `quantize_dynamic({Linear, Conv2d})`, the protocol of Khan et al. 2023. Reported with an operator census | CPU |
| INT8 PTQ | full static post-training quantization with calibration | CPU |
| INT8 QAT | quantization-aware fine-tuning, then conversion | trained on GPU, run on CPU |

**Test sets:** (1) `pv_test`: held-out PlantVillage test split; (2) `pv_segmented`: **the same test leaves with the
background removed** (PlantVillage's own segmented copy); (3) `plantdoc`: real **field** photos (PlantDoc's official test
split, comparable with prior work); (4) `plantdoc_all`: **all** PlantDoc images (train+test; PlantDoc is never used for
training, so all of it is unseen), which gives about 10× more field images and more statistical power. Both PlantDoc sets are
mapped to PlantVillage classes and scored under the *strict* (all 38 classes) and *restricted* (mapped classes only) protocols.

**Kaggle inputs:** (a) a PlantVillage mirror that has `color/` and `segmented/` folders, e.g. `abdallahalidev/plantvillage-dataset`;
(b) a PlantDoc mirror with the original `train/` and `test/` class folders (from github.com/pratikkayal/PlantDoc-Dataset).
Accelerator **GPU T4 × 2**, Internet **on** (ImageNet weights).
If (b) is missing, the PlantDoc test is skipped and logged; if `segmented/` is missing, that test is skipped and logged.
""".replace("__V__", NOTEBOOK_VERSION))

# =============================================================================
md(r"""
## 1. Environment and configuration
### 1.1 Imports
""")

code(r'''
import os, sys, re, json, time, math, copy, random, hashlib, platform, warnings, zipfile, traceback, contextlib, operator
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.fx
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.quantization import mobilenet_v2 as quantizable_mobilenet_v2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

NOTEBOOK_VERSION = "__NOTEBOOK_VERSION__"
print(f"Imports OK - notebook version {NOTEBOOK_VERSION}")
''')

md(r"""
### 1.2 Configuration — the single place to change experiments

For the 6 planned runs, change only **`ARCH`** and **`SEED`**. Keep `SPLIT_SEED` fixed so every run uses the identical split
(the run records a split fingerprint, so this can be checked).
""")

code(r'''
CFG = {
    # ---------------- run identity ----------------
    "ARCH": "mobilenet_v2",                # 'mobilenet_v2' | 'mobilenet_v3_small'
    "SEED": 0,                             # training seed: 0, 1, 2
    "SPLIT_SEED": 42,                      # data split + calibration subset seed: keep FIXED across runs
    "RUN_TAG": "",                         # optional suffix for the run id

    # ---------------- paths ----------------
    "DATA_DIR": None,                      # None = auto-detect PlantVillage under INPUT_ROOT
    "SEGMENTED_DIR": None,                 # None = sibling 'segmented' folder of the colour folder
    "PLANTDOC_DIR": None,                  # None = auto-detect; else the PlantDoc root holding train/ and test/
    "INPUT_ROOT": "/kaggle/input",
    "OUTPUT_ROOT": "/kaggle/working/plant_disease_results",
    "PREFERRED_VARIANT": "color",

    # ---------------- test sets ----------------
    "RUN_SEGMENTED_TEST": True,
    "RUN_PLANTDOC_TEST": True,
    "PLANTDOC_EVAL_ALL": True,             # also score ALL PlantDoc images (train+test; never trained on) as 'plantdoc_all'
    "PLANTDOC_CLASS_MAP": None,            # optional {plantdoc_folder: plantvillage_class} override

    # ---------------- reproducibility ----------------
    "DETERMINISTIC": False,

    # ---------------- data ----------------
    "IMG_SIZE": 224,
    "RESIZE_SIZE": 256,
    "BATCH_SIZE": 64,
    "EVAL_BATCH_SIZE": 128,
    "NUM_WORKERS": None,                   # None = min(8, CPU count)
    "TRAIN_FRAC": 0.70,
    "VAL_FRAC": 0.15,
    "TEST_FRAC": 0.15,
    "DEDUPLICATE": True,
    "MAX_IMAGES_PER_CLASS": None,          # debug only
    "SHOW_PROGRESS": True,

    # ---------------- model ----------------
    "PRETRAINED": True,
    "DROPOUT": 0.2,

    # ---------------- training ----------------
    "DEVICE": "cuda:0",
    "USE_AMP": True,
    "USE_DATA_PARALLEL": False,
    "HEAD_EPOCHS": 3,
    "HEAD_LR": 1e-3,
    "FINETUNE_EPOCHS": 15,
    "FINETUNE_LR_BACKBONE": 1e-4,
    "FINETUNE_LR_HEAD": 5e-4,
    "WARMUP_EPOCHS": 1,
    "MIN_LR_RATIO": 0.01,
    "WEIGHT_DECAY": 1e-4,
    "LABEL_SMOOTHING": 0.1,
    "GRAD_CLIP_NORM": 2.0,
    "EARLY_STOPPING_PATIENCE": 4,
    "MIN_DELTA": 1e-4,

    # ---------------- variants ----------------
    "RUN_FP16": True,
    "RUN_DYNAMIC_QUANT": True,             # Khan et al. 2023 protocol
    "QUANT_BACKEND": "auto",               # 'auto' | 'x86' | 'fbgemm' | 'qnnpack'
    "RUN_PTQ": True,
    "PTQ_METHOD": "auto",                  # 'auto' | 'eager' | 'fx'  (MobileNetV3-small always uses FX)
    "CALIBRATION_IMAGES": 1024,
    "RUN_QAT": True,
    "QAT_METHOD": "auto",
    "QAT_EPOCHS": 3,
    "QAT_LR": 1e-5,
    "QAT_FREEZE_BN_EPOCH": 1,
    "QAT_FREEZE_OBSERVER_EPOCH": 2,
    "QAT_MAX_STEPS_PER_EPOCH": None,
    "RUN_OUTLIER_AWARE": True,             # our method: keep blocks around outlier activation tensors in float
    "OUTLIER_RATIO_THRESHOLD": 16.0,       # a priori: 7-bit activations / 16 = 8 levels left for a typical channel

    # ---------------- statistics ----------------
    "BOOTSTRAP_ITERS": 1000,
    "ECE_BINS": 15,

    # ---------------- benchmarking ----------------
    "CPU_THREADS": None,                   # multi-thread setting; None = min(4, CPU count). 1 thread is always measured too
    "LATENCY_WARMUP": 10,
    "LATENCY_RUNS": 50,
    "GPU_THROUGHPUT_BATCH": 64,
    "EXPORT_TORCHSCRIPT": True,

    # ---------------- export ----------------
    "ZIP_INCLUDE_CHECKPOINTS": True,
}
print(json.dumps(CFG, indent=2))
''')

md(r"""
### 1.3 Environment report, seeds, device, run folder
""")

code(r'''
T_NOTEBOOK_START = time.time()
RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_version(v):
    nums = re.findall(r"\d+", str(v).split("+")[0])
    return tuple(int(n) for n in nums[:3])


TORCH_VERSION, TV_VERSION = parse_version(torch.__version__), parse_version(torchvision.__version__)
print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | torchvision {torchvision.__version__} | {platform.platform()}")
if TORCH_VERSION < (2, 0) or TV_VERSION < (0, 15):
    warnings.warn("This notebook targets torch>=2.0 / torchvision>=0.15.")

GPU_INFO = []
if torch.cuda.is_available():
    print(f"CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        GPU_INFO.append({"index": i, "name": p.name, "memory_gb": round(p.total_memory / 1e9, 2)})
        print(f"  cuda:{i} -> {p.name} {p.total_memory / 1e9:.1f} GB")
else:
    print("CUDA not available")

ARCH = CFG["ARCH"]
assert ARCH in ("mobilenet_v2", "mobilenet_v3_small"), f"unknown ARCH {ARCH}"
SEED, SPLIT_SEED = int(CFG["SEED"]), int(CFG["SPLIT_SEED"])
RUN_ID = f"{ARCH}_seed{SEED}" + (f"_{CFG['RUN_TAG']}" if CFG["RUN_TAG"] else "")


def set_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


set_seed(SEED, CFG["DETERMINISTIC"])

if str(CFG["DEVICE"]).startswith("cuda") and torch.cuda.is_available():
    DEVICE = torch.device(CFG["DEVICE"])
else:
    if str(CFG["DEVICE"]).startswith("cuda"):
        print("WARNING: CUDA requested but not available -> CPU (FP16 variants will be skipped).")
    DEVICE = torch.device("cpu")
USE_CUDA = DEVICE.type == "cuda"
USE_AMP = bool(CFG["USE_AMP"]) and USE_CUDA
NUM_WORKERS = CFG["NUM_WORKERS"] if CFG["NUM_WORKERS"] is not None else min(8, os.cpu_count() or 2)
CPU_THREADS = CFG["CPU_THREADS"] or min(4, os.cpu_count() or 1)
IMG_SIZE = int(CFG["IMG_SIZE"])
SHOW_PROGRESS = bool(CFG["SHOW_PROGRESS"])
print(f"RUN_ID={RUN_ID} | device {DEVICE} | AMP {USE_AMP} | workers {NUM_WORKERS} | CPU threads 1 and {CPU_THREADS}")


def autocast_ctx(enabled):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


OUT_DIR = Path(CFG["OUTPUT_ROOT"]) / RUN_ID
CKPT_DIR, MODEL_DIR = OUT_DIR / "checkpoints", OUT_DIR / "models"
REPORT_DIR, FIG_DIR, PRED_DIR = OUT_DIR / "reports", OUT_DIR / "figures", OUT_DIR / "predictions"
for d in (OUT_DIR, CKPT_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

RESULTS = OrderedDict()   # variant -> row (main table, pv_test)
PRED_STORE = {}           # (variant, test_set) -> {"y_true", "logits"}
ERRORS, TIMINGS = [], {}


def log_error(stage, exc):
    ERRORS.append({"stage": stage, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    print(f"[ERROR] {stage}: {type(exc).__name__}: {exc}")


ENV_INFO = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpus": GPU_INFO, "cpu_count": os.cpu_count(), "device": str(DEVICE), "amp": USE_AMP}
print(f"Outputs -> {OUT_DIR.resolve()}")
''')

# =============================================================================
md(r"""
## 2. Dataset preparation

### 2.1 Locate PlantVillage
The finder walks `INPUT_ROOT` and scores every folder that looks like an `ImageFolder` root. The **colour** copy is preferred,
*grayscale* and *segmented* copies score low, and anything under a *PlantDoc* path is excluded (PlantDoc is used only as an
external test set). A `train/`+`val/` layout is merged into one pool and re-split below.
""")

code(r'''
IMG_EXTS = tuple(e.lower() for e in IMG_EXTENSIONS)
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing"}


def scan_tree(root, max_depth=7):
    info = {}
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        info[dirpath] = {"subdirs": [os.path.join(dirpath, d) for d in dirnames],
                         "n_img": sum(f.lower().endswith(IMG_EXTS) for f in filenames)}
        if dirpath.rstrip(os.sep).count(os.sep) - base >= max_depth:
            dirnames[:] = []
    return info


def class_root_stats(path, info):
    subs = [s for s in info.get(path, {}).get("subdirs", []) if s in info and info[s]["n_img"] > 0]
    return len(subs), sum(info[s]["n_img"] for s in subs)


def find_dataset_candidates(root):
    info = scan_tree(root)
    cands = []
    for path, meta in info.items():
        n_cls, n_img = class_root_stats(path, info)
        if n_cls >= 2:
            cands.append({"kind": "class_root", "path": path, "roots": [path], "n_classes": n_cls, "n_images": n_img})
        splits = [s for s in meta["subdirs"] if os.path.basename(s).lower() in SPLIT_NAMES]
        split_stats = [(s,) + class_root_stats(s, info) for s in splits]
        split_stats = [t for t in split_stats if t[1] >= 2]
        if len(split_stats) >= 2:
            cands.append({"kind": "split_root", "path": path, "roots": [t[0] for t in split_stats],
                          "n_classes": max(t[1] for t in split_stats), "n_images": sum(t[2] for t in split_stats)})
    return cands


def candidate_score(c):
    p = c["path"].lower().replace("\\", "/")
    parts = p.split("/")
    s = float(c["n_images"])
    if any(k in p for k in ("grayscale", "segmented", "mask")):
        s *= 0.05
    pref = (CFG["PREFERRED_VARIANT"] or "").lower()
    if pref and pref in parts:
        s *= 2.0
    if "augment" in p:
        s *= 0.5
    if "plant" in p or "village" in p:
        s *= 1.2
    if c["n_classes"] < 4:
        s *= 0.2
    return s


def is_plantdoc_path(p):
    return "plantdoc" in p.lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_dataset():
    if CFG["DATA_DIR"]:
        p = os.path.abspath(CFG["DATA_DIR"])
        if not os.path.isdir(p):
            raise FileNotFoundError(f"CFG['DATA_DIR'] does not exist: {p}")
        own = [c for c in find_dataset_candidates(p) if os.path.abspath(c["path"]) == p]
        if not own:
            raise ValueError(f"{p} is neither an ImageFolder root nor a folder of train/val/test ImageFolder roots.")
        return max(own, key=lambda c: c["n_images"]), "manual"
    root = CFG["INPUT_ROOT"]
    if not os.path.isdir(root):
        raise FileNotFoundError(f"INPUT_ROOT {root} not found. Set CFG['DATA_DIR'].")
    cands = [c for c in find_dataset_candidates(root) if not is_plantdoc_path(c["path"])]
    if not cands:
        raise FileNotFoundError(f"No PlantVillage-like ImageFolder dataset under {root}.")
    ranked = sorted(cands, key=candidate_score, reverse=True)
    show = pd.DataFrame([{**{k: c[k] for k in ("kind", "path", "n_classes", "n_images")},
                          "score": round(candidate_score(c))} for c in ranked[:8]])
    print("Top PlantVillage candidates:")
    print(show.to_string(index=False))
    return ranked[0], "auto"


DATASET, DATASET_SOURCE = resolve_dataset()
print(f"\nSelected ({DATASET_SOURCE}): {DATASET['path']} [{DATASET['kind']}]")
if "augment" in DATASET["path"].lower():
    print("WARNING: offline-AUGMENTED mirror - augmented copies of one leaf can cross splits.")
''')

md(r"""
### 2.2 Load with `ImageFolder`, remove exact duplicates
Files are MD5-hashed. Exact duplicates are reduced to one copy, and identical files carrying different labels are dropped. Both
happen **before** splitting, and both are logged in `reports/removed_duplicates.csv`.
""")

code(r'''
t0 = time.time()
_folders = [datasets.ImageFolder(r) for r in DATASET["roots"]]
CLASS_NAMES = list(_folders[0].classes)
for f in _folders[1:]:
    if list(f.classes) != CLASS_NAMES:
        raise ValueError("Class folders differ between roots.")
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

df_all = pd.DataFrame([(p, y) for f in _folders for (p, y) in f.samples], columns=["path", "label"])
df_all = df_all.drop_duplicates("path").sort_values("path").reset_index(drop=True)
df_all["class_name"] = df_all["label"].map(dict(enumerate(CLASS_NAMES)))
print(f"ImageFolder: {len(df_all):,} images in {NUM_CLASSES} classes")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


N_DUPLICATES_REMOVED, N_CONFLICTS_REMOVED = 0, 0
if CFG["DEDUPLICATE"]:
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 2) * 4)) as ex:
        df_all["md5"] = list(ex.map(md5_file, df_all["path"].tolist()))
    n_labels = df_all.groupby("md5")["label"].nunique()
    conflict_mask = df_all["md5"].isin(set(n_labels[n_labels > 1].index))
    dup_mask = df_all.duplicated("md5", keep="first") & ~conflict_mask
    N_DUPLICATES_REMOVED, N_CONFLICTS_REMOVED = int(dup_mask.sum()), int(conflict_mask.sum())
    df_all.loc[dup_mask | conflict_mask].to_csv(REPORT_DIR / "removed_duplicates.csv", index=False)
    df = df_all.loc[~(dup_mask | conflict_mask)].reset_index(drop=True)
else:
    df_all["md5"] = df_all["path"]
    df = df_all.copy()
print(f"Exact duplicates removed: {N_DUPLICATES_REMOVED:,} | label-conflicting files removed: {N_CONFLICTS_REMOVED:,}")

if CFG["MAX_IMAGES_PER_CLASS"]:
    cap = int(CFG["MAX_IMAGES_PER_CLASS"])
    df = (df.sample(frac=1.0, random_state=SPLIT_SEED).groupby("label").head(cap)
            .sort_values("path").reset_index(drop=True))
    print(f"DEBUG cap: <= {cap} images per class -> {len(df):,}")

counts = df["label"].value_counts().sort_index()
if counts.min() < 10:
    raise ValueError(f"Class '{CLASS_NAMES[counts.idxmin()]}' has only {counts.min()} images (need >= 10).")
print(f"Final pool: {len(df):,} | per-class min/median/max = {counts.min()}/{int(counts.median())}/{counts.max()} "
      f"(imbalance {counts.max() / counts.min():.1f}x)")
TIMINGS["data_indexing_s"] = round(time.time() - t0, 1)
''')

md(r"""
### 2.3 Stratified 70 / 15 / 15 split (`SPLIT_SEED`), leakage checks, fingerprints
The split is checked to be disjoint by index, path and file hash, with every class present in every split. Two SHA-256
fingerprints are recorded: the **dataset fingerprint** (files and hashes) and the **split fingerprint** (file → split). These must be
identical across all runs that the paper aggregates.
""")

code(r'''
tr_frac, va_frac, te_frac = CFG["TRAIN_FRAC"], CFG["VAL_FRAC"], CFG["TEST_FRAC"]
assert abs(tr_frac + va_frac + te_frac - 1.0) < 1e-6
all_idx = np.arange(len(df))
labels_all = df["label"].to_numpy()
trainval_idx, test_idx = train_test_split(all_idx, test_size=te_frac, stratify=labels_all, random_state=SPLIT_SEED)
train_idx, val_idx = train_test_split(trainval_idx, test_size=va_frac / (tr_frac + va_frac),
                                      stratify=labels_all[trainval_idx], random_state=SPLIT_SEED)
train_idx, val_idx, test_idx = np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

sets = {"train": set(train_idx), "val": set(val_idx), "test": set(test_idx)}
assert sets["train"].isdisjoint(sets["val"]) and sets["train"].isdisjoint(sets["test"]) and sets["val"].isdisjoint(sets["test"])
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    ia, ib = list(sets[a]), list(sets[b])
    assert set(df.loc[ia, "path"]).isdisjoint(df.loc[ib, "path"]), f"path overlap {a}/{b}"
    assert set(df.loc[ia, "md5"]).isdisjoint(df.loc[ib, "md5"]), f"identical content in {a} and {b}"
for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
    assert not (set(range(NUM_CLASSES)) - set(labels_all[idx])), f"{name} split misses classes"
print("Leakage checks passed (disjoint by index, path, file hash; all classes in all splits).")

df["split"] = ""
df.loc[train_idx, "split"], df.loc[val_idx, "split"], df.loc[test_idx, "split"] = "train", "val", "test"
df["rel_path"] = [os.path.relpath(p, DATASET["path"]) for p in df["path"]]
df.to_csv(REPORT_DIR / "split_manifest.csv", index=False)


def sha256_lines(lines):
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


DATASET_FINGERPRINT = sha256_lines(sorted(f"{r}|{m}" for r, m in zip(df["rel_path"], df["md5"])))
SPLIT_FINGERPRINT = sha256_lines(sorted(f"{r}|{s}" for r, s in zip(df["rel_path"], df["split"])))
class_dist = (df.pivot_table(index="class_name", columns="split", values="path", aggfunc="count", fill_value=0)
                [["train", "val", "test"]])
class_dist["total"] = class_dist.sum(axis=1)
class_dist.to_csv(REPORT_DIR / "class_distribution.csv")
SPLIT_SIZES = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
print(SPLIT_SIZES)
print(f"dataset fingerprint {DATASET_FINGERPRINT[:16]}... | split fingerprint {SPLIT_FINGERPRINT[:16]}...")
''')

md(r"""
### 2.4 Transforms, datasets, loaders
Training augmentation and evaluation preprocessing are **separate pipelines on separate dataset objects**. The INT8 calibration
subset is a stratified draw from the **training** split, and its file list is saved, so mobile conversion can reuse it later.
""")

code(r'''
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(int(CFG["RESIZE_SIZE"])),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SampleListDataset(Dataset):
    def __init__(self, samples, transform, loader=default_loader):
        self.samples, self.transform, self.loader = list(samples), transform, loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.transform(self.loader(path)), y


def samples_for(idx):
    sub = df.loc[idx]
    return list(zip(sub["path"].tolist(), sub["label"].astype(int).tolist()))


train_ds = SampleListDataset(samples_for(train_idx), train_tf)
val_ds = SampleListDataset(samples_for(val_idx), eval_tf)
test_ds = SampleListDataset(samples_for(test_idx), eval_tf)

n_cal = min(int(CFG["CALIBRATION_IMAGES"]), len(train_idx))
if n_cal < len(train_idx):
    try:
        cal_idx, _ = train_test_split(train_idx, train_size=n_cal, stratify=labels_all[train_idx], random_state=SPLIT_SEED)
    except ValueError:
        cal_idx = np.random.RandomState(SPLIT_SEED).choice(train_idx, n_cal, replace=False)
else:
    cal_idx = train_idx
cal_idx = np.sort(cal_idx)
calib_ds = SampleListDataset(samples_for(cal_idx), eval_tf)
df.loc[cal_idx, ["rel_path", "label", "class_name"]].to_csv(REPORT_DIR / "calibration_manifest.csv", index=False)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, batch_size, shuffle, drop_last=False, persistent=False):
    g = torch.Generator()
    g.manual_seed(SEED)
    kw = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=NUM_WORKERS,
              pin_memory=USE_CUDA, worker_init_fn=seed_worker, generator=g)
    if NUM_WORKERS > 0:
        kw.update(persistent_workers=persistent, prefetch_factor=4)
    return DataLoader(ds, **kw)


BATCH_SIZE, EVAL_BATCH_SIZE = int(CFG["BATCH_SIZE"]), int(CFG["EVAL_BATCH_SIZE"])
train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True, drop_last=len(train_ds) > BATCH_SIZE, persistent=True)
val_loader = make_loader(val_ds, EVAL_BATCH_SIZE, shuffle=False, persistent=True)
test_loader = make_loader(test_ds, EVAL_BATCH_SIZE, shuffle=False)
calib_loader = make_loader(calib_ds, min(32, EVAL_BATCH_SIZE), shuffle=False)
print(f"train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,} | calibration {len(calib_ds):,} (from train)")
''')

md(r"""
### 2.5 Shift test sets: background-removed PlantVillage and PlantDoc field images

**`pv_segmented`.** For every *test* image, the matching file in PlantVillage's `segmented/` copy is looked up (same class, same
file stem, with a `_final_masked`-style suffix removed). Only test leaves are used; segmented copies of training images are never
loaded. Because both versions of each leaf are scored, the colour vs segmented comparison is **paired**.

**`plantdoc` and `plantdoc_all`.** Each PlantDoc folder is mapped to a PlantVillage class by crop and disease keywords. The
mapping table is saved, and unmapped folders are reported and excluded. Unreadable images are skipped and counted.
* `plantdoc` = PlantDoc's official **test** split (about 250 images), so results stay comparable with Ho-Lam & Nguyen (2026);
* `plantdoc_all` = **train + test** (about 2.5k images). PlantDoc is never used for training or selection here, so every image is
  unseen. Exact duplicates within PlantDoc are removed by MD5, and identical files with conflicting labels are dropped.
* **Cross-dataset leakage guard:** PlantDoc was collected from the web, so any PlantDoc file byte-identical to a PlantVillage
  image is removed from both sets, and the count is reported.

Two protocols are scored: *strict* = argmax over all PlantVillage classes; *restricted* = argmax over the mapped target
classes only.
""")

code(r'''
EVAL_SETS = OrderedDict([("pv_test", test_loader)])
SHIFT_INFO = {"pv_segmented": {"status": "not run"}, "plantdoc": {"status": "not run"}, "plantdoc_all": {"status": "not run"}}
SEG_POSITIONS = None          # positions in the pv_test order that have a segmented twin
PD_TARGETS = None             # PlantVillage class indices that PlantDoc folders map to

# ---------------- background-removed PlantVillage ----------------
SEG_SUFFIX_RE = re.compile(r"(_final_masked|_masked|_segmented|_seg)$", re.I)


def seg_key(cls, path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return cls, SEG_SUFFIX_RE.sub("", stem).lower()


try:
    if not CFG["RUN_SEGMENTED_TEST"]:
        SHIFT_INFO["pv_segmented"] = {"status": "disabled"}
    else:
        seg_dir = CFG["SEGMENTED_DIR"]
        if not seg_dir:
            base = DATASET["path"].rstrip("/\\")
            parent = os.path.dirname(base)
            if os.path.basename(base).lower() == "color" and os.path.isdir(parent):
                sib = [os.path.join(parent, d) for d in sorted(os.listdir(parent)) if "segment" in d.lower()]
                seg_dir = sib[0] if sib else None
        if not seg_dir or not os.path.isdir(seg_dir):
            SHIFT_INFO["pv_segmented"] = {"status": "skipped", "reason": "no segmented copy found next to the colour folder"}
        else:
            seg_index = {}
            for cls in sorted(os.listdir(seg_dir)):
                cdir = os.path.join(seg_dir, cls)
                if cls in CLASS_TO_IDX and os.path.isdir(cdir):
                    for fn in os.listdir(cdir):
                        if fn.lower().endswith(IMG_EXTS):
                            seg_index[seg_key(cls, fn)] = os.path.join(cdir, fn)
            test_rows = df.loc[test_idx]
            pos, seg_samples = [], []
            for k, (p, c, y) in enumerate(zip(test_rows["path"], test_rows["class_name"], test_rows["label"])):
                hit = seg_index.get(seg_key(c, p))
                if hit:
                    pos.append(k)
                    seg_samples.append((hit, int(y)))
            rate = len(pos) / max(1, len(test_rows))
            if rate < 0.5:
                SHIFT_INFO["pv_segmented"] = {"status": "skipped", "reason": f"only {rate:.1%} of test files matched", "dir": seg_dir}
            else:
                SEG_POSITIONS = np.array(pos)
                seg_ds = SampleListDataset(seg_samples, eval_tf)
                EVAL_SETS["pv_segmented"] = make_loader(seg_ds, EVAL_BATCH_SIZE, shuffle=False)
                pd.DataFrame({"test_position": pos, "segmented_path": [s[0] for s in seg_samples],
                              "label": [s[1] for s in seg_samples]}).to_csv(REPORT_DIR / "segmented_pairs.csv", index=False)
                SHIFT_INFO["pv_segmented"] = {"status": "ok", "dir": seg_dir, "matched": len(pos),
                                              "test_size": len(test_rows), "match_rate": round(rate, 4)}
except Exception as e:
    log_error("pv_segmented.build", e)
    SHIFT_INFO["pv_segmented"] = {"status": "failed", "reason": str(e)}
print("pv_segmented:", SHIFT_INFO["pv_segmented"])

# ---------------- PlantDoc ----------------
PLANTDOC_TO_PV_KEYS = {   # normalised PlantDoc folder -> (crop keyword, disease keyword) in PlantVillage class names
    "apple scab leaf": ("apple", "scab"), "apple leaf": ("apple", "healthy"), "apple rust leaf": ("apple", "cedar"),
    "bell pepper leaf": ("pepper", "healthy"), "bell pepper leaf spot": ("pepper", "bacterial"),
    "blueberry leaf": ("blueberry", "healthy"), "cherry leaf": ("cherry", "healthy"),
    "corn gray leaf spot": ("corn", "gray"), "corn leaf blight": ("corn", "northern"), "corn rust leaf": ("corn", "rust"),
    "peach leaf": ("peach", "healthy"), "potato leaf early blight": ("potato", "early"),
    "potato leaf late blight": ("potato", "late"), "raspberry leaf": ("raspberry", "healthy"),
    "soyabean leaf": ("soybean", "healthy"), "soybean leaf": ("soybean", "healthy"),
    "squash powdery mildew leaf": ("squash", "powdery"), "strawberry leaf": ("strawberry", "healthy"),
    "tomato early blight leaf": ("tomato", "early"), "tomato septoria leaf spot": ("tomato", "septoria"),
    "tomato leaf": ("tomato", "healthy"), "tomato leaf bacterial spot": ("tomato", "bacterial"),
    "tomato leaf late blight": ("tomato", "late"), "tomato leaf mosaic virus": ("tomato", "mosaic"),
    "tomato leaf yellow virus": ("tomato", "yellow"), "tomato mold leaf": ("tomato", "mold"),
    "tomato two spotted spider mites leaf": ("tomato", "spider"), "grape leaf": ("grape", "healthy"),
    "grape leaf black rot": ("grape", "black rot"),
}


def norm_name(s):
    return re.sub(r"\s+", " ", re.sub(r"[_,()\-]", " ", s.lower())).strip()


def pv_class_for(crop_kw, dis_kw):
    hits = []
    for i, c in enumerate(CLASS_NAMES):
        crop, _, dis = c.partition("___")
        if crop_kw in norm_name(crop) and dis_kw in norm_name(dis):
            hits.append(i)
    return hits[0] if len(hits) == 1 else None


def can_open(p):
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def plantdoc_split_dirs():
    """{'test': dir|None, 'train': dir|None}. PLANTDOC_DIR may be the PlantDoc root (with train/ and test/) or one class folder."""
    root = CFG["PLANTDOC_DIR"]
    if root and os.path.isdir(root):
        subs = {d.lower(): os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
        if "test" in subs or "train" in subs:
            return {"test": subs.get("test"), "train": subs.get("train")}
        return {"test": root, "train": None}
    found = {"test": None, "train": None}
    if os.path.isdir(CFG["INPUT_ROOT"]):
        pcs = [c["path"].rstrip("/\\") for c in find_dataset_candidates(CFG["INPUT_ROOT"])
               if c["kind"] == "class_root" and is_plantdoc_path(c["path"])]
        for split in ("test", "train"):
            hits = [p for p in pcs if os.path.basename(p).lower() == split]
            found[split] = hits[0] if hits else None
        if found["test"] and found["train"] and os.path.dirname(found["test"]) != os.path.dirname(found["train"]):
            found["train"] = None          # only use a train/ folder that is the sibling of the chosen test/
    return found


def plantdoc_samples(split_dir, split_name, override):
    rows, samples, n_bad = [], [], 0
    for folder in sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))):
        if folder in override:
            tgt = CLASS_TO_IDX.get(override[folder])
        else:
            keys = PLANTDOC_TO_PV_KEYS.get(norm_name(folder))
            tgt = pv_class_for(*keys) if keys else None
        files = [os.path.join(split_dir, folder, f) for f in sorted(os.listdir(os.path.join(split_dir, folder)))
                 if f.lower().endswith(IMG_EXTS)]
        ok_files = [f for f in files if can_open(f)] if tgt is not None else []
        n_bad += (len(files) - len(ok_files)) if tgt is not None else 0
        rows.append({"split": split_name, "plantdoc_folder": folder,
                     "plantvillage_class": CLASS_NAMES[tgt] if tgt is not None else None,
                     "n_images": len(files), "n_readable": len(ok_files)})
        samples += [(f, tgt) for f in ok_files]
    return rows, samples, n_bad


try:
    if not CFG["RUN_PLANTDOC_TEST"]:
        SHIFT_INFO["plantdoc"] = {"status": "disabled"}
    else:
        pd_dirs = plantdoc_split_dirs()
        if not pd_dirs["test"]:
            SHIFT_INFO["plantdoc"] = {"status": "skipped", "reason": "no PlantDoc 'test' class folder found"}
        else:
            override = CFG["PLANTDOC_CLASS_MAP"] or {}
            rows_t, test_s, bad_t = plantdoc_samples(pd_dirs["test"], "test", override)
            rows_tr, train_s, bad_tr = [], [], 0
            if CFG["PLANTDOC_EVAL_ALL"] and pd_dirs["train"]:
                rows_tr, train_s, bad_tr = plantdoc_samples(pd_dirs["train"], "train", override)
            map_df = pd.DataFrame(rows_t + rows_tr)
            map_df.to_csv(REPORT_DIR / "plantdoc_class_map.csv", index=False)
            print(map_df[map_df["split"] == "test"].to_string(index=False))
            # Cross-dataset leakage guard: drop PlantDoc files byte-identical to any PlantVillage file (web-scraped images).
            pv_hashes = set(df_all["md5"]) if CFG["DEDUPLICATE"] else set()
            hashed = [(f, y, md5_file(f), "test") for f, y in test_s] + [(f, y, md5_file(f), "train") for f, y in train_s]
            n_pv_overlap = sum(h in pv_hashes for _, _, h, _ in hashed)
            hashed = [t for t in hashed if t[2] not in pv_hashes]
            test_clean = [(f, y) for f, y, _, s in hashed if s == "test"]
            if not test_clean:
                SHIFT_INFO["plantdoc"] = {"status": "skipped", "reason": "no mappable PlantDoc test image", "dir": pd_dirs["test"]}
            else:
                EVAL_SETS["plantdoc"] = make_loader(SampleListDataset(test_clean, eval_tf), EVAL_BATCH_SIZE, shuffle=False)
                targets = {y for _, y in test_clean}
                SHIFT_INFO["plantdoc"] = {"status": "ok", "dir": pd_dirs["test"], "images": len(test_clean),
                                          "unreadable_skipped": bad_t, "identical_to_plantvillage_removed": n_pv_overlap,
                                          "mapped_folders": int(map_df.loc[map_df["split"] == "test", "plantvillage_class"].notna().sum()),
                                          "unmapped_folders": map_df.loc[(map_df["split"] == "test") & map_df["plantvillage_class"].isna(),
                                                                         "plantdoc_folder"].tolist()}
                if train_s:
                    # 'plantdoc_all' = test + train (never used for training here); exact duplicates removed, label conflicts dropped
                    first, conflict, n_dup = {}, set(), 0
                    for f, y, h, _ in hashed:
                        if h in first:
                            n_dup += 1
                            if first[h][1] != y:
                                conflict.add(h)
                        else:
                            first[h] = (f, y)
                    all_s = [(f, y) for h, (f, y) in first.items() if h not in conflict]
                    EVAL_SETS["plantdoc_all"] = make_loader(SampleListDataset(all_s, eval_tf), EVAL_BATCH_SIZE, shuffle=False)
                    targets |= {y for _, y in all_s}
                    SHIFT_INFO["plantdoc_all"] = {"status": "ok", "dirs": [pd_dirs["test"], pd_dirs["train"]], "images": len(all_s),
                                                  "unreadable_skipped": bad_t + bad_tr, "duplicates_removed": n_dup,
                                                  "label_conflict_hashes_dropped": len(conflict)}
                else:
                    SHIFT_INFO["plantdoc_all"] = {"status": "skipped",
                                                  "reason": "disabled" if not CFG["PLANTDOC_EVAL_ALL"] else "no sibling train/ folder"}
                PD_TARGETS = np.array(sorted(targets))
                SHIFT_INFO["plantdoc"]["target_classes"] = len(PD_TARGETS)
except Exception as e:
    log_error("plantdoc.build", e)
    SHIFT_INFO["plantdoc"] = {"status": "failed", "reason": str(e)}
print("plantdoc:", SHIFT_INFO["plantdoc"])
print("plantdoc_all:", SHIFT_INFO["plantdoc_all"])
print("Evaluation sets:", list(EVAL_SETS))

for name, loader in EVAL_SETS.items():   # one index file per test set (row order of every saved logits array)
    pd.DataFrame(loader.dataset.samples, columns=["path", "label"]).to_csv(PRED_DIR / f"index__{name}.csv", index=False)
''')

code(r'''
def denorm(t):
    m, s = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


rows = [("train (augmented)", train_ds)] + [(n, l.dataset) for n, l in EVAL_SETS.items()]
fig, axes = plt.subplots(len(rows), 6, figsize=(13, 2.3 * len(rows)), squeeze=False)
rs = np.random.RandomState(SPLIT_SEED)
for r, (name, ds) in enumerate(rows):
    pick = rs.choice(len(ds), min(6, len(ds)), replace=False)
    for c in range(6):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(pick):
            img, y = ds[int(pick[c])]
            ax.imshow(denorm(img))
            ax.set_title(CLASS_NAMES[y].replace("___", "\n").replace("_", " ")[:36], fontsize=6)
    axes[r, 0].text(-0.1, 0.5, name, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "samples_per_test_set.png", dpi=120, bbox_inches="tight")
plt.show()
''')

# =============================================================================
md(r"""
## 3. Model architecture

* **`mobilenet_v2`**: torchvision's *quantization-ready* MobileNetV2 (`quantize=False`, ImageNet weights). It has QuantStub/DeQuantStub,
  quantizable residual adds, and `fuse_model()`. ReLU6 is replaced by ReLU so Conv-BN-ReLU can be fused, and fine-tuning adapts to that.
  INT8 uses **eager** mode, with FX as the fallback.
* **`mobilenet_v3_small`**: torchvision MobileNetV3-small (ImageNet weights), the architecture of Khan et al. (2023). torchvision has no
  eager-quantizable V3-small, so INT8 uses **FX graph mode**, which handles hard-swish and squeeze-excite.

The last classifier layer is replaced with `NUM_CLASSES` outputs. FLOPs come from `torch.utils.flop_counter` (conv and matmul only,
1 MAC = 2 FLOPs).
""")

code(r'''
def build_model(arch, num_classes, pretrained, dropout):
    effective = pretrained
    def _make(use_weights):
        if arch == "mobilenet_v2":
            return quantizable_mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if use_weights else None, quantize=False)
        return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_weights else None)
    try:
        m = _make(pretrained)
    except Exception as e:
        if not pretrained:
            raise
        log_error("model.download_pretrained", e)
        print("!" * 80 + "\nWARNING: ImageNet weights unavailable (Internet off?) -> RANDOM init. Recorded in config.\n" + "!" * 80)
        m, effective = _make(False), False
    last = m.classifier[-1]
    m.classifier[-1] = nn.Linear(last.in_features, num_classes)
    nn.init.normal_(m.classifier[-1].weight, 0, 0.01)
    nn.init.zeros_(m.classifier[-1].bias)
    for mod in m.classifier:
        if isinstance(mod, nn.Dropout):
            mod.p = dropout
    return m, effective


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_flops(m, img_size):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        mm = copy.deepcopy(unwrap(m)).cpu().eval()
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            mm(torch.randn(1, 3, img_size, img_size))
        return int(counter.get_total_flops()), "torch.utils.flop_counter.FlopCounterMode"
    except Exception as e:
        return None, f"unavailable ({type(e).__name__}: {e})"


base_model, PRETRAINED_EFFECTIVE = build_model(ARCH, NUM_CLASSES, CFG["PRETRAINED"], CFG["DROPOUT"])
SUPPORTS_EAGER_QUANT = hasattr(base_model, "fuse_model")
N_PARAMS, _ = count_params(base_model)
FLOPS, FLOPS_TOOL = count_flops(base_model, IMG_SIZE)
if ARCH == "mobilenet_v2":
    assert not any(isinstance(x, nn.ReLU6) for x in base_model.modules()), "expected ReLU (quantizable variant)"

model = base_model.to(DEVICE)
if CFG["USE_DATA_PARALLEL"] and USE_CUDA and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print(f"{ARCH} | pretrained requested/effective {CFG['PRETRAINED']}/{PRETRAINED_EFFECTIVE} | eager-quantizable {SUPPORTS_EAGER_QUANT}")
print(f"classes {NUM_CLASSES} | parameters {N_PARAMS:,}" +
      (f" | {FLOPS / 1e9:.3f} GFLOPs (~{FLOPS / 2e6:.0f} M MACs) @ {IMG_SIZE}px" if FLOPS else f" | FLOPs {FLOPS_TOOL}"))
''')

# =============================================================================
md(r"""
## 4. Model training
**Phase 1:** classifier only. The feature extractor is frozen and its BatchNorm layers are kept in `eval()`, so frozen running
statistics are not updated by PlantVillage batches. **Phase 2:** full fine-tuning, with separate learning rates for backbone and
head. Both phases use AdamW (no decay on biases or norms), warm-up then cosine LR, AMP on CUDA, gradient clipping, and early
stopping on **validation macro-F1**. The best checkpoint by validation macro-F1 across both phases is restored before any test
evaluation. **The test set plays no part in training or selection.**
""")

code(r'''
criterion = nn.CrossEntropyLoss(label_smoothing=float(CFG["LABEL_SMOOTHING"]))


def set_backbone_trainable(m, trainable):
    for p in unwrap(m).features.parameters():
        p.requires_grad = trainable


def set_train_mode(m, backbone_frozen):
    m.train()
    if backbone_frozen:
        unwrap(m).features.eval()


def param_groups(modules_lrs, weight_decay):
    groups = []
    for module, lr in modules_lrs:
        decay = [p for p in module.parameters() if p.requires_grad and p.ndim > 1]
        no_decay = [p for p in module.parameters() if p.requires_grad and p.ndim <= 1]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def make_warmup_cosine(optimizer, epochs, steps_per_epoch, warmup_epochs, min_ratio):
    total = max(1, int(epochs * steps_per_epoch))
    warm = min(total - 1, int(warmup_epochs * steps_per_epoch))

    def factor(step):
        if warm > 0 and step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def compute_metrics(y_true, y_pred, labels=None):
    labels = list(range(NUM_CLASSES)) if labels is None else list(labels)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f1), "f1_weighted": float(f1w)}


@torch.inference_mode()
def evaluate(m, loader, device, use_amp=False, input_dtype=None, desc="eval"):
    m.eval()
    losses, ys, logits_all, n = 0.0, [], [], 0
    it = tqdm(loader, desc=desc, leave=False) if SHOW_PROGRESS else loader
    for x, y in it:
        x = x.to(device, non_blocking=True)
        if input_dtype is not None:
            x = x.to(input_dtype)
        with autocast_ctx(use_amp):
            logits = m(x)
        logits = logits.float().cpu()
        losses += float(criterion(logits, y)) * y.size(0)
        n += y.size(0)
        ys.append(y.numpy())
        logits_all.append(logits.numpy())
    y_true, logits_np = np.concatenate(ys), np.concatenate(logits_all)
    out = compute_metrics(y_true, logits_np.argmax(1))
    out.update({"loss": losses / max(1, n), "y_true": y_true, "y_pred": logits_np.argmax(1), "logits": logits_np})
    return out


def train_one_epoch(m, loader, optimizer, scheduler, scaler, use_amp, backbone_frozen, max_steps=None, desc="train"):
    set_train_mode(m, backbone_frozen)
    total_loss, correct, n = 0.0, 0, 0
    it = tqdm(loader, desc=desc, leave=False) if SHOW_PROGRESS else loader
    for step, (x, y) in enumerate(it):
        if max_steps is not None and step >= max_steps:
            break
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(use_amp):
            logits = m(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        if CFG["GRAD_CLIP_NORM"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], CFG["GRAD_CLIP_NORM"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.detach()) * y.size(0)
        correct += int((logits.detach().argmax(1) == y).sum())
        n += y.size(0)
    return {"loss": total_loss / max(1, n), "accuracy": correct / max(1, n)}


HISTORY = []
BEST = {"val_f1_macro": -1.0, "epoch": None, "phase": None, "path": str(CKPT_DIR / "best_fp32.pt")}


def save_checkpoint(path, m, extra):
    torch.save({"model_state": unwrap(m).state_dict(), "class_names": CLASS_NAMES, "arch": ARCH,
                "config_json": json.dumps(CFG, default=str), **extra}, path)


def run_phase(phase, epochs, backbone_frozen, optimizer, scheduler, scaler):
    patience, bad_epochs = int(CFG["EARLY_STOPPING_PATIENCE"]), 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, USE_AMP, backbone_frozen,
                             desc=f"{phase} {ep}/{epochs}")
        va = evaluate(model, val_loader, DEVICE, use_amp=USE_AMP, desc="val")
        improved = va["f1_macro"] > BEST["val_f1_macro"] + CFG["MIN_DELTA"]
        if improved:
            BEST.update({"val_f1_macro": va["f1_macro"], "val_accuracy": va["accuracy"], "epoch": ep, "phase": phase})
            save_checkpoint(BEST["path"], model, {"epoch": ep, "phase": phase, "val_f1_macro": float(va["f1_macro"]),
                                                  "val_accuracy": float(va["accuracy"])})
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {"phase": phase, "phase_epoch": ep, "global_epoch": len(HISTORY) + 1,
               "train_loss": tr["loss"], "train_accuracy": tr["accuracy"], "val_loss": va["loss"],
               "val_accuracy": va["accuracy"], "val_f1_macro": va["f1_macro"],
               "lr_max": max(g["lr"] for g in optimizer.param_groups), "epoch_time_s": round(time.time() - t0, 1),
               "is_best": improved}
        HISTORY.append(row)
        print(f"[{phase}] epoch {ep:>2}/{epochs} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} F1 {va['f1_macro']:.4f}"
              f"{'  *best*' if improved else ''} | {row['epoch_time_s']}s")
        if bad_epochs >= patience:
            print(f"[{phase}] early stopping after {patience} epochs without val macro-F1 improvement")
            break

print("Training utilities ready.")
''')

md(r"""### 4.1 Phase 1 — classifier with frozen feature extractor""")

code(r'''
t0 = time.time()
set_backbone_trainable(model, False)
print(f"Trainable parameters in phase 1: {count_params(model)[1]:,} / {count_params(model)[0]:,}")
head_epochs = int(CFG["HEAD_EPOCHS"])
if head_epochs > 0:
    opt_head = torch.optim.AdamW(param_groups([(unwrap(model).classifier, CFG["HEAD_LR"])], CFG["WEIGHT_DECAY"]))
    sched_head = make_warmup_cosine(opt_head, head_epochs, len(train_loader), 0, CFG["MIN_LR_RATIO"])
    run_phase("head", head_epochs, True, opt_head, sched_head, make_grad_scaler(USE_AMP))
TIMINGS["train_head_s"] = round(time.time() - t0, 1)
''')

md(r"""### 4.2 Phase 2 — fine-tune the complete network""")

code(r'''
t0 = time.time()
set_backbone_trainable(model, True)
print(f"Trainable parameters in phase 2: {count_params(model)[1]:,} / {count_params(model)[0]:,}")
ft_epochs = int(CFG["FINETUNE_EPOCHS"])
if ft_epochs > 0:
    m_ = unwrap(model)
    opt_ft = torch.optim.AdamW(param_groups([(m_.features, CFG["FINETUNE_LR_BACKBONE"]),
                                             (m_.classifier, CFG["FINETUNE_LR_HEAD"])], CFG["WEIGHT_DECAY"]))
    sched_ft = make_warmup_cosine(opt_ft, ft_epochs, len(train_loader), CFG["WARMUP_EPOCHS"], CFG["MIN_LR_RATIO"])
    run_phase("finetune", ft_epochs, False, opt_ft, sched_ft, make_grad_scaler(USE_AMP))
TIMINGS["train_finetune_s"] = round(time.time() - t0, 1)
''')

md(r"""### 4.3 Restore the best checkpoint (by validation macro-F1)""")

code(r'''
if not Path(BEST["path"]).exists():
    print("No checkpoint saved during training -> saving current weights as 'best'.")
    save_checkpoint(BEST["path"], model, {"epoch": 0, "phase": "none", "val_f1_macro": -1.0, "val_accuracy": -1.0})
ckpt = safe_torch_load(BEST["path"])
best_model = unwrap(model)
best_model.load_state_dict(ckpt["model_state"])
best_model = best_model.to(DEVICE).eval()
BEST_CKPT_INFO = {k: ckpt[k] for k in ("phase", "epoch", "val_f1_macro", "val_accuracy")}
print(f"Restored best checkpoint: {BEST_CKPT_INFO}")
FP32_WEIGHTS_PATH = MODEL_DIR / f"{ARCH}_fp32_state_dict.pt"
torch.save(best_model.state_dict(), FP32_WEIGHTS_PATH)
''')

# =============================================================================
md(r"""
## 5. Training curves
Train metrics come from augmented batches with dropout on. Validation metrics come from clean images in eval mode. Both use the
same label-smoothed loss. The dashed line marks the phase-1 → phase-2 switch.
""")

code(r'''
INK, INK_2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # categorical slots 1-4, fixed order
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK_2,
                     "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID_C, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

hist_df = pd.DataFrame(HISTORY)
hist_df.to_csv(REPORT_DIR / "training_history.csv", index=False)
if len(hist_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")):
        x = hist_df["global_epoch"]
        ax.plot(x, hist_df[f"train_{key}"], color=SERIES[0], lw=2, marker="o", ms=5, label="Train")
        ax.plot(x, hist_df[f"val_{key}"], color=SERIES[1], lw=2, marker="o", ms=5, label="Validation")
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)
        (y_a, l_a), (y_b, l_b) = sorted([(hist_df[f"train_{key}"].iloc[-1], "Train"),
                                         (hist_df[f"val_{key}"].iloc[-1], "Validation")])
        if y_b - y_a < gap:
            mid = (y_a + y_b) / 2
            y_a, y_b = mid - gap / 2, mid + gap / 2
        for yy, lab in ((y_a, l_a), (y_b, l_b)):
            ax.annotate(lab, (x.iloc[-1], yy), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK_2, annotation_clip=False)
        n_head = int((hist_df["phase"] == "head").sum())
        if 0 < n_head < len(hist_df):
            ax.axvline(n_head + 0.5, color=INK_2, ls="--", lw=1)
            ax.text(n_head + 0.6, ax.get_ylim()[1], " fine-tune →", va="top", fontsize=8, color=INK_2)
        ax.set_title(f"{ARCH} seed {SEED}: training and validation {title.lower()}", loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
''')

# =============================================================================
md(r"""
## 6. Evaluation framework and FP32 results

All variants go through **the same suite**. Each one is scored on every available test set, and its raw logits are saved to
`predictions/<variant>__<test_set>.npz`, row-aligned with `predictions/index__<test_set>.csv`. Every statistic can therefore be
recomputed offline without the GPU. For `pv_test` the main row also records:

* accuracy, macro precision / recall / F1 with **95 % bootstrap CIs** (`BOOTSTRAP_ITERS` resamples of the test set);
* **calibration**: top-label ECE (`ECE_BINS` equal-width bins), NLL and multi-class Brier score;
* **weight census**: `parameters()` as PyTorch reports it, *and* the number of weight elements actually stored, which includes
  INT8 weights packed inside quantized modules. The two differ for quantized models; this is the Khan 2023 issue;
* latency: CPU at **1 thread** and at `CPU_THREADS` threads (batch 1), and GPU at batch 1 plus throughput. CPU and GPU numbers
  are not comparable with each other.
""")

code(r'''
def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def calibration_stats(y, logits, n_bins):
    p = softmax_np(logits.astype(np.float64))
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf > lo) & (conf <= hi)
        if msk.any():
            ece += msk.mean() * abs(correct[msk].mean() - conf[msk].mean())
            bins.append({"lo": lo, "hi": hi, "count": int(msk.sum()), "confidence": float(conf[msk].mean()),
                         "accuracy": float(correct[msk].mean())})
    nll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    return {"ece": float(ece), "nll": nll, "brier": brier, "mean_confidence": float(conf.mean())}, bins


def fast_macro_f1(y, p, labels):
    K = NUM_CLASSES
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1[labels].mean())


def bootstrap(y, pred, labels, pred_ref=None, iters=None, seed=0):
    """95% percentile CIs for accuracy and macro-F1; with pred_ref also for the paired differences (pred - ref)."""
    iters = int(iters or CFG["BOOTSTRAP_ITERS"])
    rng = np.random.default_rng(seed)
    n = len(y)
    acc, f1, dacc, df1 = [], [], [], []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], pred[idx]
        a, f = float((yy == pp).mean()), fast_macro_f1(yy, pp, labels)
        acc.append(a)
        f1.append(f)
        if pred_ref is not None:
            rr = pred_ref[idx]
            dacc.append(a - float((yy == rr).mean()))
            df1.append(f - fast_macro_f1(yy, rr, labels))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (None, None)
    return {"accuracy_ci": ci(acc), "f1_ci": ci(f1), "delta_accuracy_ci": ci(dacc), "delta_f1_ci": ci(df1)}


def mcnemar_exact(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, float(min(1.0, p))


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


CALIBRATION_BINS = {}


def run_suite(variant, m, device, use_amp=False, input_dtype=None):
    """Evaluate one variant on every test set, save logits, return the pv_test summary."""
    for ts, loader in EVAL_SETS.items():
        r = evaluate(m, loader, device, use_amp, input_dtype, desc=f"{ts} | {variant}")
        PRED_STORE[(variant, ts)] = {"y_true": r["y_true"], "logits": r["logits"]}
        np.savez_compressed(PRED_DIR / f"{slug(variant)}__{ts}.npz", y_true=r["y_true"], logits=r["logits"].astype(np.float32))
    y, lg = PRED_STORE[(variant, "pv_test")]["y_true"], PRED_STORE[(variant, "pv_test")]["logits"]
    pred = lg.argmax(1)
    s = compute_metrics(y, pred)
    cal, bins = calibration_stats(y, lg, int(CFG["ECE_BINS"]))
    CALIBRATION_BINS[variant] = bins
    bs = bootstrap(y, pred, np.arange(NUM_CLASSES), seed=SPLIT_SEED)
    s.update(cal)
    s.update({"accuracy_ci_low": bs["accuracy_ci"][0], "accuracy_ci_high": bs["accuracy_ci"][1],
              "f1_ci_low": bs["f1_ci"][0], "f1_ci_high": bs["f1_ci"][1]})
    return s


@torch.inference_mode()
def gpu_latency(m, batch_size, dtype=torch.float32, use_autocast=False):
    m.eval()
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=dtype)
    for _ in range(int(CFG["LATENCY_WARMUP"])):
        with autocast_ctx(use_autocast):
            m(x)
    torch.cuda.synchronize(DEVICE)
    times = []
    for _ in range(int(CFG["LATENCY_RUNS"])):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with autocast_ctx(use_autocast):
            m(x)
        end.record()
        torch.cuda.synchronize(DEVICE)
        times.append(start.elapsed_time(end))
    med = float(np.median(times))
    return {"median_ms": med, "p90_ms": float(np.percentile(times, 90)), "throughput_img_s": batch_size * 1000.0 / med}


@torch.inference_mode()
def cpu_latency(m, threads, batch_size=1):
    m = m.cpu().eval()
    prev = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    try:
        x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
        for _ in range(int(CFG["LATENCY_WARMUP"])):
            m(x)
        times = []
        for _ in range(int(CFG["LATENCY_RUNS"])):
            t = time.perf_counter()
            m(x)
            times.append((time.perf_counter() - t) * 1000)
    finally:
        torch.set_num_threads(prev)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


def weight_census(m):
    reported = sum(p.numel() for p in m.parameters())
    q_elems, n_q, n_f = 0, 0, 0
    for mod in m.modules():
        cls, path = type(mod), type(mod).__module__
        if (".quantized" in path and "qat" not in path and callable(getattr(mod, "weight", None))
                and cls.__name__ != "LinearPackedParams"):
            try:
                w = mod.weight()
                b = mod.bias() if callable(getattr(mod, "bias", None)) else getattr(mod, "bias", None)
                q_elems += w.numel() + (b.numel() if b is not None else 0)
                n_q += 1
            except Exception:
                pass
        elif isinstance(mod, (nn.Conv2d, nn.Linear)) and ".quantized" not in path:
            n_f += 1
    return {"params_reported": int(reported), "stored_weight_elements": int(reported + q_elems),
            "int8_weight_layers": n_q, "fp32_weight_layers": n_f}


def file_mb(path):
    return os.path.getsize(path) / 1e6 if path and os.path.exists(path) else None


def record_result(name, status, summary=None, size_path=None, census=None, cpu1=None, cpuN=None, gpu1=None, gpu_tp=None,
                  weights="", compute="", eval_device="", quant_scope="n/a", notes=""):
    summary, census = summary or {}, census or {}
    row = {"run_id": RUN_ID, "arch": ARCH, "seed": SEED, "experiment": name, "status": status}
    for k in ("accuracy", "accuracy_ci_low", "accuracy_ci_high", "precision_macro", "recall_macro", "f1_macro",
              "f1_ci_low", "f1_ci_high", "f1_weighted", "ece", "nll", "brier", "mean_confidence"):
        row[k] = summary.get(k)
    row.update({"model_file": str(size_path) if size_path else None, "model_size_mb": file_mb(size_path),
                "params_reported": census.get("params_reported"), "stored_weight_elements": census.get("stored_weight_elements"),
                "int8_weight_layers": census.get("int8_weight_layers"), "fp32_weight_layers": census.get("fp32_weight_layers"),
                "cpu_latency_1t_ms": cpu1["median_ms"] if cpu1 else None,
                "cpu_latency_Nt_ms": cpuN["median_ms"] if cpuN else None, "cpu_threads_N": CPU_THREADS,
                "gpu_latency_bs1_ms": gpu1["median_ms"] if gpu1 else None,
                "gpu_throughput_img_s": gpu_tp["throughput_img_s"] if gpu_tp else None,
                "weights_dtype": weights, "compute": compute, "metrics_device": eval_device,
                "quantization_scope": quant_scope, "notes": notes})
    RESULTS[name] = row


print("Evaluation framework ready. Test sets:", list(EVAL_SETS))
''')

code(r'''
t0 = time.time()
fp32_summary = run_suite("FP32", best_model, DEVICE)
fp32_census = weight_census(best_model)
fp32_gpu1 = gpu_latency(best_model, 1) if USE_CUDA else None
fp32_gpu_tp = gpu_latency(best_model, int(CFG["GPU_THROUGHPUT_BATCH"])) if USE_CUDA else None
_cpu_copy = copy.deepcopy(best_model)
fp32_cpu1, fp32_cpuN = cpu_latency(_cpu_copy, 1), cpu_latency(_cpu_copy, CPU_THREADS)
del _cpu_copy
record_result("FP32", "ok", fp32_summary, FP32_WEIGHTS_PATH, fp32_census, fp32_cpu1, fp32_cpuN, fp32_gpu1, fp32_gpu_tp,
              weights="float32", compute="float32", eval_device=str(DEVICE), notes="best checkpoint by val macro-F1; unfused")
print("FP32 pv_test:", {k: round(v, 4) for k, v in fp32_summary.items() if isinstance(v, float)})

y_fp32 = PRED_STORE[("FP32", "pv_test")]["y_true"]
p_fp32 = PRED_STORE[("FP32", "pv_test")]["logits"].argmax(1)
rep = classification_report(y_fp32, p_fp32, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
                            output_dict=True, zero_division=0)
pd.DataFrame(rep).T.to_csv(REPORT_DIR / "classification_report_fp32.csv")
fp32_cm = confusion_matrix(y_fp32, p_fp32, labels=list(range(NUM_CLASSES)))
pd.DataFrame(fp32_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(REPORT_DIR / "confusion_matrix_fp32.csv")
print(classification_report(y_fp32, p_fp32, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, digits=4, zero_division=0))
TIMINGS["eval_fp32_s"] = round(time.time() - t0, 1)
''')

code(r'''
cm_norm = fp32_cm / np.clip(fp32_cm.sum(axis=1, keepdims=True), 1, None)
size = max(6, 0.32 * NUM_CLASSES + 3)
fig, ax = plt.subplots(figsize=(size + 1.5, size))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
short = [c.replace("___", ": ").replace("_", " ")[:32] for c in CLASS_NAMES]
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(short, rotation=90, fontsize=7)
ax.set_yticks(range(NUM_CLASSES))
ax.set_yticklabels(short, fontsize=7)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.grid(False)
ax.set_title(f"{ARCH} seed {SEED}: FP32 test confusion matrix (row-normalised), macro-F1 {fp32_summary['f1_macro']:.4f}", loc="left")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Share of true class")
fig.tight_layout()
fig.savefig(FIG_DIR / "confusion_matrix_fp32.png", dpi=150, bbox_inches="tight")
plt.show()
''')

# =============================================================================
md(r"""
## 7. FP16 inference (GPU)
**FP16 autocast** keeps FP32 weights (same file as FP32) and lets `torch.autocast` run convolutions and matmuls in FP16.
**FP16 converted** is a separate `.half()` copy with FP16 weights, so its file is about half the size. Both run on CUDA only.
""")

code(r'''
t0 = time.time()
if not CFG["RUN_FP16"] or not USE_CUDA:
    why = "CFG['RUN_FP16'] = False" if not CFG["RUN_FP16"] else "requires CUDA"
    for n in ("FP16 autocast", "FP16 converted"):
        record_result(n, "skipped", notes=why)
else:
    try:
        s = run_suite("FP16 autocast", best_model, DEVICE, use_amp=True)
        record_result("FP16 autocast", "ok", s, FP32_WEIGHTS_PATH, fp32_census, None, None,
                      gpu_latency(best_model, 1, use_autocast=True),
                      gpu_latency(best_model, int(CFG["GPU_THROUGHPUT_BATCH"]), use_autocast=True),
                      weights="float32", compute="mixed fp16 (torch.autocast)", eval_device=str(DEVICE),
                      notes="same FP32 weights; size = FP32 file")
        print("FP16 autocast:", round(s["accuracy"], 4), round(s["f1_macro"], 4))
    except Exception as e:
        log_error("fp16_autocast", e)
        record_result("FP16 autocast", "failed", notes=str(e))
    try:
        model_fp16 = copy.deepcopy(best_model).half().to(DEVICE).eval()
        s = run_suite("FP16 converted", model_fp16, DEVICE, input_dtype=torch.float16)
        fp16_path = MODEL_DIR / f"{ARCH}_fp16_state_dict.pt"
        torch.save(model_fp16.state_dict(), fp16_path)
        record_result("FP16 converted", "ok", s, fp16_path, weight_census(model_fp16), None, None,
                      gpu_latency(model_fp16, 1, dtype=torch.float16),
                      gpu_latency(model_fp16, int(CFG["GPU_THROUGHPUT_BATCH"]), dtype=torch.float16),
                      weights="float16", compute="float16 (model.half())", eval_device=str(DEVICE))
        print("FP16 converted:", round(s["accuracy"], 4), round(s["f1_macro"], 4))
        del model_fp16
        torch.cuda.empty_cache()
    except Exception as e:
        log_error("fp16_converted", e)
        record_result("FP16 converted", "failed", notes=str(e))
TIMINGS["fp16_s"] = round(time.time() - t0, 1)
''')

# =============================================================================
md(r"""
## 8. INT8 toolkit, and the Khan et al. (2023) dynamic-quantization protocol

### 8.1 Quantization toolkit and operator inspection
The available API (`torch.ao.quantization`, with `torch.quantization` as a fallback) and a CPU quantized engine
(`x86` → `fbgemm` → `qnnpack` → `onednn`) are picked automatically. **Every warning raised by prepare/convert is recorded**
(deprecations included) in `experiment_config.json`.

`inspect_quantization` is the only source for the *quantization scope* column. It counts quantized vs float Conv/Linear modules
and quantized vs float adds/multiplies (residual adds; squeeze-excite multiplies in MobileNetV3). "FULL INT8" is printed only
when no float compute layer is left. In torchvision's MobileNetV2, the 7 non-residual blocks still own a `skip_add` that
`forward()` never calls. Those are excluded from observers and from the counts.
""")

code(r'''
tq, QUANT_API, QUANT_IMPORT_ERROR = None, None, None
for _name in ("torch.ao.quantization", "torch.quantization"):
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            tq = __import__(_name, fromlist=["prepare"])
            _ = (tq.QuantStub, tq.prepare, tq.convert, tq.get_default_qconfig, tq.get_default_qat_qconfig,
                 tq.prepare_qat, tq.quantize_dynamic)
        QUANT_API = _name
        break
    except Exception as e:
        tq, QUANT_IMPORT_ERROR = None, f"{_name}: {type(e).__name__}: {e}"

SUPPORTED_ENGINES = [e for e in torch.backends.quantized.supported_engines if e != "none"]
if CFG["QUANT_BACKEND"] == "auto":
    QUANT_ENGINE = next((e for e in ("x86", "fbgemm", "qnnpack", "onednn") if e in SUPPORTED_ENGINES), None)
else:
    QUANT_ENGINE = CFG["QUANT_BACKEND"] if CFG["QUANT_BACKEND"] in SUPPORTED_ENGINES else None
if QUANT_ENGINE:
    torch.backends.quantized.engine = QUANT_ENGINE
try:
    import torch.ao.nn.intrinsic.qat as nniqat
except Exception:
    try:
        import torch.nn.intrinsic.qat as nniqat
    except Exception:
        nniqat = None

QUANT_INFO = {"api": QUANT_API, "import_error": QUANT_IMPORT_ERROR, "supported_engines": SUPPORTED_ENGINES,
              "engine": QUANT_ENGINE, "warnings": {}}
QUANT_READY = tq is not None and QUANT_ENGINE is not None
print(json.dumps({k: v for k, v in QUANT_INFO.items() if k != "warnings"}, indent=2))


def capture_warnings(stage, fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    msgs = sorted({f"{x.category.__name__}: {str(x.message)[:300]}" for x in w})
    if msgs:
        QUANT_INFO["warnings"][stage] = msgs
    return out


def fuse_quantizable(m, is_qat):
    try:
        m.fuse_model(is_qat=is_qat)
    except TypeError:
        m.fuse_model()
    return m


def is_unused_skip_add(name, mods):
    if not name.endswith("skip_add"):
        return False
    parent = mods.get(name.rsplit(".", 1)[0] if "." in name else "")
    return parent is not None and hasattr(parent, "use_res_connect") and not parent.use_res_connect


def exclude_unused_skip_adds(m):
    mods = dict(m.named_modules())
    for name, mod in mods.items():
        if is_unused_skip_add(name, mods):
            mod.qconfig = None
    return m


def strip_stubs(m):
    if hasattr(m, "quant"):
        m.quant, m.dequant = nn.Identity(), nn.Identity()
    return m


ADD_TARGETS = {operator.add, operator.iadd, torch.add}
MUL_TARGETS = {operator.mul, operator.imul, torch.mul}


def inspect_quantization(m):
    rep, weight_dtypes, float_layers = Counter(), Counter(), []
    mods = dict(m.named_modules())
    for name, mod in mods.items():
        cls, modpath = type(mod), type(mod).__module__
        if is_unused_skip_add(name, mods):
            rep["unused_skip_add_ignored"] += 1
            continue
        is_q = ".quantized" in modpath and "qat" not in modpath
        if is_q and callable(getattr(mod, "weight", None)) and cls.__name__ != "LinearPackedParams":
            kind = "conv" if "Conv" in cls.__name__ else ("linear" if "Linear" in cls.__name__ else "other")
            rep[f"quantized_{kind}"] += 1
            try:
                weight_dtypes[str(mod.weight().dtype)] += 1
            except Exception:
                pass
        elif isinstance(mod, (nn.Conv2d, nn.Linear)) and not is_q:
            rep["float_conv" if isinstance(mod, nn.Conv2d) else "float_linear"] += 1
            float_layers.append(name)
        if cls.__name__ == "FloatFunctional":
            rep["float_add"] += 1
        elif cls.__name__ == "QFunctional":
            rep["quantized_add"] += 1
    if isinstance(m, torch.fx.GraphModule):
        for node in m.graph.nodes:
            if node.op != "call_function":
                continue
            tgt = str(node.target)
            if node.target in ADD_TARGETS:
                rep["float_add"] += 1
            elif node.target in MUL_TARGETS:
                rep["float_mul"] += 1
            elif "quantized.add" in tgt:
                rep["quantized_add"] += 1
            elif "quantized.mul" in tgt:
                rep["quantized_mul"] += 1
    nq = rep["quantized_conv"] + rep["quantized_linear"]
    nf = rep["float_conv"] + rep["float_linear"] + rep["float_add"] + rep["float_mul"]
    if nq > 0 and nf == 0:
        scope = (f"FULL INT8: {rep['quantized_conv']} conv + {rep['quantized_linear']} linear + {rep['quantized_add']} add"
                 + (f" + {rep['quantized_mul']} mul" if rep["quantized_mul"] else "") + " quantized")
    elif nq > 0:
        scope = (f"PARTIAL INT8: quantized conv {rep['quantized_conv']}, linear {rep['quantized_linear']}; still FLOAT conv "
                 f"{rep['float_conv']}, linear {rep['float_linear']}, add {rep['float_add']}, mul {rep['float_mul']}")
    else:
        scope = "NOT QUANTIZED: no quantized Conv/Linear modules"
    return {"counts": dict(rep), "weight_dtypes": dict(weight_dtypes), "full_int8": nq > 0 and nf == 0,
            "scope": scope, "float_layers": float_layers[:20]}


def save_int8(m, tag):
    path = MODEL_DIR / f"{ARCH}_{tag}_state_dict.pt"
    torch.save(m.state_dict(), path)
    ts_path = None
    if CFG["EXPORT_TORCHSCRIPT"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ts = torch.jit.trace(m, torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
                ts_path = MODEL_DIR / f"{ARCH}_{tag}_torchscript.pt"
                ts.save(str(ts_path))
        except Exception as e:
            log_error(f"{tag}.torchscript_export", e)
    return path, ts_path


QUANT_INSPECTIONS = {}
''')

md(r"""
### 8.2 INT8 dynamic quantization (Khan et al. 2023 protocol)
Khan et al. (2023) used `torch.quantization.quantize_dynamic` and reported that it quantized "Linear" and "Conv2d" layers and
reduced the parameter count. This cell applies exactly that call, `quantize_dynamic(model, {nn.Linear, nn.Conv2d}, qint8)`, to
our trained model and records three things: which layers actually became INT8, what `model.parameters()` reports, and how many
weight elements are really stored. (The training protocol of that paper is **not** reproduced, only its quantization step.)
""")

code(r'''
DYN_NAME = "INT8 dynamic (Khan2023 protocol)"
t0 = time.time()
if not CFG["RUN_DYNAMIC_QUANT"]:
    record_result(DYN_NAME, "skipped", notes="CFG['RUN_DYNAMIC_QUANT'] = False")
elif tq is None:
    record_result(DYN_NAME, "failed", notes=f"quantization API unavailable: {QUANT_IMPORT_ERROR}")
else:
    try:
        dyn = capture_warnings("dynamic.quantize", tq.quantize_dynamic, copy.deepcopy(best_model).cpu().eval(),
                               {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        insp, census = inspect_quantization(dyn), weight_census(dyn)
        QUANT_INSPECTIONS[DYN_NAME] = insp
        s = run_suite(DYN_NAME, dyn, torch.device("cpu"))
        dyn_path, _ = save_int8(dyn, "int8_dynamic")
        record_result(DYN_NAME, "ok", s, dyn_path, census, cpu_latency(dyn, 1), cpu_latency(dyn, CPU_THREADS),
                      weights="qint8 (Linear only) + float32", compute="dynamic int8 Linear, float32 Conv",
                      eval_device="cpu", quant_scope=insp["scope"],
                      notes=f"parameters() {census['params_reported']:,} vs stored weight elements "
                            f"{census['stored_weight_elements']:,} (FP32 model: {fp32_census['params_reported']:,})")
        print(">>", insp["scope"])
        print(f"parameters() reports {census['params_reported']:,}; stored weight elements {census['stored_weight_elements']:,}; "
              f"FP32 parameters {fp32_census['params_reported']:,}")
        print("accuracy", round(s["accuracy"], 4), "macro-F1", round(s["f1_macro"], 4))
    except Exception as e:
        log_error("dynamic_quant", e)
        record_result(DYN_NAME, "failed", notes=str(e))
TIMINGS["dynamic_s"] = round(time.time() - t0, 1)
''')

# =============================================================================
md(r"""
## 9. INT8 static post-training quantization (PTQ)
Steps: fuse, then observe the calibration images (training split), then convert. MobileNetV2 tries **eager** mode first and
falls back to **FX** mode. MobileNetV3-small uses FX mode directly. Every attempt and error is logged, and the converted model has
to pass a forward pass and the operator inspection. Quantized kernels run on CPU only.
""")

code(r'''
@torch.no_grad()
def calibrate(m, loader):
    m.eval()
    for x, _ in loader:
        m(x)


def ptq_eager(float_model):
    m = copy.deepcopy(unwrap(float_model)).cpu().eval()
    fuse_quantizable(m, is_qat=False)
    m.qconfig = capture_warnings("ptq_eager.qconfig", tq.get_default_qconfig, QUANT_ENGINE)
    exclude_unused_skip_adds(m)
    capture_warnings("ptq_eager.prepare", tq.prepare, m, inplace=True)
    calibrate(m, calib_loader)
    return capture_warnings("ptq_eager.convert", tq.convert, m, inplace=False)


def ptq_fx(float_model, float_blocks=()):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    m = strip_stubs(copy.deepcopy(unwrap(float_model)).cpu().eval())
    qmap = capture_warnings("ptq_fx.qconfig", get_default_qconfig_mapping, QUANT_ENGINE)
    for n in float_blocks:
        qmap.set_module_name(n, None)
    prepared = capture_warnings("ptq_fx.prepare", prepare_fx, m, qmap, (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),))
    calibrate(prepared, calib_loader)
    return capture_warnings("ptq_fx.convert", convert_fx, prepared)


def quant_methods(setting):
    if setting == "auto":
        return ["eager", "fx"] if SUPPORTS_EAGER_QUANT else ["fx"]
    return [setting]


def check_converted(qm):
    with torch.inference_mode():
        out = qm(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
    assert out.shape == (2, NUM_CLASSES), f"unexpected output shape {tuple(out.shape)}"
    insp = inspect_quantization(qm)
    if insp["counts"].get("quantized_conv", 0) + insp["counts"].get("quantized_linear", 0) == 0:
        raise RuntimeError("conversion produced no quantized Conv/Linear modules")
    return insp


PTQ_ATTEMPTS, ptq_model = [], None
t0 = time.time()
if not CFG["RUN_PTQ"]:
    record_result("INT8 PTQ", "skipped", notes="CFG['RUN_PTQ'] = False")
elif not QUANT_READY:
    record_result("INT8 PTQ", "failed", notes=f"quantization unavailable: {QUANT_IMPORT_ERROR or SUPPORTED_ENGINES}")
else:
    for method in quant_methods(CFG["PTQ_METHOD"]):
        try:
            qm = ptq_eager(best_model) if method == "eager" else ptq_fx(best_model)
            insp = check_converted(qm)
            PTQ_ATTEMPTS.append({"method": method, "status": "ok", "scope": insp["scope"]})
            ptq_model, PTQ_METHOD_USED = qm, method
            QUANT_INSPECTIONS["INT8 PTQ"] = insp
            break
        except Exception as e:
            log_error(f"ptq_{method}", e)
            PTQ_ATTEMPTS.append({"method": method, "status": "failed", "error": f"{type(e).__name__}: {e}"})
    print("PTQ attempts:", json.dumps(PTQ_ATTEMPTS, indent=2))
    if ptq_model is None:
        record_result("INT8 PTQ", "failed", notes="; ".join(a.get("error", "") for a in PTQ_ATTEMPTS))
    else:
        insp = QUANT_INSPECTIONS["INT8 PTQ"]
        print(">>", insp["scope"])
        s = run_suite("INT8 PTQ", ptq_model, torch.device("cpu"))
        ptq_path, ptq_ts = save_int8(ptq_model, "int8_ptq")
        record_result("INT8 PTQ", "ok", s, ptq_path, weight_census(ptq_model), cpu_latency(ptq_model, 1),
                      cpu_latency(ptq_model, CPU_THREADS), weights="qint8 (per-channel)",
                      compute=f"int8 ({QUANT_ENGINE}, {PTQ_METHOD_USED})", eval_device="cpu", quant_scope=insp["scope"],
                      notes=f"static PTQ, {len(calib_ds)} calibration images from train")
        print("INT8 PTQ:", round(s["accuracy"], 4), round(s["f1_macro"], 4))
TIMINGS["ptq_s"] = round(time.time() - t0, 1)
''')

# =============================================================================
md(r"""
## 10. INT8 quantization-aware training (QAT)
Starting from the best FP32 checkpoint, the model is fused for QAT and gets the engine's QAT qconfig (fake-quantized weights and
activations). It is then fine-tuned at a low LR in FP32 on the GPU. BN statistics freeze from `QAT_FREEZE_BN_EPOCH`, and
observers freeze from `QAT_FREEZE_OBSERVER_EPOCH`. The best epoch is chosen by **validation** macro-F1 with observers paused, so
validation data never moves the quantization ranges. That epoch is converted to INT8 on CPU and tested once. The fake-quant state is
also saved (`checkpoints/qat_fakequant_state.pt`) so it can later be converted for other runtimes, e.g. the Android phone benchmark.
""")

code(r'''
def qat_prepare(method, float_model, qmap=None):
    if qmap is not None:                    # custom mapping (outlier-aware / low-bit) -> FX
        from torch.ao.quantization.quantize_fx import prepare_qat_fx
        m = strip_stubs(copy.deepcopy(unwrap(float_model)).cpu().train())
        return capture_warnings("qat_custom.prepare", prepare_qat_fx, m, qmap, (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),))
    if method == "eager":
        m = copy.deepcopy(unwrap(float_model)).cpu().train()
        fuse_quantizable(m, is_qat=True)
        m.qconfig = capture_warnings("qat_eager.qconfig", tq.get_default_qat_qconfig, QUANT_ENGINE)
        exclude_unused_skip_adds(m)
        return capture_warnings("qat_eager.prepare", tq.prepare_qat, m, inplace=False)
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx
    m = strip_stubs(copy.deepcopy(unwrap(float_model)).cpu().train())
    qmap = capture_warnings("qat_fx.qconfig", get_default_qat_qconfig_mapping, QUANT_ENGINE)
    return capture_warnings("qat_fx.prepare", prepare_qat_fx, m, qmap, (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),))


def qat_convert(method, prepared):
    m = copy.deepcopy(prepared).cpu().eval()
    if method == "eager":
        return capture_warnings("qat_eager.convert", tq.convert, m, inplace=False)
    from torch.ao.quantization.quantize_fx import convert_fx
    return capture_warnings("qat_fx.convert", convert_fx, m)


def run_qat(method, qmap=None, tag="int8"):
    qat_model = qat_prepare(method, best_model, qmap).to(DEVICE)
    opt = torch.optim.AdamW([p for p in qat_model.parameters() if p.requires_grad], lr=CFG["QAT_LR"],
                            weight_decay=CFG["WEIGHT_DECAY"])
    max_steps = CFG["QAT_MAX_STEPS_PER_EPOCH"]
    steps = min(len(train_loader), max_steps) if max_steps else len(train_loader)
    sched = make_warmup_cosine(opt, CFG["QAT_EPOCHS"], steps, 0, CFG["MIN_LR_RATIO"])
    scaler = make_grad_scaler(False)
    best_f1, best_state, hist = -1.0, None, []
    for ep in range(int(CFG["QAT_EPOCHS"])):
        t_ep = time.time()
        observers_frozen = ep >= CFG["QAT_FREEZE_OBSERVER_EPOCH"]
        qat_model.apply(tq.disable_observer if observers_frozen else tq.enable_observer)
        if ep >= CFG["QAT_FREEZE_BN_EPOCH"] and nniqat is not None:
            qat_model.apply(nniqat.freeze_bn_stats)
        tr = train_one_epoch(qat_model, train_loader, opt, sched, scaler, False, False, max_steps=max_steps,
                             desc=f"QAT {ep + 1}/{CFG['QAT_EPOCHS']}")
        qat_model.apply(tq.disable_observer)
        va = evaluate(qat_model, val_loader, DEVICE, desc="val (fake-quant)")
        if va["f1_macro"] > best_f1:
            best_f1 = va["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in qat_model.state_dict().items()}
        hist.append({"qat_epoch": ep + 1, "method": method, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                     "val_loss": va["loss"], "val_accuracy": va["accuracy"], "val_f1_macro": va["f1_macro"],
                     "observers_frozen": observers_frozen, "bn_frozen": ep >= CFG["QAT_FREEZE_BN_EPOCH"],
                     "epoch_time_s": round(time.time() - t_ep, 1)})
        hist[-1]["tag"] = tag
        print(f"[QAT-{method}-{tag}] epoch {ep + 1} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
              f"val (fake-quant) acc {va['accuracy']:.4f} F1 {va['f1_macro']:.4f}")
    if best_state is not None:
        qat_model.load_state_dict(best_state)
        torch.save(best_state, CKPT_DIR / ("qat_fakequant_state.pt" if tag == "int8" else f"qat_fakequant_state_{tag}.pt"))
    return qat_model, hist, best_f1


QAT_ATTEMPTS, qat_int8, QAT_HISTORY = [], None, []
t0 = time.time()
if not CFG["RUN_QAT"]:
    record_result("INT8 QAT", "skipped", notes="CFG['RUN_QAT'] = False")
elif not QUANT_READY:
    record_result("INT8 QAT", "failed", notes=f"quantization unavailable: {QUANT_IMPORT_ERROR or SUPPORTED_ENGINES}")
elif int(CFG["QAT_EPOCHS"]) < 1:
    record_result("INT8 QAT", "skipped", notes="QAT_EPOCHS < 1")
else:
    for method in quant_methods(CFG["QAT_METHOD"]):
        try:
            prepared, hist, val_f1 = run_qat(method)
            QAT_HISTORY.extend(hist)
            qm = qat_convert(method, prepared)
            insp = check_converted(qm)
            QAT_ATTEMPTS.append({"method": method, "status": "ok", "best_val_f1_fake_quant": val_f1, "scope": insp["scope"]})
            qat_int8, QAT_METHOD_USED = qm, method
            QUANT_INSPECTIONS["INT8 QAT"] = insp
            del prepared
            break
        except Exception as e:
            log_error(f"qat_{method}", e)
            QAT_ATTEMPTS.append({"method": method, "status": "failed", "error": f"{type(e).__name__}: {e}"})
        finally:
            if USE_CUDA:
                torch.cuda.empty_cache()
    print("QAT attempts:", json.dumps(QAT_ATTEMPTS, indent=2))
    if QAT_HISTORY:
        pd.DataFrame(QAT_HISTORY).to_csv(REPORT_DIR / "qat_history.csv", index=False)
    if qat_int8 is None:
        record_result("INT8 QAT", "failed", notes="; ".join(a.get("error", "") for a in QAT_ATTEMPTS))
    else:
        insp = QUANT_INSPECTIONS["INT8 QAT"]
        print(">>", insp["scope"])
        s = run_suite("INT8 QAT", qat_int8, torch.device("cpu"))
        qat_path, _ = save_int8(qat_int8, "int8_qat")
        record_result("INT8 QAT", "ok", s, qat_path, weight_census(qat_int8), cpu_latency(qat_int8, 1),
                      cpu_latency(qat_int8, CPU_THREADS), weights="qint8 (per-channel)",
                      compute=f"int8 ({QUANT_ENGINE}, {QAT_METHOD_USED})", eval_device="cpu", quant_scope=insp["scope"],
                      notes=f"{int(CFG['QAT_EPOCHS'])} QAT epochs; best epoch by val macro-F1 (fake-quant)")
        print("INT8 QAT:", round(s["accuracy"], 4), round(s["f1_macro"], 4))
TIMINGS["qat_s"] = round(time.time() - t0, 1)
''')

md(r"""
### 10.2 Outlier-aware INT8 (our method)

The pilots found that static INT8 of torchvision MobileNetV3-Small collapses. The cause is one stem activation channel whose range
is about 40–90× the median channel, inherited from the ImageNet weights. A single per-tensor 8-bit scale leaves the other
channels almost no levels. The rule below finds such tensors automatically, **on calibration (training) images only**:

1. prepare the model in FX graph mode and run the calibration images;
2. at every activation observer (each tensor that will be quantized), measure the per-channel range (max − min) of its input,
   skipping pooled 1×1 tensors;
3. if max/median > `OUTLIER_RATIO_THRESHOLD` (16, fixed a priori: with x86's 7-bit activations a 16× outlier leaves a typical
   channel 128/16 = 8 levels), keep the top-level blocks that produce and consume that tensor in float; repeat once.

The excluded blocks are listed and appear in the operator census as PARTIAL INT8, so the compression is never overstated.
The same block list is used for the outlier-aware QAT.
""")

code(r'''
from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx as _prepare_fx


def block_of(name):
    parts = str(name).split(".")
    return ".".join(parts[:2]) if parts[0] == "features" and len(parts) > 1 else parts[0]


def find_outlier_blocks(float_model, threshold, rounds=2):
    model = strip_stubs(copy.deepcopy(unwrap(float_model)).cpu().eval())
    example = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)
    float_blocks, log = set(), []
    for r in range(rounds):
        qmap = get_default_qconfig_mapping(QUANT_ENGINE)
        for n in float_blocks:
            qmap.set_module_name(n, None)
        prepared = capture_warnings("outlier.prepare", _prepare_fx, copy.deepcopy(model), qmap, example)
        stats, hooks = {}, []
        for name, mod in prepared.named_modules():
            if "activation_post_process" in name and hasattr(mod, "calculate_qparams"):
                def hook(m_, inp, out, k=name):
                    x = inp[0].detach()
                    if x.dim() == 4 and x.shape[2] * x.shape[3] > 1:
                        ch = x.transpose(0, 1).reshape(x.shape[1], -1)
                        mx, mn = ch.max(1).values, ch.min(1).values
                        if k in stats:
                            stats[k] = (torch.maximum(stats[k][0], mx), torch.minimum(stats[k][1], mn))
                        else:
                            stats[k] = (mx, mn)
                hooks.append(mod.register_forward_hook(hook))
        calibrate(prepared, calib_loader)
        for h in hooks:
            h.remove()
        ratios = {k: float((mx - mn).max() / (mx - mn).median().clamp_min(1e-8)) for k, (mx, mn) in stats.items()}
        flagged = {k: v for k, v in ratios.items() if v > threshold}
        new = set()
        for node in prepared.graph.nodes:
            if node.op == "call_module" and node.target in flagged:
                prod = node.args[0]
                if getattr(prod, "op", None) == "call_module":
                    new.add(block_of(prod.target))
                for user in node.users:
                    if user.op == "call_module" and "activation_post_process" not in str(user.target):
                        new.add(block_of(user.target))
        log.append({"round": r + 1, "tensors_checked": len(ratios), "max_ratio": max(ratios.values()) if ratios else None,
                    "flagged": {k: round(v, 1) for k, v in flagged.items()}, "new_float_blocks": sorted(new - float_blocks)})
        if not (new - float_blocks):
            break
        float_blocks |= new
    return sorted(float_blocks), log


OA_PTQ, OA_QAT = "INT8 PTQ outlier-aware", "INT8 QAT outlier-aware"
OUTLIER_BLOCKS, OUTLIER_LOG = [], []
t0 = time.time()
if not (CFG["RUN_OUTLIER_AWARE"] and QUANT_READY):
    for n in (OA_PTQ, OA_QAT):
        record_result(n, "skipped", notes="disabled or quantization unavailable")
else:
    try:
        OUTLIER_BLOCKS, OUTLIER_LOG = find_outlier_blocks(best_model, float(CFG["OUTLIER_RATIO_THRESHOLD"]))
        print("Outlier analysis:", json.dumps(OUTLIER_LOG, indent=2))
        print(">> blocks kept in float:", OUTLIER_BLOCKS or "none")
        if not OUTLIER_BLOCKS:
            for n in (OA_PTQ, OA_QAT):
                record_result(n, "skipped", notes="no outlier tensor above threshold: identical to the plain variant")
        else:
            qm = ptq_fx(best_model, OUTLIER_BLOCKS)
            insp = check_converted(qm)
            QUANT_INSPECTIONS[OA_PTQ] = insp
            s = run_suite(OA_PTQ, qm, torch.device("cpu"))
            pth, _ = save_int8(qm, "int8_ptq_outlier_aware")
            record_result(OA_PTQ, "ok", s, pth, weight_census(qm), cpu_latency(qm, 1), cpu_latency(qm, CPU_THREADS),
                          weights="qint8 + float islands", compute=f"int8 ({QUANT_ENGINE}, fx) except float blocks",
                          eval_device="cpu", quant_scope=insp["scope"] + f" | float blocks: {OUTLIER_BLOCKS}",
                          notes=f"outlier-aware rule, threshold {CFG['OUTLIER_RATIO_THRESHOLD']}x")
            print(f"{OA_PTQ}: acc {s['accuracy']:.4f} F1 {s['f1_macro']:.4f}")
            if CFG["RUN_QAT"] and int(CFG["QAT_EPOCHS"]) > 0:
                qmap = get_default_qat_qconfig_mapping(QUANT_ENGINE)
                for n in OUTLIER_BLOCKS:
                    qmap.set_module_name(n, None)
                prepared, hist, vf1 = run_qat("fx", qmap=qmap, tag="int8_outlier_aware")
                QAT_HISTORY.extend(hist)
                qm = qat_convert("fx", prepared)
                insp = check_converted(qm)
                QUANT_INSPECTIONS[OA_QAT] = insp
                s = run_suite(OA_QAT, qm, torch.device("cpu"))
                pth, _ = save_int8(qm, "int8_qat_outlier_aware")
                record_result(OA_QAT, "ok", s, pth, weight_census(qm), cpu_latency(qm, 1), cpu_latency(qm, CPU_THREADS),
                              weights="qint8 + float islands", compute=f"int8 ({QUANT_ENGINE}, fx) except float blocks",
                              eval_device="cpu", quant_scope=insp["scope"] + f" | float blocks: {OUTLIER_BLOCKS}",
                              notes=f"{CFG['QAT_EPOCHS']} QAT epochs, best by val macro-F1 (fake-quant {vf1:.4f})")
                print(f"{OA_QAT}: acc {s['accuracy']:.4f} F1 {s['f1_macro']:.4f}")
                del prepared
                if USE_CUDA:
                    torch.cuda.empty_cache()
    except Exception as e:
        log_error("outlier_aware", e)
        for n in (OA_PTQ, OA_QAT):
            if n not in RESULTS:
                record_result(n, "failed", notes=str(e))
if QAT_HISTORY:
    pd.DataFrame(QAT_HISTORY).to_csv(REPORT_DIR / "qat_history.csv", index=False)
TIMINGS["outlier_aware_s"] = round(time.time() - t0, 1)
''')

# =============================================================================
md(r"""
## 11. Benchmark and paired statistics

* **Main table** (`benchmark_results.csv`): one row per variant on `pv_test`, with CIs, calibration, census and latency.
* **Paired vs FP32** (same images): prediction **flip rate** and agreement, exact **McNemar** test (b = FP32 right / variant
  wrong, c = the reverse), and paired-bootstrap 95 % CIs of Δaccuracy and Δmacro-F1.
* **Per-class** (`per_class_f1.csv`): F1 of every variant for each class, with ΔF1 vs FP32 and class support. Used to check whether
  quantization hurts minority classes more.
* **Shift** (`shift_results.csv`): every variant × test set. `pv_segmented` is reported next to the *same* leaves in colour
  (`pv_test_matched`). `plantdoc` (official test split) and `plantdoc_all` (all PlantDoc images) are reported under the strict and
  restricted protocols, with paired McNemar vs FP32 on the same images.
""")

code(r'''
results_df = pd.DataFrame(list(RESULTS.values()))
ok_variants = [n for n, r in RESULTS.items() if r["status"] == "ok" and (n, "pv_test") in PRED_STORE]
y_test = PRED_STORE[("FP32", "pv_test")]["y_true"]
pred = {n: PRED_STORE[(n, "pv_test")]["logits"].argmax(1) for n in ok_variants}

paired_rows = []
for n in ok_variants:
    if n == "FP32":
        continue
    b, c, p = mcnemar_exact(y_test, pred["FP32"], pred[n])
    bs = bootstrap(y_test, pred[n], np.arange(NUM_CLASSES), pred_ref=pred["FP32"], seed=SPLIT_SEED)
    row = {"experiment": n, "flip_rate_vs_fp32": float((pred[n] != pred["FP32"]).mean()),
           "agreement_vs_fp32": float((pred[n] == pred["FP32"]).mean()),
           "mcnemar_b_fp32_right_variant_wrong": b, "mcnemar_c_fp32_wrong_variant_right": c, "mcnemar_p": p,
           "delta_accuracy": RESULTS[n]["accuracy"] - RESULTS["FP32"]["accuracy"],
           "delta_accuracy_ci_low": bs["delta_accuracy_ci"][0], "delta_accuracy_ci_high": bs["delta_accuracy_ci"][1],
           "delta_f1_macro": RESULTS[n]["f1_macro"] - RESULTS["FP32"]["f1_macro"],
           "delta_f1_ci_low": bs["delta_f1_ci"][0], "delta_f1_ci_high": bs["delta_f1_ci"][1]}
    paired_rows.append(row)
paired_df = pd.DataFrame(paired_rows)
if len(paired_df):
    results_df = results_df.merge(paired_df, on="experiment", how="left")

per_class = pd.DataFrame({"class_name": CLASS_NAMES, "support": np.bincount(y_test, minlength=NUM_CLASSES)})
for n in ok_variants:
    _, _, f1c, _ = precision_recall_fscore_support(y_test, pred[n], labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class[f"f1__{slug(n)}"] = f1c
for n in ok_variants:
    if n != "FP32":
        per_class[f"delta_f1__{slug(n)}"] = per_class[f"f1__{slug(n)}"] - per_class["f1__fp32"]
per_class.to_csv(REPORT_DIR / "per_class_f1.csv", index=False)

shift_rows = []


def add_shift(variant, test_set, protocol, y, p, labels, ref_p=None, logits=None):
    m = compute_metrics(y, p, labels)
    row = {"run_id": RUN_ID, "arch": ARCH, "seed": SEED, "experiment": variant, "test_set": test_set, "protocol": protocol,
           "n": int(len(y)), "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}
    if logits is not None:
        row.update({f"{k}": v for k, v in calibration_stats(y, logits, int(CFG["ECE_BINS"]))[0].items()})
    if ref_p is not None and variant != "FP32":
        b, c, pv = mcnemar_exact(y, ref_p, p)
        row.update({"flip_rate_vs_fp32": float((p != ref_p).mean()), "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": pv})
    shift_rows.append(row)


all_labels = np.arange(NUM_CLASSES)
for n in ok_variants:
    add_shift(n, "pv_test", "all_classes", y_test, pred[n], all_labels, pred["FP32"], PRED_STORE[(n, "pv_test")]["logits"])
    if SEG_POSITIONS is not None and (n, "pv_segmented") in PRED_STORE:
        ys = PRED_STORE[(n, "pv_segmented")]["y_true"]
        lg_s = PRED_STORE[(n, "pv_segmented")]["logits"]
        lg_c = PRED_STORE[(n, "pv_test")]["logits"][SEG_POSITIONS]
        ref_s = PRED_STORE[("FP32", "pv_segmented")]["logits"].argmax(1)
        ref_c = PRED_STORE[("FP32", "pv_test")]["logits"][SEG_POSITIONS].argmax(1)
        add_shift(n, "pv_test_matched", "all_classes", ys, lg_c.argmax(1), all_labels, ref_c, lg_c)
        add_shift(n, "pv_segmented", "all_classes", ys, lg_s.argmax(1), all_labels, ref_s, lg_s)
    for pd_name in ("plantdoc", "plantdoc_all"):
        if PD_TARGETS is None or (n, pd_name) not in PRED_STORE:
            continue
        yp, lg = PRED_STORE[(n, pd_name)]["y_true"], PRED_STORE[(n, pd_name)]["logits"]
        present = np.unique(yp)
        lg_ref = PRED_STORE[("FP32", pd_name)]["logits"]
        masked = np.full_like(lg, -np.inf)
        masked[:, PD_TARGETS] = lg[:, PD_TARGETS]
        masked_ref = np.full_like(lg_ref, -np.inf)
        masked_ref[:, PD_TARGETS] = lg_ref[:, PD_TARGETS]
        add_shift(n, pd_name, "strict", yp, lg.argmax(1), present, lg_ref.argmax(1), lg)
        add_shift(n, pd_name, "restricted", yp, masked.argmax(1), present, masked_ref.argmax(1))
shift_df = pd.DataFrame(shift_rows)
shift_df.to_csv(REPORT_DIR / "shift_results.csv", index=False)

show = ["experiment", "status", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "f1_macro", "ece", "model_size_mb",
        "params_reported", "stored_weight_elements", "cpu_latency_1t_ms", "cpu_latency_Nt_ms", "gpu_latency_bs1_ms"]
with pd.option_context("display.max_colwidth", 110, "display.width", 260):
    print(results_df[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(paired_df):
        print("\nPaired vs FP32 (pv_test):")
        print(paired_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nQuantization scope:")
    print(results_df[["experiment", "quantization_scope"]].to_string(index=False))
    print("\nShift results:")
    print(shift_df[["experiment", "test_set", "protocol", "n", "accuracy", "f1_macro"]].to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))
''')

md(r"""
### 11.1 Qualitative example images (figure assets for the paper)
The image files live on Kaggle and are not otherwise included in the zip, so this cell copies a small, deterministic selection
(resized to ≤ 320 px) into `figure_assets/`, with a `manifest.csv` holding each image's true class and every variant's prediction
and confidence. Categories:
* `gallery`: one PlantVillage test image per class (dataset figure);
* `flip_lost__<variant>` / `flip_gained__<variant>`: FP32 right and INT8 wrong / FP32 wrong and INT8 right, for each test set;
* `segpair_*`: the same leaf in colour and background-removed, where FP32's prediction changes, plus agreeing controls;
* `plantdoc_fp32_right` / `plantdoc_fp32_wrong`: field photos (strict protocol).

Check the PlantVillage and PlantDoc licences before printing their images.
""")

code(r'''
ASSET_DIR = OUT_DIR / "figure_assets"
ASSET_DIR.mkdir(exist_ok=True)
ASSET_ROWS = []
try:
    rng_a = np.random.RandomState(SPLIT_SEED)
    asset_variants = [v for v in ("FP32", DYN_NAME, "INT8 PTQ", "INT8 QAT", OA_PTQ) if v in ok_variants]
    int8_variants = [v for v in ("INT8 PTQ", "INT8 QAT", OA_PTQ) if v in ok_variants]

    def save_asset(ts, idx, category, **extra):
        src, label = EVAL_SETS[ts].dataset.samples[int(idx)]
        name = f"{category}__{ts}__{int(idx):05d}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((320, 320))
            im.save(ASSET_DIR / name, quality=90)
        row = {"asset": name, "category": category, "test_set": ts, "index": int(idx), "source_path": src,
               "label": CLASS_NAMES[label], **extra}
        for v in asset_variants:
            if (v, ts) in PRED_STORE:
                p = softmax_np(PRED_STORE[(v, ts)]["logits"][int(idx)][None].astype(np.float64))[0]
                row[f"pred__{slug(v)}"] = CLASS_NAMES[int(p.argmax())]
                row[f"conf__{slug(v)}"] = round(float(p.max()), 4)
        ASSET_ROWS.append(row)

    def pick(mask, n):
        idx = np.flatnonzero(mask)
        return rng_a.choice(idx, min(n, len(idx)), replace=False) if len(idx) else np.array([], dtype=int)

    y_pv = PRED_STORE[("FP32", "pv_test")]["y_true"]
    p_pv = PRED_STORE[("FP32", "pv_test")]["logits"].argmax(1)
    for c in range(NUM_CLASSES):
        sel = pick((y_pv == c) & (p_pv == c), 1)
        sel = sel if len(sel) else pick(y_pv == c, 1)
        for i in sel:
            save_asset("pv_test", i, "gallery")

    for ts in EVAL_SETS:
        yt = PRED_STORE[("FP32", ts)]["y_true"]
        pf = PRED_STORE[("FP32", ts)]["logits"].argmax(1)
        for v in int8_variants:
            pq = PRED_STORE[(v, ts)]["logits"].argmax(1)
            for i in pick((pf == yt) & (pq != yt), 6):
                save_asset(ts, i, f"flip_lost__{slug(v)}")
            for i in pick((pf != yt) & (pq == yt), 3):
                save_asset(ts, i, f"flip_gained__{slug(v)}")

    if SEG_POSITIONS is not None and ("FP32", "pv_segmented") in PRED_STORE:
        ys = PRED_STORE[("FP32", "pv_segmented")]["y_true"]
        ps = PRED_STORE[("FP32", "pv_segmented")]["logits"].argmax(1)
        pc = p_pv[SEG_POSITIONS]
        for tag, mask, n in (("colour_right_seg_wrong", (pc == ys) & (ps != ys), 6),
                             ("colour_wrong_seg_right", (pc != ys) & (ps == ys), 3),
                             ("both_right", (pc == ys) & (ps == ys), 3)):
            for k in pick(mask, n):
                save_asset("pv_test", SEG_POSITIONS[k], f"segpair_{tag}", pair_id=int(k), view="colour")
                save_asset("pv_segmented", k, f"segpair_{tag}", pair_id=int(k), view="segmented")

    pd_ts = "plantdoc_all" if "plantdoc_all" in EVAL_SETS else ("plantdoc" if "plantdoc" in EVAL_SETS else None)
    if pd_ts:
        yt = PRED_STORE[("FP32", pd_ts)]["y_true"]
        pf = PRED_STORE[("FP32", pd_ts)]["logits"].argmax(1)
        for i in pick(pf == yt, 6):
            save_asset(pd_ts, i, "plantdoc_fp32_right")
        for i in pick(pf != yt, 6):
            save_asset(pd_ts, i, "plantdoc_fp32_wrong")
except Exception as e:
    log_error("figure_assets", e)
pd.DataFrame(ASSET_ROWS).to_csv(ASSET_DIR / "manifest.csv", index=False)
print(f"Saved {len(ASSET_ROWS)} figure assets ->", ASSET_DIR)
if ASSET_ROWS:
    print(pd.DataFrame(ASSET_ROWS)["category"].value_counts().to_string())
''')

# =============================================================================
md(r"""
## 12. Figures, saved records, limitations
### 12.1 Figures
Each chart shows one measure. Accuracy and macro-F1 appear as dot plots with 95 % CIs on a zoomed axis. Size and latency are
zero-based bars. CPU 1-thread and N-thread latencies share one ms axis. Shift accuracy is grouped by test set.
""")

code(r'''
ok_df = results_df[results_df["status"] == "ok"].reset_index(drop=True)


def dot_ci_chart(data, col, lo_col, hi_col, title, xlabel, fname):
    d = data.dropna(subset=[col]).iloc[::-1]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(d) + 1.4))
    y = np.arange(len(d))
    v = d[col].to_numpy(float)
    lo, hi = d[lo_col].to_numpy(float), d[hi_col].to_numpy(float)
    ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", color=SERIES[0], ms=8, elinewidth=2, capsize=4)
    for yi, vi in zip(y, v):
        ax.annotate(f"{vi:.4f}", (vi, yi), xytext=(8, 8), textcoords="offset points", fontsize=9, color=INK)
    span = max(hi.max() - lo.min(), 0.004)
    ax.set_xlim(max(0, lo.min() - span * 0.4), min(1.0, hi.max() + span * 0.6))
    ax.set_yticks(y)
    ax.set_yticklabels(d["experiment"])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def bar_chart(data, cols, labels, title, xlabel, fname, fmt):
    d = data.dropna(subset=cols, how="all").iloc[::-1]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(d) + 1.5))
    y = np.arange(len(d))
    h = 0.8 / len(cols)
    vmax = np.nanmax(d[cols].to_numpy(float))
    for k, (c, lab) in enumerate(zip(cols, labels)):
        vals = d[c].to_numpy(float)
        pos = y + (k - (len(cols) - 1) / 2) * h
        ax.barh(pos, np.nan_to_num(vals), height=h * 0.9, color=SERIES[k], label=lab)
        for pi, vi in zip(pos, vals):
            if not np.isnan(vi):
                ax.annotate(fmt.format(vi), (vi, pi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xlim(0, vmax * 1.25)
    ax.set_yticks(y)
    ax.set_yticklabels(d["experiment"])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    if len(cols) > 1:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


tag = f"{ARCH}, seed {SEED}"
dot_ci_chart(ok_df, "accuracy", "accuracy_ci_low", "accuracy_ci_high", f"PlantVillage test accuracy, 95% CI ({tag})",
             "Accuracy", "chart_accuracy.png")
dot_ci_chart(ok_df, "f1_macro", "f1_ci_low", "f1_ci_high", f"PlantVillage test macro-F1, 95% CI ({tag})", "Macro-F1",
             "chart_macro_f1.png")
bar_chart(ok_df[ok_df["experiment"] != "FP16 autocast"], ["model_size_mb"], ["size"], f"Model file size ({tag})", "MB",
          "chart_model_size.png", "{:.2f} MB")
bar_chart(ok_df, ["cpu_latency_1t_ms", "cpu_latency_Nt_ms"], ["CPU, 1 thread", f"CPU, {CPU_THREADS} threads"],
          f"CPU latency, batch 1, median ({tag})", "ms / image", "chart_latency_cpu.png", "{:.1f}")
bar_chart(ok_df, ["gpu_latency_bs1_ms"], ["GPU"], f"GPU latency, batch 1, median ({tag})", "ms / image",
          "chart_latency_gpu.png", "{:.2f}")

if len(shift_df):
    piv = shift_df.assign(key=shift_df["test_set"] + " / " + shift_df["protocol"]).pivot_table(
        index="experiment", columns="key", values="accuracy")
    pd_key = "plantdoc_all" if "plantdoc_all / strict" in piv.columns else "plantdoc"   # chart the larger PlantDoc set
    order = [k for k in ("pv_test / all_classes", "pv_segmented / all_classes", f"{pd_key} / strict", f"{pd_key} / restricted")
             if k in piv.columns]
    piv = piv.reindex([n for n in ok_variants if n in piv.index])[order].reset_index()
    bar_chart(piv, order, order, f"Accuracy under shift ({tag})", "Accuracy", "chart_shift_accuracy.png", "{:.3f}")

rel = [n for n in ("FP32", DYN_NAME, "INT8 PTQ", "INT8 QAT") if n in CALIBRATION_BINS]
if rel:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1)
    for k, n in enumerate(rel):
        b = CALIBRATION_BINS[n]
        ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], marker="o", ms=5, lw=2, color=SERIES[k],
                label=f"{n} (ECE {RESULTS[n]['ece']:.4f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability on PlantVillage test ({tag})", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.show()

dcols = [c for c in per_class.columns if c.startswith("delta_f1__") and ("ptq" in c or "qat" in c)]
if dcols:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, c in enumerate(dcols):
        ax.scatter(per_class["support"], per_class[c], s=40, color=SERIES[k], label=c.replace("delta_f1__", ""),
                   edgecolors=SURFACE, linewidths=1.5)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Test support of the class (log scale)")
    ax.set_ylabel("ΔF1 vs FP32")
    ax.set_title(f"Per-class F1 change after INT8 vs class size ({tag})", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_delta_f1.png", dpi=150, bbox_inches="tight")
    plt.show()
''')

md(r"""### 12.2 Save results, configuration, environment""")

code(r'''
TIMINGS["total_s"] = round(time.time() - T_NOTEBOOK_START, 1)
results_df.to_csv(REPORT_DIR / "benchmark_results.csv", index=False)


def to_jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (Path, torch.device)):
        return str(o)
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    return o


with open(REPORT_DIR / "benchmark_results.json", "w") as f:
    json.dump(to_jsonable({"results": results_df.to_dict(orient="records"), "shift": shift_df.to_dict(orient="records"),
                           "ptq_attempts": PTQ_ATTEMPTS, "qat_attempts": QAT_ATTEMPTS,
                           "quant_inspections": QUANT_INSPECTIONS, "calibration_bins": CALIBRATION_BINS}), f, indent=2)

EXPERIMENT_CONFIG = {
    "notebook_version": NOTEBOOK_VERSION, "run_id": RUN_ID, "run_started_utc": RUN_STARTED_UTC,
    "config": CFG,
    "derived": {"arch": ARCH, "seed": SEED, "split_seed": SPLIT_SEED, "device": str(DEVICE), "amp": USE_AMP,
                "num_workers": NUM_WORKERS, "cpu_threads_N": CPU_THREADS, "pretrained_effective": PRETRAINED_EFFECTIVE,
                "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES, "parameters": N_PARAMS, "flops": FLOPS,
                "flops_tool": FLOPS_TOOL, "dataset_path": DATASET["path"], "dataset_roots": DATASET["roots"],
                "dataset_source": DATASET_SOURCE, "dataset_fingerprint": DATASET_FINGERPRINT,
                "split_fingerprint": SPLIT_FINGERPRINT, "images_before_dedup": len(df_all),
                "duplicates_removed": N_DUPLICATES_REMOVED, "label_conflicts_removed": N_CONFLICTS_REMOVED,
                "split_sizes": SPLIT_SIZES, "calibration_images": len(calib_ds), "best_checkpoint": BEST_CKPT_INFO,
                "shift_test_sets": SHIFT_INFO, "quantization": QUANT_INFO},
    "environment": ENV_INFO,
    "timings_s": TIMINGS,
}
with open(OUT_DIR / "experiment_config.json", "w") as f:
    json.dump(to_jsonable(EXPERIMENT_CONFIG), f, indent=2)
with open(REPORT_DIR / "errors.json", "w") as f:
    json.dump(ERRORS, f, indent=2)
print(f"Saved. {len(ERRORS)} logged error(s). Quantization warning stages: {list(QUANT_INFO['warnings'])}")
''')

md(r"""
### 12.3 Limitations, threats to validity, reproducibility

**PlantVillage is a controlled dataset.** It contains single detached leaves, centred, on uniform backgrounds under uniform
lighting. In-distribution accuracy does not show field performance. Capture conditions correlate with labels (Noyan, 2022, reports
49 % accuracy from 8 background pixels against 2.6 % chance). The `pv_segmented` and `plantdoc` test sets measure two parts of that
gap. Notes for reading them:
* *Segmented images* come from PlantVillage's own segmentation. Black backgrounds and mask artefacts are a different shift, not
  "the leaf alone". Noyan (2022) argues capture bias survives background removal.
* *PlantDoc* is small (a few hundred test images), its labels are noisy, and the class mapping is by keyword, which covers a subset
  of the 38 classes. Report the mapping table with the results, and use per-image paired tests rather than small accuracy
  differences.
* *Exact-duplicate removal only.* Near-duplicates (several photos of one leaf, re-encodings) are not detected, so in-distribution
  scores may be optimistic.

**Experimental limitations.** There are three training seeds on one fixed split, so split variance is not covered. Latency is
measured on a shared Kaggle vCPU (x86 kernels) and a T4 GPU, not on an edge device; the phone benchmark is a separate stage. The
FP32 CPU timing uses the unfused model while INT8 models are fused. MobileNetV2 uses the torchvision quantizable variant (ReLU
instead of ReLU6). `torch.ao.quantization` is in maintenance mode, and its warnings are recorded. Mixed precision and cuDNN
autotuning make GPU runs non-bit-exact.

**Reproducibility record per run:** `experiment_config.json` (full config, versions, dataset and split fingerprints, shift-set
status, quantization API, engine and warnings, timings), `reports/split_manifest.csv`, `calibration_manifest.csv`,
`removed_duplicates.csv`, `segmented_pairs.csv`, `plantdoc_class_map.csv`, training/QAT histories, every variant's raw logits on
every test set (`predictions/`), checkpoints and models, `errors.json`.
""")

# =============================================================================
md(r"""## 13. Export""")

code(r'''
zip_path = Path(CFG["OUTPUT_ROOT"]) / f"{RUN_ID}.zip"
if zip_path.exists():
    zip_path.unlink()
n_files = 0
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        if not CFG["ZIP_INCLUDE_CHECKPOINTS"] and CKPT_DIR in p.parents:
            continue
        zf.write(p, Path(RUN_ID) / p.relative_to(OUT_DIR))
        n_files += 1
with zipfile.ZipFile(zip_path) as zf:
    assert zf.testzip() is None, "corrupt zip entry"
print(f"Run folder: {OUT_DIR.resolve()}")
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.stat().st_size / 1e6:9.3f} MB  {p.relative_to(OUT_DIR)}")
print(f"\nZIP ({n_files} files, {zip_path.stat().st_size / 1e6:.2f} MB): {zip_path.resolve()}")
print(f"Total time: {(time.time() - T_NOTEBOOK_START) / 60:.1f} min")
summary_cols = ["experiment", "status", "accuracy", "f1_macro", "ece", "model_size_mb", "params_reported",
                "stored_weight_elements", "cpu_latency_1t_ms", "quantization_scope"]
with pd.option_context("display.max_colwidth", 90, "display.width", 260):
    print(results_df[summary_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
''')

# =============================================================================
nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
}


def validate(notebook):
    nbformat.validate(notebook)
    n_code = 0
    for i, c in enumerate(notebook.cells):
        if c.cell_type == "code":
            n_code += 1
            try:
                ast.parse(c.source)
            except SyntaxError as e:
                raise SystemExit(f"Syntax error in code cell #{i}: {e}")
    return n_code


if __name__ == "__main__":
    n_code = validate(nb)
    nbformat.write(nb, str(NB_PATH))
    validate(nbformat.read(str(NB_PATH), as_version=4))
    print(f"Wrote {NB_PATH} v{NOTEBOOK_VERSION} ({len(nb.cells)} cells: {n_code} code, {len(nb.cells) - n_code} markdown) - schema + syntax OK")
    sys.exit(0)
