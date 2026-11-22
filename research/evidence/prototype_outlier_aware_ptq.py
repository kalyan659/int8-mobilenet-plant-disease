"""Prototype of the outlier-aware selective INT8 rule, applied to the REAL pilot checkpoints (diagnostic only).

Rule (threshold fixed a priori at 16x): prepare the model with FX, run calibration images, and for every activation
observer (= every tensor that will be quantized) measure the per-channel range of its input. If max/median > 16, keep
that tensor in float by setting qconfig None on its producer and consumer modules. Repeat once to catch new boundaries.
"""
import copy
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torchvision.models.quantization import mobilenet_v2 as qmnv2
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
ENGINE = next(e for e in ("x86", "fbgemm", "onednn", "qnnpack") if e in torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = ENGINE
RUNS = Path(__file__).resolve().parents[1] / "runs" / "_kaggle"
CK = {"mobilenet_v2": RUNS.parent / "_pilot_v2.2.0/kaggle_download_p1/plant_disease_results/mobilenet_v2_seed0",
      "mobilenet_v3_small": RUNS / "plantvillage-int8-mobilenetv3-small-pilot-diag/plant_disease_results/mobilenet_v3_small_seed0_pilotdiag"}
T = 16.0
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
paths = []
for d in CK.values():
    m = pd.read_csv(d / "figure_assets" / "manifest.csv")
    m = m[m["test_set"] == "pv_test"].drop_duplicates("source_path")
    paths += list(zip(m["asset"].map(lambda a, d=d: d / "figure_assets" / a), m["source_path"]))
seen, imgs = set(), []
for p, s in paths:
    if s not in seen:
        seen.add(s)
        imgs.append(tf(Image.open(p).convert("RGB")))
X = torch.stack(imgs)[torch.randperm(len(seen))]
cal = X[: len(X) // 2]
print(f"engine {ENGINE} | {len(X)} images ({len(cal)} calibration) | threshold {T}x\n")


def block_of(name):
    parts = str(name).split(".")
    return ".".join(parts[:2]) if parts[0] == "features" and len(parts) > 1 else parts[0]


def outlier_float_modules(model, qmap_fn, cal, threshold, rounds=2):
    float_mods = set()
    for _ in range(rounds):
        qmap = qmap_fn()
        for n in float_mods:
            qmap.set_module_name(n, None)
        prepared = prepare_fx(copy.deepcopy(model), qmap, (cal[:1],))
        stats, hooks = {}, []
        for name, mod in prepared.named_modules():
            if "activation_post_process" in name and hasattr(mod, "calculate_qparams"):
                def f(m_, inp, out, k=name):
                    x = inp[0].detach()
                    if x.dim() == 4 and x.shape[2] * x.shape[3] > 1:   # skip pooled 1x1 tensors (SE)
                        ch = x.transpose(0, 1).reshape(x.shape[1], -1)
                        r = ch.max(1).values - ch.min(1).values
                        stats[k] = r if k not in stats else torch.maximum(stats[k], r)
                hooks.append(mod.register_forward_hook(f))
        with torch.no_grad():
            prepared(cal)
        for h in hooks:
            h.remove()
        flagged = {k: float(r.max() / r.median().clamp_min(1e-8)) for k, r in stats.items()
                   if float(r.max() / r.median().clamp_min(1e-8)) > threshold}
        new = set()
        for node in prepared.graph.nodes:
            if node.op == "call_module" and node.target in flagged:
                prod = node.args[0]
                if getattr(prod, "op", None) == "call_module":
                    new.add(block_of(prod.target))
                for user in node.users:
                    if user.op == "call_module" and "activation_post_process" not in str(user.target):
                        new.add(block_of(user.target))
        print(f"   round: flagged tensors {[(k, round(v, 1)) for k, v in flagged.items()]} -> float modules {sorted(new)}")
        if not (new - float_mods):
            break
        float_mods |= new
    return sorted(float_mods)


for arch, d in CK.items():
    ck = torch.load(d / "checkpoints" / "best_fp32.pt", map_location="cpu", weights_only=True)
    m = qmnv2(weights=None, quantize=False) if arch == "mobilenet_v2" else mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(ck["class_names"]))
    m.load_state_dict(ck["model_state"])
    if hasattr(m, "quant"):
        m.quant, m.dequant = nn.Identity(), nn.Identity()
    m.eval()
    with torch.no_grad():
        ref = m(X)
    print(f"{arch}:")
    fm = outlier_float_modules(m, lambda: get_default_qconfig_mapping(ENGINE), cal, T)
    for tag, mods in (("plain FX PTQ", []), ("outlier-aware PTQ (block-level)", fm)):
        qmap = get_default_qconfig_mapping(ENGINE)
        for n in mods:
            qmap.set_module_name(n, None)
        p = prepare_fx(copy.deepcopy(m), qmap, (cal[:1],))
        with torch.no_grad():
            p(cal)
        q = convert_fx(p)
        with torch.no_grad():
            out = q(X)
        print(f"   {tag:34s} float modules {len(mods):2d} | agreement with FP32 {(out.argmax(1) == ref.argmax(1)).float().mean():.3f}"
              f" | logit cos {F.cosine_similarity(out, ref, dim=1).mean():.3f}")
    print()
