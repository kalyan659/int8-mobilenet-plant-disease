"""Diagnose the MobileNetV3-Small static-INT8 collapse seen in Kaggle pilot P2 (CPU only, no training).

Uses P2's trained FP32 checkpoint and the real PlantVillage *test* images exported in figure_assets/ of P1 and P2
(same fixed split). Half the images calibrate, all are evaluated. Each ablation keeps one suspect part in float and
reports top-1 agreement with FP32 and mean logit cosine. Small sample => diagnostic only, never a paper number.
"""
import copy
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torchvision.ops.misc import SqueezeExcitation
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qconfig_mapping
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
RUNS = Path(__file__).resolve().parents[1] / "runs" / "_kaggle"
P2 = RUNS / "plantvillage-int8-mobilenetv3-small-pilot-diag" / "plant_disease_results" / "mobilenet_v3_small_seed0_pilotdiag"
P1 = RUNS.parent / "_pilot_v2.2.0" / "kaggle_download_p1" / "plant_disease_results" / "mobilenet_v2_seed0"
ENGINE = next(e for e in ("x86", "fbgemm", "onednn", "qnnpack") if e in torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = ENGINE
print(f"torch {torch.__version__} torchvision {torchvision.__version__} engine {ENGINE} (Kaggle pilot used x86)")

ck = torch.load(P2 / "checkpoints" / "best_fp32.pt", map_location="cpu", weights_only=True)
classes = ck["class_names"]
fp32 = mobilenet_v3_small(weights=None)
fp32.classifier[-1] = nn.Linear(fp32.classifier[-1].in_features, len(classes))
fp32.load_state_dict(ck["model_state"])
fp32.eval()

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
rows = []
for run in (P1, P2):
    m = pd.read_csv(run / "figure_assets" / "manifest.csv")
    m = m[m["test_set"] == "pv_test"].drop_duplicates("source_path")
    for _, r in m.iterrows():
        rows.append((run / "figure_assets" / r["asset"], classes.index(r["label"]), r["source_path"]))
seen, imgs, ys = set(), [], []
for p, y, src in rows:
    if src in seen:
        continue
    seen.add(src)
    imgs.append(tf(Image.open(p).convert("RGB")))
    ys.append(y)
X, Y = torch.stack(imgs), torch.tensor(ys)
perm = torch.randperm(len(X))
X, Y = X[perm], Y[perm]
cal = X[: len(X) // 2]
print(f"{len(X)} unique PlantVillage test images ({len(cal)} for calibration); assets are ≤320 px re-encodes (slight shift)")

with torch.no_grad():
    ref = fp32(X)
print(f"FP32 accuracy on these images: {(ref.argmax(1) == Y).float().mean():.3f}\n")

se_names = [n for n, mod in fp32.named_modules() if isinstance(mod, SqueezeExcitation)]
hs_names = [n for n, mod in fp32.named_modules() if isinstance(mod, (nn.Hardswish,))]
first_last = ["features.0", "classifier"]


def run(tag, qmap):
    m = copy.deepcopy(fp32)
    p = prepare_fx(m, qmap, (X[:1],))
    with torch.no_grad():
        for i in range(0, len(cal), 16):
            p(cal[i:i + 16])
    q = convert_fx(p)
    with torch.no_grad():
        out = q(X)
    agree = (out.argmax(1) == ref.argmax(1)).float().mean().item()
    acc = (out.argmax(1) == Y).float().mean().item()
    cos = F.cosine_similarity(out, ref, dim=1).mean().item()
    print(f"{tag:58s} agree {agree:.3f} | acc {acc:.3f} | logit cos {cos:.3f}")


base = lambda: get_default_qconfig_mapping(ENGINE)
run("A  default qconfig mapping (as in the pilot)", base())
run("B  squeeze-excite blocks kept in float", (lambda q: [q.set_module_name(n, None) for n in se_names] and q)(base()))
run("C  hard-swish kept in float", base().set_object_type(nn.Hardswish, None))
run("D  hard-swish + squeeze-excite in float", (lambda q: [q.set_module_name(n, None) for n in se_names] and q)(
    base().set_object_type(nn.Hardswish, None)))
run("E  first conv + classifier in float", (lambda q: [q.set_module_name(n, None) for n in first_last] and q)(base()))
pt = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
             weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
run("F  per-tensor weights, full-range activations", QConfigMapping().set_global(pt))
pc = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
             weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
run("G  per-channel weights, full-range activations", QConfigMapping().set_global(pc))
print(f"\nSE modules: {len(se_names)} | Hardswish modules: {len(hs_names)}")

print("\n[Localisation] quantize only feature blocks [0, k); everything after stays float")
n_blocks = len(fp32.features)
for k in range(1, n_blocks + 1):
    q = base()
    for i in range(k, n_blocks):
        q.set_module_name(f"features.{i}", None)
    q.set_module_name("classifier", None)
    run(f"   blocks 0..{k - 1} quantized", q)
print("\n[Single block] quantize ONE feature block, everything else float")
for i in range(n_blocks):
    q = QConfigMapping().set_module_name(f"features.{i}", get_default_qconfig_mapping(ENGINE).global_qconfig)
    run(f"   only features.{i} quantized ({type(fp32.features[i]).__name__})", q)

print("\n[Mitigation] selective INT8: stem + first block kept in float, everything else INT8")
q = base()
for n in ("features.0", "features.1"):
    q.set_module_name(n, None)
run("   features.0 + features.1 float, rest INT8", q)

print("\n[Range evidence] per-channel activation ranges on the calibration images (FP32 model)")
acts = {}
hooks = [fp32.features[i].register_forward_hook(lambda m, i_, o, k=i: acts.__setitem__(k, o.detach())) for i in range(4)]
with torch.no_grad():
    fp32(cal)
for h in hooks:
    h.remove()
for k, a in acts.items():
    ch = a.transpose(0, 1).reshape(a.shape[1], -1)
    rng = ch.max(1).values - ch.min(1).values
    print(f"   features.{k} output: {a.shape[1]} ch | per-channel range median {rng.median():.3f}, max {rng.max():.3f}, "
          f"max/median {rng.max() / rng.median():.1f}x | tensor range [{a.min():.2f}, {a.max():.2f}]")
for k in (0, 1):
    for name, mod in fp32.features[k].named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            g = (mod.weight / torch.sqrt(mod.running_var + mod.eps)).abs()
            print(f"   features.{k}.{name}: |gamma/sqrt(var+eps)| median {g.median():.2f}, max {g.max():.2f} "
                  f"(BN-fold weight scale outliers {g.max() / g.median():.1f}x)")

print("\n[Origin] stem output ranges: torchvision ImageNet weights (before any fine-tuning) vs our fine-tuned checkpoint")
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v2, MobileNet_V2_Weights
for tag, model in (("MobileNetV3-S ImageNet", mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)),
                   ("MobileNetV3-S fine-tuned (P2)", fp32), ("MobileNetV2 ImageNet", mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1))):
    model.eval()
    out = {}
    h = model.features[0].register_forward_hook(lambda m, i_, o: out.__setitem__("a", o.detach()))
    with torch.no_grad():
        model(cal)
    h.remove()
    a = out["a"]
    ch = a.transpose(0, 1).reshape(a.shape[1], -1)
    rng = ch.max(1).values - ch.min(1).values
    print(f"   {tag:32s} stem out {a.shape[1]} ch | range median {rng.median():.3f}, max {rng.max():.3f}, max/median {rng.max() / rng.median():.1f}x")
