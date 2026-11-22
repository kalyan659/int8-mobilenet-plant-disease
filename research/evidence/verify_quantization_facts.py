"""Evidence script for the research record (CPU only, random/ImageNet weights, no training).
F1: torch.ao.quantization.quantize_dynamic on MobileNetV3-small (38-class head), the protocol of
    Khan et al. 2023: which layers get quantized, and what model.parameters() reports afterwards.
F2: whether FX-graph-mode static INT8 PTQ can fully convert MobileNetV3-small (hardswish + SE)."""
import collections, io, sys, warnings
import torch, torch.nn as nn, torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
warnings.filterwarnings("ignore")
torch.manual_seed(0)
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | engines {torch.backends.quantized.supported_engines}")
eng = next(e for e in ("x86", "fbgemm", "qnnpack", "onednn") if e in torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = eng

def n_params(m): return sum(p.numel() for p in m.parameters())
def sd_mb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1e6
def census(m):
    c = collections.Counter()
    for x in m.modules():
        n, mod = type(x).__name__, type(x).__module__
        if "Conv" in n or "Linear" in n:
            c[("int8 " if ".quantized" in mod else "fp32 ") + n] += 1
    return dict(c)

m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
m.classifier[3] = nn.Linear(1024, 38)
m.eval()
lin = sum(p.numel() for x in m.modules() if isinstance(x, nn.Linear) for p in x.parameters())
print("\n[F1] FP32 MobileNetV3-small, 38 classes")
print(f"  parameters() = {n_params(m):,} | Linear-layer params = {lin:,} | state_dict = {sd_mb(m):.3f} MB")
q = torch.ao.quantization.quantize_dynamic(m, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
print("  after quantize_dynamic({nn.Linear, nn.Conv2d}, qint8):")
print(f"  parameters() = {n_params(q):,} (= FP32 total - Linear params: {n_params(m) - lin:,}) | state_dict = {sd_mb(q):.3f} MB")
print(f"  layer census: {census(q)}")

print("\n[F2] FX static PTQ of MobileNetV3-small")
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
x = torch.randn(16, 3, 224, 224)
p = prepare_fx(m, get_default_qconfig_mapping(eng), (x[:1],))
with torch.no_grad():
    p(x)
qs = convert_fx(p)
fn = collections.Counter(str(n.target) for n in qs.graph.nodes if n.op == "call_function")
print(f"  engine {eng} | module census: {census(qs)}")
print(f"  float ops left in graph: { {k: v for k, v in fn.items() if 'quantized' not in k and 'quantize' not in k and 'dequantize' not in k} }")
with torch.no_grad():
    a, b = m(x).argmax(1), qs(x).argmax(1)
print(f"  state_dict = {sd_mb(qs):.3f} MB | top-1 agreement FP32 vs INT8 on random inputs (ImageNet weights, not a PlantVillage result): {(a == b).float().mean():.3f}")

print("\n[F3] Control: same FX static PTQ on MobileNetV2 vs MobileNetV3-small, ImageNet weights, 1000-class heads,")
print("     smooth synthetic images (low-frequency noise, ImageNet-normalised); calibrate on 64, evaluate on 64 others")
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn.functional as F
def smooth_images(n, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(n, 3, 14, 14, generator=g)
    img = F.interpolate(z, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (img - mean) / std
cal, ev = smooth_images(64, 1), smooth_images(64, 2)
for name, ctor, w in (("MobileNetV2", mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1),
                      ("MobileNetV3-small", mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1)):
    fm = ctor(weights=w).eval()
    pm = prepare_fx(fm, get_default_qconfig_mapping(eng), (cal[:1],))
    with torch.no_grad():
        for i in range(0, 64, 16):
            pm(cal[i:i + 16])
    qm = convert_fx(pm)
    with torch.no_grad():
        lf, lq = fm(ev), qm(ev)
    agree = (lf.argmax(1) == lq.argmax(1)).float().mean().item()
    cos = F.cosine_similarity(lf, lq, dim=1).mean().item()
    print(f"  {name:18s} top-1 agreement {agree:.3f} | mean logit cosine {cos:.3f} | engine {eng}")
