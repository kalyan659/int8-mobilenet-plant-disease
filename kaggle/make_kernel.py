"""Build a Kaggle kernel folder from the experiment notebook with explicit run overrides.

    python make_kernel.py <run_name>        # run_name is a key of RUNS below

Writes kaggle/kernels/<slug>/ with the notebook (config cell + a visible '# --- KAGGLE RUN OVERRIDES ---' block) and
kernel-metadata.json (private, GPU T4 x2, Internet on, datasets attached). Push with:
    kaggle kernels push -p kaggle/kernels/<slug> --accelerator NvidiaTeslaT4
"""
import json
import re
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "plant_disease_mobilenetv2_tinyml.ipynb"
USER = "your-kaggle-username"   # set to your Kaggle username
DATASETS = ["abdallahalidev/plantvillage-dataset", "nirmalsankalana/plantdoc-dataset"]
COMMON = {"SHOW_PROGRESS": False}           # keep batch logs readable

RUNS = {   # title must slugify to the kernel id
    "p1": ("PlantVillage INT8 MobileNetV2 Seed0", {"ARCH": "mobilenet_v2", "SEED": 0}),
    "p2": ("PlantVillage INT8 MobileNetV3 Small Pilot Diag",
           {"ARCH": "mobilenet_v3_small", "SEED": 0, "RUN_TAG": "pilotdiag", "MAX_IMAGES_PER_CLASS": 300,
            "FINETUNE_EPOCHS": 6}),
    "mnv2_s1": ("PlantVillage INT8 MobileNetV2 Seed1", {"ARCH": "mobilenet_v2", "SEED": 1}),
    "mnv2_s2": ("PlantVillage INT8 MobileNetV2 Seed2", {"ARCH": "mobilenet_v2", "SEED": 2}),
    "mnv3_s0": ("PlantVillage INT8 MobileNetV3 Small Seed0", {"ARCH": "mobilenet_v3_small", "SEED": 0}),
    "mnv3_s1": ("PlantVillage INT8 MobileNetV3 Small Seed1", {"ARCH": "mobilenet_v3_small", "SEED": 1}),
    "mnv3_s2": ("PlantVillage INT8 MobileNetV3 Small Seed2", {"ARCH": "mobilenet_v3_small", "SEED": 2}),
}


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build(run):
    title, overrides = RUNS[run]
    slug = slugify(title)
    out = HERE / "kernels" / slug
    out.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(str(NB), as_version=4)
    idx = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.startswith("CFG = {"))
    ov = {**COMMON, **overrides}
    nb.cells[idx].source += ("\n\n# --- KAGGLE RUN OVERRIDES (set by kaggle/make_kernel.py, run '" + run + "') ---\n"
                             f"CFG.update({ov!r})\nprint('Run overrides:', {ov!r})\n")
    code_file = f"{slug}.ipynb"
    nbformat.write(nb, str(out / code_file))
    meta = {"id": f"{USER}/{slug}", "title": title, "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": DATASETS,
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{run}: {out}  id={meta['id']}  overrides={ov}")
    return out


if __name__ == "__main__":
    for r in sys.argv[1:] or ["p1"]:
        build(r)
