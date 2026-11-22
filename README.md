# INT8 MobileNets for plant disease recognition: code and results

Code, run records and results for the manuscript

> Kalyanbrat Medhi and Sanjib Kr. Kalita. *Auditing INT8 quantization of MobileNets for plant disease recognition: compression
> accounting, an outlier-aware fix and behavior under field shift.* Submitted to *Computers and Electronics in Agriculture*.

The study trains MobileNetV2 and MobileNetV3-Small on PlantVillage (38 classes, three seeds, one fixed de-duplicated split) and
compares FP32, FP16, dynamic INT8, static INT8 post-training quantization (PTQ), quantization-aware training (QAT) and the proposed
outlier-aware selective INT8 on the PlantVillage test set, on the same test leaves with their backgrounds removed, and on PlantDoc
field images. Every number in the paper is computed by the scripts in this repository from the saved run outputs.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24443000.svg)](https://doi.org/10.5281/zenodo.24443000)

This repository is archived on Zenodo: https://doi.org/10.5281/zenodo.24443000 (all versions; v1.0.0 is https://doi.org/10.5281/zenodo.24443001). Large files (saved model outputs, model weights, checkpoints) are archived in a separate Zenodo dataset record (CC BY-NC-SA 4.0): https://doi.org/10.5281/zenodo.24443981.

## Contents

| Path | What it is |
|---|---|
| `build_notebook.py` | Generates the experiment notebook (edit this, not the notebook). Notebook version 2.3.0 was used for all runs. |
| `plant_disease_mobilenetv2_tinyml.ipynb` | The generated notebook: data audit and split, training, all precision variants, evaluation, export. |
| `kaggle/` | Scripts that build, push and fetch the Kaggle runs; `kernels/` holds the exact notebooks and metadata that ran. |
| `research/aggregate_runs.py` | Aggregates the six runs into `research/aggregated/` (refuses runs with different data or split fingerprints). |
| `research/make_paper_figures.py` | Produces every figure and table of the paper into `research/paper_figures/` (with `MANIFEST.md` provenance). |
| `research/evidence/` | Smaller analyses cited in the paper (quantization facts, collapse diagnosis, outlier rule, flip-confidence, fusion latency), each with its saved output (`.out.txt`). |
| `research/runs/<run>/` | Per-run configuration (library versions, fingerprints, quantization warnings) and reports: metrics, shift results, training and QAT histories, per-class F1, calibration subset, duplicate list, PlantDoc class map. |
| `research/split_manifest.csv` | The train/validation/test assignment of every PlantVillage image (identical in all runs). |
| `research/pilot_plan.md` | Go/no-go criteria written **before** any real result (unchanged since). |
| `research/experiment_log.md`, `research/claims_ledger.md` | Research log: what was run, what it showed, and the evidence behind each statement in the paper. |

## Data

The datasets are not redistributed here. Both are public on Kaggle:

* PlantVillage (color and segmented): https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset (Hughes and Salathé, 2015)
* PlantDoc: https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset (Singh et al., 2020)

## Reproducing the results

1. **Notebook.** `python build_notebook.py` regenerates `plant_disease_mobilenetv2_tinyml.ipynb`.
2. **Training runs (Kaggle, GPU).** Set `USER` in `kaggle/make_kernel.py` to your Kaggle username and
   `export KAGGLE_USER=<your username>` for the fetch scripts, then for each run
   (`p1` = MobileNetV2 seed 0, `mnv2_s1`, `mnv2_s2`, `mnv3_s0`, `mnv3_s1`, `mnv3_s2`; `p2` is the diagnostic pilot):
   ```
   python kaggle/make_kernel.py p1
   kaggle kernels push -p kaggle/kernels/plantvillage-int8-mobilenetv2-seed0 --accelerator NvidiaTeslaT4
   bash kaggle/poll_and_fetch.sh plantvillage-int8-mobilenetv2-seed0
   ```
   Each run takes about 1.4–2.3 h on a Tesla T4 and writes a `<run>.zip` into `research/runs/`. The runs used Python 3.13,
   PyTorch 2.11.0 (CUDA 13.0), torchvision 0.26.0 and PyTorch's x86 quantized engine. GPU training is not bit-exact.
3. **Tables and figures (CPU).** With the six run folders in `research/runs/` (from step 2, or by unzipping the Zenodo dataset
   archive https://doi.org/10.5281/zenodo.24443981 at the repository root):
   ```
   python research/aggregate_runs.py
   python research/make_paper_figures.py
   ```
4. **Evidence scripts (CPU).** Run each `research/evidence/*.py` and compare with its `.out.txt`. Several need the model files,
   checkpoints or saved outputs from the Zenodo archive; the latency check gives machine-specific timings (only ratios are used).

Without retraining, steps 3 and 4 reproduce every table, figure and statistic from the archived outputs.

## Use of AI assistance

As stated in the paper, the code in this repository was written with an AI assistant (Claude, Anthropic), which also ran the
Kaggle experiments and the analyses; the research question, choice of experiments, analysis and conclusions are the authors'.

## License

Code: MIT (see `LICENSE`). Results, tables, figures and run records in this repository: CC BY 4.0. Model weights, checkpoints and
saved model outputs on Zenodo: CC BY-NC-SA 4.0, because the models were trained on PlantVillage images (CC BY-NC-SA 4.0). See
`LICENSE-DATA.md`. The datasets keep their own licenses.
