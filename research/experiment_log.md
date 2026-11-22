# Experiment log

*Public copy (lightly edited for release): session housekeeping such as download commands and queue restarts was removed; results and dates are unchanged. Pilot (v2.2.0) numbers are pilot evidence and are not part of the paper's 3-seed results. Files under `literature/` that some entries mention are not shared (copyright).*

One entry per run, newest last. Real-data results go here only after the run's outputs are saved under `runs/<run_id>/`.

---

### 2026-11-15 · notebook smoke tests (no real data, not results)
- Purpose: check that every cell runs in order. Synthetic 4-class, 64-px images, CPU, torch 2.15.0 / torchvision 0.29.2.
- Outcome: all cells ran; eager and FX quantization paths both gave full INT8 (52 conv + 1 linear + 10 adds); errors.json empty.
- Learned: `torch.ao.quantization` still works on 2.15 but warns "will be removed in 2.10", and the notebook records the warning. torchvision's MobileNetV2 has 7 unused `skip_add` modules, now excluded from observers and counts.
- **Accuracy numbers from these runs mean nothing and must never be cited.**

### 2026-11-15 · evidence/verify_quantization_facts.py (CPU, no training)
- Result: supports C1, C2 and C4 in the claims ledger; C5 is inconclusive.

### 2026-11-15 · notebook v2.0.0 smoke tests (no real data, not results)
- Runs: mobilenet_v2 seed 0 and mobilenet_v3_small seed 1, with synthetic 4-class data, a segmented copy and a PlantDoc-style folder
  (including one unmappable folder and one corrupt file). CPU, torch 2.15.0, oneDNN engine.
- Outcome: all cells ran for both architectures; errors.json empty; dataset and split fingerprints identical across the two runs;
  aggregate_runs.py combined both zips.
- Structural observations (valid beyond toy data): MobileNetV2 eager PTQ/QAT = FULL INT8 (52 conv + 1 linear + 10 add);
  MobileNetV3-small FX PTQ/QAT = FULL INT8 (52 conv + 2 linear + 6 add + 9 mul); Khan protocol = PARTIAL (0/52 conv).
  `parameters()` = 0 after full conversion (C6b).
- Watch item, not a result: on this PC's oneDNN backend, INT8 MobileNetV3-small CPU latency was ~6× slower than FP32
  (about 40 ms vs 6.7 ms at 64 px). Check on Kaggle's x86 engine; it may become an RQ4 finding about kernel support for
  hard-swish and SE.
- **Accuracy numbers from these runs mean nothing and must never be cited.**

### 2026-11-16 · notebook v2.1.0: PlantDoc power upgrade + cross-dataset leakage guard (smoke test, not results)
- Change: new test set `plantdoc_all` = all PlantDoc images (train+test; PlantDoc is never used for training or selection),
  with exact duplicates removed by MD5 and conflicting-label files dropped. `plantdoc` stays as the official test split so
  results remain comparable with Ho-Lam 2026. New guard: PlantDoc files byte-identical to any PlantVillage file are removed
  from both sets, and the count is recorded as `identical_to_plantvillage_removed`.
- Smoke check (mobilenet_v2, synthetic data with planted cases): 1 PV-identical file removed, 2 duplicates removed,
  1 conflicting-label file dropped, 1 corrupt file skipped. All counts match the planted cases. All cells ran; errors.json empty.
- Runs made with v2.0.0 must not be aggregated with v2.1.0 runs (aggregate_runs.py lists notebook versions per run).

### 2026-11-16 · notebook v2.2.0 (figure assets) + make_paper_figures.py (smoke test, not results)
- Notebook: new §11.1 saves `figure_assets/` (≤ 320 px JPEGs + manifest.csv with the true class and every variant's prediction
  and confidence): one gallery image per class, FP32-right/INT8-wrong and FP32-wrong/INT8-right flips per test set, paired
  colour vs background-removed leaves, and PlantDoc right/wrong examples. Selection is deterministic (SPLIT_SEED).
- `research/make_paper_figures.py`: from the run folders, writes 14 figures (300-dpi PNG + vector PDF) and 5 tables
  (CSV / Markdown / LaTeX), plus MANIFEST.md tracing each output to its source files. Refuses mixed fingerprints.
- Smoke check: 4 synthetic runs (mobilenet_v2 and mobilenet_v3_small × seeds 0, 1), all cells executed, errors.json empty.
  Figures were inspected by eye; fixed text overflow in Fig 1, the legend overlap in Fig 5, error bars below 0 in Fig 6, and
  sparse-bin noise in Fig 7 (bins with < 20 images are no longer plotted; ECE still uses all bins).
- Table 3 (quantization accounting) reproduces the C2 artefact inside the pipeline: Khan protocol `parameters()` = 927,008 vs
  1,521,956 stored (MobileNetV3-S, 4-class smoke head); PTQ/QAT = FULL INT8 with `parameters()` = 0.

### 2026-11-16 · Kaggle pilots pushed (notebook v2.2.0)
- Built with `kaggle/make_kernel.py` (adds a visible run-overrides block to the config cell; metadata private, GPU T4 ×2,
  Internet on). Datasets: `abdallahalidev/plantvillage-dataset` (color/ + segmented/), `nirmalsankalana/plantdoc-dataset`
  (original PlantDoc train/ + test/ class folders, names with underscores, which the notebook's name normaliser handles).
- **P1** `plantvillage-int8-mobilenetv2-seed0`: ARCH=mobilenet_v2, SEED=0, full data. Counts as run 1 of 6.
  Status at push: RUNNING.
- **P2** `plantvillage-int8-mobilenetv3-small-pilot-diag`: ARCH=mobilenet_v3_small, SEED=0,
  MAX_IMAGES_PER_CLASS=300, FINETUNE_EPOCHS=6. Diagnostic only, not a paper run. Status at push: QUEUED (it appears only
  one GPU session runs at a time on this account).
- Results: pending. Evaluate against the pre-registered checks K0–K6 in pilot_plan.md.

### 2026-11-16 · RESULTS: Kaggle pilots P1 + P2 (notebook v2.2.0, torch 2.11.0+cu130, 2× Tesla T4, quant engine x86)
**P1, mobilenet_v2 seed 0, full data** (run 1 of 6; `runs/mobilenet_v2_seed0.zip`; 1.84 h). errors.json empty.
- Data: 54,305 PlantVillage images; **21 exact duplicates removed**, 0 label conflicts; split 37,998 / 8,143 / 8,143;
  segmented twins matched 7,964 / 8,143 (97.8%); PlantDoc test 252 images (27/27 folders mapped, 0 unmapped);
  PlantDoc-all 2,911 (6 duplicates and 5 label-conflict hashes dropped); **0 PlantDoc files identical to PlantVillage**.
- Best checkpoint: fine-tune epoch 11, val macro-F1 0.9952.
- pv_test (n = 8,143): FP32 acc 0.9969 (95% CI 0.9958–0.9980), macro-F1 0.9958. FP16 autocast / converted are identical
  (flip ≤ 0.02%). **Khan-protocol dynamic** acc 0.9968, `parameters()` 2,223,872 vs 2,272,550 stored, size 9.20 vs 9.34 MB,
  CPU latency unchanged (24.0 vs 23.9 ms): PARTIAL INT8 (0/52 conv). **PTQ** acc 0.9957 (Δ −0.12 pp, McNemar 17 vs 7,
  p = 0.064), 2.70 MB (3.46× smaller), CPU 1-thread 15.1 ms (1.58× faster), FULL INT8. **QAT** acc 0.9942 (Δ −0.27 pp,
  McNemar 24 vs 2, p = 1e-5).
- Shift, PTQ flip rate vs FP32: pv_test **0.29%** → pv_segmented **7.81%** (27×) → PlantDoc-all **22.8%** (77×).
  pv_segmented accuracy: FP32 0.8675, **PTQ 0.8356 (McNemar 361 vs 107, p ≈ 0)**, **QAT 0.9008 (93 vs 358, p ≈ 0)**.
  PlantDoc-all strict: FP32 0.2800, PTQ 0.2783, QAT 0.2704 (no significant accuracy change despite 23–25% flips).
  FP32 PlantDoc test strict 0.3373 (Ho-Lam 2026 MobileNetV2: 0.2857).
- ECE: FP32 0.095, PTQ 0.173, QAT 0.104 (caution: label smoothing 0.1 makes FP32 under-confident; ECE comparisons
  carry this confound).

**P2, mobilenet_v3_small seed 0, 300 img/class, 6 FT epochs** (diagnostic only; `runs/_diagnostic/`; 0.41 h). errors.json empty.
- FP32 acc 0.9733. Khan dynamic 0.9715, `parameters()` **927,008** vs 1,556,806 stored (= Khan 2023's "0.93 M", now
  reproduced for the 38-class head), 4.49 MB. **PTQ collapses to 0.0279 (≈ chance 1/38)** although FULL INT8 by census.
  **QAT 0.4433**; QAT fake-quant validation was also low (0.12 → 0.43), so the collapse is already present in quantization
  simulation, not introduced by conversion. INT8 CPU latency 9.70 ms vs FP32 9.60 ms (**no speed-up on x86**).

**Collapse diagnosis** (`evidence/diagnose_mnv3_int8.py` + `.out.txt`; CPU, oneDNN, 127 real test images from figure_assets;
diagnostic only): the collapse reproduces on oneDNN (agreement with FP32 0.197). Keeping squeeze-excite, hard-swish or the
first conv + classifier in float does not help. Single-block quantization pins it to **features.0 (stem) and features.1**
(agreement 0.157 / 0.173 when quantized alone; every other block ≥ 0.969). The stem output has one channel spanning 145 vs a
3.6 median (**41× outlier**), and the outlier is **already present in torchvision's ImageNet weights (45×)**. MobileNetV2's
stem spread is 6×. Block-1 BN-fold scale outliers 14×. **Keeping those two blocks in float restores agreement to 0.961.**

**Pre-registered checks (pilot_plan.md):** K0 PASS · K1 PASS (927,008) · K2 PASS (33.7% vs 28.6%, within ±15 pp) ·
K3 PASS (PTQ −0.12 pp; QAT −0.27 pp, but QAT's drop is significant) · K4 **collapse > 5 pp → RQ4 finding**, mechanism
identified · K5 flip rate under shift ≥ 3× clean, and McNemar on pv_segmented significant → headline *"INT8 is free
in-distribution but not under shift"*, with the twist that QAT *improves* background robustness while PTQ hurts it
(one seed) · K6 MobileNetV2 INT8 1.58× faster; MobileNetV3-S INT8 no faster on x86. **Decision rule: GO.**

### 2026-11-16 · notebook v2.3.0: outlier-aware INT8 (our method); low-bit dropped
- The first author approved the outlier-aware method and all 6 runs. INT4/INT2 was considered and **dropped at the first author's request**: PyTorch has no
  real INT4/INT2 convolution kernels, and only really runnable models were wanted (no simulation).
- Rule (§10.2): FX-prepare the model, run the calibration (train) images, measure the per-channel range at every activation observer
  input (pooled 1×1 tensors skipped); if max/median > 16 (fixed a priori: 7-bit x86 activations / 16 = 8 levels), keep the producer
  and consumer top-level blocks in float; repeat once. New variants: `INT8 PTQ outlier-aware`, `INT8 QAT outlier-aware` (real INT8
  kernels with float islands, reported as PARTIAL INT8 with the block list).
- Prototype on the real pilot checkpoints (`evidence/prototype_outlier_aware_ptq.py` + `.out.txt`; 127 test-split images;
  diagnostic only): MobileNetV3-S → float blocks [features.0, features.1], FP32 agreement 0.197 → 0.937; MobileNetV2 → [features.17]
  (21.7× tensor), agreement 0.984 → 0.984 (no harm). An earlier, module-level version gave only 0.370 on V3-S (islands too small;
  pooled SE tensors produced false positives); both issues were fixed before the rule was frozen.
- Smoke tests (both archs, synthetic data): all cells ran, errors.json empty, pyflakes clean.
- Pilot v2.2.0 files moved to `runs/_pilot_v2.2.0/` (the new mobilenet_v2_seed0 run reuses the same RUN_ID). Pilot numbers stay
  valid as pilot evidence but must not be aggregated with v2.3.0 runs.

### 2026-11-16 · Full runs (notebook v2.3.0) started on Kaggle
- Kaggle allows **2 concurrent batch GPU sessions** on this account ("Maximum batch GPU session count of 2 reached").
- Running: `plantvillage-int8-mobilenetv3-small-seed0`, `plantvillage-int8-mobilenetv2-seed0` (version 2, v2.3.0).
- The remaining four runs were queued with `kaggle/run_queue.sh`.

### 2026-11-16 · RESULTS: v2.3.0 runs 1–2 of 6 (mobilenet_v3_small seed 0, mobilenet_v2 seed 0)
Both errors.json empty; split fingerprint 32b732ea792d… identical; 1.38 h and 2.22 h. (The run zip of mnv3-small-seed0 was verified with testzip.)
**MobileNetV3-Small seed 0 (full data):** FP32 0.9946. Khan dynamic 0.9945 (4.49 MB). **Plain PTQ 0.0742 (collapse confirmed at full
scale)**, plain QAT 0.6300. **Outlier-aware rule selected [features.0, features.1] automatically, on calibration images**:
**OA-PTQ 0.9887 (Δ −0.59 pp, McNemar p ≈ 0)**, 1.877 MB, CPU 7.53 ms vs FP32 8.39 ms (1.11× faster; plain INT8 8.76 ms, slower than FP32).
**OA-QAT 0.9894.** Shift: background removed: FP32 0.7283, OA-PTQ 0.6144 (−11.4 pp, flips 17.1%), OA-QAT 0.7191 (flips 14.8%);
PlantDoc-all strict: FP32 0.1917, OA-PTQ 0.1776 (flips 31.0%), OA-QAT 0.1786 (flips 34.2%). Clean flip rate 0.9%.
**MobileNetV2 seed 0 (v2.3.0 re-run of the pilot config):** FP32 0.9969. PTQ 0.9971 (Δ +0.01 pp, p = 1.0), QAT 0.9953 (p = 0.029).
The OA rule selected [features.17] (one late block), so OA-PTQ is 4.07 MB with no accuracy benefit (0.9966): **the rule over-triggers on
MobileNetV2** (an honest limitation to report). Shift: background removed: FP32 0.8518, **PTQ 0.8619 (+1.0 pp)**, QAT 0.8882 (+3.6 pp);
flips 7.3% / 7.9% vs 0.23% / 0.38% clean; PlantDoc-all flips 17.7% (PTQ), 22.7% (QAT).
**Replication note:** the pilot's "PTQ lowers background-removed accuracy (−3.2 pp)" is **NOT replicated** (+1.0 pp here, same seed,
non-deterministic cuDNN). Replicated so far: flip-rate amplification under shift (~30× background-removed, ~60–80× PlantDoc) and QAT
raising background-removed accuracy (+3.3, +3.6 pp).

### 2026-11-16 · RESULTS: v2.3.0 run 3 of 6 (mobilenet_v3_small seed 1) + queue fix
- mnv3-small seed 1: errors.json empty, split fingerprint 32b732ea792d (same), 1.61 h. FP32 0.9957; Khan dynamic 0.9957 (4.49 MB);
  **plain PTQ 0.0298 (collapse replicated)**, plain QAT 0.6568; the OA rule again selected **[features.0, features.1]**;
  **OA-PTQ 0.9899**, OA-QAT 0.9916; OA-PTQ 1.877 MB, CPU 8.50 ms vs FP32 9.37 ms. Background removed: FP32 0.6994, OA-PTQ 0.5652
  (−13.4 pp), OA-QAT 0.6705. PlantDoc-all: FP32 0.1831, OA-PTQ 0.1608, OA-QAT 0.1738.
- Data integrity: one run zip (mnv2 seed 1) was first downloaded incompletely. run_queue.sh now counts a run as fetched only if
  its zip exists, has no partial leftover and passes `testzip`; the zip was downloaded again.

### 2026-11-16 · RESULTS: ALL 6 RUNS COMPLETE (v2.3.0), aggregated
All 6 zips pass testzip; errors.json empty in all; notebook 2.3.0, torch 2.11.0+cu130; split fingerprint 32b732ea792d… identical.
Run times: mnv2 2.22 / 2.31 / 2.11 h; mnv3-small 1.38 / 1.61 / 1.53 h. Outputs: `aggregated/` (aggregate_runs.py) and
`paper_figures/` (make_paper_figures.py; 14 figures, 5 tables, MANIFEST.md). Mean ± s.d. over 3 seeds:
- **MobileNetV2:** FP32 0.9971 ± 0.0002 | Khan dynamic 0.9971, 9.20 MB, CPU 25.1 ms (FP32 23.2 ms) | **PTQ 0.9962 ± 0.0010**, 2.70 MB,
  15.5 ms (1.50×) | QAT 0.9952 ± 0.0006. Flip rate PTQ 0.29% clean → 5.5% background-removed → 17.8% PlantDoc-all; QAT 0.37% →
  7.9% → 24.7%. Background-removed accuracy: FP32 0.8578; PTQ 0.8542; **QAT 0.8920 ± 0.0182 (+3.4 pp; McNemar p<0.05 in 3/3)**.
  PlantDoc-all strict: FP32 0.2692, PTQ 0.2667, QAT 0.2768. The OA rule selected 1–2 late blocks (seed 0 [17]; seeds 1–2 [16, 17]):
  OA-PTQ 4.67 ± 0.53 MB, 0.9961, no benefit → over-triggering confirmed.
- **MobileNetV3-Small:** FP32 0.9952 ± 0.0006 | Khan dynamic 0.9950, 4.49 MB, 9.13 ms (FP32 8.84 ms) | **plain PTQ 0.0458 ± 0.0246
  (collapse 3/3)**, plain QAT 0.6457 ± 0.0140 | OA rule selected **[features.0, features.1] in 3/3 seeds** | **OA-PTQ 0.9889 ± 0.0010**
  (Δ −0.63 pp), 1.88 MB (3.39× smaller), 8.05 ms (1.10× faster) | **OA-QAT 0.9908 ± 0.0012**. Background-removed: FP32 0.6953 ±
  0.0353, OA-PTQ 0.5645 ± 0.0502 (−13.1 pp, flips 20.4%), OA-QAT 0.6706 ± 0.0484 (−2.5 pp, flips 15.3%). PlantDoc-all strict:
  FP32 0.1892, OA-PTQ 0.1727 (flips 33.1%), OA-QAT 0.1768 (flips 32.1%).
- **Pre-registered headline (pilot_plan.md K5):** flip rate under shift ≫ 3× clean in every INT8 variant → *"INT8 is (almost) free
  in-distribution but not under shift"*. **Additional replicated pattern (both archs, 3/3 seeds):** under background removal, QAT
  preserves or improves accuracy while PTQ does not (V2: QAT +3.4 pp vs PTQ −0.4 pp; V3-S: OA-QAT −2.5 pp vs OA-PTQ −13.1 pp).

### 2026-11-17 · Pre-submission review (minor revision) and fusion-latency check
- A pre-submission review recommended minor revision. Concerns: QAT extra-epoch confound; FP32 timed unfused vs fused INT8
  and virtual-CPU latency; outlier rule over-triggering on MobileNetV2; single split; low-support classes; numbered equation; PT2E.
- New evidence (no new training): `evidence/fusion_latency_check.py` (C36) and the validation-gate check from the recorded QAT
  histories (C37). Result that changed the text: the 1.10x CPU advantage of OA-PTQ MobileNetV3-S on Kaggle is smaller than fusion
  alone (1.18-1.22x on the second machine), so "faster than FP32" is no longer presented as an INT8 benefit (intro, 4.5, 5.3).
- QAT confound: no FP32 control was trained; the text now presents the QAT/background-robustness result as an association (abstract,
  intro, 5.4, 5.5, conclusions, highlight 5 replaced). An FP32 control (+3 epochs, same schedule, no fake-quant) was not run.

