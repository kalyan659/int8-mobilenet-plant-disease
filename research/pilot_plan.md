# Pilot plan: early go/no-go before the full 6-run budget

Written 2026-11-15, **before any real result exists**. The criteria below are fixed now so the pilot cannot be read in our favour
after the fact.

## What "better than existing" can and cannot mean here
- **It cannot mean higher clean PlantVillage accuracy.** The benchmark is saturated: MobileNetV2 99.77% (Ho-Lam 2026),
  MobileNetV3-S 99.50% (Khan 2023), EfficientNet 99.97% (Atila 2021, to fetch). Comparing accuracies taken from papers with
  different splits is the very practice we criticise (Khan 2023, Table 3).
- **It can mean, measured in one pipeline on identical data:**
  (a) a *truly* smaller model at equal accuracy: full INT8 vs the partial INT8 of the Khan protocol, reproduced on the same
  weights;
  (b) *valid* evidence: validation-based selection, seeds, paired tests, leakage control;
  (c) *new knowledge* about INT8 under real shift: PlantDoc and background removal, plus calibration and per-class effects.
- **Optional method contribution** (needs your decision after the pilot): background-randomised training/QAT, using
  PlantVillage's segmented masks to paste leaves onto varied backgrounds. Ahmad 2023 reports that background removal and
  lab+field mixing help corn models (FP32). Combining it with INT8 edge models is untested in the collected literature. It would
  give a *method* that can beat baselines on the field metric, which is where improvement is still possible.

## Pilot runs (≈ 1.5–2 GPU-hours, estimate; real timing is part of the pilot)
| Run | Config | Counts toward the paper? |
|---|---|---|
| P1 | `ARCH=mobilenet_v2, SEED=0`, full data, default config | **Yes**, it is run 1 of 6 (nothing wasted) |
| P2 | `ARCH=mobilenet_v3_small, SEED=0`, `MAX_IMAGES_PER_CLASS=300`, `FINETUNE_EPOCHS=6` | No, diagnostic only (V3 INT8 stability and latency) |

## Pre-registered checks and decisions
| ID | Check | Expected | If not as expected |
|---|---|---|---|
| K0 | `errors.json` empty; GPU/AMP/FP16 cells ran; PlantDoc map covers ≥ 20 folders; `plantdoc_all` built; note `identical_to_plantvillage_removed` | yes | fix pipeline before any more runs |
| K1 | Khan protocol on MobileNetV3-S, 38 classes: `parameters()` = 927,008, conv layers FP32 | yes (C1/C2) | investigate; RQ1 weakens |
| K2 | FP32 PlantDoc strict accuracy (MobileNetV2) within ±15 pp of Ho-Lam 2026's 28.57% | roughly | check the class mapping and preprocessing before trusting any shift numbers |
| K3 | Full INT8 PTQ ΔAcc on pv_test (MobileNetV2) ≥ −0.5 pp; QAT ≥ −0.3 pp | yes | INT8 is costly even in-distribution: a finding, re-centre RQ2 |
| K4 | MobileNetV3-S static PTQ (P2): ΔAcc vs FP32 | unknown (C5) | a collapse > 5 pp **is** an RQ4 finding: log it, then test the mitigations (per-tensor qconfig, more calibration, QAT) |
| K5 | Shift sensitivity: INT8 flip rate on pv_segmented vs pv_test (MobileNetV2, PTQ) | unknown | sets the headline (below) |
| K6 | INT8 CPU latency on Kaggle x86 vs FP32, for both archs | INT8 faster | slower INT8 (as seen on local oneDNN) = deployment finding for RQ1/RQ4 |

**Decision rule.** Run the remaining 5 runs if K0–K2 pass. K3–K6 do **not** decide go/no-go. They decide the headline:
- K5 flip rate under shift ≥ 3× the clean flip rate, or a significant McNemar on pv_segmented → headline *"INT8 is free
  in-distribution but not under shift"*;
- otherwise → headline *"full INT8 is safe under shift; the field gap is a domain problem, not a precision problem"*, plus the
  accounting audit. Both are publishable. The second is a stronger case for adding the optional background-randomised method.

## After the pilot
1. Put the zips in `research/runs/`, run `aggregate_runs.py`, and record K0–K6 in `experiment_log.md` against this table.
2. Decide on the optional method (yes/no) **before** the remaining runs, so every run uses the same notebook version.
