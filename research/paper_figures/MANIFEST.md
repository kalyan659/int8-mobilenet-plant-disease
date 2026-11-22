# Paper figures and tables: provenance (generated, do not edit)

Generated 2026-11-16T17:08:47+00:00 by research/make_paper_figures.py

Runs (6): mobilenet_v2_seed0 (v2.3.0, torch 2.11.0+cu130), mobilenet_v2_seed1 (v2.3.0, torch 2.11.0+cu130), mobilenet_v2_seed2 (v2.3.0, torch 2.11.0+cu130), mobilenet_v3_small_seed0 (v2.3.0, torch 2.11.0+cu130), mobilenet_v3_small_seed1 (v2.3.0, torch 2.11.0+cu130), mobilenet_v3_small_seed2 (v2.3.0, torch 2.11.0+cu130)
Dataset fingerprint f0dbac1404290232c53929f0095814f960b3e401a4a23aac474673c72f052508
Split fingerprint 32b732ea792d09c3ac705313f2cd83a43538abd6452e1e18d01c36254b50d645

| Output | Source |
|---|---|
| figures/fig01_method_overview.png/.pdf | static diagram (no data) |
| figures/fig02a_dataset_gallery.png/.pdf | mobilenet_v2_seed0/figure_assets (gallery) |
| figures/fig02b_test_set_examples.png/.pdf | mobilenet_v2_seed0/figure_assets (segpair, plantdoc) |
| figures/fig03_clean_accuracy_f1.png/.pdf | runs/*/reports/benchmark_results.csv |
| figures/fig04_size_latency.png/.pdf | runs/*/reports/benchmark_results.csv (FP16 rows: GPU-only, no CPU latency) |
| figures/fig05_accuracy_under_shift.png/.pdf | runs/*/reports/shift_results.csv |
| figures/fig06_flip_rate_by_test_set.png/.pdf | runs/*/reports/shift_results.csv (flip_rate_vs_fp32) |
| figures/fig07_reliability_clean_vs_field.png/.pdf | runs/*/predictions/*.npz (pooled over seeds) |
| figures/fig08_per_class_delta_f1.png/.pdf | runs/*/reports/per_class_f1.csv |
| figures/fig09_training_curves.png/.pdf | runs/*/reports/training_history.csv |
| figures/fig10_flips_mobilenet_v2_plantdoc_all.png/.pdf | mobilenet_v2_seed0/figure_assets/manifest.csv | LICENCE CHECK: PlantDoc images are web-scraped (some carry stock-photo watermarks); do not print without rights |
| figures/fig11_background_pairs_mobilenet_v2.png/.pdf | mobilenet_v2_seed0/figure_assets/manifest.csv |
| figures/fig10_flips_mobilenet_v3_small_plantdoc_all.png/.pdf | mobilenet_v3_small_seed0/figure_assets/manifest.csv | LICENCE CHECK: PlantDoc images are web-scraped (some carry stock-photo watermarks); do not print without rights |
| figures/fig11_background_pairs_mobilenet_v3_small.png/.pdf | mobilenet_v3_small_seed0/figure_assets/manifest.csv |
| tables/tab01_dataset.csv/.md/.tex | mobilenet_v2_seed0/experiment_config.json |
| tables/tab02_main_results.csv/.md/.tex | runs/*/reports/benchmark_results.csv |
| tables/tab03_quantization_accounting.csv/.md/.tex | runs/*/reports/benchmark_results.csv |
| tables/tab04_shift.csv/.md/.tex | runs/*/reports/shift_results.csv |
| tables/tab05_paired_tests.csv/.md/.tex | runs/*/reports/benchmark_results.csv |
