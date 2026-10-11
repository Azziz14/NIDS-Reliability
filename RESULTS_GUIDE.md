# Results Guide: CICIDS2017 Paper-Mode Outputs

## Output directory

All generated files are in `nids-reliability/results/`. This directory contains the
completed five-seed, full-data paper-mode run. Do not modify or delete these files.

A separate archive of the preliminary quick-validation outputs (seed 42 only,
100,000-row training subset) is preserved in `nids-reliability/results_quickval_seed42/`
and must not be overwritten.

---

## Complete file inventory

### Primary experiment results

| File | Size | Purpose |
|---|---|---|
| `all_results_seed42.csv` | 118,775 B | Per-row metrics for seed 42 only (the primary run). 36 rows: 2 tasks × 3 models × 6 conditions. Also used internally as the seed-42 contribution to the repeated-seed tables. |
| `repeated_seed_results.csv` | 593,712 B | All 180 raw result rows across all five seeds. The source of truth for all aggregated statistics. |
| `repeated_seed_summary.csv` | 10,539 B | Five-seed mean, standard deviation, and 95% confidence intervals. 36 rows, N=5 for every cell. |
| `final_condition_metrics.csv` | 5,873 B | Five-seed mean values per task/model/condition used as input to degradation and interaction calculations. 36 rows. |

### Analysis tables

| File | Size | Purpose |
|---|---|---|
| `degradation_macro_f1.csv` | 3,633 B | Relative Macro-F1 degradation from reference for each condition. 36 rows. |
| `degradation_macro_recall.csv` | 3,637 B | Relative Macro Recall degradation. 36 rows. |
| `degradation_fpr.csv` | 3,771 B | Relative FPR change from reference. 36 rows. Positive value = FPR increased (worsened). |
| `interaction_results.csv` | 1,407 B | Temporal × missingness interaction term. 12 rows: 2 tasks × 3 models × 2 missingness rates. |
| `attack_level_recall.csv` | 42,917 B | Per-class mean recall (aggregated across 5 seeds) for every model/condition combination. 486 rows: 27 classes × 6 conditions × 3 models. |
| `attack_recall_degradation.csv` | 59,874 B | Per-class recall degradation relative to the reference condition. |

### Figures

| File | Size | Description |
|---|---|---|
| `figure1_reference_vs_temporal_macro_f1.png` | 96,231 B | Multiclass Macro-F1: Reference vs Temporal, all three models. |
| `figure2_degradation.png` | 148,686 B | Relative Macro-F1 degradation across all six conditions, multiclass. |
| `figure3_interaction.png` | 85,631 B | Temporal × missingness interaction effect, multiclass. |
| `figure4_attack_recall_degradation.png` | 261,621 B | Ten largest per-class recall degradation values across all model/condition/class combinations. |
| `figure5_representative_confusion_matrix.png` | 62,751 B | Confusion matrix for the best-performing model on the seed-42 multiclass reference run. |
| `figure6_repeated_seed_variability.png` | 105,836 B | Five-seed Macro-F1 with 95% t-CI error bars for Reference and Temporal conditions. Only present when N ≥ 2; absent from the quick-validation archive. |

### Metadata

| File | Size | Purpose |
|---|---|---|
| `validation.json` | 2,157 B | Run metadata, dataset provenance, configuration flags, and PASS/FAIL status. |
| `experiment_config.json` | 1,942 B | Full configuration snapshot: models, tasks, conditions, formulas, preprocessing policy. |
| `dataset_file_validation.json` | 1,069 B | Per-file loading statistics: rows retained, capture-day rank, timestamp availability. |

---

## Understanding the three result levels

### Level 1 — all_results_seed42.csv

Contains the 36 rows from seed 42 only. Each row represents one (task, model, condition)
combination. Columns include Confusion Matrix and Per-class Recall as serialised objects,
which makes this file larger than the summary tables.

Use this file for inspecting raw seed-42 predictions or the confusion matrix for a specific
condition. Do not treat it as the final paper results — it is one seed.

### Level 2 — repeated_seed_results.csv

Contains all 180 rows (5 seeds × 36 combinations). This is the authoritative source for
all aggregated statistics.

Columns: Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, FPR, Confusion
Matrix, Per-class Recall, Task, Model, Condition, Missingness, Seed.

Core metrics (Accuracy through FPR) have zero NaN values. Confusion Matrix and Per-class
Recall are serialised strings that reflect the per-seed model behaviour.

### Level 3 — repeated_seed_summary.csv

Contains 36 rows (2 tasks × 3 models × 6 conditions), each aggregated from N=5 seeds.

Columns provided per metric (Macro F1, Macro Recall, FPR):
- `Mean_*`: arithmetic mean across 5 seeds
- `Std_*`: sample standard deviation (ddof=1, Bessel-corrected)
- `CI95_*_Low` and `CI95_*_High`: 95% t-based confidence interval bounds

All 36 rows have N=5. All Std and CI columns are fully populated — none are NaN. This
distinguishes the paper-mode outputs from the quick-validation archive, where N=1 produced
NaN for all Std and CI fields.

---

## How to read mean, standard deviation and confidence intervals

The summary statistics reflect variability across the five reference-split seeds. Each seed
uses the same full dataset but a different stratified 80/20 reference partition and different
model random seeds.

The 95% t-CI is computed as:

```
mean ± t(0.975, df=4) × std / sqrt(5)
```

A narrower CI indicates that the metric is stable across seeds. A wider CI indicates
sensitivity to the specific reference split chosen.

Example from the completed run (multiclass, Random Forest, Reference condition):
- Mean Macro-F1: 0.929297
- Std: 0.009690
- 95% CI: [0.917265, 0.941328]

This means that across five independently seeded runs on the full training partition, Random
Forest multiclass reference performance ranged from approximately 0.917 to 0.941.

---

## How to interpret temporal Macro-F1 and unseen-class effects

The temporal evaluation uses a chronological hold-out: Monday–Thursday as training,
Friday as test. The Friday test partition contains only five of the 27 class labels:
BENIGN, Botnet, Botnet - Attempted, DDoS, and Portscan.

**26 of 27 classes show zero temporal recall.** This breaks down into two distinct groups:

**Group 1 — genuinely absent from temporal training (4 classes):**
Botnet, Botnet - Attempted, DDoS, Portscan. These appear only on Friday and were never
seen during Monday–Thursday training. Zero recall on these classes reflects a true
generalisation failure to unseen attack types.

**Group 2 — absent from Friday test (22 classes):**
DoS variants, FTP-Patator, SSH-Patator, Web Attacks, Infiltration, Heartbleed, and others.
These classes appear in Monday–Thursday training but have zero test rows on Friday. Recall
is reported as 0.0 because `zero_division=0` is used and there are no true positive
opportunities. This is not a model failure — it is a dataset characteristic.

**Consequence for Macro-F1:** Because Macro-F1 averages recall equally across all 27 classes
and 26 of them contribute 0.0, temporal Macro-F1 is structurally bounded near 1/27 ≈ 0.037
regardless of model quality. The small between-seed variance confirms this.

**Correct interpretation:** The temporal Macro-F1 values (~0.036–0.037) do not indicate that
models are failing on attack types they were trained on. They reflect the class composition
of the Friday test partition. Weighted F1 (which weights by test-set class frequency) gives
a more useful single-number summary of temporal detection performance for the classes that
actually appear in the Friday test set.

This experiment measures temporal reliability under the specific CICIDS2017 chronological
structure. It does not constitute evidence of concept drift.

---

## Degradation and interaction interpretation

`degradation_macro_f1.csv` contains relative Macro-F1 degradation for every task/model/
condition combination, computed from the five-seed mean values.

Positive degradation = performance is worse under that condition than under reference.
Zero degradation = the Reference condition row itself.

`interaction_results.csv` contains the non-additive interaction term:

```
Interaction = D_Combined - (D_Temporal + D_Missingness)
```

A negative multiclass interaction (e.g. −0.058 to −0.229 in the completed run) means the
combined stress condition is less damaging than the sum of individual stressors. This is
expected when temporal degradation already approaches 1.0 (near-complete degradation), so
adding missingness cannot proportionally worsen what is already near-total. It is a
ceiling/floor effect, not an indication of robustness.

---

## Archive vs live results

| Directory | Contents | Status |
|---|---|---|
| `results/` | Five-seed full-data paper-mode outputs | Final — do not modify |
| `results_quickval_seed42/` | Single-seed 100k-row quick-validation outputs | Archive — do not modify |

Key differences between the two:

| Property | Quick-validation | Paper-mode |
|---|---|---|
| Seeds | 42 only | 42, 52, 62, 72, 82 |
| Training rows (Reference) | 100,000 (stratified sample) | 1,671,939 (full partition) |
| Training rows (Temporal) | 100,000 (stratified sample) | 1,543,723 (full partition) |
| repeated_seed_results.csv rows | 36 | 180 |
| Std/CI fields in summary | all NaN (N=1) | all populated (N=5) |
| figure6 | absent | present |

---

## Expected limitations

- Temporal Macro-F1 near 0.037 is structurally determined by the Friday class distribution,
  not by model quality. This is documented and expected.
- LR convergence warnings were suppressed globally. Convergence on the full training partition
  was not independently verified.
- All findings are specific to CICIDS2017. No cross-dataset conclusions are supported.
- N=5 seeds provide a useful but limited estimate of between-run variability.
- The five reference splits are not independent datasets.

---

## Safe reproduction instructions

To reproduce the paper-mode results:

1. Confirm `results_quickval_seed42/` is intact before proceeding (it will not be affected).
2. In `src/01_setup.py`, set `QUICK_VALIDATION = False` and `PAPER_MODE = True`.
   (These are currently already set to the paper-mode values in the repository.)
3. Confirm the CICIDS2017 CSV files are accessible at the path reported in `results/validation.json`.
4. Run: `python nids-reliability/run_experiments.py`
5. The run will overwrite all files in `results/`. Archive the current `results/` first if
   the existing outputs must be preserved.
6. Expected runtime: approximately 2–3 hours on a machine with 16 GB RAM and 2–4 CPU cores.
7. After completion, verify using: `python nids-reliability/diagnostic_audit.py`

---

## Planned future work (not yet implemented)

The following components are described in project planning documents but are **not present in
any source file, configuration, or output in this repository**:

- UNSW-NB15 dataset integration
- CSE-CIC-IDS2018 optional external validation
- Cross-dataset comparative analysis
- SHAP explainability
- Automated reliability report generation
