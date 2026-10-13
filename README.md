# NIDS Reliability Evaluation Framework
## Evaluating ML-Based Network Intrusion Detection Systems Under Temporal Shift and Controlled Stress Conditions

This repository contains a reproducible research framework evaluating the reliability of machine learning-based Network Intrusion Detection Systems (NIDS) when subjected to chronological holdout shift, test-time feature missingness, and combined stress conditions.

The evaluation covers two premier intrusion detection benchmarks:
1. **CICIDS2017 (Part A - Checkpoint `2fe4892`):** Captures multi-day enterprise network attacks across 27 classes, evaluating structural class shifts under capture-day provenance (Monday–Thursday train vs. Friday test).
2. **UNSW-NB15 (Part B):** Evaluates contemporary network attacks across 10 classes under continuous class support using discrete capture session provenance.
3. **Cross-Dataset Synthesis:** Comparative evaluation disentangling structural class-absence failure from continuous feature drift, evaluating model resilience hierarchies, and analyzing non-additive combined stress interactions.

---

## 1. Research Protocol & Experimental Scope

- **Datasets:**
  - `CICIDS2017`: 2,089,924 records across 87 retained numeric features; 27 multiclass targets.
  - `UNSW-NB15`: 257,673 records across 42 retained features (39 numeric + 3 categorical); 10 multiclass targets.
- **Model Architectures:**
  - Logistic Regression (L-BFGS solver, `tol=1e-3`, `max_iter=300`, training `StandardScaler`).
  - Random Forest (`n_estimators=65`, `max_depth=18`, `class_weight="balanced_subsample"`).
  - XGBoost (`n_estimators=85`, `max_depth=5`, `tree_method="hist"`, `learning_rate=0.08`).
- **Evaluation Tasks:** Strictly partitioned Binary (Normal vs. Attack) and Multiclass classifications.
- **Evaluation Conditions (6 Total):**
  1. `Reference` (Stratified 80/20 train/test split, clean test data)
  2. `Reference + 10% Missingness` (Stratified split, 10% test-time Bernoulli missingness)
  3. `Reference + 20% Missingness` (Stratified split, 20% test-time Bernoulli missingness)
  4. `Temporal` (Chronological session holdout, clean test data)
  5. `Temporal + 10% Missingness` (Chronological holdout, 10% test-time missingness)
  6. `Temporal + 20% Missingness` (Chronological holdout, 20% test-time missingness)
- **Paper Mode Statistical Protocol:**
  - Evaluated across 5 independent random seeds: `[42, 52, 62, 72, 82]`.
  - Statistical aggregation reporting sample means, standard deviations, and 95% Student-$t$ confidence intervals ($df=4$).

---

## 2. Directory Structure & Key Files

```
NIDS-Reliability/
├── configs/
│   ├── experiment_config.yaml         # Preserved CICIDS2017 reference configuration
│   └── unsw_experiment_config.yaml    # Standalone UNSW-NB15 experimental parameters
├── src/
│   ├── 01_setup.py ... 07_*.py        # CICIDS2017 pipeline (Part A, preserved intact)
│   ├── unsw_01_setup.py               # UNSW-NB15 setup, hyperparams, & exclusion rationale
│   ├── unsw_02_data_and_temporal.py   # UNSW-NB15 ingestion, class audit, & session splits
│   ├── unsw_03_preprocessing.py       # Imputers, encoders, model factories, & LR convergence audit
│   ├── unsw_04_experiments.py         # UNSW-NB15 6-condition experiment runner
│   ├── unsw_05_analysis.py            # Degradation, FPR, & attack-level recall calculations
│   ├── unsw_06_repeated_figures.py    # 5-seed aggregation (95% CI) & Figures 1-6
│   ├── unsw_07_validation_export.py   # Zero-leakage assertions & structured CSV/JSON exports
│   └── cross_dataset_analysis.py      # Cross-dataset comparative synthesis & visualizations
├── run_experiments.py                 # Master runner for CICIDS2017 (Part A)
├── run_unsw_experiments.py            # Master runner for UNSW-NB15 (Part B)
├── run_cross_analysis.py              # Master runner for cross-dataset synthesis
├── diagnostic_audit.py                # Automated audit script for outputs & temporal health
├── EXPERIMENT_DESIGN.md               # Detailed methodology for CICIDS2017
├── RESULTS_GUIDE.md                   # Complete results guide for CICIDS2017
├── UNSW_EXPERIMENT_DESIGN.md          # Comprehensive methodology for UNSW-NB15
├── CROSS_DATASET_REPORT.md            # In-depth comparative research report
└── AGENTS.md                          # Mandatory scientific constraints & agent guardrails
```

---

## 3. Reproduction & Execution Instructions

### A. Environment Setup
Install required dependencies:
```bash
pip install -r requirements.txt
pip install huggingface_hub xgboost scikit-learn scipy matplotlib pandas numpy
```

### B. Running UNSW-NB15 Experiments (Part B)
Execute the complete five-seed paper evaluation for UNSW-NB15:
```bash
python run_unsw_experiments.py
```
Outputs are exported to `results/unsw_nb15/`:
- `repeated_seed_results.csv`: Authoritative 180-row dataset across all 5 seeds.
- `repeated_seed_summary.csv`: 5-seed means, standard deviations, and 95% Student-$t$ CIs.
- `degradation_*.csv`: Relative degradation tables for Macro-F1, Recall, and FPR.
- `interaction_results.csv`: Combined stress non-additive interaction terms.
- `attack_level_recall.csv`: Per-attack detection recall tracking across all 10 classes.
- `figure1_*.png` to `figure6_*.png`: Publication charts.
- `validation.json`: Audit log confirming zero leakage, condition completeness, and status `PASS`.

### C. Running Cross-Dataset Analysis
To generate side-by-side comparison tables and cross-dataset figures:
```bash
python run_cross_analysis.py
```
Outputs are exported to `results/cross_dataset/`:
- `comparison_metrics.csv`
- `comparison_degradation.csv`
- `comparison_interaction.csv`
- `comparison_class_dynamics.csv`
- `figure_cross_1_reference_vs_temporal.png` to `figure_cross_4_model_resilience_ranking.png`

---

## 4. Key Scientific Findings & Guardrails

1. **Continuous vs. Structural Temporal Failure:**
   - **CICIDS2017:** Friday test partition contains only 5 of 27 classes (22 attack classes absent, 4 unseen). Temporal Macro-F1 collapses to a structural ceiling near ~0.037.
   - **UNSW-NB15:** All 10 classes are populated across both partitions. Temporal degradation reflects true continuous distribution and feature drift.
2. **Feature Exclusion Policy:**
   - All spatial identifiers (`srcip`, `dstip`) and synthetic row keys (`id`) are excluded to prevent memorization shortcuts.
   - Ephemeral `sport` is excluded as dynamic client socket noise.
   - Raw `dsport` is excluded to prevent lab server bias; behavioral port context is preserved via 6 engineered features (`service`, `ct_srv_src`, `ct_srv_dst`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `is_sm_ips_ports`).
3. **Leakage-Safe Preprocessing:**
   - Categorical `OneHotEncoder(handle_unknown="ignore")` and numeric `SimpleImputer(strategy="median")` are fitted strictly on training data.
   - Missingness masks are applied exclusively to evaluation matrix copies at test time.
4. **Research Boundaries (AGENTS.md):**
   - No fabricated experimental results.
   - No temporal proxies (e.g. flow duration as fake timestamps).
   - Findings represent benchmark reliability profiles; they do not constitute proof of concept drift, adversarial robustness, universal model superiority, or production readiness.
