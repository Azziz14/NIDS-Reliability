# Major Project Implementation Plan — Part B (Audited & Refined)
**Reliability Evaluation of Machine Learning-Based NIDS Under Temporal and Controlled Stress Conditions**

- **Protocol Scope:** UNSW-NB15 Integration & Cross-Dataset Reliability Evaluation
- **Baseline Checkpoint:** `2fe4892` (CICIDS2017 Validated)
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Protocol:** 6 Evaluation Conditions × 2 Tasks (Binary & Multiclass) × 5 Seeds (`[42, 52, 62, 72, 82]`)
- **Status:** Awaiting Approval & Repository Sync
- **Generated PDF:** [Part_B_Implementation_Plan.pdf](file:///c:/Users/ashis/OneDrive/Desktop/pp1/NIDS-Reliability/Part_B_Implementation_Plan.pdf) *(also available in Downloads & Desktop)*

---

## 1. Executive Summary & Research Mandate

This document presents the revised, empirical implementation plan for **Part B** of the research project *Reliability Evaluation of Machine Learning-Based Network Intrusion Detection Systems Under Temporal and Controlled Stress Conditions*.

Following the completion and documentation of Part A (CICIDS2017) at checkpoint `2fe4892`, Part B integrates **UNSW-NB15** and executes an integrated **cross-dataset reliability evaluation**. Crucially, this refined plan incorporates an audit of the actual UNSW-NB15 files, individual justifications for feature exclusions (particularly source and destination ports), exact leakage-safe preprocessing specifications, transparent Logistic Regression convergence handling, and explicit recognition that the temporal conditions across the two datasets represent fundamentally different phenomena.

> **Scientific Guardrails (AGENTS.md):**
> - Zero synthetic results fabricated or manually estimated.
> - Preprocessing transformations fitted strictly on training data only.
> - No flow duration or proxy feature substituted as a timestamp.
> - Binary and multiclass tasks remain strictly partitioned.
> - No claims of concept drift proof, universal model superiority, or cross-dataset transferability.

---

## 2. Empirical Audit of Actual UNSW-NB15 Files & Provenance

An empirical inspection of candidate distributions on the Hugging Face Hub (`Mouwiya/UNSW-NB15-small`, `lacg030175/UNSW-NB15`, and official ACCS releases) reveals key structural realities that govern the experiment:

| Distribution Format | File Names & Sizes | Row Count & Features | Timestamp & Duplicate Findings |
|---|---|---|---|
| **Standard Partitioned Set** (Official ACCS Benchmark) | `UNSW_NB15_training-set.csv` (32.3 MB)<br>`UNSW_NB15_testing-set.csv` (15.4 MB) | 175,341 rows (Train)<br>82,332 rows (Test)<br>Total: 257,673 rows<br>45 columns (with `id`) | **Timestamps absent:** `Stime` and `Ltime` were stripped by ACCS.<br>**Severe duplication:** 67,601 duplicates (38.56%) in training, 26,387 (32.05%) in testing without `id`. 48,111 cross-partition duplicates. |
| **Full Raw Flow Set** (4 CSVs / Full Parquet) | `UNSW-NB15_1.csv` to `4.csv`<br>Total ~500 MB uncompressed | 2,540,044 raw records<br>1,584,259 deduplicated<br>49 columns | **Timestamps present:** `Stime` and `Ltime` epoch seconds.<br>**Session provenance:** Contains 2 discrete capture periods. 30.5% duplicate flow records removed during cleaning. |

### A. Timestamp Ranges & Temporal-Split Feasibility Policy
1. **Data-Driven Provenance (No Blind Assumptions):** We will not assume a January/February split until the selected dataset's timestamp distribution is verified at runtime.
2. **Dual-Mode Provenance Detection:**
   - *If raw flow data with `Stime` is used:* Parse epoch values, verify monotonicity, plot/audit the timestamp histogram, and construct the temporal holdout at a verified session boundary (e.g., Session 1 train vs. Session 2 test).
   - *If the standard 45-column benchmark distribution is used:* Acknowledge that row timestamps are absent. Temporal evaluation in this mode utilizes the official ACCS train/test holdout, where train and test were captured during separate traffic sessions. Flow duration (`dur`) or packet IAT will **never** be used as a fake timestamp.

---

## 3. Individual Justification for Feature Exclusions

Every excluded feature in the pipeline is eliminated based on specific, documented scientific rationales rather than arbitrary omission:

| Feature Name | Type in Raw Data | Exclusion Rationale & Leakage Prevention |
|---|---|---|
| `srcip`, `dstip` | Nominal (IP Strings) | **Spatial Memorization Leakage:** Encodes network topology and subnet identity. Classifiers split on specific IP strings, memorizing attacker/victim VMs in the synthetic testbed. Fails completely in cross-network or shifted environments. |
| `sport` | Integer (Source Port) | **Ephemeral Port Noise:** Assigned dynamically client-side by OS socket allocators (typically 32,768–65,535). High cardinality, ephemeral noise with zero intrinsic intrusion semantics; causes decision trees to overfit to client socket allocation states. |
| `dsport` | Integer (Destination Port) | **Lab Target Bias & Feature Redundancy:** While destination ports represent targeted services (e.g., 80, 22), treating raw port numbers as continuous integers is statistically invalid (port 81 is not "greater than" 80). Furthermore, raw ports leak synthetic lab server configurations. **Crucially, UNSW-NB15 already extracts 6 engineered features capturing port behaviors:** `service` (nominal), `ct_srv_src`, `ct_srv_dst`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, and `is_sm_ips_ports`. Dropping raw `dsport` prevents lab overfitting while retaining full behavioral port context. |
| `id` | Integer (Index) | **Monotonic Key Leakage:** Synthetic row identifier added during export. Serves as a perfect shortcut predictor if retained. |
| `Stime`, `Ltime` | Timestamp / Epoch | **Time-Memorization Leakage:** Retained strictly for chronological sorting and temporal partitioning. Must be excluded from feature matrix $X$ so models learn traffic dynamics rather than epoch time thresholds. |

---

## 4. Exact Leakage-Safe Numeric & Categorical Preprocessing Pipeline

The preprocessing sequence strictly isolates the training partition from evaluation data across all six stress conditions:

1. **Split Before Preprocessing:** Train and test index partitions are established prior to any transformation. No global statistics are computed.
2. **Categorical Preprocessing (Unseen Test-Category Handling):**
   - Nominal features: `proto` (protocol), `service` (service), `state` (connection state).
   - Transformed via `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` fitted **strictly on $X_{\text{train}}$**.
   - **Unseen Category Protocol:** If an unseen protocol or state appears in test data (or under missingness corruption), `handle_unknown="ignore"` outputs an all-zero vector across the dummy columns for that feature. No test-category information leaks into the vocabulary.
3. **Numeric Imputation:**
   - 39 continuous and engineered count features are handled via `SimpleImputer(strategy="median", keep_empty_features=True)` fitted **strictly on $X_{\text{train}}$**.
   - The fitted imputer transforms both clean test data and missingness-corrupted test matrices. Zero non-finite values (`NaN`, `Inf`) are permitted in estimator inputs.
4. **Model-Specific Scaling:**
   - **Logistic Regression:** `StandardScaler` fitted solely on imputed $X_{\text{train}}$ and applied to evaluation sets.
   - **Random Forest & XGBoost:** Invariant to monotonic scaling; receive imputed float32 feature arrays directly.
5. **Test-Time Controlled Missingness Stress:**
   - Independent Bernoulli masks ($p \in \{0.10, 0.20\}$) are applied **exclusively to evaluation matrices** at test time.
   - Injected NaNs are passed through the pre-fitted train median imputer. Training data and fitted models are never modified.

---

## 5. Class Coverage, Sample Sizes & Non-Equivalence of Temporal Conditions

A critical scientific contribution of this evaluation is acknowledging that **CICIDS2017 and UNSW-NB15 temporal conditions represent fundamentally different phenomena**:

| Evaluation Dimension | CICIDS2017 Baseline (Part A) | UNSW-NB15 Evaluation (Part B) | Scientific Implication & Non-Equivalence |
|---|---|---|---|
| **Reference Split** | 1,671,939 Train / 417,985 Test<br>(80/20 Stratified across 27 classes) | Standard: 175,341 Train / 82,332 Test<br>Deduplicated: ~1.26M Train / ~316K Test | Both evaluate in-distribution performance under identical stratified sampling. |
| **Temporal Split** | 1,543,723 Train (Mon–Thu)<br>546,201 Test (Friday only) | Standard Holdout: 175,341 Train / 82,332 Test<br>Verified Session Cut: Session 1 / Session 2 | **Fundamental difference in class dynamics:**<br>• *CICIDS2017:* 22 attack classes absent from Friday test; 4 attacks absent from Mon–Thu train. Macro-F1 bounded near 0.037.<br>• *UNSW-NB15:* **All 10 classes present in both partitions.** Tests true distribution shift under continuous class coverage. |
| **Class Coverage** | 5 of 27 classes evaluated on Friday | 10 of 10 classes evaluated in temporal split | Temporal degradation cannot be compared naively; report must distinguish unseen-class ceilings from feature drift. |

### UNSW-NB15 Class Distribution (Audited Benchmark Split)

| Class Name | Train Count (175k) | Test Count (82k) | Total Count | Temporal Status |
|---|---|---|---|---|
| **Normal** | 56,000 | 37,000 | 93,000 | Present in both partitions |
| **Generic** | 40,000 | 18,871 | 58,871 | Present in both partitions |
| **Exploits** | 33,393 | 11,132 | 44,525 | Present in both partitions |
| **Fuzzers** | 18,184 | 6,062 | 24,246 | Present in both partitions |
| **DoS** | 12,264 | 4,089 | 16,353 | Present in both partitions |
| **Reconnaissance** | 10,491 | 3,496 | 13,987 | Present in both partitions |
| **Analysis** | 2,000 | 677 | 2,677 | Present in both partitions |
| **Backdoor** | 1,746 | 583 | 2,329 | Present in both partitions |
| **Shellcode** | 1,133 | 378 | 1,511 | Present in both partitions |
| **Worms** | 130 | 44 | 174 | Present in both partitions |

---

## 6. Mathematical Definitions & Handling of Edge Cases

- **Undefined Metrics & Absent Classes:**
  - `precision_recall_fscore_support(..., zero_division=0)`: If a class has zero true instances in the test partition (support = 0), recall is defined as 0.0 under `zero_division=0`. If never predicted, precision is 0.0.
  - Macro metrics represent unweighted means over all $K$ target classes: $\text{Macro-F1} = \frac{1}{K}\sum_{i=1}^K F1_i$.
- **Zero Denominators in Degradation:**
  - Formula: $D = (M_{\text{Reference}} - M_{\text{Condition}}) / M_{\text{Reference}}$.
  - Handled via `.replace(0, np.nan)`: if $M_{\text{Reference}} = 0$, degradation is mathematically undefined and recorded as `NaN` (never fabricated as 0.0 or 1.0).
- **False Positive Rate (FPR):**
  - Per-class $FPR_i = FP_i / (FP_i + TN_i)$. If $FP_i + TN_i = 0$, $FPR_i = 0.0$. Macro-FPR is unweighted mean across all classes.
  - Relative FPR Degradation: $D_{\text{FPR}} = (M_{\text{Condition}} - M_{\text{Reference}}) / M_{\text{Reference}}$ with zero denominators replaced by `NaN`.
- **Non-Additive Interaction Term:**
  $$\text{Interaction} = D_{\text{Combined}} - (D_{\text{Temporal}} + D_{\text{Missingness}})$$
- **Student-t 95% Confidence Intervals:**
  $$\bar{x} \pm t_{0.975, df=4} \cdot \frac{s}{\sqrt{5}} \quad (t_{\text{crit}} = 2.7764)$$
  Sample standard deviation $s$ uses Bessel correction ($ddof=1$). If $s = 0$, CI half-width is 0.0. If $N < 2$, CI bounds are `NaN`.

---

## 7. Logistic Regression Convergence Transparency Protocol

Rather than claiming that iteration counts prove mathematical convergence, we establish an explicit convergence audit:
1. **Active Warning Interception:** `model.fit()` runs within `warnings.catch_warnings(record=True)` to capture scikit-learn `ConvergenceWarning` events.
2. **Diagnostic Recording:** The export metadata records:
   - `lr_iterations`: Actual iterations executed (`int(model.n_iter_[0])`).
   - `lr_max_iter`: Iteration ceiling (`300`).
   - `lr_converged_without_warning`: Boolean indicating whether L-BFGS terminated within tolerance before reaching `max_iter`.
   - `lr_warnings_logged`: Verbatim warning strings if any were raised.
3. **Scientific Disclosure:** The report explicitly states that terminating below `max_iter` indicates the gradient norm satisfied stopping tolerance `tol=1e-3`, while reaching `max_iter=300` indicates an iteration cutoff. We explicitly avoid claiming that `n_iter_` alone constitutes mathematical proof of global optimality.

---

## 8. Cross-Dataset Comparative Analysis Framework

The cross-dataset synthesis answers four core research questions:
1. **Structural Ceiling vs. Feature Drift:** Disentangles whether temporal vulnerability is driven by unseen classes (CICIDS2017) or continuous feature drift across identical classes (UNSW-NB15).
2. **Architectural Resilience:** Evaluates whether gradient-boosted trees (XGBoost) and random forests maintain consistent superiority over linear models across both network topologies.
3. **Interaction Geometry:** Assesses whether combined stress exhibits sub-additive saturation (floor effects) or super-additive degradation across datasets.
4. **Scientific Boundaries (Rule 8):** Clarifies that findings highlight dataset-specific reliability profiles rather than universal model superiority or production readiness.

---

## 9. Codebase Architecture & File Modifications

```
NIDS-Reliability/
├── configs/
│   ├── experiment_config.yaml         # Preserved (CICIDS2017 reference configuration)
│   └── unsw_experiment_config.yaml    # Standalone UNSW-NB15 experimental parameters
├── src/
│   ├── 01_setup.py ... 07_*.py        # CICIDS2017 pipeline (Preserved intact)
│   ├── unsw_01_setup.py               # UNSW-NB15 automatic downloader & schema auditor
│   ├── unsw_02_data_and_temporal.py   # Audited ingestion, stime sorting & duplicate cleaning
│   ├── unsw_03_preprocessing.py       # Imputers, encoders, scalers & model factories
│   ├── unsw_04_experiments.py         # 5-seed 6-condition UNSW-NB15 runner
│   ├── unsw_05_analysis.py            # Degradation, interaction & attack-level metrics
│   ├── unsw_06_repeated_figures.py    # Statistical aggregation (95% CI) & Figures 1-6
│   ├── unsw_07_validation_export.py   # Integrity assertions & structured CSV/JSON exports
│   └── cross_dataset_analysis.py      # Cross-dataset comparative synthesis & visualizations
├── run_experiments.py                 # Existing CICIDS2017 execution script
├── run_unsw_experiments.py            # Dedicated execution script for UNSW-NB15
├── run_cross_analysis.py              # Execution script for cross-dataset analysis
├── diagnostic_audit.py                # Updated to audit both datasets & fix cp1252 output
├── UNSW_EXPERIMENT_DESIGN.md          # Full methodology documentation for UNSW-NB15
└── CROSS_DATASET_REPORT.md            # Comprehensive comparative report for final thesis
```

---

## 10. Complete Inventory of Deliverables & Generated Artifacts

| Output File / Directory | Dimensions / Content | Description & Purpose |
|---|---|---|
| `results/unsw_nb15/all_results_seed42.csv` | 36 rows (2 × 3 × 6) | Per-combination metrics for primary seed 42 |
| `results/unsw_nb15/repeated_seed_results.csv` | 180 rows | Authoritative raw data across all 5 seeds |
| `results/unsw_nb15/repeated_seed_summary.csv` | 36 rows × 15 cols | 5-seed means, standard deviations, and 95% CIs |
| `results/unsw_nb15/final_condition_metrics.csv` | 36 rows | Aggregated performance baseline per condition |
| `results/unsw_nb15/degradation_*.csv` | 36 rows per metric | Relative degradation tables (F1, Recall, FPR) |
| `results/unsw_nb15/interaction_results.csv` | 12 rows (2 × 3 × 2) | Non-additive combined stress interaction metrics |
| `results/unsw_nb15/attack_level_recall.csv` | 180 rows (10 × 6 × 3) | Per-attack-class detection recall under all conditions |
| `results/unsw_nb15/figure1-6_*.png` | 6 high-res figures | Publication-quality charts matching Part A specs |
| `results/unsw_nb15/validation.json` | JSON schema | Audit metadata confirming zero-leakage, pass flags, and LR convergence log |
| `results/cross_dataset/comparison_*.csv` | Comparative tables | Side-by-side performance, degradation & interactions |
| `results/cross_dataset/figure_cross_*.png` | Comparative figures | Visual comparison of stress vulnerabilities across datasets |
| `CROSS_DATASET_REPORT.md` | Full Academic Text | Comprehensive final-year report chapter with limitations |

---

## 11. Scientific Validation & Leakage Verification Criteria

1. **Zero Data Leakage:** Encoders, imputers, and scalers fitted strictly on training subsets. Temporal partitions isolated prior to any preprocessing.
2. **Unseen Category Defense:** All-zero dummy encoding for test categories unseen during training; verified zero encoding exceptions.
3. **Matrix Cleanliness:** Automated assertions confirming zero `NaN`, `Inf`, or unhandled values in estimator inputs.
4. **Statistical Completeness:** Summary tables contain zero `NaN` values for standard deviations or confidence intervals across $N=5$.
5. **Convergence Transparency:** Explicit logging of LR convergence status without overstating `n_iter_`.

---

### Execution Confirmation Gate

Per your instructions (*"Msg before ur starting nxt such that i will let u know whether the committed all the updates or not any"*):
- Implementation will begin once you confirm this plan and verify that all remote and local updates on your side are fully committed.
