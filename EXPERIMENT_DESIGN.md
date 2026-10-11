# Experiment Design: CICIDS2017 Reliability Evaluation

## Research objective and scope

This framework evaluates how the reliability of ML-based Network Intrusion Detection Systems
degrades under temporal shift and controlled feature-missingness stress.

The core research question is: by how much does detection performance degrade when a model is
evaluated on data from a later time period than it was trained on, when feature values are
partially missing at evaluation time, or when both stressors are present simultaneously?

**In-scope for this implementation (CICIDS2017 conference results):**
- Dataset: CICIDS2017
- Models: Logistic Regression, Random Forest, XGBoost
- Tasks: binary and multiclass classification
- Reliability factors: temporal shift and controlled feature missingness
- Six evaluation conditions (see below)
- Five-seed repeated protocol with 95% confidence intervals

**Out of scope for this implementation (not implemented in this repository):**
- UNSW-NB15 dataset
- CSE-CIC-IDS2018 dataset
- SHAP explainability
- Cross-dataset validation
- Automated reliability report generation

---

## Dataset: CICIDS2017

Source: Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017.

The dataset was loaded from five daily CSV files (Monday through Friday), each corresponding to
one day of network capture. The executed run used a local copy at
`C:\content\CICIDS2017_timestamped` (reported in `results/validation.json` as
`local:/content/CICIDS2017_timestamped`).

After loading, deduplication and cleaning:
- Total rows: 2,089,924
- Numeric features retained: 87
- Distinct class labels: 27
- Raw NaN cells in feature matrix: 0 (confirmed by validation.json)
- Raw Inf cells in feature matrix: 0 (confirmed by validation.json)

**Temporal provenance:** The Timestamp column in the available CSV files was only 41–52%
parseable across files, which is below the 90% validity threshold required for row-level
ordering. The pipeline therefore used capture-day/file-order provenance: Monday (rank 0),
Tuesday (rank 1), Wednesday (rank 2), Thursday (rank 3), Friday (rank 4). This is documented
as `capture_day_file_order` in `results/validation.json`.

Flow Duration, Total TCP Flow Time, packet counts, and IAT features are explicitly forbidden
as timestamp substitutes. This is enforced in `src/01_setup.py` via `FORBIDDEN_TIMESTAMP_FEATURES`.

---

## Reference split

The reference split is a stratified random 80/20 partition of the full dataset.

- Train: 1,671,939 rows
- Test:  417,985 rows
- Stratified by class label to preserve class proportions
- `random_state` is set to the current seed for each of the five runs, so the reference split
  varies reproducibly across seeds

Implementation: `sklearn.model_selection.train_test_split` with `stratify=y_sorted`,
`test_size=0.20`, `random_state=seed` — in `src/04_experiments.py`, `run_one_seed()`.

---

## Temporal split

The temporal split uses the documented capture-day ordering as chronological provenance.

- Train: Monday through Thursday — 1,543,723 rows
- Test:  Friday only — 546,201 rows
- Whole capture days are kept intact; no within-day timestamp is fabricated

The temporal split is fixed by capture-day provenance and does not change across seeds. Only
the reference split varies per seed.

**Class coverage in temporal test (Friday):** The Friday test partition contains only five of
the 27 class labels:

| Class | Train count | Test count |
|---|---|---|
| BENIGN | 1,288,473 | 287,198 |
| Botnet | 0 | 736 |
| Botnet - Attempted | 0 | 4,064 |
| DDoS | 0 | 95,144 |
| Portscan | 0 | 159,059 |

Four attack classes (Botnet, Botnet - Attempted, DDoS, Portscan) are genuinely absent from
Monday–Thursday training. The remaining 22 attack classes appear only in Monday–Thursday
and have zero test-set rows on Friday. Both groups produce zero recall in temporal evaluation.

This is a dataset characteristic, not a code defect. It means temporal Macro-F1 is
structurally bounded near 1/27 ≈ 0.037, because only BENIGN recall contributes.

Implementation: `src/03_preprocessing_models.py`, capture-day branching in the temporal split
block, with per-class diagnostic table printed at runtime.

---

## Binary and multiclass tasks

Both tasks are run independently for every seed and condition.

**Multiclass:** raw string class labels are used directly. XGBoost requires integer labels
internally; `np.unique(..., return_inverse=True)` encodes them and predictions are decoded
back to strings. 27 classes total.

**Binary:** BENIGN maps to 0, all other labels map to 1. Implemented in `prepare_binary_labels()`
in `src/03_preprocessing_models.py`.

The two tasks share the same train/test index partitions and the same fitted imputer per split,
but each task's label array is prepared independently. Binary and multiclass results are kept
in separate rows and never aggregated together.

---

## Models

Three estimators are evaluated. Hyperparameters below are the paper-mode values (used in the
completed five-seed run).

**Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- solver: lbfgs
- max_iter: 300
- tol: 1e-3
- random_state: per-seed value

Note: `warnings.filterwarnings("ignore")` is set globally, so convergence warnings are
suppressed. Convergence on the full 1.67M-row training partition was not independently verified.

**Random Forest** (`sklearn.ensemble.RandomForestClassifier`)
- n_estimators: 65
- max_depth: 18
- min_samples_leaf: 2
- max_samples: 0.65
- class_weight: balanced_subsample
- n_jobs: max(1, min(2, cpu_count))
- random_state: per-seed value

**XGBoost** (`xgboost.XGBClassifier`)
- n_estimators: 85
- max_depth: 5
- learning_rate: 0.08
- subsample: 0.75
- colsample_bytree: 0.75
- min_child_weight: 2
- max_bin: 64
- tree_method: hist
- binary objective: binary:logistic, eval_metric: logloss
- multiclass objective: multi:softmax, eval_metric: mlogloss
- n_jobs: max(1, min(2, cpu_count))
- random_state: per-seed value

---

## Six evaluation conditions

For each seed, the experiment evaluates six conditions across both splits:

| Condition name | Split | Feature missingness |
|---|---|---|
| Reference | Reference (stratified 80/20) | None |
| Reference+10% Missingness | Reference | 10% at test time |
| Reference+20% Missingness | Reference | 20% at test time |
| Temporal | Temporal (Mon–Thu / Fri) | None |
| Temporal+10% Missingness | Temporal | 10% at test time |
| Temporal+20% Missingness | Temporal | 20% at test time |

### Feature missingness protocol

Missingness is injected only into the test/evaluation matrix, never into training data. This
makes it a controlled test-time stress factor.

For each feature column independently, a Bernoulli mask with probability equal to the
missingness rate is applied to the test matrix copy. The original test matrix is not modified.
The resulting NaN values are then passed through the already-fitted train-derived median
imputer.

Implementation: `inject_feature_missingness()` in `src/03_preprocessing_models.py`, using
`np.random.default_rng(seed)` for reproducibility. The same model fitted on the clean training
partition is reused for all three stress levels (0%, 10%, 20%).

### Leakage prevention

- The median imputer is fitted on the training partition only (`fit_train_imputer()` in
  `src/03_preprocessing_models.py`), then applied to test data via `transform_with_imputer()`.
- The StandardScaler for Logistic Regression is fitted on the imputed training data only and
  applied to test data.
- No label, timestamp, identifier, or prohibited feature leaks into model inputs.
- The temporal split is constructed from capture-day provenance before any preprocessing step.

---

## Five-seed design

Seeds: 42, 52, 62, 72, 82

Seed 42 runs as the primary experiment in `src/04_experiments.py`. Seeds 52, 62, 72, and 82
run in `run_repeated_seeds()` inside `src/06_repeated_and_figures.py`, which is only called
when `PAPER_MODE = True`.

Seed 42 is not re-executed in the repeated-seed loop. The `run_repeated_seeds()` function
starts its loop at `SEEDS[1:]` = [52, 62, 72, 82] and prepends the seed-42 results from the
primary run. Seed keys in the combined report dictionary include the seed value, preventing
any collision.

Each seed independently controls:
- The reference split (`random_state=seed` in `train_test_split`)
- The stratified training-sample selection (when applicable)
- Each model's `random_state`
- The missingness injection RNG (`np.random.default_rng(seed)`)

The temporal split does not change across seeds because it is determined by capture-day
provenance, not by a random seed.

---

## Metrics

Computed by `evaluate_predictions()` in `src/03_preprocessing_models.py` for every
condition, task, model, and seed.

| Metric | Definition |
|---|---|
| Accuracy | Fraction of correctly classified samples |
| Macro Precision | Unweighted mean of per-class precision |
| Macro Recall | Unweighted mean of per-class recall |
| Macro F1 | Unweighted mean of per-class F1 |
| Weighted F1 | Class-frequency-weighted mean of per-class F1 |
| FPR | One-vs-rest macro-averaged false positive rate |
| Per-class Recall | Dict mapping each class label to its recall value |
| Confusion Matrix | Full K×K confusion matrix |

`zero_division=0` is used throughout `sklearn` metric calls, so classes with no predicted or
true samples contribute 0.0 rather than raising an error.

**FPR calculation** (`calculate_fpr_from_cm()` in `src/03_preprocessing_models.py`):
For each class i, FPR_i = FP_i / (FP_i + TN_i). The macro-averaged FPR is the unweighted
mean across all classes.

---

## Degradation formulas

Relative degradation for higher-is-better metrics (Accuracy, Precision, Recall, F1):

```
Degradation = (M_Reference - M_Condition) / M_Reference
```

Positive degradation means the condition performs worse than reference.

Relative degradation for FPR (lower is better):

```
Degradation = (M_Condition - M_Reference) / M_Reference
```

Positive degradation means FPR increased relative to reference.

Zero denominators are replaced with NaN (`replace(0, np.nan)`), so degradation is undefined
when the reference metric is zero. No zero denominators were observed in the completed run.

Implementation: `degradation_from_summary()` in `src/06_repeated_and_figures.py`.
Outputs: `results/degradation_macro_f1.csv`, `degradation_macro_recall.csv`,
`degradation_fpr.csv`.

---

## Interaction formula

The interaction term isolates the non-additive component of combined stress:

```
Interaction = D_Combined - (D_Temporal + D_Missingness)
```

Where all three degradation values are computed from the same relative degradation formula
above, using five-seed mean Macro-F1.

A positive interaction means combined stress is worse than the sum of individual stressors.
A negative interaction means combined stress is less damaging than expected additively, which
can reflect ceiling or floor effects.

Implementation: `interaction_from_degradation()` in `src/06_repeated_and_figures.py`.
Output: `results/interaction_results.csv` (12 rows: 2 tasks × 3 models × 2 missingness rates).

---

## Confidence intervals

95% t-based confidence intervals are computed from the five-seed repeated results.

```
mean ± t(0.975, df=4) × std / sqrt(5)
```

Where std is the sample standard deviation with Bessel correction (ddof=1) and the critical
value uses the Student-t distribution with df = n - 1 = 4.

Implementation: `mean_ci()` in `src/06_repeated_and_figures.py`, using `scipy.stats.t`.
Output: `results/repeated_seed_summary.csv` (all 36 rows have N=5, all CI columns populated).

The LR Temporal CI width is effectively zero because temporal Macro-F1 is structurally
determined by the class distribution of the Friday test set and does not vary across seeds.
This is correct behaviour, not a defect.

---

## Scientific limitations

1. **Temporal Macro-F1 is structurally bounded near 1/27.** The Friday test partition contains
   only BENIGN and four attack classes. The other 22 attack classes have zero Friday test rows.
   All models predict 0.0 recall for those 22 classes. Temporal Macro-F1 reflects class absence,
   not model failure on seen attack types.

2. **The temporal evaluation does not constitute proof of concept drift.** It demonstrates
   performance under a specific chronological holdout. AGENTS.md rule 7 prohibits describing
   this as concept drift evidence.

3. **LR convergence is not verified.** Convergence warnings are suppressed globally. The
   reported LR metrics reflect the model state after 300 iterations regardless of convergence.

4. **Single dataset.** All findings are specific to CICIDS2017. Cross-dataset generalization
   is not claimed.

5. **Five seeds are not independent datasets.** Between-seed variance reflects the reference
   split randomness and model initialisation variability only.

6. **N_JOBS is capped at 2.** RF and XGBoost training used at most two parallel workers,
   which is conservative for multi-core machines.
