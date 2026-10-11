# Auto-extracted from CICIDS2017_Reliability_Framework_Final_Optimized_v6_FIXED.ipynb
# Cell order is preserved intentionally for reproducibility.


# ================= NOTEBOOK CELL 21 =================
# 18. Experiment runner
#
# The runner explicitly separates:
# - reference vs temporal training data,
# - binary vs multiclass labels,
# - normal vs stressed test inputs.
#
# The fitted model is reused for normal, 10% missingness, and 20% missingness
# evaluation. Missingness is a test-time stress factor, not a second training protocol.


# ================= NOTEBOOK CELL 22 =================
# 18. Experiment runner — fit once per model/split; reuse split-level preprocessing
CONDITION_SPECS = [
    ("Reference", 0.0),
    ("Reference+10% Missingness", 0.10),
    ("Reference+20% Missingness", 0.20),
    ("Temporal", 0.0),
    ("Temporal+10% Missingness", 0.10),
    ("Temporal+20% Missingness", 0.20),
]

def run_one_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    y_all = y_sorted

    # Reference split changes by seed; temporal split is fixed by provenance.
    all_idx = np.arange(len(y_all), dtype=np.int64)
    ref_train_idx_s, ref_test_idx_s = train_test_split(
        all_idx,
        test_size=0.20,
        random_state=seed,
        stratify=y_all
    )

    temp_train_idx_s = temporal_train_idx
    temp_test_idx_s = temporal_test_idx

    ref_fit_idx_s = make_training_indices(
        ref_train_idx_s, y_all, FAST_TRAIN_SAMPLES, seed
    )
    temp_fit_idx_s = make_training_indices(
        temp_train_idx_s, y_all, FAST_TRAIN_SAMPLES, seed
    )

    split_indices = {
        "Reference": (ref_fit_idx_s, ref_test_idx_s),
        "Temporal": (temp_fit_idx_s, temp_test_idx_s),
    }

    # Materialize each split once and fit the median imputer once.
    prepared = {}
    for split_name, (fit_idx, test_idx) in split_indices.items():
        X_train_raw = take_sorted_rows(fit_idx)
        X_test_raw = take_sorted_rows(test_idx)
        X_train_imputed, imputer = fit_train_imputer(X_train_raw)

        prepared[split_name] = {
            "fit_idx": fit_idx,
            "test_idx": test_idx,
            "X_train_imputed": X_train_imputed,
            "X_test_raw": X_test_raw,
            "imputer": imputer,
        }
        del X_train_raw
        gc.collect()

    rows = []
    reports = {}

    for task in ["multiclass", "binary"]:
        binary = task == "binary"
        y_task = prepare_binary_labels(y_all) if binary else y_all
        labels = [0, 1] if binary else sorted(np.unique(y_task))

        for split_name in ["Reference", "Temporal"]:
            p = prepared[split_name]
            fit_idx = p["fit_idx"]
            test_idx = p["test_idx"]
            y_train = y_task[fit_idx]
            y_test = y_task[test_idx]

            print(
                f"[Seed {seed}] {task} | {split_name} | "
                f"train={len(fit_idx):,} test={len(test_idx):,}"
            )

            for model_name in MODEL_NAMES:
                print(f"  -> {model_name}")

                # One fit per model/split/task. All three stress levels reuse it.
                fitted = fit_model_for_split(
                    model_name=model_name,
                    X_train_imputed=p["X_train_imputed"],
                    imputer=p["imputer"],
                    y_train=y_train,
                    seed=seed,
                    binary=binary,
                )

                for condition, missingness in CONDITION_SPECS:
                    if condition.startswith(split_name):
                        result = evaluate_fitted(
                            fitted=fitted,
                            X_test_raw=p["X_test_raw"],
                            y_test=y_test,
                            labels=labels,
                            condition=condition,
                            model_name=model_name,
                            binary=binary,
                            missingness=missingness,
                            seed=seed,
                        )
                        rows.append(result)

                        # Keep every multiclass condition for attack-level analysis.
                        if task == "multiclass":
                            reports[(task, condition, model_name)] = result

                del fitted
                gc.collect()

    del prepared
    gc.collect()

    return pd.DataFrame(rows), reports


# ================= NOTEBOOK CELL 23 =================
# 19. Run the primary experiment
results_all = pd.DataFrame()
reports_all = {}

if RUN_EXPERIMENTS:
    results_all, reports_all = run_one_seed(SEED)
    print("\nPrimary experiment complete.")

    display(
        results_all[
            ["Task", "Model", "Condition", "Missingness", "Accuracy",
             "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "FPR", "Seed"]
        ].sort_values(["Task", "Model", "Condition"])
    )
else:
    print("RUN_EXPERIMENTS=False: experiment execution skipped.")

