# ============================================================================
# UNSW-NB15 Reliability Evaluation Pipeline — Experiment Runner (04)
# ============================================================================

import gc
import random
import numpy as np
import pandas as pd

CONDITION_SPECS = [
    ("Reference", 0.0),
    ("Reference+10% Missingness", 0.10),
    ("Reference+20% Missingness", 0.20),
    ("Temporal", 0.0),
    ("Temporal+10% Missingness", 0.10),
    ("Temporal+20% Missingness", 0.20),
]

def make_unsw_training_indices(indices, y_values, max_samples, seed):
    """
    PAPER_MODE: Returns the complete, verified training partition.
    QUICK_VALIDATION: Returns a reproducible stratified sample preserving class proportions.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if PAPER_MODE or len(indices) <= max_samples:
        return indices.copy()

    y_arr = np.asarray(y_values)
    labels = y_arr[indices]
    unique, counts = np.unique(labels, return_counts=True)

    if not FAST_SAMPLING_STRATIFIED:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(indices, size=max_samples, replace=False))

    rng = np.random.default_rng(seed)
    # Largest-remainder quota allocation
    target = max_samples * counts / counts.sum()
    quotas = np.floor(target).astype(int)
    quotas = np.maximum(quotas, 1)

    while quotas.sum() > max_samples:
        reducible = np.where(quotas > 1)[0]
        if len(reducible) == 0:
            break
        j = reducible[np.argmax(quotas[reducible] - target[reducible])]
        quotas[j] -= 1

    remaining = max_samples - quotas.sum()
    fractions = target - np.floor(target)
    order = np.argsort(-fractions)
    for j in order:
        if remaining <= 0:
            break
        if quotas[j] < counts[j]:
            quotas[j] += 1
            remaining -= 1

    selected_parts = []
    for cls, quota in zip(unique, quotas):
        cls_idx = indices[labels == cls]
        quota = min(int(quota), len(cls_idx))
        selected_parts.append(rng.choice(cls_idx, size=quota, replace=False))

    selected = np.concatenate(selected_parts).astype(np.int64)
    rng.shuffle(selected)
    return selected

def run_unsw_one_seed(seed):
    """
    Executes all 36 condition combinations for a single seed.
    Reference split varies reproducibly with seed; Temporal split is fixed by provenance.
    """
    np.random.seed(seed)
    random.seed(seed)

    y_all_mc = df_master["attack_cat"].to_numpy()
    y_all_bin = df_master["label"].to_numpy()
    all_indices = np.arange(len(df_master), dtype=np.int64)

    # 1. Reference partition (stratified 80/20 by multiclass attack category)
    ref_train_raw_idx, ref_test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        stratify=y_all_mc
    )

    # 2. Temporal partition (ACCS session holdout: Session 0 train, Session 1 test)
    temp_train_raw_idx = np.flatnonzero(df_master["_session_partition"] == 0).astype(np.int64)
    temp_test_idx = np.flatnonzero(df_master["_session_partition"] == 1).astype(np.int64)

    # Apply training size policy
    ref_fit_idx = make_unsw_training_indices(ref_train_raw_idx, y_all_mc, FAST_TRAIN_SAMPLES, seed)
    temp_fit_idx = make_unsw_training_indices(temp_train_raw_idx, y_all_mc, FAST_TRAIN_SAMPLES, seed)

    split_configs = {
        "Reference": (ref_fit_idx, ref_test_idx),
        "Temporal": (temp_fit_idx, temp_test_idx),
    }

    # Materialize features and fit preprocessors once per split
    prepared_splits = {}
    for split_name, (fit_idx, test_idx) in split_configs.items():
        df_train_sub = df_master.iloc[fit_idx]
        df_test_sub = df_master.iloc[test_idx]

        # Fit preprocessors on training split only (leakage-safe)
        preprocessors = fit_train_preprocessors(
            df_train_sub, cat_cols=categorical_cols, num_cols=numeric_cols
        )
        
        # Transform clean training data
        X_train_transformed = transform_feature_matrix(preprocessors, df_train_sub)

        prepared_splits[split_name] = {
            "fit_idx": fit_idx,
            "test_idx": test_idx,
            "preprocessors": preprocessors,
            "X_train": X_train_transformed,
            "df_test_raw": df_test_sub,
        }

    rows = []
    reports = {}

    for task in ["multiclass", "binary"]:
        binary = (task == "binary")
        y_task = y_all_bin if binary else y_all_mc
        labels = [0, 1] if binary else sorted(np.unique(y_task))

        for split_name in ["Reference", "Temporal"]:
            prep = prepared_splits[split_name]
            y_train = y_task[prep["fit_idx"]]
            y_test = y_task[prep["test_idx"]]

            print(
                f"[UNSW Seed {seed}] {task:10s} | {split_name:10s} | "
                f"train={len(prep['fit_idx']):,} test={len(prep['test_idx']):,}"
            )

            for model_name in MODEL_NAMES:
                # Fit model once per (seed, task, split, model)
                fitted = fit_model_for_unsw(
                    model_name=model_name,
                    X_train=prep["X_train"],
                    y_train=y_train,
                    seed=seed,
                    binary=binary,
                )

                # Evaluate across all 3 conditions for this split (0%, 10%, 20% missingness)
                for condition, missingness in CONDITION_SPECS:
                    if condition.startswith(split_name):
                        result = evaluate_fitted_model(
                            fitted=fitted,
                            preprocessors=prep["preprocessors"],
                            df_test_raw=prep["df_test_raw"],
                            y_test=y_test,
                            labels=labels,
                            condition=condition,
                            model_name=model_name,
                            binary=binary,
                            missingness=missingness,
                            seed=seed,
                        )
                        rows.append(result)

                        if task == "multiclass":
                            reports[(task, condition, model_name)] = result

                del fitted
                gc.collect()

    del prepared_splits
    gc.collect()

    return pd.DataFrame(rows), reports

results_all = pd.DataFrame()
reports_all = {}

if RUN_EXPERIMENTS:
    print("\nStarting primary experiment on Seed 42...")
    results_all, reports_all = run_unsw_one_seed(SEED)
    print("\nPrimary seed experiment complete. Results overview:")
    
    display_cols = [
        "Task", "Model", "Condition", "Missingness",
        "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "FPR", "Seed"
    ]
    summary_preview = results_all[display_cols].sort_values(["Task", "Model", "Condition"])
    print(summary_preview.to_string(index=False))
else:
    print("RUN_EXPERIMENTS=False: execution skipped.")
