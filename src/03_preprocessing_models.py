# Auto-extracted from CICIDS2017_Reliability_Framework_Final_Optimized_v6_FIXED.ipynb
# Cell order is preserved intentionally for reproducibility.


# ================= NOTEBOOK CELL 13 =================
# 11. Leakage-safe preprocessing
#
# There must be no missing values in the matrices actually passed to the estimators.
#
# For each split:
# - fit the median imputer on training data only,
# - transform train and test,
# - replace any residual non-finite values defensively,
# - fit the scaler on training data only for Logistic Regression.
#
# For missingness stress, NaNs are injected only into the test matrix, then passed
# through the already-fitted train-derived imputer. This keeps the stress factor
# controlled and prevents leakage.


# ================= NOTEBOOK CELL 14 =================
# 11. Leakage-safe preprocessing + optimized model factories
from sklearn.base import clone
from scipy.stats import t

def fit_train_imputer(X_train_raw):
    """Fit the imputer exactly once for a split and reuse it across models/tasks."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    Xtr = imputer.fit_transform(X_train_raw).astype(np.float32, copy=False)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(Xtr).all():
        raise ValueError("Non-finite values remain after train-only imputation.")
    return Xtr, imputer

def transform_with_imputer(imputer, X_raw):
    out = imputer.transform(X_raw).astype(np.float32, copy=False)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite values remain after train-fitted imputation.")
    return out

def make_lr(seed):
    return LogisticRegression(
        # max_iter=120 is sufficient for quick validation.
        # In PAPER_MODE with full data, lbfgs needs more iterations to converge.
        max_iter=120 if not PAPER_MODE else 300,
        solver="lbfgs",
        tol=1e-3,
        random_state=seed
    )

def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=45 if not PAPER_MODE else 65,
        max_depth=18,
        min_samples_leaf=2,
        max_samples=0.65,
        n_jobs=N_JOBS,
        random_state=seed,
        class_weight="balanced_subsample"
    )

def make_xgb(seed, binary=False):
    if not XGB_AVAILABLE:
        raise RuntimeError("XGBoost is required for this experiment.")

    common = dict(
        n_estimators=55 if not PAPER_MODE else 85,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=2,
        max_bin=64,
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=seed,
        verbosity=0
    )

    if binary:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common
        )

    return XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        **common
    )

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]
print("Optimized model factories ready.")


# ================= NOTEBOOK CELL 15 =================
# 12. Build reference and capture-day chronological splits using indices only
n = len(y_sorted)

class_counts_for_split = pd.Series(y_sorted).value_counts()
if class_counts_for_split.min() < 2:
    rare = class_counts_for_split[class_counts_for_split < 2].to_dict()
    raise ValueError(
        "Reference stratified split cannot be formed because some classes have "
        f"fewer than 2 rows: {rare}. The notebook will not silently delete classes."
    )

all_idx = np.arange(n, dtype=np.int64)
ref_train_idx, ref_test_idx = train_test_split(
    all_idx,
    test_size=0.20,
    random_state=SEED,
    stratify=y_sorted
)

if ROW_TIMESTAMP_MODE:
    # Row-level chronological 80/20 split.
    cut = int(0.80 * n)
    temporal_train_idx = np.arange(cut, dtype=np.int64)
    temporal_test_idx = np.arange(cut, n, dtype=np.int64)
else:
    # File/day-level chronological split. Whole capture days remain intact.
    # Monday->Thursday are training and Friday is held out when Friday exists.
    # This prevents within-day leakage and does not fabricate row timestamps.
    day_sequence = [0, 1, 2, 3, 4]
    available_days = sorted(np.unique(source_day_ids).tolist())

    if len(available_days) < 2:
        raise ValueError("At least two capture days are required for temporal evaluation.")

    test_day = available_days[-1]
    train_mask = source_day_ids[ordered_idx] < test_day
    test_mask = source_day_ids[ordered_idx] == test_day

    temporal_train_idx = np.flatnonzero(train_mask).astype(np.int64)
    temporal_test_idx = np.flatnonzero(test_mask).astype(np.int64)

    if len(temporal_train_idx) == 0 or len(temporal_test_idx) == 0:
        raise ValueError("Capture-day temporal split produced an empty partition.")

print("Reference:", len(ref_train_idx), "train /", len(ref_test_idx), "test")
print(
    "Temporal:",
    len(temporal_train_idx), "train /", len(temporal_test_idx), "test",
    "| mode =", "row-level Timestamp" if ROW_TIMESTAMP_MODE else "capture-day"
)

temporal_train_classes = set(y_sorted[temporal_train_idx])
temporal_test_classes = set(y_sorted[temporal_test_idx])
unseen_temporal_classes = sorted(temporal_test_classes - temporal_train_classes)

print("Temporal test-only classes:", len(unseen_temporal_classes))
if unseen_temporal_classes:
    print("These classes are genuinely absent from temporal training:")
    print(unseen_temporal_classes)

# Per-class temporal train/test counts for the audit record.
# This separates unseen-class effects from degradation on seen classes.
all_temporal_classes = sorted(temporal_train_classes | temporal_test_classes)
print("\nPer-class temporal split counts:")
print(f"  {'Class':40s} {'Train':>8} {'Test':>8} {'Unseen':>8}")
y_temp_train = y_sorted[temporal_train_idx]
y_temp_test = y_sorted[temporal_test_idx]
for cls in all_temporal_classes:
    n_train = int(np.sum(y_temp_train == cls))
    n_test = int(np.sum(y_temp_test == cls))
    unseen_flag = "YES" if cls in unseen_temporal_classes else ""
    print(f"  {cls:40s} {n_train:>8,} {n_test:>8,} {unseen_flag:>8}")

def take_sorted_rows(indices):
    """Materialize only requested rows; no second full feature matrix is retained."""
    indices = np.asarray(indices, dtype=np.int64)
    return X_matrix[ordered_idx[indices]]


# ================= NOTEBOOK CELL 16 =================
# 14. Training-size policy — full data for paper, stratified bounded data for validation
def make_training_indices(indices, y_values, max_samples, seed):
    """
    PAPER_MODE: return the complete training partition.

    QUICK_VALIDATION: use a reproducible stratified sample so the fast run
    preserves approximate class proportions. This is only a pipeline check,
    never the final paper protocol.
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
    # Largest-remainder allocation so the selected sample approximately follows
    # the original class distribution while guaranteeing at least one example
    # per class when the budget permits.
    if max_samples < len(unique):
        raise ValueError("FAST_TRAIN_SAMPLES must be >= number of classes.")

    target = max_samples * counts / counts.sum()
    quotas = np.floor(target).astype(int)
    quotas = np.maximum(quotas, 1)

    # Reduce if minimum-one allocation exceeded the budget.
    while quotas.sum() > max_samples:
        reducible = np.where(quotas > 1)[0]
        if len(reducible) == 0:
            break
        # Remove from the class with the largest quota relative to its target.
        j = reducible[np.argmax(quotas[reducible] - target[reducible])]
        quotas[j] -= 1

    # Distribute remaining slots by largest fractional remainder.
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

ref_fit_idx = make_training_indices(
    ref_train_idx, y_sorted, FAST_TRAIN_SAMPLES, SEED
)
temp_fit_idx = make_training_indices(
    temporal_train_idx, y_sorted, FAST_TRAIN_SAMPLES, SEED
)

print("Reference fit rows:", len(ref_fit_idx))
print("Temporal fit rows:", len(temp_fit_idx))
print("Paper mode:", PAPER_MODE)
print("Fast sampling:", "stratified" if FAST_SAMPLING_STRATIFIED else "uniform")


# ================= NOTEBOOK CELL 17 =================
# 14. Evaluation metrics
def calculate_fpr_from_cm(cm):
    cm = np.asarray(cm)
    total = cm.sum()
    fprs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = fp + tn
        fprs.append(fp / denom if denom else 0.0)
    return float(np.mean(fprs))

def evaluate_predictions(y_true, y_pred, labels):
    labels = list(labels)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "Accuracy": float(acc),
        "Macro Precision": float(np.mean(p)),
        "Macro Recall": float(np.mean(r)),
        "Macro F1": float(np.mean(f1)),
        "Weighted F1": float(
            precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )[2]
        ),
        "FPR": calculate_fpr_from_cm(cm),
        "Confusion Matrix": cm,
        "Per-class Recall": dict(zip(labels, r))
    }


# ================= NOTEBOOK CELL 18 =================
# 15. Memory-efficient feature missingness stress
def inject_feature_missingness(X, rate, seed):
    """
    Inject independent feature-value missingness into the evaluation matrix.
    The original matrix is never modified.

    A column-wise 1-D mask avoids allocating a full rows x features boolean matrix.
    """
    if rate <= 0:
        return X

    rng = np.random.default_rng(seed)
    X_stress = X.copy()

    for j in range(X_stress.shape[1]):
        mask = rng.random(X_stress.shape[0]) < rate
        X_stress[mask, j] = np.nan

    return X_stress

def apply_fitted_imputer(imputer, X):
    return transform_with_imputer(imputer, X)


# ================= NOTEBOOK CELL 19 =================
# 16. Fit model using a split-level, already-fitted imputer
def fit_model_for_split(
    model_name, X_train_imputed, imputer, y_train, seed, binary=False
):
    if model_name == "Logistic Regression":
        model = make_lr(seed)
    elif model_name == "Random Forest":
        model = make_rf(seed)
    elif model_name == "XGBoost":
        model = make_xgb(seed, binary=binary)
    else:
        raise ValueError("Unknown model: " + model_name)

    X_train = X_train_imputed
    scaler = None

    if model_name == "Logistic Regression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)

    label_classes = None
    if model_name == "XGBoost" and not binary:
        label_classes, y_encoded = np.unique(
            np.asarray(y_train).astype(str), return_inverse=True
        )
        model.fit(X_train, y_encoded)
    else:
        model.fit(X_train, y_train)

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "label_classes": label_classes,
        "binary": binary,
        "model_name": model_name,
    }

def transform_test(fitted, X_test_raw):
    X_test = apply_fitted_imputer(fitted["imputer"], X_test_raw)

    if fitted["scaler"] is not None:
        X_test = fitted["scaler"].transform(X_test).astype(np.float32, copy=False)

    if not np.isfinite(X_test).all():
        raise ValueError("Non-finite values remain in the estimator input.")

    return X_test


# ================= NOTEBOOK CELL 20 =================
# 17. Safe multiclass/binary prediction wrapper
def prepare_binary_labels(y):
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.number):
        unique = set(np.unique(arr).tolist())
        if unique.issubset({0, 1}):
            return arr.astype(np.int32)
    return np.asarray(
        [0 if str(v).strip().casefold() == "benign" else 1 for v in arr],
        dtype=np.int32
    )

def evaluate_fitted(
    fitted, X_test_raw, y_test, labels, condition, model_name, binary=False,
    missingness=0.0, seed=42
):
    X_eval_raw = X_test_raw
    if missingness > 0:
        X_eval_raw = inject_feature_missingness(X_test_raw, missingness, seed)

    X_eval = transform_test(fitted, X_eval_raw)
    raw_pred = fitted["model"].predict(X_eval)

    if binary:
        y_true_eval = prepare_binary_labels(y_test)
        labels_eval = [0, 1]
        y_pred = np.asarray(raw_pred).astype(int)
    else:
        y_true_eval = np.asarray(y_test).astype(str)
        labels_eval = list(labels)

        if model_name == "XGBoost" and fitted.get("label_classes") is not None:
            encoded_pred = np.asarray(raw_pred).astype(int)
            known_classes = np.asarray(fitted["label_classes"]).astype(str)
            y_pred = known_classes[encoded_pred]
        else:
            y_pred = np.asarray(raw_pred).astype(str)

    metrics = evaluate_predictions(y_true_eval, y_pred, labels_eval)
    metrics.update({
        "Task": "binary" if binary else "multiclass",
        "Model": model_name,
        "Condition": condition,
        "Missingness": float(missingness),
        "Seed": int(seed),
    })
    return metrics

