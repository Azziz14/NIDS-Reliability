# ============================================================================
# UNSW-NB15 Reliability Evaluation Pipeline — Preprocessing & Models (03)
# ============================================================================

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# ============================================================================
# 1. LEAKAGE-SAFE PREPROCESSOR FACTORIES
# ============================================================================

def fit_train_preprocessors(df_train, cat_cols, num_cols):
    """
    Fits OneHotEncoder and SimpleImputer STRICTLY on the training partition.
    Unseen categories in test data will produce an all-zeros vector (zero leakage).
    """
    # 1. Categorical encoder
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(df_train[cat_cols])
    
    # 2. Median imputer for numeric columns
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(df_train[num_cols])
    
    cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    all_feature_names = list(num_cols) + cat_feature_names
    
    return {
        "encoder": encoder,
        "imputer": imputer,
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
        "feature_names": all_feature_names,
    }

def transform_feature_matrix(preprocessors, df_input):
    """
    Transforms any dataframe (train, clean test, or stressed test) using
    the pre-fitted training transformers. Zero NaN/Inf permitted in output.
    """
    imputer = preprocessors["imputer"]
    encoder = preprocessors["encoder"]
    num_cols = preprocessors["num_cols"]
    cat_cols = preprocessors["cat_cols"]
    
    # Impute numeric
    X_num = imputer.transform(df_input[num_cols]).astype(np.float32, copy=False)
    
    # Encode categorical (unseen categories are safely mapped to zeros)
    X_cat = encoder.transform(df_input[cat_cols]).astype(np.float32, copy=False)
    
    # Concatenate
    X_out = np.hstack([X_num, X_cat])
    
    # Defensive replacement of any lingering non-finite values
    X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)
    
    if not np.isfinite(X_out).all():
        raise ValueError("Non-finite values remain after transformation.")
        
    return X_out

# ============================================================================
# 2. CONTROLLED TEST-TIME MISSINGNESS INJECTION
# ============================================================================

def inject_feature_missingness(df_eval, rate, seed, num_cols):
    """
    Injects independent Bernoulli missingness (NaNs) into numeric features of
    the test partition only. The original DataFrame is never modified.
    """
    if rate <= 0.0:
        return df_eval
        
    rng = np.random.default_rng(seed)
    df_stressed = df_eval.copy()
    
    for col in num_cols:
        mask = rng.random(len(df_stressed)) < rate
        df_stressed.loc[mask, col] = np.nan
        
    return df_stressed

# ============================================================================
# 3. OPTIMIZED MODEL FACTORIES
# ============================================================================

def make_lr(seed):
    return LogisticRegression(
        max_iter=120 if not PAPER_MODE else 300,
        solver="lbfgs",
        tol=1e-3,
        random_state=seed,
    )

def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=45 if not PAPER_MODE else 65,
        max_depth=18,
        min_samples_leaf=2,
        max_samples=0.65,
        n_jobs=N_JOBS,
        random_state=seed,
        class_weight="balanced_subsample",
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
        verbosity=0,
    )

    if binary:
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    return XGBClassifier(objective="multi:softmax", eval_metric="mlogloss", **common)

# ============================================================================
# 4. MODEL FITTING & LOGISTIC REGRESSION CONVERGENCE AUDIT
# ============================================================================

def fit_model_for_unsw(model_name, X_train, y_train, seed, binary=False):
    """
    Fits the requested estimator on preprocessed training data.
    For Logistic Regression:
      - Fits StandardScaler on X_train only.
      - Intercepts warnings to record convergence state transparently.
    """
    scaler = None
    lr_diag = None
    label_classes = None

    if model_name == "Logistic Regression":
        model = make_lr(seed)
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train).astype(np.float32, copy=False)
        
        # Actively intercept ConvergenceWarning without suppressing silently
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            model.fit(X_train_fit, y_train)
            
        n_iter_val = int(model.n_iter_[0])
        has_conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in captured_warnings)
        
        # Record explicit audit metrics
        lr_diag = {
            "n_iter": n_iter_val,
            "max_iter": int(model.max_iter),
            "converged_without_warning": bool(not has_conv_warn and n_iter_val < model.max_iter),
            "convergence_warning_raised": bool(has_conv_warn),
            "warnings_text": [str(w.message) for w in captured_warnings if issubclass(w.category, ConvergenceWarning)],
        }
        
    elif model_name == "Random Forest":
        model = make_rf(seed)
        model.fit(X_train, y_train)
        
    elif model_name == "XGBoost":
        model = make_xgb(seed, binary=binary)
        if not binary:
            label_classes, y_encoded = np.unique(np.asarray(y_train).astype(str), return_inverse=True)
            model.fit(X_train, y_encoded)
        else:
            model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return {
        "model": model,
        "scaler": scaler,
        "label_classes": label_classes,
        "binary": binary,
        "model_name": model_name,
        "lr_diag": lr_diag,
    }

# ============================================================================
# 5. EVALUATION METRICS & PREDICTION HANDLER
# ============================================================================

def calculate_fpr_from_cm(cm):
    """Calculates macro-averaged False Positive Rate across all classes from confusion matrix."""
    cm = np.asarray(cm)
    total = cm.sum()
    fprs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = fp + tn
        fprs.append(fp / denom if denom > 0 else 0.0)
    return float(np.mean(fprs))

def evaluate_predictions(y_true, y_pred, labels):
    """Computes all required metrics with zero_division=0 guardrails."""
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
        "Per-class Recall": dict(zip(labels, r)),
    }

def evaluate_fitted_model(
    fitted, preprocessors, df_test_raw, y_test, labels, condition,
    model_name, binary=False, missingness=0.0, seed=42
):
    """Transforms test data, injects missingness if requested, predicts, and evaluates."""
    # 1. Stress injection if missingness > 0
    if missingness > 0.0:
        df_eval_raw = inject_feature_missingness(
            df_test_raw, rate=missingness, seed=seed, num_cols=preprocessors["num_cols"]
        )
    else:
        df_eval_raw = df_test_raw

    # 2. Transform through train-fitted imputer and encoder
    X_eval = transform_feature_matrix(preprocessors, df_eval_raw)
    
    # 3. Model-specific scaling
    if fitted["scaler"] is not None:
        X_eval = fitted["scaler"].transform(X_eval).astype(np.float32, copy=False)
        
    # 4. Inference
    raw_pred = fitted["model"].predict(X_eval)
    
    if binary:
        y_true_eval = np.asarray(y_test).astype(int)
        y_pred = np.asarray(raw_pred).astype(int)
        labels_eval = [0, 1]
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
        "LR_Diag": fitted.get("lr_diag"),
    })
    
    return metrics

print("Preprocessing pipelines, model factories, and evaluation wrappers ready.\n")
