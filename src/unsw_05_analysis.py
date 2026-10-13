# ============================================================================
# UNSW-NB15 Reliability Evaluation Pipeline — Analysis & Degradation (05)
# ============================================================================

import numpy as np
import pandas as pd

# ============================================================================
# 1. DEGRADATION ANALYSIS TABLES (SEED 42 PREVIEW)
# ============================================================================

def degradation_table(results, metric="Macro F1"):
    """
    Computes relative degradation relative to the 'Reference' condition.
    Positive value indicates degraded performance.
    Zero denominators are replaced with NaN (never fabricated).
    """
    if results.empty:
        return pd.DataFrame()

    base = (
        results[results["Condition"] == "Reference"]
        [["Task", "Model", metric]]
        .rename(columns={metric: "Reference"})
    )
    merged = results.merge(base, on=["Task", "Model"], how="left")

    if metric == "FPR":
        # For FPR, higher is worse: Degradation = (Condition - Reference) / Reference
        merged["Degradation"] = (
            (merged[metric] - merged["Reference"]) /
            merged["Reference"].replace(0, np.nan)
        )
    else:
        # Higher is better: Degradation = (Reference - Condition) / Reference
        merged["Degradation"] = (
            (merged["Reference"] - merged[metric]) /
            merged["Reference"].replace(0, np.nan)
        )

    cols = ["Task", "Model", "Condition", metric, "Reference", "Degradation"]
    if "Seed" in merged.columns:
        cols.append("Seed")
    return merged[cols].copy()

if not results_all.empty:
    primary_degradation_f1 = degradation_table(results_all, "Macro F1")
    primary_degradation_recall = degradation_table(results_all, "Macro Recall")
    primary_degradation_fpr = degradation_table(results_all, "FPR")
    print("\nSeed-42 Macro-F1 Degradation Preview:")
    print(primary_degradation_f1[primary_degradation_f1["Task"] == "multiclass"].to_string(index=False))
else:
    primary_degradation_f1 = pd.DataFrame()
    primary_degradation_recall = pd.DataFrame()
    primary_degradation_fpr = pd.DataFrame()

# ============================================================================
# 2. ATTACK-LEVEL RECALL ANALYSIS
# ============================================================================

def attack_recall_table(reports):
    """Formats per-class recall into structured tabular format."""
    rows = []
    for (task, condition, model_name), result in reports.items():
        if task != "multiclass":
            continue
        for cls, recall in result["Per-class Recall"].items():
            rows.append({
                "Task": task,
                "Condition": condition,
                "Model": model_name,
                "Class": cls,
                "Recall": float(recall),
                "Seed": int(result["Seed"]),
            })
    return pd.DataFrame(rows)

def attack_recall_condition_degradation(attack_df):
    """Computes per-attack degradation relative to reference condition."""
    if attack_df.empty:
        return pd.DataFrame()

    ref_col = "Recall" if "Recall" in attack_df.columns else "Mean_Recall"
    base = (
        attack_df[attack_df["Condition"] == "Reference"]
        [["Model", "Class", ref_col]]
        .rename(columns={ref_col: "Reference_Recall"})
    )

    out = attack_df.merge(base, on=["Model", "Class"], how="left")
    out["Recall_Degradation"] = out["Reference_Recall"] - out[ref_col]
    out["Relative_Recall_Degradation"] = (
        out["Recall_Degradation"] / out["Reference_Recall"].replace(0, np.nan)
    )
    return out

attack_recall = attack_recall_table(reports_all) if reports_all else pd.DataFrame()
attack_recall_degradation = (
    attack_recall_condition_degradation(attack_recall)
    if not attack_recall.empty else pd.DataFrame()
)

if not attack_recall_degradation.empty:
    print("\nAttack-Level Recall Degradation (Largest Drops, Seed 42):")
    top_drops = (
        attack_recall_degradation[attack_recall_degradation["Condition"] != "Reference"]
        .sort_values("Recall_Degradation", ascending=False)
        .head(15)
        [["Model", "Condition", "Class", "Reference_Recall", "Recall", "Recall_Degradation"]]
    )
    print(top_drops.to_string(index=False))

print("\nAnalysis stage (05) ready.\n")
