# ============================================================================
# UNSW-NB15 Reliability Evaluation Pipeline — Repeated Seeds & Figures (06)
# ============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

# ============================================================================
# 1. FIVE-SEED REPEATED RUNNER
# ============================================================================

def run_unsw_repeated_seeds():
    all_runs = [results_all.copy()] if not results_all.empty else []
    all_report_runs = [reports_all.copy()] if reports_all else []

    # Seed 42 is already executed; run remaining seeds in SEEDS[1:]
    for s in SEEDS[1:]:
        print(f"\n==================== UNSW-NB15 SEED {s} ====================")
        run_df, run_reports = run_unsw_one_seed(s)
        all_runs.append(run_df)
        all_report_runs.append(run_reports)

    if not all_runs:
        return pd.DataFrame(), {}

    combined_reports = {}
    for report_dict in all_report_runs:
        for (task, condition, model_name), result in report_dict.items():
            key = (task, condition, model_name, int(result["Seed"]))
            combined_reports[key] = result

    return pd.concat(all_runs, ignore_index=True), combined_reports

def mean_ci(series, confidence=CI_LEVEL):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(np.mean(values))
    if n == 1:
        return mean, np.nan, np.nan, np.nan, 1
    std = float(np.std(values, ddof=1))
    critical = float(t.ppf((1 + confidence) / 2, df=n - 1))
    half_width = critical * std / np.sqrt(n)
    return mean, std, mean - half_width, mean + half_width, n

def summarize_repeated_results(results):
    rows = []
    for keys, g in results.groupby(["Task", "Model", "Condition"]):
        task, model, condition = keys
        mf1, sf1, lo_f1, hi_f1, n = mean_ci(g["Macro F1"])
        mr, sr, lo_r, hi_r, _ = mean_ci(g["Macro Recall"])
        fpr, sfpr, lo_fpr, hi_fpr, _ = mean_ci(g["FPR"])

        rows.append({
            "Task": task,
            "Model": model,
            "Condition": condition,
            "Mean_Macro_F1": mf1,
            "Std_Macro_F1": sf1,
            "CI95_Macro_F1_Low": lo_f1,
            "CI95_Macro_F1_High": hi_f1,
            "Mean_Macro_Recall": mr,
            "Std_Macro_Recall": sr,
            "CI95_Macro_Recall_Low": lo_r,
            "CI95_Macro_Recall_High": hi_r,
            "Mean_FPR": fpr,
            "Std_FPR": sfpr,
            "CI95_FPR_Low": lo_fpr,
            "CI95_FPR_High": hi_fpr,
            "N": n,
        })
    return pd.DataFrame(rows)

if FINAL_MODE and RUN_EXPERIMENTS:
    print("\nRunning repeated seeds protocol for paper mode...")
    repeated_results, repeated_reports = run_unsw_repeated_seeds()
else:
    repeated_results = results_all.copy() if not results_all.empty else pd.DataFrame()
    repeated_reports = {}

repeated_summary = (
    summarize_repeated_results(repeated_results)
    if not repeated_results.empty else pd.DataFrame()
)

# Mean condition metrics for degradation & interaction
if FINAL_MODE and not repeated_results.empty:
    final_condition_metrics = (
        repeated_results
        .groupby(["Task", "Model", "Condition"], as_index=False)
        .agg(
            Accuracy=("Accuracy", "mean"),
            Macro_Precision=("Macro Precision", "mean"),
            Macro_Recall=("Macro Recall", "mean"),
            Macro_F1=("Macro F1", "mean"),
            Weighted_F1=("Weighted F1", "mean"),
            FPR=("FPR", "mean"),
        )
    )
else:
    final_condition_metrics = results_all.rename(
        columns={
            "Macro Precision": "Macro_Precision",
            "Macro Recall": "Macro_Recall",
            "Macro F1": "Macro_F1",
            "Weighted F1": "Weighted_F1",
        }
    )[[
        "Task", "Model", "Condition", "Accuracy",
        "Macro_Precision", "Macro_Recall", "Macro_F1",
        "Weighted_F1", "FPR"
    ]].copy()

# ============================================================================
# 2. FINAL DEGRADATION & INTERACTION TABLES
# ============================================================================

def degradation_from_summary(summary, metric_col):
    base = (
        summary[summary["Condition"] == "Reference"]
        [["Task", "Model", metric_col]]
        .rename(columns={metric_col: "Reference"})
    )
    merged = summary.merge(base, on=["Task", "Model"], how="left")

    if metric_col == "FPR":
        merged["Degradation"] = (
            (merged[metric_col] - merged["Reference"]) /
            merged["Reference"].replace(0, np.nan)
        )
    else:
        merged["Degradation"] = (
            (merged["Reference"] - merged[metric_col]) /
            merged["Reference"].replace(0, np.nan)
        )

    return merged[["Task", "Model", "Condition", metric_col, "Reference", "Degradation"]].copy()

final_degradation_f1 = degradation_from_summary(final_condition_metrics, "Macro_F1")
final_degradation_recall = degradation_from_summary(final_condition_metrics, "Macro_Recall")
final_degradation_fpr = degradation_from_summary(final_condition_metrics, "FPR")

def interaction_from_degradation(deg):
    out = []
    for (task, model), g in deg.groupby(["Task", "Model"]):
        lookup = dict(zip(g["Condition"], g["Degradation"]))
        d_temp = lookup.get("Temporal")
        for rate in ("10%", "20%"):
            d_m = lookup.get(f"Reference+{rate} Missingness")
            d_c = lookup.get(f"Temporal+{rate} Missingness")
            if d_temp is not None and d_m is not None and d_c is not None:
                out.append({
                    "Task": task,
                    "Model": model,
                    "Missingness": rate,
                    "Temporal Degradation": d_temp,
                    "Missingness Degradation": d_m,
                    "Combined Degradation": d_c,
                    "Interaction": d_c - (d_temp + d_m),
                })
    return pd.DataFrame(out)

final_interaction_results = interaction_from_degradation(final_degradation_f1)

# Backward-compatible names
degradation_f1 = final_degradation_f1
degradation_recall = final_degradation_recall
degradation_fpr = final_degradation_fpr
interaction_results = final_interaction_results

# ============================================================================
# 3. ATTACK-LEVEL AGGREGATION ACROSS SEEDS
# ============================================================================

def attack_table_from_reports_aggregated(report_dict):
    rows = []
    for key, result in report_dict.items():
        if len(key) == 4:
            task, condition, model_name, seed = key
        else:
            task, condition, model_name = key
            seed = result["Seed"]
        if task != "multiclass":
            continue
        for cls, recall in result["Per-class Recall"].items():
            rows.append({
                "Task": task,
                "Condition": condition,
                "Model": model_name,
                "Class": cls,
                "Recall": float(recall),
                "Seed": int(seed),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(["Task", "Condition", "Model", "Class"], as_index=False)
        .agg(
            Mean_Recall=("Recall", "mean"),
            Std_Recall=("Recall", "std"),
            N=("Recall", "count")
        )
    )

if FINAL_MODE and repeated_reports:
    attack_recall_final = attack_table_from_reports_aggregated(repeated_reports)
else:
    attack_recall_final = attack_table_from_reports_aggregated(reports_all) if reports_all else pd.DataFrame()

if not attack_recall_final.empty:
    ref = (
        attack_recall_final[attack_recall_final["Condition"] == "Reference"]
        [["Model", "Class", "Mean_Recall"]]
        .rename(columns={"Mean_Recall": "Reference_Recall"})
    )
    attack_recall_final_degradation = attack_recall_final.merge(ref, on=["Model", "Class"], how="left")
    attack_recall_final_degradation["Recall_Degradation"] = (
        attack_recall_final_degradation["Reference_Recall"] - attack_recall_final_degradation["Mean_Recall"]
    )
    attack_recall_final_degradation["Relative_Recall_Degradation"] = (
        attack_recall_final_degradation["Recall_Degradation"] /
        attack_recall_final_degradation["Reference_Recall"].replace(0, np.nan)
    )
else:
    attack_recall_final_degradation = pd.DataFrame()

attack_recall = attack_recall_final
attack_recall_degradation = attack_recall_final_degradation

# ============================================================================
# 4. PUBLICATION FIGURES GENERATION
# ============================================================================

if not final_condition_metrics.empty:
    print("\nGenerating publication figures...")

    # Figure 1 — Reference vs Temporal Macro-F1
    plot_df = final_condition_metrics[
        (final_condition_metrics["Task"] == "multiclass") &
        (final_condition_metrics["Condition"].isin(["Reference", "Temporal"]))
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in MODEL_NAMES:
        g = plot_df[plot_df["Model"] == model]
        ax.plot(g["Condition"], g["Macro_F1"], marker="o", linewidth=2, label=model)
    ax.set_title("UNSW-NB15: Reference vs Chronological Macro-F1 (Multiclass)")
    ax.set_ylabel("Macro-F1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure1_reference_vs_temporal_macro_f1.png", dpi=180)
    plt.close(fig)

    # Figure 2 — Degradation by Condition
    d = degradation_f1[
        (degradation_f1["Task"] == "multiclass") &
        (~degradation_f1["Condition"].eq("Reference"))
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in MODEL_NAMES:
        g = d[d["Model"] == model]
        ax.plot(g["Condition"], g["Degradation"], marker="o", linewidth=2, label=model)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("UNSW-NB15: Relative Macro-F1 Degradation Across Conditions")
    ax.set_ylabel("Relative Degradation")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure2_degradation.png", dpi=180)
    plt.close(fig)

    # Figure 3 — Interaction Effects
    if not final_interaction_results.empty:
        g = final_interaction_results[final_interaction_results["Task"] == "multiclass"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model in MODEL_NAMES:
            z = g[g["Model"] == model]
            ax.plot(z["Missingness"], z["Interaction"], marker="s", linewidth=2, label=model)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title("UNSW-NB15: Temporal x Missingness Interaction (Multiclass)")
        ax.set_ylabel("Non-Additive Interaction Term")
        ax.set_xlabel("Missingness Rate")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure3_interaction.png", dpi=180)
        plt.close(fig)

    # Figure 4 — Attack-Level Recall Degradation
    if not attack_recall_degradation.empty:
        top_drops = (
            attack_recall_degradation[attack_recall_degradation["Condition"] != "Reference"]
            .sort_values("Recall_Degradation", ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{r['Model'][:3]}-{r['Class']}\n({r['Condition'][:12]})" for _, r in top_drops.iterrows()]
        bars = ax.bar(range(len(top_drops)), top_drops["Recall_Degradation"], color="#e11d48", alpha=0.85)
        ax.set_xticks(range(len(top_drops)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Recall Degradation (Reference - Stressed)")
        ax.set_title("UNSW-NB15: Largest Attack-Level Recall Drops")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure4_attack_recall_degradation.png", dpi=180)
        plt.close(fig)

    # Figure 5 — Representative Confusion Matrix (Seed 42 Best Model)
    if reports_all:
        ref_reports = [
            (m, r) for (t, c, m), r in reports_all.items()
            if t == "multiclass" and c == "Reference"
        ]
        if ref_reports:
            best_model, best_res = max(ref_reports, key=lambda x: x[1]["Macro F1"])
            cm = best_res["Confusion Matrix"]
            classes = list(best_res["Per-class Recall"].keys())
            
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
            ax.set_title(f"UNSW-NB15: {best_model} Confusion Matrix (Reference, Seed 42)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "figure5_representative_confusion_matrix.png", dpi=180)
            plt.close(fig)

    # Figure 6 — Repeated Seed Variability with 95% CIs
    if not repeated_summary.empty and repeated_summary["N"].max() >= 2:
        fig_df = repeated_summary[
            (repeated_summary["Task"] == "multiclass") &
            (repeated_summary["Condition"].isin(["Reference", "Temporal"]))
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model in MODEL_NAMES:
            g = fig_df[fig_df["Model"] == model]
            yerr = g["CI95_Macro_F1_High"] - g["Mean_Macro_F1"]
            ax.errorbar(
                g["Condition"], g["Mean_Macro_F1"], yerr=yerr,
                marker="o", capsize=5, linewidth=2, label=model
            )
        ax.set_title("UNSW-NB15: 5-Seed Macro-F1 with 95% Student-t CIs")
        ax.set_ylabel("Mean Macro-F1")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure6_repeated_seed_variability.png", dpi=180)
        plt.close(fig)

    print("Figures generated successfully.\n")
