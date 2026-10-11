# Auto-extracted from CICIDS2017_Reliability_Framework_Final_Optimized_v6_FIXED.ipynb
# Cell order is preserved intentionally for reproducibility.


# ================= NOTEBOOK CELL 27 =================
# 23. Repeated seeds
#
# Paper-mode reproducibility uses seeds: 42, 52, 62, 72, 82
#
# Seed 42 is already produced by the primary experiment. To avoid training the same
# models twice, the repeated-seed stage runs only the remaining four seeds and then
# combines them with the primary seed-42 results.

# ================= NOTEBOOK CELL 28 =================
# 23. Repeated seeds + statistically correct final aggregation
def run_repeated_seeds():
    all_runs = [results_all.copy()] if not results_all.empty else []
    all_report_runs = [reports_all.copy()] if reports_all else []

    # Seed 42 is already produced by the primary run; avoid duplicate fitting.
    for s in SEEDS[1:]:
        print(f"\n========== SEED {s} ==========")
        run_df, run_reports = run_one_seed(s)
        all_runs.append(run_df)
        all_report_runs.append(run_reports)

    if not all_runs:
        return pd.DataFrame(), {}

    combined_reports = {}
    for report_dict in all_report_runs:
        # Include seed in the key so repeated conditions remain distinct.
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
    repeated_results, repeated_reports = run_repeated_seeds()
else:
    repeated_results = results_all.copy() if not results_all.empty else pd.DataFrame()
    repeated_reports = {}

repeated_summary = (
    summarize_repeated_results(repeated_results)
    if not repeated_results.empty else pd.DataFrame()
)

# Final paper tables are based on the five-seed means in PAPER_MODE.
# Validation mode uses its single seed only so the notebook remains fast.
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

    return merged[
        ["Task", "Model", "Condition", metric_col, "Reference", "Degradation"]
    ].copy()

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

# Display interaction results here (forward reference in 05_analysis.py was removed).
if not final_interaction_results.empty:
    display(final_interaction_results)
else:
    print("Interaction table is empty — check that all 6 conditions produced results.")

# Backward-compatible names used by export/figure cells.
degradation_f1 = final_degradation_f1
degradation_recall = final_degradation_recall
degradation_fpr = final_degradation_fpr
interaction_results = final_interaction_results


# Attack-level reliability analysis is also aggregated across all seeds in paper mode.
def attack_table_from_reports(report_dict, aggregate=True):
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
    if aggregate:
        return (
            df.groupby(["Task", "Condition", "Model", "Class"], as_index=False)
            .agg(
                Mean_Recall=("Recall", "mean"),
                Std_Recall=("Recall", "std"),
                N=("Recall", "count")
            )
        )
    return df

if FINAL_MODE and repeated_reports:
    attack_recall_final = attack_table_from_reports(repeated_reports, aggregate=True)
else:
    attack_recall_final = attack_table_from_reports(
        reports_all, aggregate=True
    ) if reports_all else pd.DataFrame()

if not attack_recall_final.empty:
    ref = (
        attack_recall_final[attack_recall_final["Condition"] == "Reference"]
        [["Model", "Class", "Mean_Recall"]]
        .rename(columns={"Mean_Recall": "Reference_Recall"})
    )
    attack_recall_final_degradation = attack_recall_final.merge(
        ref, on=["Model", "Class"], how="left"
    )
    attack_recall_final_degradation["Recall_Degradation"] = (
        attack_recall_final_degradation["Reference_Recall"] -
        attack_recall_final_degradation["Mean_Recall"]
    )
    attack_recall_final_degradation["Relative_Recall_Degradation"] = (
        attack_recall_final_degradation["Recall_Degradation"] /
        attack_recall_final_degradation["Reference_Recall"].replace(0, np.nan)
    )
else:
    attack_recall_final_degradation = pd.DataFrame()

# Use the all-condition final attack tables for downstream figures/exports.
attack_recall = attack_recall_final
attack_recall_degradation = attack_recall_final_degradation

display(repeated_summary if not repeated_summary.empty else final_condition_metrics)


# ================= NOTEBOOK CELL 29 =================
# 24. Final-paper figures — compact, reproducible, and based on final summaries
import matplotlib
matplotlib.use("Agg")   # non-interactive backend: saves files without calling plt.show()
import matplotlib.pyplot as plt

if not final_condition_metrics.empty:
    # Figure 1 — Reference vs temporal Macro-F1
    plot_df = final_condition_metrics[
        (final_condition_metrics["Task"] == "multiclass") &
        (final_condition_metrics["Condition"].isin(["Reference", "Temporal"]))
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in MODEL_NAMES:
        g = plot_df[plot_df["Model"] == model]
        ax.plot(g["Condition"], g["Macro_F1"], marker="o", label=model)
    ax.set_title("Reference vs Chronological Macro-F1")
    ax.set_ylabel("Macro-F1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure1_reference_vs_temporal_macro_f1.png", dpi=180)
    plt.close(fig)

    # Figure 2 — degradation across all reliability conditions
    d = degradation_f1[
        (degradation_f1["Task"] == "multiclass") &
        (~degradation_f1["Condition"].eq("Reference"))
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    for model in MODEL_NAMES:
        g = d[d["Model"] == model]
        ax.plot(g["Condition"], g["Degradation"], marker="o", label=model)
    ax.axhline(0, linewidth=1)
    ax.set_title("Macro-F1 Degradation by Reliability Condition")
    ax.set_ylabel("Relative degradation")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure2_degradation.png", dpi=180)
    plt.close(fig)

    # Figure 3 — temporal × missingness interaction
    if not final_interaction_results.empty:
        g = final_interaction_results[final_interaction_results["Task"] == "multiclass"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in MODEL_NAMES:
            z = g[g["Model"] == model]
            ax.plot(z["Missingness"], z["Interaction"], marker="o", label=model)
        ax.axhline(0, linewidth=1)
        ax.set_title("Temporal × Missingness Interaction")
        ax.set_ylabel("Interaction effect")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure3_interaction.png", dpi=180)
        plt.close(fig)

    # Figure 4 — worst attack-level recall degradation by condition.
    if not attack_recall_degradation.empty:
        g = attack_recall_degradation[
            attack_recall_degradation["Condition"] != "Reference"
        ].copy()
        fig, ax = plt.subplots(figsize=(12, 6))
        # Show the ten largest degradation values across all model/condition/class combinations.
        z = (
            g.sort_values("Recall_Degradation", ascending=False)
            .head(10)
        )
        labels = (
            z["Model"].astype(str) + " | " +
            z["Condition"].astype(str) + " | " +
            z["Class"].astype(str)
        )
        ax.bar(np.arange(len(z)), z["Recall_Degradation"])
        ax.set_xticks(np.arange(len(z)))
        ax.set_xticklabels(labels, rotation=75, ha="right")
        ax.set_title("Largest Attack-Class Recall Degradation")
        ax.set_ylabel("Reference Recall − Condition Recall")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure4_attack_recall_degradation.png", dpi=180)
        plt.close(fig)

    # Figure 5 — representative confusion matrix from seed-42 reference run.
    cm_candidates = results_all[
        (results_all["Task"] == "multiclass") &
        (results_all["Condition"] == "Reference")
    ].sort_values("Macro F1", ascending=False)

    if not cm_candidates.empty:
        best_row = cm_candidates.iloc[0]
        cm = np.asarray(best_row["Confusion Matrix"])
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, aspect="auto")
        ax.set_title(f"Reference Confusion Matrix — {best_row['Model']}")
        ax.set_xlabel("Predicted class index")
        ax.set_ylabel("True class index")
        fig.colorbar(im, ax=ax, label="Count")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure5_representative_confusion_matrix.png", dpi=180)
        plt.close(fig)

    # Figure 6 — five-seed Macro-F1 variability with t-based 95% CI.
    if not repeated_summary.empty and repeated_summary["N"].max() > 1:
        g = repeated_summary[
            (repeated_summary["Task"] == "multiclass") &
            (repeated_summary["Condition"].isin(["Reference", "Temporal"]))
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        for model in MODEL_NAMES:
            z = g[g["Model"] == model]
            x = np.arange(len(z))
            yerr_low = z["Mean_Macro_F1"] - z["CI95_Macro_F1_Low"]
            yerr_high = z["CI95_Macro_F1_High"] - z["Mean_Macro_F1"]
            ax.errorbar(
                x, z["Mean_Macro_F1"],
                yerr=[yerr_low, yerr_high],
                marker="o", capsize=4, label=model
            )

        conditions = sorted(g["Condition"].unique())
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions)
        ax.set_title("Repeated-Seed Macro-F1 with 95% t-Confidence Intervals")
        ax.set_ylabel("Mean Macro-F1")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure6_repeated_seed_variability.png", dpi=180)
        plt.close(fig)
else:
    print("Figures skipped because no experiment results are available.")

