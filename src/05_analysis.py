# Auto-extracted from CICIDS2017_Reliability_Framework_Final_Optimized_v6_FIXED.ipynb
# Cell order is preserved intentionally for reproducibility.


# ================= NOTEBOOK CELL 24 =================
# 20. Degradation analysis
# `degradation_f1`, `degradation_recall`, and `degradation_fpr` are finalized
# after repeated-seed aggregation in Cell 23. This cell provides a quick
# seed-42 view for validation runs.
def degradation_table(results, metric="Macro F1"):
    base = (
        results[results["Condition"] == "Reference"]
        [["Task", "Model", metric]]
        .rename(columns={metric: "Reference"})
    )
    merged = results.merge(base, on=["Task", "Model"], how="left")

    if metric == "FPR":
        merged["Degradation"] = (
            (merged[metric] - merged["Reference"]) /
            merged["Reference"].replace(0, np.nan)
        )
    else:
        merged["Degradation"] = (
            (merged["Reference"] - merged[metric]) /
            merged["Reference"].replace(0, np.nan)
        )

    return merged[
        ["Task", "Model", "Condition", metric, "Reference", "Degradation", "Seed"]
    ].copy()

if not results_all.empty:
    primary_degradation_f1 = degradation_table(results_all, "Macro F1")
    primary_degradation_recall = degradation_table(results_all, "Macro Recall")
    primary_degradation_fpr = degradation_table(results_all, "FPR")
    print("Seed-42 Macro-F1 degradation preview:")
    display(primary_degradation_f1)
else:
    primary_degradation_f1 = pd.DataFrame()
    primary_degradation_recall = pd.DataFrame()
    primary_degradation_fpr = pd.DataFrame()


# ================= NOTEBOOK CELL 25 =================
# 21. Interaction analysis
# final_interaction_results is computed in 06_repeated_and_figures.py after the
# repeated-seed aggregation completes. A display of that table is emitted there,
# immediately after the variable is defined.
# This placeholder preserves the original cell boundary marker only.
print("Interaction analysis: results will be displayed after repeated-seed aggregation (06).")


# ================= NOTEBOOK CELL 26 =================
# 22. Attack-level recall analysis across ALL reliability conditions
def attack_recall_table(reports):
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
    if attack_df.empty:
        return pd.DataFrame()

    base = (
        attack_df[attack_df["Condition"] == "Reference"]
        [["Model", "Class", "Recall"]]
        .rename(columns={"Recall": "Reference Recall"})
    )

    out = attack_df.merge(base, on=["Model", "Class"], how="left")
    out["Recall Degradation"] = (
        out["Reference Recall"] - out["Recall"]
    )
    out["Relative Recall Degradation"] = (
        out["Recall Degradation"] /
        out["Reference Recall"].replace(0, np.nan)
    )
    return out

attack_recall = attack_recall_table(reports_all) if reports_all else pd.DataFrame()
attack_recall_degradation = (
    attack_recall_condition_degradation(attack_recall)
    if not attack_recall.empty else pd.DataFrame()
)

if not attack_recall_degradation.empty:
    display(
        attack_recall_degradation[
            attack_recall_degradation["Condition"] != "Reference"
        ].sort_values(
            ["Model", "Condition", "Recall Degradation"],
            ascending=[True, True, False]
        ).head(60)
    )
else:
    print("Attack-level condition table will populate after the main experiment.")

