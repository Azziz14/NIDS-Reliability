"""
CICIDS2017 Quick-Run Output Validation & Temporal Diagnosis

This script audits the results from the most recent quick-validation run
without re-running the full experiment.

It inspects:
1. Generated output files and their completeness
2. Schema alignment and feature handling
3. Temporal split diagnostics (classes, train/test counts)
4. Experimental condition coverage
5. Metric calculations
6. Attack-level analysis coverage

Usage: python diagnostic_audit.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

print("=" * 80)
print("NIDS RELIABILITY PROJECT — QUICK-RUN DIAGNOSTIC AUDIT")
print("=" * 80)
print()

# ============================================================================
# 1. OUTPUT FILE INVENTORY
# ============================================================================
print("1. OUTPUT FILE INVENTORY")
print("-" * 80)

expected_outputs = [
    "all_results_seed42.csv",
    "final_condition_metrics.csv",
    "degradation_macro_f1.csv",
    "degradation_macro_recall.csv",
    "degradation_fpr.csv",
    "interaction_results.csv",
    "attack_level_recall.csv",
    "attack_recall_degradation.csv",
    "validation.json",
    "experiment_config.json",
    "dataset_file_validation.json",
    "figure1_reference_vs_temporal_macro_f1.png",
]

found = []
missing = []

for fname in expected_outputs:
    fpath = RESULTS / fname
    if fpath.exists():
        found.append(fname)
        size = fpath.stat().st_size
        print(f"  [OK]  {fname:45s} {size:>10,} bytes")
    else:
        missing.append(fname)
        print(f"  [MISS] {fname}")

print()
print(f"Found: {len(found)}/{len(expected_outputs)}")
if missing:
    print(f"Missing: {missing}")
    print()
    print("WARNING: Export stage may have failed or run wrote to /content/ (Colab mode).")
    print("         Re-run with OUTPUT_DIR correctly set to results/")
    print()

# ============================================================================
# 2. VALIDATION JSON — DATASET & PROVENANCE
# ============================================================================
print()
print("2. VALIDATION JSON — DATASET & PROVENANCE")
print("-" * 80)

val_path = RESULTS / "validation.json"
if val_path.exists():
    with open(val_path, "r") as f:
        validation = json.load(f)
    
    print(f"Dataset source:       {validation.get('dataset_source')}")
    print(f"Temporal provenance:  {validation.get('temporal_provenance_mode')}")
    print(f"Timestamp column:     {validation.get('timestamp_column')}")
    print(f"Total rows:           {validation.get('rows_final'):,}")
    print(f"Features:             {validation.get('numeric_features')}")
    print(f"Classes:              {validation.get('classes')}")
    print(f"Quick validation:     {validation.get('quick_validation')}")
    print(f"Paper mode:           {validation.get('paper_mode')}")
    print(f"Train sample limit:   {validation.get('fast_train_samples'):,}")
    print(f"Seeds:                {validation.get('seeds')}")
    print(f"Status:               {validation.get('validation_status')}")
else:
    print("  [MISS] validation.json not found — export failed.")
    validation = None

# ============================================================================
# 3. EXPERIMENT CONFIG
# ============================================================================
print()
print("3. EXPERIMENT CONFIG")
print("-" * 80)

cfg_path = RESULTS / "experiment_config.json"
if cfg_path.exists():
    with open(cfg_path, "r") as f:
        config = json.load(f)
    
    print(f"Models:       {config.get('models')}")
    print(f"Tasks:        {config.get('tasks')}")
    print(f"Conditions:   {len(config.get('conditions', []))}")
    for cond in config.get("conditions", []):
        print(f"  - {cond}")
    print(f"Missingness:  {config.get('missingness_levels')}")
else:
    print("  [MISS] experiment_config.json not found.")
    config = None

# ============================================================================
# 4. RESULTS COVERAGE — ALL CONDITIONS & MODELS
# ============================================================================
print()
print("4. RESULTS COVERAGE — ALL CONDITIONS & MODELS")
print("-" * 80)

results_path = RESULTS / "all_results_seed42.csv"
if results_path.exists():
    results = pd.read_csv(results_path)
    print(f"Total result rows:  {len(results)}")
    print()
    
    print("Task coverage:")
    print(results["Task"].value_counts().to_string())
    print()
    
    print("Model coverage:")
    print(results["Model"].value_counts().to_string())
    print()
    
    print("Condition coverage:")
    print(results["Condition"].value_counts().to_string())
    print()
    
    print("Seeds:")
    print(results["Seed"].unique())
    print()
    
    # Expected: 2 tasks × 3 models × 6 conditions = 36 rows
    expected_rows = 2 * 3 * 6
    if len(results) == expected_rows:
        print(f"✓ Row count matches expected: {expected_rows}")
    else:
        print(f"⚠ Row count mismatch: expected {expected_rows}, got {len(results)}")
else:
    print("  [MISS] all_results_seed42.csv not found.")
    results = None

# ============================================================================
# 5. MULTICLASS TEMPORAL PERFORMANCE
# ============================================================================
print()
print("5. MULTICLASS TEMPORAL PERFORMANCE")
print("-" * 80)

if results is not None:
    mc_temporal = results[
        (results["Task"] == "multiclass") &
        (results["Condition"] == "Temporal")
    ][["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "FPR"]]
    
    if not mc_temporal.empty:
        print("Multiclass temporal results:")
        print(mc_temporal.to_string(index=False))
        print()
        
        min_f1 = mc_temporal["Macro F1"].min()
        max_f1 = mc_temporal["Macro F1"].max()
        print(f"Macro-F1 range: {min_f1:.4f} to {max_f1:.4f}")
        
        if min_f1 < 0.30:
            print("⚠ WARNING: Very low Macro-F1 detected in temporal evaluation.")
            print("  This may indicate:")
            print("  - Unseen classes in temporal test (legitimate dataset characteristic)")
            print("  - Feature schema mismatch (check schema intersection log)")
            print("  - Temporal leakage or incorrect split")
        else:
            print("✓ Macro-F1 within reasonable range.")
    else:
        print("  [MISS] No multiclass temporal results found.")

# ============================================================================
# 6. ATTACK-LEVEL RECALL ANALYSIS
# ============================================================================
print()
print("6. ATTACK-LEVEL RECALL ANALYSIS — UNSEEN CLASSES")
print("-" * 80)

attack_path = RESULTS / "attack_level_recall.csv"
if attack_path.exists():
    attack = pd.read_csv(attack_path)
    
    print(f"Total attack-level rows: {len(attack)}")
    print(f"Unique classes: {attack['Class'].nunique()}")
    print()
    
    # Identify zero-recall classes in temporal (indicating unseen)
    temporal_zero = attack[
        (attack["Condition"] == "Temporal") &
        (attack["Mean_Recall"] == 0.0)
    ]
    
    if not temporal_zero.empty:
        print(f"Classes with zero temporal recall: {len(temporal_zero['Class'].unique())}")
        print("These are likely absent from temporal training:")
        for cls in temporal_zero["Class"].unique():
            print(f"  - {cls}")
        print()
    
    # Reference performance for those classes
    if not temporal_zero.empty:
        unseen_classes = temporal_zero["Class"].unique()
        ref_unseen = attack[
            (attack["Condition"] == "Reference") &
            (attack["Class"].isin(unseen_classes))
        ][["Model", "Class", "Mean_Recall"]]
        
        if not ref_unseen.empty:
            print("Reference performance on those classes:")
            print(ref_unseen.to_string(index=False))
            print()
else:
    print("  [MISS] attack_level_recall.csv not found.")

# ============================================================================
# 7. DEGRADATION ANALYSIS
# ============================================================================
print()
print("7. DEGRADATION ANALYSIS — TEMPORAL VS REFERENCE")
print("-" * 80)

deg_path = RESULTS / "degradation_macro_f1.csv"
if deg_path.exists():
    deg = pd.read_csv(deg_path)
    
    temporal_deg = deg[
        (deg["Task"] == "multiclass") &
        (deg["Condition"] == "Temporal")
    ][["Model", "Macro_F1", "Reference", "Degradation"]]
    
    if not temporal_deg.empty:
        print("Temporal degradation (Macro-F1):")
        print(temporal_deg.to_string(index=False))
        print()
        
        max_deg = temporal_deg["Degradation"].max()
        if max_deg > 0.50:
            print(f"⚠ Large degradation detected: {max_deg:.2%}")
            print("  Check for unseen classes or temporal distribution shift.")
        else:
            print(f"✓ Degradation within expected range: max {max_deg:.2%}")
    else:
        print("  [MISS] No temporal degradation data found.")
else:
    print("  [MISS] degradation_macro_f1.csv not found.")

# ============================================================================
# 8. INTERACTION ANALYSIS
# ============================================================================
print()
print("8. INTERACTION ANALYSIS — TEMPORAL × MISSINGNESS")
print("-" * 80)

int_path = RESULTS / "interaction_results.csv"
if int_path.exists():
    interaction = pd.read_csv(int_path)
    
    mc_interaction = interaction[interaction["Task"] == "multiclass"]
    
    if not mc_interaction.empty:
        print("Multiclass interaction effects:")
        print(mc_interaction.to_string(index=False))
        print()
        
        # Expected: 3 models × 2 missingness levels = 6 rows
        if len(mc_interaction) == 6:
            print("✓ Complete interaction coverage.")
        else:
            print(f"⚠ Interaction coverage incomplete: expected 6 rows, got {len(mc_interaction)}")
    else:
        print("  [MISS] No multiclass interaction results found.")
else:
    print("  [MISS] interaction_results.csv not found.")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print()
print("=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)
print()

issues = []
ok = []

if missing:
    issues.append(f"Missing {len(missing)} expected output files")
else:
    ok.append("All expected output files present")

if validation and validation.get("validation_status") == "PASS":
    ok.append("Validation assertions passed")
else:
    issues.append("Validation status not confirmed")

if results is not None and len(results) == 36:
    ok.append("Complete condition×model×task coverage")
else:
    issues.append("Incomplete experimental coverage")

if results is not None:
    min_temporal_f1 = results[
        (results["Task"] == "multiclass") &
        (results["Condition"] == "Temporal")
    ]["Macro F1"].min()
    
    if min_temporal_f1 < 0.30:
        issues.append(f"Very low temporal Macro-F1: {min_temporal_f1:.4f} — investigate unseen classes")

print("✓ VERIFIED:")
for item in ok:
    print(f"  - {item}")

if issues:
    print()
    print("⚠ REQUIRES ATTENTION:")
    for item in issues:
        print(f"  - {item}")

print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()

if missing:
    print("1. Re-run quick validation to generate missing outputs:")
    print("   python run_experiments.py")
    print()

if not issues:
    print("Quick validation passed all checks.")
    print()
    print("Ready for PAPER_MODE five-seed run:")
    print("  1. Edit src/01_setup.py: QUICK_VALIDATION = False, PAPER_MODE = True")
    print("  2. Confirm FAST_TRAIN_SAMPLES is not artificially limiting training")
    print("  3. Estimate runtime: ~5x the quick-validation time")
    print("  4. Run: python run_experiments.py")
else:
    print("Address the issues above before launching the full five-seed experiment.")

print()
print("Audit complete.")
