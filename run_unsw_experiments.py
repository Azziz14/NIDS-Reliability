# ============================================================================
# UNSW-NB15 Reliability Evaluation Framework — Main Execution Runner
# ============================================================================

from pathlib import Path
import sys

# Ensure UTF-8 clean output across Windows console runtimes
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
MODULES = [
    "src/unsw_01_setup.py",
    "src/unsw_02_data_and_temporal.py",
    "src/unsw_03_preprocessing.py",
    "src/unsw_04_experiments.py",
    "src/unsw_05_analysis.py",
    "src/unsw_06_repeated_figures.py",
    "src/unsw_07_validation_export.py",
]

def _display_shim(obj=None):
    """Fallback display helper for execution outside Jupyter/Colab environments."""
    import pandas as pd
    if obj is None:
        return
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        print(obj.to_string())
    else:
        print(repr(obj))

shared_namespace = {
    "__file__": str(ROOT / "run_unsw_experiments.py"),
    "__name__": "__main__",
    "display": _display_shim,
}

print("=" * 80)
print("STARTING UNSW-NB15 RELIABILITY EVALUATION PIPELINE")
print("=" * 80)

for rel_path in MODULES:
    full_path = ROOT / rel_path
    print(f"\n>> EXECUTING {rel_path}...")
    code = compile(full_path.read_text(encoding="utf-8"), str(full_path), "exec")
    exec(code, shared_namespace, shared_namespace)

print("\n" + "=" * 80)
print("UNSW-NB15 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 80)
