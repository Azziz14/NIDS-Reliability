from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = [
    "src/01_setup.py",
    "src/02_data_and_temporal.py",
    "src/03_preprocessing_models.py",
    "src/04_experiments.py",
    "src/05_analysis.py",
    "src/06_repeated_and_figures.py",
    "src/07_validation_export.py",
]

def _display_shim(obj=None):
    """Fallback for display() when running outside Jupyter/Colab."""
    import pandas as pd
    import numpy as np
    if obj is None:
        return
    if isinstance(obj, pd.DataFrame):
        print(obj.to_string())
    elif isinstance(obj, pd.Series):
        print(obj.to_string())
    else:
        print(repr(obj))

# Execute extracted modules in one shared namespace because the original notebook
# intentionally builds state sequentially across cells. This preserves behavior
# while giving coding agents clean, searchable files.
shared = {
    "__file__": str(ROOT / "run_experiments.py"),
    "__name__": "__main__",
    # Provide display() so src files run identically outside Jupyter/Colab.
    "display": _display_shim,
}

for relative in MODULES:
    path = ROOT / relative
    print(f"\n\n===== RUNNING {relative} =====")
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, shared, shared)
