# ============================================================================
# UNSW-NB15 Reliability Evaluation Pipeline — Setup & Configuration (01)
# ============================================================================

from pathlib import Path
import os
import gc
import json
import warnings
import random
import sys
import subprocess
import time
import numpy as np
import pandas as pd

# Ensure Hugging Face hub client is available for automatic dataset download
try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.24.0"])
    from huggingface_hub import snapshot_download, hf_hub_download

# Base seeds for reproducibility (matching CICIDS2017 Part A protocol)
SEED = 42
SEEDS = [42, 52, 62, 72, 82]

# ---------------- RUN MODES ----------------
# QUICK_VALIDATION: bounded training partition (e.g., 50k rows) and single seed for rapid verification
# PAPER_MODE: full verified training partition across all five seeds [42, 52, 62, 72, 82]
QUICK_VALIDATION = False
PAPER_MODE = True
RUN_EXPERIMENTS = True

FAST_TRAIN_SAMPLES = 50_000
FAST_SAMPLING_STRATIFIED = True

# Controlled test-time missingness rates
MISSINGNESS_LEVELS = (0.0, 0.10, 0.20)

# Confidence interval specification
CI_LEVEL = 0.95

# Threading constraint (conservative parallelism for local & shared runtimes)
N_JOBS = max(1, min(2, (os.cpu_count() or 2)))

# Define root directory robustly whether executed directly or via runner
_cur_file = Path(__file__).resolve()
if _cur_file.parent.name == "src":
    ROOT_DIR = _cur_file.parent.parent
else:
    ROOT_DIR = _cur_file.parent

OUTPUT_DIR = ROOT_DIR / "results" / "unsw_nb15"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- DATASET SOURCING & PATHS ----------------
LOCAL_DATA_DIR = ROOT_DIR / "data" / "unsw_nb15"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--Mouwiya--UNSW-NB15-small" / "snapshots" / "6f5f54594dfc80c84264aec4c7cc3d9b162f1e2f"
HF_SOURCE_PREFERENCE = [
    "Mouwiya/UNSW-NB15-small",
    "Mireu-Lab/UNSW-NB15",
    "lacg030175/UNSW-NB15",
]

# Forbidden proxy features: Flow duration, packet counts, bytes, or IAT must NEVER serve as fake timestamps
FORBIDDEN_TIMESTAMP_FEATURES = {
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload", "sinpkt", "dinpkt"
}

# Explicitly documented feature exclusion rationales
EXCLUDED_FEATURES_RATIONALE = {
    "id": "Synthetic monotonic row index; perfect shortcut leakage if retained.",
    "srcip": "Spatial memorization leakage; models split on specific testbed IP identities.",
    "dstip": "Spatial memorization leakage; models split on specific target lab VM addresses.",
    "sport": "Ephemeral port noise; client-side OS dynamic socket allocator artifact (high cardinality, zero intrusion semantics).",
    "dsport": "Target server port bias + continuous integer fallacy on nominal values. Note: Behavioral port context is fully retained via 6 engineered features: service, ct_srv_src, ct_srv_dst, ct_src_dport_ltm, ct_dst_sport_ltm, and is_sm_ips_ports.",
    "stime": "Temporal metadata; reserved strictly for chronological ordering / session partitioning, excluded from feature matrix X.",
    "ltime": "Temporal metadata; reserved strictly for chronological ordering, excluded from feature matrix X.",
}

CATEGORICAL_FEATURES = ["proto", "service", "state"]

# Model names
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]

# Core estimator imports with safe XGBoost verification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    print("Installing XGBoost...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True

# Reproducibility initialization
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

FINAL_MODE = PAPER_MODE
if QUICK_VALIDATION and PAPER_MODE:
    raise ValueError("Choose either QUICK_VALIDATION=True or PAPER_MODE=True, not both.")

print("=" * 80)
print("UNSW-NB15 RELIABILITY FRAMEWORK -- SETUP & CONFIGURATION")
print("=" * 80)
print("Root directory:       ", ROOT_DIR)
print("Output directory:     ", OUTPUT_DIR)
print("QUICK_VALIDATION:     ", QUICK_VALIDATION)
print("PAPER_MODE:           ", PAPER_MODE)
print("Seeds:                ", SEEDS if PAPER_MODE else [SEED])
print("Missingness levels:   ", MISSINGNESS_LEVELS)
print("N_JOBS:               ", N_JOBS)
print("XGBoost available:    ", XGB_AVAILABLE)
print("Setup complete.\n")
