# Auto-extracted from CICIDS2017_Reliability_Framework_Final_Optimized_v6_FIXED.ipynb
# Cell order is preserved intentionally for reproducibility.


# ================= NOTEBOOK CELL 1 =================
# 1. Configuration — final reliability experiment (optimized + validated)
from pathlib import Path
import os, gc, json, warnings, random, sys, subprocess
import numpy as np
import pandas as pd


# Hugging Face client — imported here before any downloader functions are defined.
try:
    from huggingface_hub import snapshot_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.24.0"])
    from huggingface_hub import snapshot_download

SEED = 42
SEEDS = [42, 52, 62, 72, 82]

# ---------------- RUN MODES ----------------
# QUICK_VALIDATION: finishes much faster and verifies the entire pipeline.
# PAPER_MODE: uses the full verified training partitions and all five seeds.
QUICK_VALIDATION = False
PAPER_MODE = True
RUN_EXPERIMENTS = True

# Bounded validation size. This is NOT used in PAPER_MODE.
FAST_TRAIN_SAMPLES = 100_000
# In quick validation, preserve approximate class proportions rather than
# using one-example-per-class plus random fill. This setting is NEVER used
# in PAPER_MODE.
FAST_SAMPLING_STRATIFIED = True

# Exact test-time stress rates used by the reliability matrix.
MISSINGNESS_LEVELS = (0.0, 0.10, 0.20)

# Statistical reporting for the five-seed final experiment.
CI_LEVEL = 0.95

# Full-data paper protocol.
# Repeated seeds are expensive, so they run only in PAPER_MODE.
REPEATED_SEEDS = SEEDS

# ---------------- DATA / MEMORY ----------------
HF_SOURCE_PREFERENCE = ["kumaranimesh1901/CICIDS-2017", "bvk/CICIDS-2017", "San0160/CICIDS-2017"]

# Robust Hugging Face networking for Colab. snapshot_download resumes interrupted transfers.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

# Never use these as temporal substitutes.
FORBIDDEN_TIMESTAMP_FEATURES = {
    "flow duration", "total tcp flow time", "total fwd packets",
    "total bwd packets", "flow iat mean", "flow iat std",
    "flow iat max", "flow iat min"
}

# Threading is intentionally conservative for shared Colab runtimes.
N_JOBS = max(1, min(2, (os.cpu_count() or 2)))
# Resolve OUTPUT_DIR relative to this file so the pipeline works outside Colab.
# Falls back to /content/reliability_outputs only when running inside Colab.
_in_colab = os.path.isdir("/content") and os.path.isdir("/usr") and sys.platform != "win32"
if _in_colab:
    OUTPUT_DIR = Path("/content/reliability_outputs")
else:
    OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)

FINAL_MODE = PAPER_MODE
if QUICK_VALIDATION and PAPER_MODE:
    raise ValueError("Choose either QUICK_VALIDATION=True or PAPER_MODE=True, not both.")

print("Configuration ready.")
print("HF source preference:", HF_SOURCE_PREFERENCE)
print("QUICK_VALIDATION:", QUICK_VALIDATION)
print("PAPER_MODE:", PAPER_MODE)
print("RUN_EXPERIMENTS:", RUN_EXPERIMENTS)
print("N_JOBS:", N_JOBS)


# ================= NOTEBOOK CELL 2 =================
# 2. Imports with a safe XGBoost fallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    print("XGBoost is not currently available. Installing it...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True

print("Core imports loaded. XGBoost available:", XGB_AVAILABLE)


# ================= NOTEBOOK CELL 3 =================
# 3. Dataset source — automatic Hugging Face import
#
# The notebook follows the same automatic import pattern as the proven FAST v4 notebook:
# huggingface_hub.snapshot_download() downloads CICIDS2017 directly into the Colab runtime.
#
# No Google Drive, manual upload, or manually entered dataset path is required.
#
# The loader accepts CICIDS2017 CSV variants with a valid Label column. If a genuine row-level
# Timestamp is present and valid, it is used. If the selected source does not preserve a usable
# row-level timestamp, temporal evaluation uses the documented CICIDS2017 capture-day/file order.
#
# Important: Flow Duration, Total TCP Flow Time, packet counts, and other flow features are
# never substituted for a timestamp.


# ================= NOTEBOOK CELL 4 =================
# 4. Automatic CICIDS2017 download — same proven pattern as the FAST v4 notebook
# No Google Drive, no manual upload, no manual dataset path.
#
# We intentionally use snapshot_download directly instead of HfApi/list_repo_files.
# This is more robust in Colab and avoids the HfApi import-order failure seen in v2/v3.
#
# Source preference:
#   1) local /content/CICIDS2017_timestamped if already present
#   2) kumaranimesh1901/CICIDS-2017
#   3) bvk/CICIDS-2017
#   4) San0160/CICIDS-2017
#
# A repository is accepted when it contains at least 5 CICIDS2017-style CSV files
# with a usable Label column. Timestamp is checked separately later.
# If no real row-level Timestamp exists, the notebook uses capture-day/file order.
# It NEVER substitutes Flow Duration / Total TCP Flow Time as a timestamp.

import time

try:
    from huggingface_hub import snapshot_download
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "huggingface_hub>=0.24.0"
    ])
    from huggingface_hub import snapshot_download

# Robust Hugging Face networking for Colab.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

def normalize_header_name(value):
    return (
        str(value)
        .replace("\ufeff", "")
        .replace("\u00a0", " ")
        .strip()
        .casefold()
    )

def find_exact_column(columns, target):
    target_norm = normalize_header_name(target)
    return next(
        (c for c in columns if normalize_header_name(c) == target_norm),
        None
    )

def discover_csv_files(data_dir):
    data_dir = Path(data_dir)
    paths = sorted(
        list(data_dir.rglob("*.csv")) +
        list(data_dir.rglob("*.CSV"))
    )
    # Remove duplicates caused by case-insensitive filesystems.
    return list(dict.fromkeys(paths))

def inspect_cicids_files(csv_paths):
    usable = []
    diagnostics = []

    for path in csv_paths:
        try:
            head = pd.read_csv(
                path,
                nrows=0,
                encoding="cp1252",
                engine="c",
                low_memory=False,
            )
            label = find_exact_column(head.columns, "Label")
            timestamp = find_exact_column(head.columns, "Timestamp")

            if label is None:
                diagnostics.append({
                    "file": str(path),
                    "status": "Missing Label",
                    "columns": list(head.columns),
                })
                continue

            usable.append({
                "path": path,
                "label": label,
                "timestamp": timestamp,
                "columns": list(head.columns),
            })
        except Exception as exc:
            diagnostics.append({
                "file": str(path),
                "status": f"{type(exc).__name__}: {exc}",
                "columns": [],
            })

    return usable, diagnostics

def download_cicids_source(repo_id, local_dir, attempts=3):
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            print(
                f"Downloading/checking {repo_id} "
                f"(attempt {attempt}/{attempts})..."
            )

            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            # This is the same direct Hugging Face import strategy as the
            # previously working FAST v4 notebook.
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                allow_patterns=["*.csv", "*.CSV"],
                ignore_patterns=[
                    "*.md", "*.txt", "*.json", "*.parquet",
                    "*.zip", "*.pcap", "*.pcapng"
                ],
            )

            csv_paths = discover_csv_files(local_dir)

            if not csv_paths:
                raise RuntimeError("No CSV files were downloaded.")

            usable, diagnostics = inspect_cicids_files(csv_paths)

            if len(usable) < 5:
                raise RuntimeError(
                    f"Only {len(usable)} CSVs contain Label. "
                    f"Diagnostics: {diagnostics[:5]}"
                )

            print(
                f"Accepted {repo_id}: {len(usable)} CSVs contain a valid Label column."
            )
            return local_dir, csv_paths, usable, diagnostics

        except Exception as exc:
            last_error = exc
            print(
                f"  Download/check failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < attempts:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Failed after {attempts} attempts: {last_error}"
    )

# First use an already-downloaded timestamped/local directory if it exists.
local_timestamped_dir = Path("/content/CICIDS2017_timestamped")
local_paths = discover_csv_files(local_timestamped_dir) if local_timestamped_dir.exists() else []

selected_repo = None
csv_candidates = []
schema_infos = []
repo_errors = {}

if local_paths:
    local_infos, local_diagnostics = inspect_cicids_files(local_paths)
    if len(local_infos) >= 5:
        selected_repo = "local:/content/CICIDS2017_timestamped"
        csv_candidates = local_paths
        schema_infos = local_infos
        print(
            f"Using existing CICIDS2017 files from {local_timestamped_dir}"
        )

# Otherwise download automatically.
if selected_repo is None:
    source_specs = [
        ("kumaranimesh1901/CICIDS-2017", "/content/CICIDS2017_timestamped"),
        ("bvk/CICIDS-2017", "/content/CICIDS2017_bvk"),
        ("San0160/CICIDS-2017", "/content/CICIDS2017_san0160"),
    ]

    for repo_id, target_dir in source_specs:
        try:
            _, paths, infos, diagnostics = download_cicids_source(
                repo_id,
                target_dir,
                attempts=3,
            )
            selected_repo = repo_id
            csv_candidates = paths
            schema_infos = infos
            break
        except Exception as exc:
            repo_errors[repo_id] = (
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"Rejected {repo_id} -> {repo_errors[repo_id]}"
            )

if selected_repo is None:
    raise RuntimeError(
        "No usable CICIDS2017 Hugging Face source was found. "
        "No manual dataset upload is allowed by this notebook. "
        f"Diagnostics: {repo_errors}"
    )

# Reinspect the final selected files so all later cells use one authoritative schema list.
schema_infos, candidate_diagnostics = inspect_cicids_files(csv_candidates)

if len(schema_infos) < 5:
    raise RuntimeError(
        f"Selected source has only {len(schema_infos)} usable Label CSVs. "
        f"Diagnostics: {candidate_diagnostics[:10]}"
    )

print("\nSelected CICIDS2017 source:", selected_repo)
print("CSV files found:", len(csv_candidates))
for p in csv_candidates:
    print(" -", p.name)


# ================= NOTEBOOK CELL 5 =================
# 5. Efficient schema discovery + schema harmonization

def normalized_columns(columns):
    return [normalize_header_name(c) for c in columns]

schema_infos = []
candidate_diagnostics = []

for path in csv_candidates:
    try:
        head = pd.read_csv(path, nrows=0, encoding="cp1252", engine="c", low_memory=False)
        label_candidate = find_exact_column(head.columns, "Label")
        timestamp_candidate = find_exact_column(head.columns, "Timestamp")

        if label_candidate is None:
            candidate_diagnostics.append({
                "file": str(path),
                "status": "Missing Label",
                "columns": list(head.columns)
            })
            continue

        schema_infos.append({
            "path": path,
            "label": label_candidate,
            "timestamp": timestamp_candidate,
            "columns": list(head.columns)
        })
    except Exception as exc:
        candidate_diagnostics.append({
            "file": str(path),
            "status": f"Header read error: {type(exc).__name__}: {exc}",
            "columns": []
        })

if not schema_infos:
    raise ValueError(
        "No CICIDS2017 CSV containing a usable Label column was loaded. "
        f"Diagnostics: {candidate_diagnostics[:10]}"
    )

print("Usable CSV files:", len(schema_infos))
print("Files with actual Timestamp:", sum(x["timestamp"] is not None for x in schema_infos))

# We do not require identical raw column names because CICIDS2017 mirrors may
# rename/remove identity fields. The model feature intersection is established
# after loading, while Label remains mandatory.


# ================= NOTEBOOK CELL 6 =================
# 5b. Dataset schema checks
for info in schema_infos:
    if normalize_header_name(info["label"]) != "label":
        raise ValueError(f"Unexpected Label column in {info['path']}.")

timestamp_candidates = [
    info["timestamp"] for info in schema_infos if info["timestamp"] is not None
]

if timestamp_candidates:
    print("At least one source file exposes an actual Timestamp column.")
else:
    print("No source file exposes a row-level Timestamp.")
    print("Temporal evaluation will therefore use verified CICIDS2017 capture-day/file order.")
    print("No duration/flow feature will be substituted as a timestamp.")

