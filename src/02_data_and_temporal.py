# Auto-extracted from CICIDS2017_Reliability_Framework_Final_Optimized_v6_FIXED.ipynb
# Cell order is preserved intentionally for reproducibility.


# ================= NOTEBOOK CELL 7 =================
# 6. Temporal provenance policy
#
# The notebook uses the strongest temporal provenance actually available in the selected
# CICIDS2017 mirror:
#
# 1. Row-level mode: if every selected source file contains a real Timestamp, it is
#    parsed and validated against the July 2017 capture period, then rows are sorted by it.
# 2. Capture-day mode: if the selected mirror is the standard CICIDS2017 CSV variant
#    without row-level timestamps, temporal evaluation is performed by the original
#    capture-day/file order (Monday -> Tuesday -> Wednesday -> Thursday -> Friday).
# 3. Never: Flow Duration, Total TCP Flow Time, packet counts, IAT values, or any
#    other model feature is treated as a timestamp.
#
# This is important because several public CICIDS2017 distributions differ in whether the
# identity/timestamp fields are retained. The original eight-file organization is stable,
# and the capture-day ordering is legitimate temporal provenance when row timestamps are
# absent.

# ================= NOTEBOOK CELL 8 =================
# 6. Parse the actual Timestamp when available
def parse_cicids_timestamp(series):
    raw = (
        series.astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    candidates = []
    formats = [
        "%d/%m/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d/%m/%Y %I:%M:%S.%f %p",
        "%m/%d/%Y %I:%M:%S.%f %p",
    ]
    for fmt in formats:
        candidates.append(pd.to_datetime(raw, format=fmt, errors="coerce"))
    try:
        candidates.append(pd.to_datetime(raw, format="mixed", errors="coerce"))
    except Exception:
        # Compatible fallback for older pandas.
        candidates.append(pd.to_datetime(raw, errors="coerce"))

    validity = [int(x.notna().sum()) for x in candidates]
    best = candidates[int(np.argmax(validity))]
    return best, max(validity) / max(len(raw), 1)


# ================= NOTEBOOK CELL 9 =================
# 7. Load files one at a time, validate schema, and compact to float32 arrays

def capture_day_rank(filename):
    name = filename.casefold()
    if "monday" in name:
        return 0
    if "tuesday" in name:
        return 1
    if "wednesday" in name:
        return 2
    if "thursday" in name:
        return 3
    if "friday" in name:
        return 4
    return 99

# Sort source files by documented CICIDS2017 capture-day order, not lexicographic filename order.
schema_infos = sorted(
    schema_infos,
    key=lambda x: (capture_day_rank(x["path"].name), x["path"].name.casefold())
)

# Decide temporal provenance before loading the full feature matrix.
TIMESTAMP_VALIDITY_THRESHOLD = 0.90
TIMESTAMP_SAMPLE_ROWS = 20000
preflight_timestamp = []

has_timestamp_flags = [info["timestamp"] is not None for info in schema_infos]
ROW_TIMESTAMP_MODE = all(has_timestamp_flags)

if any(has_timestamp_flags) and not ROW_TIMESTAMP_MODE:
    print("Mixed Timestamp availability -> using capture-day temporal mode.")

if ROW_TIMESTAMP_MODE:
    print("\nPreflight: checking actual Timestamp quality...")
    for info in schema_infos:
        sample = pd.read_csv(
            info["path"], usecols=[info["timestamp"]], nrows=TIMESTAMP_SAMPLE_ROWS,
            encoding="cp1252", engine="c"
        )
        parsed_sample, sample_validity = parse_cicids_timestamp(sample[info["timestamp"]])
        valid_sample = parsed_sample.dropna()
        sample_min = valid_sample.min() if not valid_sample.empty else None
        sample_max = valid_sample.max() if not valid_sample.empty else None
        preflight_timestamp.append({
            "file": str(info["path"]), "sample_validity": float(sample_validity),
            "sample_min": None if sample_min is None else str(sample_min),
            "sample_max": None if sample_max is None else str(sample_max)
        })
        print(f"  {info['path'].name}: {sample_validity:.2%} valid")
        del sample, parsed_sample
        gc.collect()

    if any(x["sample_validity"] < TIMESTAMP_VALIDITY_THRESHOLD for x in preflight_timestamp):
        print("Actual Timestamp values are incomplete/invalid for row-level ordering.")
        print("Using whole CICIDS2017 capture-day/source-file order instead.")
        print("No timestamp will be fabricated from partial values.")
        ROW_TIMESTAMP_MODE = False
    else:
        for x in preflight_timestamp:
            for key in ("sample_min", "sample_max"):
                if x[key] is not None:
                    dt = pd.Timestamp(x[key])
                    if not (pd.Timestamp("2017-07-01") <= dt <= pd.Timestamp("2017-07-10")):
                        print("Suspicious Timestamp date range detected -> capture-day mode.")
                        ROW_TIMESTAMP_MODE = False
                        break
            if not ROW_TIMESTAMP_MODE:
                break

print("Temporal provenance selected:", "ROW_TIMESTAMP" if ROW_TIMESTAMP_MODE else "CAPTURE_DAY")

feature_parts = []
label_parts = []
timestamp_parts = []
source_file_parts = []
source_day_parts = []
loaded_files = []
file_validation = []
reference_feature_names = None

for file_id, info in enumerate(schema_infos):
    path = info["path"]
    print(f"\nReading: {path.name}")

    part = pd.read_csv(path, encoding="cp1252", low_memory=False, engine="c")

    part.columns = [
        str(c).replace("\ufeff", "").replace("\u00a0", " ").strip()
        for c in part.columns
    ]

    label_col_file = find_exact_column(part.columns, "Label")
    timestamp_col_file = find_exact_column(part.columns, "Timestamp")

    if label_col_file is None:
        raise ValueError(f"{path.name}: Label column missing after loading.")

    labels = part[label_col_file].astype("string").str.strip()
    keep = labels.notna() & (labels != "")

    parsed_ts = None
    validity = None

    if ROW_TIMESTAMP_MODE:
        parsed_ts, validity = parse_cicids_timestamp(part[timestamp_col_file])
        print(f"  Timestamp parse validity: {validity:.2%}")

        if validity < 0.90:
            raise AssertionError("Timestamp preflight selected row-level mode, but full-load validity is below threshold.")

        valid_ts = parsed_ts.dropna()
        if valid_ts.empty:
            raise ValueError(f"{path.name}: no valid timestamps remain.")

        fmin, fmax = valid_ts.min(), valid_ts.max()
        if not (
            pd.Timestamp("2017-07-01") <= fmin <= pd.Timestamp("2017-07-10")
            and pd.Timestamp("2017-07-01") <= fmax <= pd.Timestamp("2017-07-10")
        ):
            raise ValueError(
                f"{path.name}: suspicious timestamp range {fmin} -> {fmax}."
            )
        keep &= parsed_ts.notna()
    else:
        fmin = fmax = None
        print("  Row-level Timestamp: unavailable; using capture-day provenance.")

    feature_cols = [
        c for c in part.columns
        if c != label_col_file and c != timestamp_col_file
    ]

    numeric_dict = {}
    for c in feature_cols:
        values = pd.to_numeric(part[c], errors="coerce")
        if values.notna().any():
            numeric_dict[c] = values.astype(np.float32)

    if reference_feature_names is None:
        reference_feature_names = list(numeric_dict.keys())
    elif set(numeric_dict.keys()) != set(reference_feature_names):
        # CICIDS2017 mirrors legitimately differ by a small number of columns
        # (e.g. the Inbound column is absent in some daily files). Reduce to the
        # common intersection rather than aborting. A hard error is only raised
        # when the intersection is empty, which indicates a schema mismatch that
        # cannot be recovered.
        common = [c for c in reference_feature_names if c in numeric_dict]
        only_in_new = set(numeric_dict.keys()) - set(reference_feature_names)
        only_in_ref = set(reference_feature_names) - set(numeric_dict.keys())
        if only_in_ref or only_in_new:
            print(
                f"  Schema difference in {path.name}: "
                f"{len(only_in_ref)} cols dropped (absent in this file), "
                f"{len(only_in_new)} cols ignored (absent in reference). "
                f"Intersection: {len(common)} features."
            )
        if not common:
            raise ValueError(
                f"{path.name}: numeric feature intersection with reference is empty."
            )
        # Narrow reference feature names to the intersection so all subsequent
        # files and the final matrix stay consistent.
        reference_feature_names = common

    X_part = pd.DataFrame(
        {c: numeric_dict[c] for c in reference_feature_names},
        index=part.index
    ).loc[keep]

    X_arr = X_part.to_numpy(dtype=np.float32, copy=True)
    X_arr[~np.isfinite(X_arr)] = np.nan

    feature_parts.append(X_arr)
    label_parts.append(labels.loc[keep].to_numpy(dtype=str))
    source_file_parts.append(np.full(int(keep.sum()), file_id, dtype=np.int16))
    source_day_parts.append(np.full(
        int(keep.sum()), capture_day_rank(path.name), dtype=np.int8
    ))

    if ROW_TIMESTAMP_MODE:
        timestamp_parts.append(
            parsed_ts.loc[keep].to_numpy(dtype="datetime64[ns]")
        )

    loaded_files.append(str(path))
    file_validation.append({
        "file": str(path),
        "rows_loaded": int(keep.sum()),
        "capture_day": capture_day_rank(path.name),
        "timestamp_validity": None if validity is None else float(validity),
        "timestamp_min": None if fmin is None else str(fmin),
        "timestamp_max": None if fmax is None else str(fmax),
    })

    print("  Retained rows:", int(keep.sum()))
    print("  Features:", X_arr.shape[1])

    del part, numeric_dict, X_part, X_arr, parsed_ts, labels
    gc.collect()

X_matrix = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
y_multiclass = np.concatenate(label_parts)
source_file_ids = np.concatenate(source_file_parts)
source_day_ids = np.concatenate(source_day_parts)

if ROW_TIMESTAMP_MODE:
    timestamps = np.concatenate(timestamp_parts)
else:
    timestamps = None

del feature_parts, label_parts, source_file_parts, source_day_parts
gc.collect()

timestamp_col = "Timestamp" if ROW_TIMESTAMP_MODE else None
label_col = "Label"
numeric_features = reference_feature_names

print("\nCombined shape:", X_matrix.shape)
print("Classes:", len(np.unique(y_multiclass)))
print("Temporal provenance mode:", "ROW_TIMESTAMP" if ROW_TIMESTAMP_MODE else "CAPTURE_DAY")
if ROW_TIMESTAMP_MODE:
    print("Timestamp range:", timestamps.min(), "->", timestamps.max())


# ================= NOTEBOOK CELL 10 =================
# 8. Compact data cleaning
# Exact duplicate removal is performed with row hashes in chunks.
# Timestamp/source provenance is included in the hash when available.

n_before = len(y_multiclass)
CHUNK = 200_000
hash_parts = []

for start in range(0, n_before, CHUNK):
    end = min(start + CHUNK, n_before)

    frame = pd.DataFrame(X_matrix[start:end], columns=numeric_features)
    frame["_label"] = y_multiclass[start:end]
    frame["_file_id"] = source_file_ids[start:end]
    frame["_day_id"] = source_day_ids[start:end]

    if ROW_TIMESTAMP_MODE:
        frame["_timestamp"] = timestamps[start:end]

    h = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    hash_parts.append(h)

    del frame, h
    gc.collect()

row_hashes = np.concatenate(hash_parts)
del hash_parts
gc.collect()

_, unique_idx = np.unique(row_hashes, return_index=True)
unique_idx.sort()

if len(unique_idx) < n_before:
    X_matrix = X_matrix[unique_idx]
    y_multiclass = y_multiclass[unique_idx]
    source_file_ids = source_file_ids[unique_idx]
    source_day_ids = source_day_ids[unique_idx]
    if ROW_TIMESTAMP_MODE:
        timestamps = timestamps[unique_idx]

print("Duplicate rows removed:", n_before - len(unique_idx))
print("Rows after cleaning:", len(y_multiclass))
print("Classes:", len(np.unique(y_multiclass)))

del row_hashes, unique_idx
gc.collect()


# ================= NOTEBOOK CELL 11 =================
# 9. Final numeric feature matrix
if X_matrix.ndim != 2 or X_matrix.shape[0] != len(y_multiclass):
    raise ValueError("Feature/label row alignment is invalid.")

if source_file_ids.shape[0] != len(y_multiclass):
    raise ValueError("Source-file provenance is misaligned with features/labels.")

# Remove columns that are entirely missing.
all_missing = np.all(np.isnan(X_matrix), axis=0)
if all_missing.any():
    removed_features = [
        numeric_features[i] for i, flag in enumerate(all_missing) if flag
    ]
    X_matrix = X_matrix[:, ~all_missing]
    numeric_features = [
        c for i, c in enumerate(numeric_features) if not all_missing[i]
    ]
else:
    removed_features = []

X_matrix[~np.isfinite(X_matrix)] = np.nan

print("Numeric features retained:", len(numeric_features))
print("Completely missing features removed:", len(removed_features))
print("Raw missing cells:", int(np.isnan(X_matrix).sum()))
print("Matrix dtype:", X_matrix.dtype)

if len(numeric_features) == 0:
    raise ValueError("No usable numeric features remain.")


# ================= NOTEBOOK CELL 12 =================
# 10. Class summary and chronological ordering
class_counts = pd.Series(y_multiclass).value_counts()
print("Number of classes:", len(class_counts))
display(class_counts.to_frame("count").head(30))

if ROW_TIMESTAMP_MODE:
    ordered_idx = np.argsort(timestamps, kind="mergesort")
    temporal_order_description = "Actual row-level Timestamp"
else:
    # Preserve whole capture-day/file provenance. Rows inside a source file are
    # intentionally NOT given a fabricated within-file timestamp.
    ordered_idx = np.lexsort((
        np.arange(len(y_multiclass), dtype=np.int64),
        source_file_ids
    ))
    temporal_order_description = "CICIDS2017 capture-day/source-file order"

y_sorted = y_multiclass[ordered_idx]

if ROW_TIMESTAMP_MODE:
    timestamps_sorted = timestamps[ordered_idx]
    if not np.all(timestamps_sorted[:-1] <= timestamps_sorted[1:]):
        raise AssertionError("Chronological sorting failed.")
    print("\nActual timestamp range:")
    print(timestamps_sorted[0], "->", timestamps_sorted[-1])
else:
    timestamps_sorted = None
    print("\nTemporal provenance:", temporal_order_description)
    print("Source files in temporal order:")
    for fid in np.unique(source_file_ids[ordered_idx]):
        print(" ", fid, Path(loaded_files[int(fid)]).name)

print("Temporal order verified.")

